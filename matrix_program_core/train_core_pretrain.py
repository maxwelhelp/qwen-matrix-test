#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pretrain the actual UniversalMatrixProgramCore on synthetic program tasks.

This is the corrected pretrain: the checkpoint contains `core.state_dict()`, not a
separate decoder. The same core is later loaded by transfer_audio_v4.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from matrix_program_core.universal_core import (  # noqa: E402
    PHASES,
    PRIMITIVES,
    ProgramCoreConfig,
    UniversalMatrixProgramCore,
    core_skill_loss,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


class SyntheticEvidenceProgramTask(Dataset):
    """Synthetic task with evidence/context, not only final W.

    Each sample has class-dependent evidence patterns. The labels force the same
    core to learn useful primitive/transition/pair-flow priors and execution.
    """

    def __init__(self, n: int, classes: int, evidence_cells: int, dim: int, seed: int = 0, noise: float = 0.08):
        self.n = int(n)
        self.classes = int(classes)
        self.evidence_cells = int(evidence_cells)
        self.dim = int(dim)
        self.seed = int(seed)
        self.noise = float(noise)
        g = torch.Generator().manual_seed(seed + 100)
        self.class_proto = F.normalize(torch.randn(classes, dim, generator=g), dim=-1)
        self.pos = torch.linspace(0, 1, evidence_cells).view(evidence_cells, 1)
        self.freq_bank = torch.linspace(1.0, 5.0, dim).view(1, dim)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y = int(idx % self.classes)
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + int(idx))
        phase = 0.17 * float(idx % 23)
        wave = torch.sin(2 * math.pi * (1.0 + y % 5) * self.pos + phase * self.freq_bank)
        local = torch.cos(2 * math.pi * (1.0 + (y // 2) % 4) * self.pos * self.freq_bank)
        proto = self.class_proto[y].view(1, self.dim)
        evidence = 0.55 * proto + 0.30 * wave + 0.15 * local
        # class-specific bump forces block/phase context, not only global mean.
        center = (y + 1) / float(self.classes + 1)
        bump = torch.exp(-((self.pos - center) ** 2) / 0.015)
        evidence = evidence + 0.35 * bump * proto
        evidence = evidence + self.noise * torch.randn(self.evidence_cells, self.dim, generator=g)
        evidence = (evidence - evidence.mean(dim=0, keepdim=True)) / evidence.std(dim=0, keepdim=True).clamp_min(1e-4)
        # Regression target: a stable class prototype transformed by evidence stats.
        target_vec = F.normalize(proto.squeeze(0) + 0.25 * evidence.mean(dim=0), dim=0)
        return evidence.float(), torch.tensor(y, dtype=torch.long), target_vec.float()


class SyntheticProgramHead(nn.Module):
    def __init__(self, dim: int, classes: int, dropout: float = 0.05):
        super().__init__()
        self.query = nn.Parameter(torch.randn(classes, dim) * 0.04)
        self.value = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.cls = nn.Linear(dim, classes)
        self.recon = nn.Linear(dim, dim)

    def forward(self, slots: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.query.to(device=slots.device, dtype=slots.dtype)
        score = torch.einsum("cd,bsd->bcs", q, slots) / math.sqrt(slots.shape[-1])
        attn = torch.softmax(score.float(), dim=-1).to(slots.dtype)
        read = torch.einsum("bcs,bsd->bcd", attn, self.value(slots))
        pooled = self.norm(read.mean(dim=1))
        pooled = self.drop(pooled)
        return self.cls(pooled), self.recon(pooled), attn.detach()


def build_skill_targets(steps_total: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    p = len(PRIMITIVES)
    idx = {n: i for i, n in enumerate(PRIMITIVES)}
    prim = torch.full((steps_total, p), 0.03, device=device)
    pair = torch.full((steps_total, p, p), 0.005, device=device)

    def add(t: int, names, weight: float = 1.0):
        for n in names:
            if n in idx:
                prim[t, idx[n]] += weight

    def link(t: int, a: str, b: str, weight: float = 1.0):
        if a in idx and b in idx:
            pair[t, idx[a], idx[b]] += weight

    for t in range(steps_total):
        phase = PHASES[min(t // 2, len(PHASES) - 1)]
        if phase == "extract":
            add(t, ["ctx_matrix", "channel_butterfly", "phase_matrix"], 0.8)
            link(t, "ctx_matrix", "channel_butterfly", 1.0)
            link(t, "channel_butterfly", "phase_matrix", 0.8)
        elif phase == "compare":
            add(t, ["low_rank", "product_gate", "phase_matrix"], 0.8)
            link(t, "low_rank", "product_gate", 1.0)
            link(t, "ctx_matrix", "phase_matrix", 0.7)
        elif phase == "suppress":
            add(t, ["block_butterfly", "product_gate", "phase_matrix"], 0.8)
            link(t, "block_butterfly", "phase_matrix", 1.0)
            link(t, "product_gate", "phase_matrix", 0.8)
        else:
            add(t, ["block_butterfly", "channel_butterfly", "phase_matrix"], 0.8)
            link(t, "phase_matrix", "block_butterfly", 1.0)
            link(t, "block_butterfly", "channel_butterfly", 0.8)
    prim = prim / prim.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    pair = pair / pair.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    return prim, pair


def evaluate(core, head, loader, device, dtype, args, prim_t, pair_t) -> Dict[str, float]:
    core.eval(); head.eval()
    correct = 0; n = 0; loss_sum = 0.0; skill_sum = 0.0
    use_amp = device.startswith("cuda") and dtype != torch.float32
    with torch.no_grad():
        for evidence, y, target_vec in loader:
            evidence = evidence.to(device)
            y = y.to(device)
            target_vec = target_vec.to(device)
            with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=use_amp):
                slots, aux = core(evidence)
                logits, vec, _ = head(slots)
                ce = F.cross_entropy(logits.float(), y)
                rec = F.mse_loss(F.normalize(vec.float(), dim=-1), target_vec.float())
                skill = core_skill_loss(aux, prim_t, pair_t, args.lambda_pair)
                loss = ce + args.lambda_recon * rec + args.lambda_skill * skill
            bs = y.numel()
            correct += int((logits.argmax(-1) == y).sum().cpu())
            n += bs
            loss_sum += float(loss.detach().cpu()) * bs
            skill_sum += float(skill.detach().cpu()) * bs
    return {"loss": loss_sum / max(1, n), "acc": correct / max(1, n), "skill_loss": skill_sum / max(1, n)}


def run(args) -> None:
    set_seed(args.seed)
    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(args.amp, torch.float32)
    out = ensure_dir(args.out_dir)

    cfg = ProgramCoreConfig(
        dim=args.dim,
        evidence_cells=args.evidence_cells,
        layers=args.layers,
        blocks=args.blocks,
        steps=args.steps,
        variants=args.variants,
        channel_stages=args.channel_stages,
        dropout=args.dropout,
        use_deltas=True,
    )
    core = UniversalMatrixProgramCore(cfg).to(device)
    head = SyntheticProgramHead(args.dim, args.classes, args.head_dropout).to(device)

    if args.init_core:
        ckpt = torch.load(args.init_core, map_location=device)
        state = ckpt.get("core", ckpt)
        missing, unexpected = core.load_state_dict(state, strict=False)
        print(f"loaded init core: {args.init_core} missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    train_ds = SyntheticEvidenceProgramTask(args.train_n, args.classes, args.evidence_cells, args.dim, args.seed, args.noise)
    val_ds = SyntheticEvidenceProgramTask(args.val_n, args.classes, args.evidence_cells, args.dim, args.seed + 999, args.noise)
    train = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=args.pin_memory)
    val = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.workers, pin_memory=args.pin_memory)
    prim_t, pair_t = build_skill_targets(args.layers * args.steps, torch.device(device))

    params = list(core.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda") and dtype == torch.float16)
    use_amp = device.startswith("cuda") and dtype != torch.float32

    fields = ["epoch", "train_loss", "train_ce", "train_acc", "skill_loss", "val_loss", "val_acc", "val_skill_loss", "best_acc"]
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()
    best = -1.0
    best_epoch = 0
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        core.train(); head.train()
        total = 0.0; ce_total = 0.0; skill_total = 0.0; correct = 0; n = 0
        for step, (evidence, y, target_vec) in enumerate(train, 1):
            evidence = evidence.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            target_vec = target_vec.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=use_amp):
                slots, aux = core(evidence)
                logits, vec, _ = head(slots)
                ce = F.cross_entropy(logits.float(), y)
                rec = F.mse_loss(F.normalize(vec.float(), dim=-1), target_vec.float())
                skill = core_skill_loss(aux, prim_t, pair_t, args.lambda_pair)
                loss = ce + args.lambda_recon * rec + args.lambda_skill * skill
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            scaler.step(opt); scaler.update()
            bs = y.numel()
            total += float(loss.detach().cpu()) * bs
            ce_total += float(ce.detach().cpu()) * bs
            skill_total += float(skill.detach().cpu()) * bs
            correct += int((logits.argmax(-1) == y).sum().detach().cpu())
            n += bs
            if args.log_every and step % args.log_every == 0:
                print(f"epoch {ep:03d} step {step:05d} loss={total/max(1,n):.4f} ce={ce_total/max(1,n):.4f} acc={100*correct/max(1,n):.2f}% skill={skill_total/max(1,n):.4f}", flush=True)
        va = evaluate(core, head, val, device, dtype, args, prim_t, pair_t)
        train_acc = correct / max(1, n)
        if va["acc"] > best:
            best = va["acc"]; best_epoch = ep
            torch.save({"core": core.state_dict(), "head": head.state_dict(), "config": cfg.__dict__, "args": vars(args), "best_acc": best, "epoch": ep, "primitives": list(PRIMITIVES)}, out / "core_pretrain_best.pt")
        torch.save({"core": core.state_dict(), "head": head.state_dict(), "config": cfg.__dict__, "args": vars(args), "best_acc": best, "epoch": ep, "primitives": list(PRIMITIVES)}, out / "core_pretrain_last.pt")
        row = {
            "epoch": ep,
            "train_loss": total / max(1, n),
            "train_ce": ce_total / max(1, n),
            "train_acc": train_acc,
            "skill_loss": skill_total / max(1, n),
            "val_loss": va["loss"],
            "val_acc": va["acc"],
            "val_skill_loss": va["skill_loss"],
            "best_acc": best,
        }
        with (out / "metrics.csv").open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        print(f"epoch {ep:03d}/{args.epochs} train={row['train_loss']:.4f}/{100*train_acc:.2f}% val={va['loss']:.4f}/{100*va['acc']:.2f}% skill={row['skill_loss']:.4f} best={100*best:.2f}%@{best_epoch}", flush=True)

    write_json(out / "final_report.json", {"best_acc": best, "best_epoch": best_epoch, "elapsed_sec": time.time() - t0, "config": cfg.__dict__, "args": vars(args)})


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="matrix_program_core/runs/core_pretrain_debug")
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", choices=["fp16", "bf16", "fp32", "off"], default="bf16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--classes", type=int, default=10)
    p.add_argument("--train-n", type=int, default=12000)
    p.add_argument("--val-n", type=int, default=2000)
    p.add_argument("--dim", type=int, default=96)
    p.add_argument("--evidence-cells", type=int, default=48)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--variants", type=int, default=3)
    p.add_argument("--channel-stages", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.04)
    p.add_argument("--head-dropout", type=float, default=0.05)
    p.add_argument("--noise", type=float, default=0.08)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=0.75)
    p.add_argument("--lambda-skill", type=float, default=0.35)
    p.add_argument("--lambda-pair", type=float, default=0.45)
    p.add_argument("--lambda-recon", type=float, default=0.20)
    p.add_argument("--init-core", default="")
    p.add_argument("--log-every", type=int, default=25)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
