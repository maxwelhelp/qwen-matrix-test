#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pretrain Factorized Matrix Program AssemblerCore.

This trains the actual transferable assembly logic:

    evidence -> MatrixProgramAssemblerCore -> synthetic head

The saved checkpoint contains `assembler_core`, not a separate decoder.
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

from matrix_program_core.assembler_core import (  # noqa: E402
    PHASES,
    AssemblerConfig,
    MatrixProgramAssemblerCore,
    assembler_skill_loss,
)
from simple_butterfly_matrix.simple_butterfly_matrix import PRIMITIVES  # noqa: E402


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


class SyntheticAssemblerTask(Dataset):
    """Synthetic task with evidence and execution target.

    The target class depends on structured evidence so the core cannot solve it by
    a scalar shortcut only. Flow targets are global program-prior supervision.
    """

    def __init__(self, n: int, classes: int, evidence_cells: int, dim: int, seed: int = 0, noise: float = 0.08):
        self.n = int(n)
        self.classes = int(classes)
        self.evidence_cells = int(evidence_cells)
        self.dim = int(dim)
        self.seed = int(seed)
        self.noise = float(noise)
        g = torch.Generator().manual_seed(seed + 777)
        self.class_proto = F.normalize(torch.randn(classes, dim, generator=g), dim=-1)
        self.pos = torch.linspace(0.0, 1.0, evidence_cells).view(evidence_cells, 1)
        self.freq = torch.linspace(1.0, 5.0, dim).view(1, dim)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        y = int(idx % self.classes)
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + int(idx))
        proto = self.class_proto[y].view(1, self.dim)
        phase = 0.13 * float(idx % 31)
        wave1 = torch.sin(2 * math.pi * (1 + y % 5) * self.pos + phase * self.freq)
        wave2 = torch.cos(2 * math.pi * (1 + (y // 2) % 4) * self.pos * self.freq)
        center = (y + 1) / float(self.classes + 1)
        bump = torch.exp(-((self.pos - center) ** 2) / 0.012)
        evidence = 0.45 * proto + 0.25 * wave1 + 0.20 * wave2 + 0.35 * bump * proto
        evidence = evidence + self.noise * torch.randn(self.evidence_cells, self.dim, generator=g)
        evidence = (evidence - evidence.mean(dim=0, keepdim=True)) / evidence.std(dim=0, keepdim=True).clamp_min(1e-4)
        target_vec = F.normalize(proto.squeeze(0) + 0.20 * evidence.mean(dim=0), dim=0)
        return evidence.float(), torch.tensor(y, dtype=torch.long), target_vec.float()


class SyntheticAssemblerHead(nn.Module):
    def __init__(self, dim: int, classes: int, dropout: float = 0.05):
        super().__init__()
        self.classes = int(classes)
        self.query = nn.Parameter(torch.randn(classes, dim) * 0.04)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(dim, classes)
        self.recon = nn.Linear(dim, dim)

    def forward(self, slots: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.query.to(device=slots.device, dtype=slots.dtype)
        k = self.key(slots)
        v = self.value(slots)
        score = torch.einsum("cd,nsd->ncs", q, k) / math.sqrt(slots.shape[-1])
        attn = torch.softmax(score.float(), dim=-1).to(slots.dtype)
        read = torch.einsum("ncs,nsd->ncd", attn, v)
        pooled = self.norm(read.mean(dim=1))
        pooled = self.drop(pooled)
        return self.classifier(pooled), self.recon(pooled)


def _one_hot_soft(size: int, hot: int, value: float = 1.0, floor: float = 0.02, device=None) -> torch.Tensor:
    x = torch.full((size,), floor, device=device)
    if 0 <= hot < size:
        x[hot] += value
    return x / x.sum().clamp_min(1e-8)


def build_flow_targets(cfg: AssemblerConfig, device: torch.device) -> Dict[str, torch.Tensor]:
    T = cfg.layers * cfg.steps
    B, K, A = cfg.blocks, cfg.primitive_slots, cfg.address_cells
    P = len(PRIMITIVES)
    mem0 = cfg.blocks
    glob0 = cfg.blocks + cfg.memory_cells
    idx = {n: i for i, n in enumerate(PRIMITIVES)}

    read = torch.full((T, B, K, A), 0.015, device=device)
    prim = torch.full((T, B, K, P), 0.02, device=device)
    slot_tr = torch.full((T, B, K, K), 0.02, device=device)
    prim_tr = torch.full((T, P, P), 0.01, device=device)
    comp = torch.full((T, B, K), 0.04, device=device)
    write = torch.full((T, B, A), 0.015, device=device)

    phase_prims = {
        "extract": ["ctx_matrix", "channel_butterfly", "phase_matrix"],
        "compare": ["low_rank", "product_gate", "phase_matrix"],
        "suppress": ["block_butterfly", "product_gate", "phase_matrix"],
        "aggregate": ["block_butterfly", "channel_butterfly", "phase_matrix"],
    }
    phase_pairs = {
        "extract": [("ctx_matrix", "channel_butterfly", 1.2), ("channel_butterfly", "phase_matrix", 0.9)],
        "compare": [("low_rank", "product_gate", 1.2), ("ctx_matrix", "phase_matrix", 0.8)],
        "suppress": [("block_butterfly", "phase_matrix", 1.2), ("product_gate", "phase_matrix", 0.9)],
        "aggregate": [("phase_matrix", "block_butterfly", 1.1), ("block_butterfly", "channel_butterfly", 0.9)],
    }

    for t in range(T):
        layer = t // cfg.steps
        phase = PHASES[min(layer, len(PHASES) - 1)]
        names = phase_prims.get(phase, list(PRIMITIVES))
        for b in range(B):
            for k in range(K):
                # Read from own state, nearby state, memory and global; all dense.
                read[t, b, k, b] += 1.0
                if b > 0:
                    read[t, b, k, b - 1] += 0.15
                if b + 1 < B:
                    read[t, b, k, b + 1] += 0.15
                if cfg.memory_cells > 0:
                    read[t, b, k, mem0 + (b + k + t) % cfg.memory_cells] += 0.45
                if cfg.global_cells > 0:
                    read[t, b, k, glob0 + (k + t) % cfg.global_cells] += 0.30

                for j, name in enumerate(names):
                    if name in idx:
                        prim[t, b, k, idx[name]] += 0.90 / (1 + abs(k - j))
                slot_tr[t, b, k, k] += 0.95
                slot_tr[t, b, k, (k - 1) % K] += 0.25
                slot_tr[t, b, k, (k + 1) % K] += 0.18
                comp[t, b, k] += 1.0 / (1 + abs(k - (t + b) % K))

            # Write to own state, memory, global; still dense.
            write[t, b, b] += 1.1
            if cfg.memory_cells > 0:
                write[t, b, mem0 + (b + t) % cfg.memory_cells] += 0.35
            if cfg.global_cells > 0:
                write[t, b, glob0 + b % cfg.global_cells] += 0.25

        for a, bname, w in phase_pairs.get(phase, []):
            if a in idx and bname in idx:
                prim_tr[t, idx[a], idx[bname]] += w
        prim_tr[t] += torch.eye(P, device=device) * 0.20

    read = read / read.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    prim = prim / prim.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    slot_tr = slot_tr / slot_tr.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    prim_tr = prim_tr / prim_tr.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    comp = comp / comp.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    write = write / write.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return {
        "read_flow": read,
        "primitive_slot_flow": prim,
        "slot_transition_flow": slot_tr,
        "primitive_transition_flow": prim_tr,
        "slot_composition_flow": comp,
        "write_flow": write,
    }


def flow_entropy_report(aux) -> Dict[str, float]:
    return {f"entropy_{k}": float(v.detach().cpu()) for k, v in aux.entropies.items()}


def evaluate(core, head, loader, device, dtype, args, flow_targets, skill_weights):
    core.eval(); head.eval()
    use_amp = device.startswith("cuda") and dtype != torch.float32
    total = ce_total = rec_total = skill_total = 0.0
    correct = n = 0
    last_aux = None
    with torch.no_grad():
        for evidence, y, target_vec in loader:
            evidence = evidence.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            target_vec = target_vec.to(device, non_blocking=True)
            with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=use_amp):
                _cells, aux = core(evidence)
                logits, vec = head(aux.slots)
                ce = F.cross_entropy(logits.float(), y)
                rec = F.mse_loss(F.normalize(vec.float(), dim=-1), target_vec.float())
                skill, flow_losses = assembler_skill_loss(aux, flow_targets, skill_weights)
                loss = ce + args.lambda_recon * rec + args.lambda_skill * skill
            bs = y.numel()
            correct += int((logits.argmax(-1) == y).sum().detach().cpu())
            n += bs
            total += float(loss.detach().cpu()) * bs
            ce_total += float(ce.detach().cpu()) * bs
            rec_total += float(rec.detach().cpu()) * bs
            skill_total += float(skill.detach().cpu()) * bs
            last_aux = aux
    out = {
        "loss": total / max(1, n),
        "ce": ce_total / max(1, n),
        "recon": rec_total / max(1, n),
        "skill": skill_total / max(1, n),
        "acc": correct / max(1, n),
    }
    if last_aux is not None:
        out.update(flow_entropy_report(last_aux))
        out["memory_usage"] = float(last_aux.memory_usage.detach().cpu())
        out["global_usage"] = float(last_aux.global_usage.detach().cpu())
    return out


def run(args) -> None:
    set_seed(args.seed)
    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(args.amp, torch.float32)
    out = ensure_dir(args.out_dir)

    cfg = AssemblerConfig(
        dim=args.dim,
        evidence_cells=args.evidence_cells,
        layers=args.layers,
        blocks=args.blocks,
        steps=args.steps,
        primitive_slots=args.primitive_slots,
        memory_cells=args.memory_cells,
        global_cells=args.global_cells,
        channel_stages=args.channel_stages,
        dropout=args.dropout,
        use_deltas=True,
    )
    core = MatrixProgramAssemblerCore(cfg).to(device)
    head = SyntheticAssemblerHead(args.dim, args.classes, args.head_dropout).to(device)

    if args.init_assembler:
        ckpt = torch.load(args.init_assembler, map_location=device)
        state = ckpt.get("assembler_core", ckpt.get("core", ckpt))
        missing, unexpected = core.load_state_dict(state, strict=False)
        print(f"loaded init assembler: {args.init_assembler} missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    train_ds = SyntheticAssemblerTask(args.train_n, args.classes, args.evidence_cells, args.dim, args.seed, args.noise)
    val_ds = SyntheticAssemblerTask(args.val_n, args.classes, args.evidence_cells, args.dim, args.seed + 1000, args.noise)
    train = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=args.pin_memory)
    val = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.workers, pin_memory=args.pin_memory)

    flow_targets = build_flow_targets(cfg, torch.device(device))
    skill_weights = {
        "read_flow_kl": args.w_read,
        "primitive_slot_kl": args.w_primitive,
        "slot_transition_kl": args.w_slot_transition,
        "primitive_transition_kl": args.w_primitive_transition,
        "slot_composition_kl": args.w_composition,
        "write_flow_kl": args.w_write,
    }

    params = list(core.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda") and dtype == torch.float16)
    use_amp = device.startswith("cuda") and dtype != torch.float32

    fields = [
        "epoch", "train_loss", "train_ce", "train_acc", "train_recon", "train_skill",
        "val_loss", "val_ce", "val_acc", "val_recon", "val_skill", "best_acc",
        "entropy_read", "entropy_primitive", "entropy_slot_transition", "entropy_primitive_transition", "entropy_write",
        "memory_usage", "global_usage",
    ]
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    best = -1.0
    best_epoch = 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        core.train(); head.train()
        total = ce_total = rec_total = skill_total = 0.0
        correct = n = 0
        for step, (evidence, y, target_vec) in enumerate(train, 1):
            evidence = evidence.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            target_vec = target_vec.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=use_amp):
                _cells, aux = core(evidence)
                logits, vec = head(aux.slots)
                ce = F.cross_entropy(logits.float(), y)
                rec = F.mse_loss(F.normalize(vec.float(), dim=-1), target_vec.float())
                skill, _flow_losses = assembler_skill_loss(aux, flow_targets, skill_weights)
                loss = ce + args.lambda_recon * rec + args.lambda_skill * skill
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            scaler.step(opt); scaler.update()
            bs = y.numel()
            total += float(loss.detach().cpu()) * bs
            ce_total += float(ce.detach().cpu()) * bs
            rec_total += float(rec.detach().cpu()) * bs
            skill_total += float(skill.detach().cpu()) * bs
            correct += int((logits.argmax(-1) == y).sum().detach().cpu())
            n += bs
            if args.log_every and step % args.log_every == 0:
                print(f"epoch {ep:03d} step {step:05d} loss={total/max(1,n):.4f} ce={ce_total/max(1,n):.4f} acc={100*correct/max(1,n):.2f}% skill={skill_total/max(1,n):.4f}", flush=True)

        va = evaluate(core, head, val, device, dtype, args, flow_targets, skill_weights)
        train_acc = correct / max(1, n)
        if va["acc"] > best:
            best = va["acc"]
            best_epoch = ep
            torch.save({
                "assembler_core": core.state_dict(),
                "head": head.state_dict(),
                "config": cfg.__dict__,
                "args": vars(args),
                "best_acc": best,
                "epoch": ep,
                "primitives": list(PRIMITIVES),
            }, out / "assembler_core_best.pt")
        torch.save({
            "assembler_core": core.state_dict(),
            "head": head.state_dict(),
            "config": cfg.__dict__,
            "args": vars(args),
            "best_acc": best,
            "epoch": ep,
            "primitives": list(PRIMITIVES),
        }, out / "assembler_core_last.pt")

        row = {
            "epoch": ep,
            "train_loss": total / max(1, n),
            "train_ce": ce_total / max(1, n),
            "train_acc": train_acc,
            "train_recon": rec_total / max(1, n),
            "train_skill": skill_total / max(1, n),
            "val_loss": va["loss"],
            "val_ce": va["ce"],
            "val_acc": va["acc"],
            "val_recon": va["recon"],
            "val_skill": va["skill"],
            "best_acc": best,
            "entropy_read": va.get("entropy_read", 0.0),
            "entropy_primitive": va.get("entropy_primitive", 0.0),
            "entropy_slot_transition": va.get("entropy_slot_transition", 0.0),
            "entropy_primitive_transition": va.get("entropy_primitive_transition", 0.0),
            "entropy_write": va.get("entropy_write", 0.0),
            "memory_usage": va.get("memory_usage", 0.0),
            "global_usage": va.get("global_usage", 0.0),
        }
        with (out / "metrics.csv").open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        print(f"epoch {ep:03d}/{args.epochs} train={row['train_loss']:.4f}/{100*train_acc:.2f}% val={va['loss']:.4f}/{100*va['acc']:.2f}% skill={row['train_skill']:.4f} best={100*best:.2f}%@{best_epoch}", flush=True)

    write_json(out / "final_report.json", {
        "best_acc": best,
        "best_epoch": best_epoch,
        "elapsed_sec": time.time() - t0,
        "config": cfg.__dict__,
        "args": vars(args),
        "checkpoint": str(out / "assembler_core_best.pt"),
    })


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="matrix_program_core/runs/assembler_pretrain_debug")
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", choices=["fp16", "bf16", "fp32", "off"], default="bf16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--classes", type=int, default=10)
    p.add_argument("--train-n", type=int, default=6000)
    p.add_argument("--val-n", type=int, default=1000)
    p.add_argument("--dim", type=int, default=96)
    p.add_argument("--evidence-cells", type=int, default=48)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--primitive-slots", type=int, default=4)
    p.add_argument("--memory-cells", type=int, default=4)
    p.add_argument("--global-cells", type=int, default=2)
    p.add_argument("--channel-stages", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.04)
    p.add_argument("--head-dropout", type=float, default=0.05)
    p.add_argument("--noise", type=float, default=0.08)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=0.75)
    p.add_argument("--lambda-skill", type=float, default=0.35)
    p.add_argument("--lambda-recon", type=float, default=0.20)
    p.add_argument("--w-read", type=float, default=0.35)
    p.add_argument("--w-primitive", type=float, default=0.45)
    p.add_argument("--w-slot-transition", type=float, default=0.30)
    p.add_argument("--w-primitive-transition", type=float, default=0.35)
    p.add_argument("--w-composition", type=float, default=0.25)
    p.add_argument("--w-write", type=float, default=0.35)
    p.add_argument("--init-assembler", default="")
    p.add_argument("--log-every", type=int, default=25)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
