#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audio transfer v4 using the same UniversalMatrixProgramCore.

This is the clean transfer experiment:

    AudioInputAdapter(MatrixEvidence) -> UniversalMatrixProgramCore -> TaskHead

The core can be trained in three modes:

    --train-mode freeze_core   only input adapter + task head train
    --train-mode delta         LoRA-like: only *_delta core params + adapter/head train
    --train-mode full          all core params train

The pretrain checkpoint must come from train_core_pretrain.py and contains
`ckpt["core"]`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from matrix_program_core.train_core_pretrain import build_skill_targets  # noqa: E402
from matrix_program_core.universal_core import ProgramCoreConfig, UniversalMatrixProgramCore, core_skill_loss  # noqa: E402
from simple_butterfly_matrix.simple_butterfly_matrix import (  # noqa: E402
    MatrixEvidence,
    amp_dtype,
    ensure_dir,
    make_loaders,
    set_seed,
    write_json,
)


class MatrixProgramTaskHead(nn.Module):
    def __init__(self, dim: int, classes: int, dropout: float = 0.05):
        super().__init__()
        self.classes = int(classes)
        self.query = nn.Parameter(torch.randn(classes, dim) * 0.04)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.mix = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm = nn.LayerNorm(dim)
        self.logit_w = nn.Parameter(torch.randn(classes, dim) * 0.04)
        self.logit_bias = nn.Parameter(torch.zeros(classes))

    def forward(self, slots: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        q = self.query.to(device=slots.device, dtype=slots.dtype)
        k = self.key(slots)
        v = self.value(slots)
        score = torch.einsum("cd,bsd->bcs", q, k) / math.sqrt(slots.shape[-1])
        attn = torch.softmax(score.float(), dim=-1).to(slots.dtype)
        read = torch.einsum("bcs,bsd->bcd", attn, v)
        global_read = slots.mean(dim=1, keepdim=True).expand_as(read)
        cls_state = self.query.to(device=slots.device, dtype=slots.dtype).view(1, self.classes, -1).expand(slots.shape[0], -1, -1)
        h = self.norm(cls_state + self.mix(torch.cat([cls_state, read, global_read], dim=-1)))
        logits = (h * self.logit_w.to(device=slots.device, dtype=slots.dtype).view(1, self.classes, -1)).sum(dim=-1)
        logits = logits + self.logit_bias.to(device=slots.device, dtype=slots.dtype)
        return logits, {"class_slot_attention": attn.detach(), "class_read": read.detach()}


class AudioMatrixProgramV4(nn.Module):
    def __init__(self, classes: int, args):
        super().__init__()
        self.input_adapter = MatrixEvidence(args.sample_rate, args.n_mels, args.hop_length, args.evidence_cells, args.dim)
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
        self.core = UniversalMatrixProgramCore(cfg)
        self.head = MatrixProgramTaskHead(args.dim, classes, args.head_dropout)

    def forward(self, wav: torch.Tensor):
        evidence = self.input_adapter(wav)
        slots, caux = self.core(evidence)
        logits, haux = self.head(slots)
        return logits, caux, haux


def trainable_summary(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    core_total = sum(p.numel() for p in model.core.parameters())
    core_train = sum(p.numel() for p in model.core.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "core_total": core_total, "core_trainable": core_train}


def configure_train_mode(model: AudioMatrixProgramV4, mode: str, train_input_adapter: bool, train_head: bool) -> None:
    model.core.freeze_for_mode(mode)
    for p in model.input_adapter.parameters():
        p.requires_grad = bool(train_input_adapter)
    for p in model.head.parameters():
        p.requires_grad = bool(train_head)


def aux_losses(logits: torch.Tensor, caux, args, prim_t: torch.Tensor, pair_t: torch.Tensor) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    out["skill"] = core_skill_loss(caux, prim_t, pair_t, args.lambda_pair)
    gate = caux.write_gates.float()
    upd = caux.update_norms.float()
    out["write_budget"] = (gate.mean() - args.write_target).pow(2) if gate.numel() else torch.zeros((), device=logits.device)
    out["update_alive"] = F.relu(torch.tensor(float(args.min_update_norm), device=logits.device) - upd.mean()).pow(2) if upd.numel() else torch.zeros((), device=logits.device)
    out["logit_norm"] = logits.float().pow(2).mean()
    out["transport_entropy"] = caux.transport_entropy.float()
    return out


def train_epoch(model, loader, opt, scaler, device, dtype, args, epoch: int, prim_t: torch.Tensor, pair_t: torch.Tensor):
    model.train()
    use_amp = device.startswith("cuda") and dtype != torch.float32
    totals = {"loss": 0.0, "ce": 0.0, "correct": 0, "n": 0}
    aux_sum: Dict[str, float] = {}
    for step, (wav, y) in enumerate(loader, 1):
        if args.max_train_batches and step > args.max_train_batches:
            break
        wav = wav.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=use_amp):
            logits, caux, _ = model(wav)
            ce = F.cross_entropy(logits.float(), y)
            losses = aux_losses(logits, caux, args, prim_t, pair_t)
            loss = ce
            loss = loss + args.lambda_skill * losses["skill"]
            loss = loss + args.lambda_write_budget * losses["write_budget"]
            loss = loss + args.lambda_update_alive * losses["update_alive"]
            loss = loss + args.lambda_logit_norm * losses["logit_norm"]
        if not torch.isfinite(loss):
            print("NONFINITE_LOSS skip", flush=True)
            continue
        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
        scaler.step(opt)
        scaler.update()
        bs = y.numel()
        totals["loss"] += float(loss.detach().cpu()) * bs
        totals["ce"] += float(ce.detach().cpu()) * bs
        totals["correct"] += int((logits.argmax(-1) == y).sum().detach().cpu())
        totals["n"] += bs
        for k, v in losses.items():
            aux_sum[k] = aux_sum.get(k, 0.0) + float(v.detach().cpu()) * bs
        if args.log_every and step % args.log_every == 0:
            print(f"epoch {epoch:03d} step {step:05d} loss={totals['loss']/max(1,totals['n']):.4f} ce={totals['ce']/max(1,totals['n']):.4f} acc={100*totals['correct']/max(1,totals['n']):.2f}%", flush=True)
    out = {k: v / max(1, totals["n"]) for k, v in totals.items() if k != "correct"}
    out["acc"] = totals["correct"] / max(1, totals["n"])
    for k, v in aux_sum.items():
        out[k] = v / max(1, totals["n"])
    return out


@torch.no_grad()
def evaluate(model, loader, device, dtype, args, prim_t: torch.Tensor, pair_t: torch.Tensor):
    model.eval()
    use_amp = device.startswith("cuda") and dtype != torch.float32
    total_loss, correct, n = 0.0, 0, 0
    last_report = None
    for step, (wav, y) in enumerate(loader, 1):
        if args.max_val_batches and step > args.max_val_batches:
            break
        wav = wav.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=use_amp):
            logits, caux, haux = model(wav)
            ce = F.cross_entropy(logits.float(), y)
            losses = aux_losses(logits, caux, args, prim_t, pair_t)
            loss = ce + args.lambda_skill * losses["skill"]
        pred = logits.argmax(-1)
        bs = y.numel()
        total_loss += float(loss.detach().cpu()) * bs
        correct += int((pred == y).sum().detach().cpu())
        n += bs
        prim = caux.primitive_probs.float().mean(dim=1).cpu() if caux.primitive_probs.numel() else torch.empty(0)
        pair = caux.pair_flow.float().cpu() if caux.pair_flow.numel() else torch.empty(0)
        last_report = {
            "skill_loss": float(losses["skill"].detach().cpu()),
            "write_gate_mean": float(caux.write_gates.float().mean().detach().cpu()) if caux.write_gates.numel() else 0.0,
            "update_norm_mean": float(caux.update_norms.float().mean().detach().cpu()) if caux.update_norms.numel() else 0.0,
            "transport_entropy": float(caux.transport_entropy.detach().cpu()),
            "primitive_mean_by_step": prim.tolist() if prim.numel() else [],
            "pair_flow_shape": list(pair.shape),
            "slot_count": len(caux.slot_names),
        }
    return {"loss": total_loss / max(1, n), "acc": correct / max(1, n), "n": n, "report": last_report}


def run(args) -> None:
    set_seed(args.seed)
    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    dtype = amp_dtype(args.amp)
    out_dir = ensure_dir(Path(args.out_dir))
    train_loader, val_loader, classes, train_counts, val_counts = make_loaders(args)
    args.num_classes = len(classes)

    model = AudioMatrixProgramV4(len(classes), args).to(device)
    if args.core_checkpoint:
        ckpt = torch.load(args.core_checkpoint, map_location=device)
        state = ckpt.get("core", ckpt)
        missing, unexpected = model.core.load_state_dict(state, strict=False)
        print(f"loaded core checkpoint: {args.core_checkpoint} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location=device)
        state = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"loaded full task checkpoint: {args.init_checkpoint} missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    configure_train_mode(model, args.train_mode, args.train_input_adapter, args.train_head)
    summary = trainable_summary(model)
    print(f"loaded datasets: train={len(train_loader.dataset)} val={len(val_loader.dataset)} classes={classes}", flush=True)
    print(f"AudioMatrixProgramV4 params={summary} mode={args.train_mode} device={device} amp={args.amp}", flush=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable parameters; change --train-mode or adapter/head flags")
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda") and dtype == torch.float16)
    prim_t, pair_t = build_skill_targets(args.layers * args.steps, torch.device(device))

    fields = ["epoch", "train_loss", "train_ce", "train_acc", "val_loss", "val_acc", "best_acc", "skill", "write_budget", "update_alive", "logit_norm"]
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()
    best, best_epoch = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        tr = train_epoch(model, train_loader, opt, scaler, device, dtype, args, epoch, prim_t, pair_t)
        va = evaluate(model, val_loader, device, dtype, args, prim_t, pair_t)
        if va["acc"] > best:
            best, best_epoch = va["acc"], epoch
            torch.save({"model": model.state_dict(), "core": model.core.state_dict(), "args": vars(args), "classes": classes, "epoch": epoch, "best_acc": best, "trainable_summary": summary}, out_dir / "best.pt")
        torch.save({"model": model.state_dict(), "core": model.core.state_dict(), "args": vars(args), "classes": classes, "epoch": epoch, "best_acc": best, "trainable_summary": summary}, out_dir / "last.pt")
        row = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "train_ce": tr["ce"],
            "train_acc": tr["acc"],
            "val_loss": va["loss"],
            "val_acc": va["acc"],
            "best_acc": best,
            "skill": tr.get("skill", 0.0),
            "write_budget": tr.get("write_budget", 0.0),
            "update_alive": tr.get("update_alive", 0.0),
            "logit_norm": tr.get("logit_norm", 0.0),
        }
        with (out_dir / "metrics.csv").open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        write_json(out_dir / f"analysis_epoch_{epoch:03d}.json", {"epoch": epoch, "train": tr, "val": va, "best_acc": best, "best_epoch": best_epoch, "classes": classes, "train_counts": train_counts, "val_counts": val_counts, "trainable_summary": summary})
        print(f"epoch {epoch:03d}/{args.epochs} train={tr['loss']:.4f}/{100*tr['acc']:.2f}% val={va['loss']:.4f}/{100*va['acc']:.2f}% best={100*best:.2f}%@{best_epoch} skill={tr.get('skill',0.0):.4f}", flush=True)
    write_json(out_dir / "final_report.json", {"best_acc": best, "best_epoch": best_epoch, "args": vars(args), "classes": classes, "trainable_summary": summary})


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--synthetic-length", type=int, default=512)
    p.add_argument("--data-root", default="../architecture_builder/data/speechcommands")
    p.add_argument("--download", action="store_true")
    p.add_argument("--classes", default="yes,no,up,down,left,right,on,off,stop,go")
    p.add_argument("--train-limit", type=int, default=12000)
    p.add_argument("--val-limit", type=int, default=2000)
    p.add_argument("--seconds", type=float, default=1.0)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--n-mels", type=int, default=64)
    p.add_argument("--hop-length", type=int, default=160)
    p.add_argument("--dim", type=int, default=96)
    p.add_argument("--evidence-cells", type=int, default=48)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--variants", type=int, default=3)
    p.add_argument("--channel-stages", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.04)
    p.add_argument("--head-dropout", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.012)
    p.add_argument("--grad-clip", type=float, default=0.75)
    p.add_argument("--write-target", type=float, default=0.44)
    p.add_argument("--min-update-norm", type=float, default=0.20)
    p.add_argument("--lambda-skill", type=float, default=0.05)
    p.add_argument("--lambda-pair", type=float, default=0.45)
    p.add_argument("--lambda-write-budget", type=float, default=0.025)
    p.add_argument("--lambda-update-alive", type=float, default=0.005)
    p.add_argument("--lambda-logit-norm", type=float, default=0.0007)
    p.add_argument("--amp", choices=["fp16", "bf16", "fp32", "off"], default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--max-val-batches", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--out-dir", default="./matrix_program_core/runs/audio_v4")
    p.add_argument("--core-checkpoint", default="")
    p.add_argument("--init-checkpoint", default="")
    p.add_argument("--train-mode", choices=["freeze_core", "delta", "full"], default="delta")
    p.add_argument("--train-input-adapter", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--train-head", action=argparse.BooleanOptionalAction, default=True)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
