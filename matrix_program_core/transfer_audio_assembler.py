#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audio transfer using Factorized MatrixProgramAssemblerCore.

Clean transfer:

    MatrixEvidence(audio) -> MatrixProgramAssemblerCore -> AudioHead

Train modes:

    freeze_core  : train only input adapter + head
    delta        : train only *_delta inside assembler + adapter + head
    full         : train full assembler + adapter + head
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from matrix_program_core.assembler_core import (  # noqa: E402
    AssemblerConfig,
    MatrixProgramAssemblerCore,
    assembler_skill_loss,
)
from matrix_program_core.train_assembler_pretrain import build_flow_targets  # noqa: E402
from simple_butterfly_matrix.simple_butterfly_matrix import (  # noqa: E402
    MatrixEvidence,
    amp_dtype,
    ensure_dir,
    make_loaders,
    set_seed,
    write_json,
)


class AudioAssemblerHead(nn.Module):
    def __init__(self, dim: int, classes: int, dropout: float = 0.05):
        super().__init__()
        self.classes = int(classes)
        self.query = nn.Parameter(torch.randn(classes, dim) * 0.04)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.update = nn.Sequential(
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
        score = torch.einsum("cd,nsd->ncs", q, k) / math.sqrt(slots.shape[-1])
        attn = torch.softmax(score.float(), dim=-1).to(slots.dtype)
        read = torch.einsum("ncs,nsd->ncd", attn, v)
        global_read = slots.mean(dim=1, keepdim=True).expand_as(read)
        cls_state = q.view(1, self.classes, -1).expand(slots.shape[0], -1, -1)
        h = self.norm(cls_state + self.update(torch.cat([cls_state, read, global_read], dim=-1)))
        logits = (h * self.logit_w.to(device=slots.device, dtype=slots.dtype).view(1, self.classes, -1)).sum(dim=-1)
        logits = logits + self.logit_bias.to(device=slots.device, dtype=slots.dtype)
        return logits, {"class_slot_attention": attn.detach(), "class_read": read.detach()}


class AudioAssemblerModel(nn.Module):
    def __init__(self, classes: int, args):
        super().__init__()
        self.input_adapter = MatrixEvidence(args.sample_rate, args.n_mels, args.hop_length, args.evidence_cells, args.dim)
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
        self.assembler_core = MatrixProgramAssemblerCore(cfg)
        self.head = AudioAssemblerHead(args.dim, classes, args.head_dropout)

    def forward(self, wav: torch.Tensor):
        evidence = self.input_adapter(wav)
        _cells, aux = self.assembler_core(evidence)
        logits, haux = self.head(aux.slots)
        return logits, aux, haux


def trainable_summary(model: AudioAssemblerModel) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    core_total = sum(p.numel() for p in model.assembler_core.parameters())
    core_train = sum(p.numel() for p in model.assembler_core.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "core_total": core_total, "core_trainable": core_train}


def configure_train_mode(model: AudioAssemblerModel, mode: str, train_input_adapter: bool, train_head: bool) -> None:
    model.assembler_core.freeze_for_mode(mode)
    for p in model.input_adapter.parameters():
        p.requires_grad = bool(train_input_adapter)
    for p in model.head.parameters():
        p.requires_grad = bool(train_head)


def aux_losses(logits: torch.Tensor, aux, args, flow_targets, skill_weights) -> Dict[str, torch.Tensor]:
    skill, flow_losses = assembler_skill_loss(aux, flow_targets, skill_weights)
    gate = aux.write_gates.float()
    upd = aux.update_norms.float()
    out: Dict[str, torch.Tensor] = dict(flow_losses)
    out["skill"] = skill
    out["write_budget"] = (gate.mean() - args.write_target).pow(2) if gate.numel() else torch.zeros((), device=logits.device)
    out["update_alive"] = F.relu(torch.tensor(float(args.min_update_norm), device=logits.device) - upd.mean()).pow(2) if upd.numel() else torch.zeros((), device=logits.device)
    out["logit_norm"] = logits.float().pow(2).mean()
    return out


def train_epoch(model, loader, opt, scaler, device, dtype, args, epoch: int, flow_targets, skill_weights):
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
            logits, aux, _haux = model(wav)
            ce = F.cross_entropy(logits.float(), y)
            losses = aux_losses(logits, aux, args, flow_targets, skill_weights)
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
            trainable = [p for p in model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
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
            print(f"epoch {epoch:03d} step {step:05d} loss={totals['loss']/max(1, totals['n']):.4f} ce={totals['ce']/max(1, totals['n']):.4f} acc={100*totals['correct']/max(1, totals['n']):.2f}%", flush=True)
    out = {k: v / max(1, totals["n"]) for k, v in totals.items() if k != "correct"}
    out["acc"] = totals["correct"] / max(1, totals["n"])
    for k, v in aux_sum.items():
        out[k] = v / max(1, totals["n"])
    return out


@torch.no_grad()
def evaluate(model, loader, device, dtype, args, flow_targets, skill_weights):
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
            logits, aux, haux = model(wav)
            ce = F.cross_entropy(logits.float(), y)
            losses = aux_losses(logits, aux, args, flow_targets, skill_weights)
            loss = ce + args.lambda_skill * losses["skill"]
        pred = logits.argmax(-1)
        bs = y.numel()
        total_loss += float(loss.detach().cpu()) * bs
        correct += int((pred == y).sum().detach().cpu())
        n += bs
        last_report = {
            "skill": float(losses["skill"].detach().cpu()),
            "write_gate_mean": float(aux.write_gates.float().mean().detach().cpu()) if aux.write_gates.numel() else 0.0,
            "update_norm_mean": float(aux.update_norms.float().mean().detach().cpu()) if aux.update_norms.numel() else 0.0,
            "memory_usage": float(aux.memory_usage.detach().cpu()),
            "global_usage": float(aux.global_usage.detach().cpu()),
            "entropy": {k: float(v.detach().cpu()) for k, v in aux.entropies.items()},
            "slot_count": len(aux.slot_names),
            "cell_names": aux.cell_names,
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

    model = AudioAssemblerModel(len(classes), args).to(device)
    if args.assembler_checkpoint:
        ckpt = torch.load(args.assembler_checkpoint, map_location=device)
        state = ckpt.get("assembler_core", ckpt.get("core", ckpt))
        missing, unexpected = model.assembler_core.load_state_dict(state, strict=False)
        print(f"loaded assembler checkpoint: {args.assembler_checkpoint} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location=device)
        state = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"loaded full task checkpoint: {args.init_checkpoint} missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    configure_train_mode(model, args.train_mode, args.train_input_adapter, args.train_head)
    summary = trainable_summary(model)
    print(f"loaded datasets: train={len(train_loader.dataset)} val={len(val_loader.dataset)} classes={classes}", flush=True)
    print(f"AudioAssemblerModel params={summary} mode={args.train_mode} device={device} amp={args.amp}", flush=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable parameters; change train mode")
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda") and dtype == torch.float16)

    cfg = model.assembler_core.cfg
    flow_targets = build_flow_targets(cfg, torch.device(device))
    skill_weights = {
        "read_flow_kl": args.w_read,
        "primitive_slot_kl": args.w_primitive,
        "slot_transition_kl": args.w_slot_transition,
        "primitive_transition_kl": args.w_primitive_transition,
        "slot_composition_kl": args.w_composition,
        "write_flow_kl": args.w_write,
    }

    fields = [
        "epoch", "train_loss", "train_ce", "train_acc", "val_loss", "val_acc", "best_acc",
        "skill", "read_flow_kl", "primitive_slot_kl", "slot_transition_kl", "primitive_transition_kl", "slot_composition_kl", "write_flow_kl",
        "write_budget", "update_alive", "logit_norm",
    ]
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    best, best_epoch = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        tr = train_epoch(model, train_loader, opt, scaler, device, dtype, args, epoch, flow_targets, skill_weights)
        va = evaluate(model, val_loader, device, dtype, args, flow_targets, skill_weights)
        if va["acc"] > best:
            best, best_epoch = va["acc"], epoch
            torch.save({
                "model": model.state_dict(),
                "assembler_core": model.assembler_core.state_dict(),
                "args": vars(args),
                "classes": classes,
                "epoch": epoch,
                "best_acc": best,
                "trainable_summary": summary,
                "assembler_config": cfg.__dict__,
            }, out_dir / "best.pt")
        torch.save({
            "model": model.state_dict(),
            "assembler_core": model.assembler_core.state_dict(),
            "args": vars(args),
            "classes": classes,
            "epoch": epoch,
            "best_acc": best,
            "trainable_summary": summary,
            "assembler_config": cfg.__dict__,
        }, out_dir / "last.pt")
        row = {k: 0.0 for k in fields}
        row.update({
            "epoch": epoch,
            "train_loss": tr["loss"],
            "train_ce": tr["ce"],
            "train_acc": tr["acc"],
            "val_loss": va["loss"],
            "val_acc": va["acc"],
            "best_acc": best,
        })
        for k in row.keys():
            if k in tr:
                row[k] = tr[k]
        with (out_dir / "metrics.csv").open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        write_json(out_dir / f"analysis_epoch_{epoch:03d}.json", {
            "epoch": epoch,
            "train": tr,
            "val": va,
            "best_acc": best,
            "best_epoch": best_epoch,
            "classes": classes,
            "train_counts": train_counts,
            "val_counts": val_counts,
            "trainable_summary": summary,
        })
        print(f"epoch {epoch:03d}/{args.epochs} train={tr['loss']:.4f}/{100*tr['acc']:.2f}% val={va['loss']:.4f}/{100*va['acc']:.2f}% best={100*best:.2f}%@{best_epoch} skill={tr.get('skill',0.0):.4f}", flush=True)

    write_json(out_dir / "final_report.json", {
        "best_acc": best,
        "best_epoch": best_epoch,
        "args": vars(args),
        "classes": classes,
        "trainable_summary": summary,
        "assembler_config": cfg.__dict__,
    })


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
    p.add_argument("--primitive-slots", type=int, default=4)
    p.add_argument("--memory-cells", type=int, default=4)
    p.add_argument("--global-cells", type=int, default=2)
    p.add_argument("--channel-stages", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.04)
    p.add_argument("--head-dropout", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=3)
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
    p.add_argument("--lambda-write-budget", type=float, default=0.025)
    p.add_argument("--lambda-update-alive", type=float, default=0.005)
    p.add_argument("--lambda-logit-norm", type=float, default=0.0007)
    p.add_argument("--w-read", type=float, default=0.25)
    p.add_argument("--w-primitive", type=float, default=0.30)
    p.add_argument("--w-slot-transition", type=float, default=0.20)
    p.add_argument("--w-primitive-transition", type=float, default=0.25)
    p.add_argument("--w-composition", type=float, default=0.15)
    p.add_argument("--w-write", type=float, default=0.25)
    p.add_argument("--amp", choices=["fp16", "bf16", "fp32", "off"], default="fp32")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--max-val-batches", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--out-dir", default="./matrix_program_core/runs/audio_assembler")
    p.add_argument("--assembler-checkpoint", default="")
    p.add_argument("--init-checkpoint", default="")
    p.add_argument("--train-mode", choices=["freeze_core", "delta", "full"], default="delta")
    p.add_argument("--train-input-adapter", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--train-head", action=argparse.BooleanOptionalAction, default=True)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
