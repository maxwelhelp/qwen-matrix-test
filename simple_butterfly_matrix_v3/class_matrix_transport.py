#!/usr/bin/env python3
"""v3: soft transport backbone with differentiable class matrices.

This version keeps the v2 matrix-transport backbone and replaces the passive
classification head with a ClassMatrix head:

  slots -> phase read matrix -> class states -> class pair repair -> logits

No hard router and no top-k are used. Class specialization is trained as matrix
state dynamics instead of a late aggregate readout shortcut.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simple_butterfly_matrix.simple_butterfly_matrix import (  # noqa: E402
    BackboneAux,
    amp_dtype,
    ensure_dir,
    make_loaders,
    set_seed,
    slot_diversity_loss,
    write_json,
)
from simple_butterfly_matrix_v2.soft_matrix_transport import SoftMatrixTransportBackbone  # noqa: E402


PHASE_NAMES = ("input", "extract", "compare", "suppress", "aggregate")


def phase_slot_matrix(layers: int, steps: int, variants: int, blocks: int, device: torch.device, normalize_rows: bool) -> torch.Tensor:
    """Return [phase_count, slot_count] phase-slot map."""

    phase_ids: List[int] = []
    phase_ids.extend([0] * (variants * blocks))
    for l in range(layers):
        phase = min(l + 1, len(PHASE_NAMES) - 1)
        for _s in range(steps):
            phase_ids.extend([phase] * (variants * blocks))
    mat = torch.zeros(len(PHASE_NAMES), len(phase_ids), device=device)
    for slot, phase in enumerate(phase_ids):
        mat[phase, slot] = 1.0
    if normalize_rows:
        mat = mat / mat.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return mat


def class_phase_prior(classes: int) -> torch.Tensor:
    """Weak initial class/phase coverage prior.

    This is not a hard assignment. It only prevents all classes from starting at
    the same late aggregate read.
    """

    p = torch.zeros(classes, len(PHASE_NAMES))
    for c in range(classes):
        mode = c % 5
        if mode == 0:
            p[c, 2] = 0.45
            p[c, 3] = 0.35
            p[c, 4] = 0.25
        elif mode == 1:
            p[c, 1] = 0.35
            p[c, 2] = 0.35
            p[c, 3] = 0.30
        elif mode == 2:
            p[c, 1] = 0.45
            p[c, 4] = 0.30
            p[c, 2] = 0.20
        elif mode == 3:
            p[c, 2] = 0.30
            p[c, 3] = 0.35
            p[c, 4] = 0.25
        else:
            p[c, 1] = 0.30
            p[c, 3] = 0.35
            p[c, 4] = 0.20
    p[:, 0] = 0.08
    return p


class ClassMatrixHead(nn.Module):
    """Differentiable class-state head.

    Classes are trainable matrix states. They read slots through a class-phase
    matrix and receive pair-memory repair before producing logits.
    """

    def __init__(self, dim: int, classes: int, layers: int, steps: int, variants: int, blocks: int, pair_slots: int, dropout: float, phase_prior_strength: float):
        super().__init__()
        self.dim = int(dim)
        self.classes = int(classes)
        self.layers = int(layers)
        self.steps = int(steps)
        self.variants = int(variants)
        self.blocks = int(blocks)
        self.pair_slots = int(pair_slots)
        self.phase_prior_strength = float(phase_prior_strength)

        self.class_state = nn.Parameter(torch.randn(classes, dim) * 0.05)
        self.class_phase_logits = nn.Parameter(class_phase_prior(classes))
        self.pair_state = nn.Parameter(torch.randn(pair_slots, dim) * 0.04)
        self.class_pair_logits = nn.Parameter(torch.randn(classes, pair_slots) * 0.04)

        self.key_w = nn.Parameter(torch.randn(dim, dim) * 0.04)
        self.value_w = nn.Parameter(torch.randn(dim, dim) * 0.04)
        self.class_q = nn.Parameter(torch.randn(dim, dim) * 0.04)
        self.pair_q = nn.Parameter(torch.randn(dim, dim) * 0.04)
        self.pair_k = nn.Parameter(torch.randn(dim, dim) * 0.04)
        self.pair_v = nn.Parameter(torch.randn(dim, dim) * 0.04)

        self.class_update = nn.Sequential(
            nn.LayerNorm(dim * 5),
            nn.Linear(dim * 5, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.class_norm = nn.LayerNorm(dim)
        self.logit_w = nn.Parameter(torch.randn(classes, dim) * 0.04)
        self.logit_bias = nn.Parameter(torch.zeros(classes))
        self.write_logit = nn.Parameter(torch.tensor(-0.35))
        self.drop = nn.Dropout(dropout)

    def forward(self, slots: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, S, D = slots.shape
        phase_map_prior = phase_slot_matrix(self.layers, self.steps, self.variants, self.blocks, slots.device, normalize_rows=True).to(slots.dtype)
        phase_map_mass = phase_slot_matrix(self.layers, self.steps, self.variants, self.blocks, slots.device, normalize_rows=False).to(slots.dtype)
        phase_w = torch.softmax(self.class_phase_logits.float(), dim=-1).to(slots.dtype)  # [C,P]
        slot_prior = torch.matmul(phase_w, phase_map_prior).clamp_min(1e-8)  # [C,S]

        keys = slots @ self.key_w.to(device=slots.device, dtype=slots.dtype)
        values = self.drop(slots @ self.value_w.to(device=slots.device, dtype=slots.dtype))
        q = self.class_state.to(device=slots.device, dtype=slots.dtype) @ self.class_q.to(device=slots.device, dtype=slots.dtype)
        score = torch.einsum("cd,bsd->bcs", q, keys) / math.sqrt(D)
        score = score + self.phase_prior_strength * slot_prior.log().view(1, self.classes, S)
        attn = torch.softmax(score.float(), dim=-1).to(slots.dtype)
        class_read = torch.einsum("bcs,bsd->bcd", attn, values)

        pair_w = torch.softmax(self.class_pair_logits.float(), dim=-1).to(slots.dtype)  # [C,P]
        pair_base = torch.matmul(pair_w, self.pair_state.to(device=slots.device, dtype=slots.dtype))  # [C,D]
        pair_q = class_read @ self.pair_q.to(device=slots.device, dtype=slots.dtype)
        pair_k = pair_base @ self.pair_k.to(device=slots.device, dtype=slots.dtype)
        pair_score = torch.einsum("bcd,ed->bce", pair_q, pair_k) / math.sqrt(D)
        pair_attn = torch.softmax(pair_score.float(), dim=-1).to(slots.dtype)
        pair_v = pair_base @ self.pair_v.to(device=slots.device, dtype=slots.dtype)
        pair_ctx = torch.einsum("bce,ed->bcd", pair_attn, pair_v)

        cls = self.class_state.to(device=slots.device, dtype=slots.dtype).view(1, self.classes, D).expand(B, -1, -1)
        delta = self.class_update(torch.cat([cls, class_read, pair_ctx, cls * class_read, class_read - pair_ctx], dim=-1))
        write = torch.sigmoid(self.write_logit.to(device=slots.device, dtype=slots.dtype))
        class_next = self.class_norm(cls + write * self.drop(delta))
        logits = (class_next * class_read * self.logit_w.to(device=slots.device, dtype=slots.dtype).view(1, self.classes, D)).sum(dim=-1)
        logits = logits + self.logit_bias.to(device=slots.device, dtype=slots.dtype)

        phase_mass = torch.einsum("bcs,ps->bcp", attn.float(), phase_map_mass.float())
        return logits, {
            "class_slot_attention": attn.detach(),
            "class_phase_mass": phase_mass.detach(),
            "class_read": class_read.detach(),
            "class_next": class_next.detach(),
            "pair_attention": pair_attn.detach(),
            "pair_update_norm": pair_ctx.detach().float().norm(dim=-1).mean(),
            "class_write": write.detach(),
        }


class ClassMatrixTransportClassifier(nn.Module):
    def __init__(self, classes: int, args):
        super().__init__()
        self.backbone = SoftMatrixTransportBackbone(
            dim=args.dim,
            evidence_cells=args.evidence_cells,
            layers=args.layers,
            blocks=args.blocks,
            steps=args.steps,
            variants=args.variants,
            sample_rate=args.sample_rate,
            n_mels=args.n_mels,
            hop_length=args.hop_length,
            channel_stages=args.channel_stages,
            dropout=args.dropout,
        )
        self.head = ClassMatrixHead(
            dim=args.dim,
            classes=classes,
            layers=args.layers,
            steps=args.steps,
            variants=args.variants,
            blocks=args.blocks,
            pair_slots=args.pair_slots,
            dropout=args.head_dropout,
            phase_prior_strength=args.phase_prior_strength,
        )

    def forward(self, wav: torch.Tensor):
        slots, baux = self.backbone(wav)
        logits, haux = self.head(slots)
        return logits, baux, haux


def class_read_diversity_loss(attn: torch.Tensor) -> torch.Tensor:
    a = attn.float().mean(dim=0)
    a = F.normalize(a, dim=-1)
    sim = a @ a.t()
    offdiag = sim - torch.eye(sim.shape[0], device=sim.device)
    return F.relu(offdiag - 0.20).mean()


def phase_balance_loss(phase_mass: torch.Tensor, min_early: float, max_aggregate: float) -> torch.Tensor:
    usage = phase_mass.float().mean(dim=(0, 1))
    early_loss = F.relu(torch.tensor(float(min_early), device=usage.device) - usage[:4]).pow(2).mean()
    agg_loss = F.relu(usage[4] - float(max_aggregate)).pow(2)
    ent = -(usage.clamp_min(1e-8) * usage.clamp_min(1e-8).log()).sum()
    ent_floor = F.relu(torch.tensor(1.25, device=usage.device) - ent).pow(2)
    return early_loss + agg_loss + 0.25 * ent_floor


def aux_losses(logits: torch.Tensor, baux: BackboneAux, haux: Dict[str, torch.Tensor], args) -> Dict[str, torch.Tensor]:
    gate = baux.write_gates.float()
    upd = baux.update_norms.float()
    target = torch.tensor(float(args.write_target), device=logits.device)
    return {
        "write_budget": (gate.mean() - target).pow(2) if gate.numel() else torch.zeros((), device=logits.device),
        "update_alive": F.relu(torch.tensor(float(args.min_update_norm), device=logits.device) - upd.mean()).pow(2) if upd.numel() else torch.zeros((), device=logits.device),
        "class_read_div": class_read_diversity_loss(haux["class_slot_attention"]),
        "phase_balance": phase_balance_loss(haux["class_phase_mass"], args.min_early_phase_mass, args.max_aggregate_phase_mass),
        "slot_div": slot_diversity_loss(baux.slots),
        "logit_norm": logits.float().pow(2).mean(),
        "pair_update_norm": haux["pair_update_norm"].float(),
    }


def train_epoch(model, loader, opt, scaler, device, dtype, args, epoch: int):
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
            logits, baux, haux = model(wav)
            ce = F.cross_entropy(logits.float(), y)
            losses = aux_losses(logits, baux, haux, args)
            loss = ce
            loss = loss + args.lambda_write_budget * losses["write_budget"]
            loss = loss + args.lambda_update_alive * losses["update_alive"]
            loss = loss + args.lambda_class_read_div * losses["class_read_div"]
            loss = loss + args.lambda_phase_balance * losses["phase_balance"]
            loss = loss + args.lambda_slot_div * losses["slot_div"]
            loss = loss + args.lambda_logit_norm * losses["logit_norm"]
        if not torch.isfinite(loss):
            print("NONFINITE_LOSS skip", flush=True)
            continue
        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
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
            print(
                f"epoch {epoch:03d} step {step:05d} "
                f"loss={totals['loss']/max(1, totals['n']):.4f} "
                f"ce={totals['ce']/max(1, totals['n']):.4f} "
                f"acc={100*totals['correct']/max(1, totals['n']):.2f}%",
                flush=True,
            )
    out = {k: v / max(1, totals["n"]) for k, v in totals.items() if k != "correct"}
    out["acc"] = totals["correct"] / max(1, totals["n"])
    for k, v in aux_sum.items():
        out[k] = v / max(1, totals["n"])
    return out


@torch.no_grad()
def evaluate(model, loader, device, dtype, args):
    model.eval()
    use_amp = device.startswith("cuda") and dtype != torch.float32
    total_loss, correct, n = 0.0, 0, 0
    conf = torch.zeros(args.num_classes, args.num_classes, dtype=torch.long)
    last_report = None
    for step, (wav, y) in enumerate(loader, 1):
        if args.max_val_batches and step > args.max_val_batches:
            break
        wav = wav.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=use_amp):
            logits, baux, haux = model(wav)
            loss = F.cross_entropy(logits.float(), y)
        pred = logits.argmax(-1)
        bs = y.numel()
        total_loss += float(loss.detach().cpu()) * bs
        correct += int((pred == y).sum().detach().cpu())
        n += bs
        conf += torch.bincount((y.cpu() * args.num_classes + pred.cpu()), minlength=args.num_classes ** 2).view(args.num_classes, args.num_classes)

        attn = haux["class_slot_attention"].float().mean(dim=0).cpu()
        phase = haux["class_phase_mass"].float().mean(dim=0).cpu()
        gates = baux.write_gates.float().mean(dim=(0, 2)).cpu() if baux.write_gates.numel() else torch.empty(0)
        updates = baux.update_norms.float().mean(dim=(0, 2)).cpu() if baux.update_norms.numel() else torch.empty(0)
        top_reads = []
        for ci in range(attn.shape[0]):
            vals, idxs = torch.topk(attn[ci], k=min(5, attn.shape[1]))
            top_reads.append([{"slot": baux.slot_names[int(j)], "weight": float(v)} for v, j in zip(vals.tolist(), idxs.tolist())])
        last_report = {
            "write_gate_by_step": gates.tolist(),
            "update_norm_by_step": updates.tolist(),
            "class_top_reads": top_reads,
            "class_phase_mass": [
                {PHASE_NAMES[pi]: float(phase[ci, pi]) for pi in range(len(PHASE_NAMES))}
                for ci in range(phase.shape[0])
            ],
            "phase_mass_mean": {PHASE_NAMES[pi]: float(phase[:, pi].mean()) for pi in range(len(PHASE_NAMES))},
            "pair_update_norm": float(haux["pair_update_norm"].detach().cpu()),
            "class_write": float(haux["class_write"].detach().cpu()),
            "slot_count": len(baux.slot_names),
        }
    return {"loss": total_loss / max(1, n), "acc": correct / max(1, n), "n": n, "confusion": conf.tolist(), "report": last_report}


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
    model = ClassMatrixTransportClassifier(len(classes), args).to(device)
    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location=device)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=True)
        print(f"loaded init checkpoint: {args.init_checkpoint}", flush=True)
    print(f"loaded datasets: train={len(train_loader.dataset)} val={len(val_loader.dataset)} classes={classes}", flush=True)
    print(f"ClassMatrixTransport params={sum(p.numel() for p in model.parameters())} V={args.variants} L={args.layers} B={args.blocks} S={args.steps} D={args.dim} pairs={args.pair_slots} device={device} amp={args.amp}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda") and dtype == torch.float16)
    fields = [
        "epoch", "train_loss", "train_ce", "train_acc", "val_loss", "val_acc", "best_acc",
        "write_budget", "update_alive", "class_read_div", "phase_balance", "slot_div", "logit_norm", "pair_update_norm",
    ]
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()
    best, best_epoch = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        tr = train_epoch(model, train_loader, opt, scaler, device, dtype, args, epoch)
        va = evaluate(model, val_loader, device, dtype, args)
        if va["acc"] > best:
            best, best_epoch = va["acc"], epoch
            torch.save({"model": model.state_dict(), "args": vars(args), "classes": classes, "epoch": epoch, "best_acc": best}, out_dir / "best.pt")
        torch.save({"model": model.state_dict(), "args": vars(args), "classes": classes, "epoch": epoch, "best_acc": best}, out_dir / "last.pt")
        row = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "train_ce": tr["ce"],
            "train_acc": tr["acc"],
            "val_loss": va["loss"],
            "val_acc": va["acc"],
            "best_acc": best,
            "write_budget": tr.get("write_budget", 0.0),
            "update_alive": tr.get("update_alive", 0.0),
            "class_read_div": tr.get("class_read_div", 0.0),
            "phase_balance": tr.get("phase_balance", 0.0),
            "slot_div": tr.get("slot_div", 0.0),
            "logit_norm": tr.get("logit_norm", 0.0),
            "pair_update_norm": tr.get("pair_update_norm", 0.0),
        }
        with (out_dir / "metrics.csv").open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        write_json(out_dir / f"analysis_epoch_{epoch:03d}.json", {
            "epoch": epoch,
            "train": tr,
            "val": {"loss": va["loss"], "acc": va["acc"], "n": va["n"], "confusion": va["confusion"]},
            "best_acc": best,
            "best_epoch": best_epoch,
            "classes": classes,
            "train_counts": train_counts,
            "val_counts": val_counts,
            "matrix_report": va["report"],
        })
        print(f"epoch {epoch:03d}/{args.epochs} train={tr['loss']:.4f}/{100*tr['acc']:.2f}% val={va['loss']:.4f}/{100*va['acc']:.2f}% best={100*best:.2f}%@{best_epoch}", flush=True)
    write_json(out_dir / "final_report.json", {"best_acc": best, "best_epoch": best_epoch, "args": vars(args), "classes": classes})


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
    p.add_argument("--pair-slots", type=int, default=12)
    p.add_argument("--channel-stages", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.04)
    p.add_argument("--head-dropout", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.012)
    p.add_argument("--grad-clip", type=float, default=0.75)
    p.add_argument("--write-target", type=float, default=0.46)
    p.add_argument("--min-update-norm", type=float, default=0.20)
    p.add_argument("--phase-prior-strength", type=float, default=0.85)
    p.add_argument("--min-early-phase-mass", type=float, default=0.08)
    p.add_argument("--max-aggregate-phase-mass", type=float, default=0.42)
    p.add_argument("--lambda-write-budget", type=float, default=0.025)
    p.add_argument("--lambda-update-alive", type=float, default=0.005)
    p.add_argument("--lambda-class-read-div", type=float, default=0.035)
    p.add_argument("--lambda-phase-balance", type=float, default=0.045)
    p.add_argument("--lambda-slot-div", type=float, default=0.002)
    p.add_argument("--lambda-logit-norm", type=float, default=0.0007)
    p.add_argument("--amp", choices=["fp16", "bf16", "fp32", "off"], default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--max-val-batches", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--out-dir", default="./simple_butterfly_matrix_v3/runs/speechcommands_v3")
    p.add_argument("--init-checkpoint", default="")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
