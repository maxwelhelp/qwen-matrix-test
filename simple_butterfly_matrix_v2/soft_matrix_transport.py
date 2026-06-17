#!/usr/bin/env python3
"""v2: fully differentiable soft matrix-transport butterfly architecture.

This version removes the top-k idea entirely. It keeps all primitive and variant
paths soft:

  primitive_outputs -> primitive_flow_matrix -> variant_transport_matrix -> slots

No branch is discretely selected. The compute preference is a continuous matrix
flow, initialized from a phase basis and trained end-to-end.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simple_butterfly_matrix.simple_butterfly_matrix import (  # noqa: E402
    BackboneAux,
    BlockButterfly,
    ChannelButterfly,
    MatrixClassificationHead,
    MatrixEvidence,
    PHASES,
    PRIMITIVES,
    amp_dtype,
    ensure_dir,
    evaluate,
    make_loaders,
    set_seed,
    train_epoch,
    write_json,
)


class SoftMatrixTransportStep(nn.Module):
    """All primitives and variants remain alive through soft matrices."""

    def __init__(self, dim: int, blocks: int, variants: int, phase: str, channel_stages: int, dropout: float):
        super().__init__()
        self.dim = int(dim)
        self.blocks = int(blocks)
        self.variants = int(variants)
        self.phase = str(phase)
        self.channel = ChannelButterfly(dim, channel_stages)
        self.block = BlockButterfly(blocks)
        rank = max(8, dim // 4)
        self.low_a = nn.Parameter(torch.randn(dim, rank) * 0.04)
        self.low_b = nn.Parameter(torch.randn(rank, dim) * 0.04)
        self.ctx_w = nn.Parameter(torch.randn(dim, dim) * 0.04)
        self.gate_h = nn.Parameter(torch.randn(dim, dim) * 0.02)
        self.gate_c = nn.Parameter(torch.randn(dim, dim) * 0.02)
        self.gate_bias = nn.Parameter(torch.full((dim,), -0.35))

        self.primitive_flow = nn.Parameter(self._primitive_prior(phase))
        self.variant_transport = nn.Parameter(self._variant_prior(variants))
        self.variant_primitive_bias = nn.Parameter(torch.zeros(variants, len(PRIMITIVES)))
        self.write_logit = nn.Parameter(torch.tensor(-0.15))
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    @staticmethod
    def _primitive_prior(phase: str) -> torch.Tensor:
        p = len(PRIMITIVES)
        m = torch.eye(p) * 0.55
        idx = {name: i for i, name in enumerate(PRIMITIVES)}
        if phase == "extract":
            m[idx["channel_butterfly"], idx["ctx_matrix"]] = 0.35
            m[idx["phase_matrix"], idx["channel_butterfly"]] = 0.35
        elif phase == "compare":
            m[idx["phase_matrix"], idx["low_rank"]] = 0.35
            m[idx["product_gate"], idx["ctx_matrix"]] = 0.25
        elif phase == "suppress":
            m[idx["phase_matrix"], idx["block_butterfly"]] = 0.45
            m[idx["product_gate"], idx["phase_matrix"]] = 0.25
        elif phase == "aggregate":
            m[idx["block_butterfly"], idx["channel_butterfly"]] = 0.35
            m[idx["phase_matrix"], idx["block_butterfly"]] = 0.40
        return m + 0.01 * torch.randn(p, p)

    @staticmethod
    def _variant_prior(variants: int) -> torch.Tensor:
        m = torch.eye(variants) * 1.30
        for i in range(variants):
            m[i, (i + 1) % variants] = 0.20
            m[i, (i - 1) % variants] = 0.20
        return m

    def _primitive_outputs(self, h: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        # h/ctx: [B,V,N,D], output [B,V,N,P,D]
        B, V, N, D = h.shape
        flat_h = h.reshape(B * V, N, D)
        flat_ctx = ctx.reshape(B * V, N, D)
        ctx_m = flat_ctx @ self.ctx_w.to(device=h.device, dtype=h.dtype)
        channel = self.channel(flat_h + ctx_m)
        block = self.block(flat_h)
        low = (flat_h @ self.low_a.to(device=h.device, dtype=h.dtype)) @ self.low_b.to(device=h.device, dtype=h.dtype)
        product = flat_h * torch.tanh(ctx_m)
        gate = torch.sigmoid(
            flat_h @ self.gate_h.to(device=h.device, dtype=h.dtype)
            + ctx_m @ self.gate_c.to(device=h.device, dtype=h.dtype)
            + self.gate_bias.to(device=h.device, dtype=h.dtype)
        )
        if self.phase == "extract":
            phase = channel + 0.50 * ctx_m
        elif self.phase == "compare":
            phase = self.channel(flat_h - ctx_m)
        elif self.phase == "suppress":
            phase = -gate * block.mean(dim=1, keepdim=True)
        elif self.phase == "aggregate":
            phase = self.block(flat_h + flat_h.mean(dim=1, keepdim=True))
        else:
            phase = channel
        cands = torch.stack([channel, block, low, ctx_m, product, phase], dim=2)
        return cands.reshape(B, V, N, len(PRIMITIVES), D)

    def forward(self, h: torch.Tensor, ctx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cands = self._primitive_outputs(h, ctx)
        flow = self.primitive_flow.to(device=h.device, dtype=h.dtype)
        mixed = torch.einsum("pq,bvnqd->bvnpd", flow, cands)
        primitive_gate = torch.sigmoid(self.variant_primitive_bias.to(device=h.device, dtype=h.dtype)).view(1, self.variants, 1, len(PRIMITIVES), 1)
        update = (primitive_gate * mixed).sum(dim=3) / primitive_gate.sum(dim=3).clamp_min(1e-4)

        write = torch.sigmoid(self.write_logit.to(device=h.device, dtype=h.dtype))
        local_next = self.norm(h + write * self.drop(update))

        transport = torch.softmax(self.variant_transport.float(), dim=-1).to(h.dtype)
        transported = torch.einsum("uv,bvnd->bund", transport, local_next)
        h_next = self.norm(0.65 * local_next + 0.35 * transported)

        update_norm = update.detach().float().norm(dim=-1).mean(dim=1)
        variant_entropy = -(transport.float().clamp_min(1e-8) * transport.float().clamp_min(1e-8).log()).sum(dim=-1).mean()
        return h_next, write.expand(h.shape[0], self.blocks), update_norm, variant_entropy.detach()


class SoftMatrixTransportBackbone(nn.Module):
    """Backbone with parallel soft variants and differentiable transport."""

    def __init__(self, dim: int, evidence_cells: int, layers: int, blocks: int, steps: int, variants: int, sample_rate: int, n_mels: int, hop_length: int, channel_stages: int, dropout: float):
        super().__init__()
        self.dim = int(dim)
        self.layers = int(layers)
        self.blocks = int(blocks)
        self.steps = int(steps)
        self.variants = int(variants)
        self.evidence = MatrixEvidence(sample_rate, n_mels, hop_length, evidence_cells, dim)
        self.block_query = nn.Parameter(torch.randn(blocks, dim) * 0.04)
        self.variant_bias = nn.Parameter(torch.randn(variants, blocks, dim) * 0.025)
        self.phase_ctx = nn.Parameter(torch.randn(layers, steps, variants, dim, dim) * 0.022)
        self.layer_bias = nn.Parameter(torch.randn(layers, 1, 1, dim) * 0.02)
        self.step_bias = nn.Parameter(torch.randn(steps, 1, 1, dim) * 0.02)
        self.norm = nn.LayerNorm(dim)
        phase_names = [PHASES[min(i, len(PHASES) - 1)] for i in range(layers)]
        self.units = nn.ModuleList([
            SoftMatrixTransportStep(dim, blocks, variants, phase_names[l], channel_stages, dropout)
            for l in range(layers)
            for _s in range(steps)
        ])

    def _block_init(self, evidence: torch.Tensor) -> torch.Tensor:
        q = self.block_query.to(device=evidence.device, dtype=evidence.dtype)
        score = torch.einsum("nd,bed->bne", q, evidence) / math.sqrt(evidence.shape[-1])
        attn = torch.softmax(score.float(), dim=-1).to(evidence.dtype)
        base = torch.einsum("bne,bed->bnd", attn, evidence)
        h = base[:, None, :, :] + self.variant_bias.to(device=evidence.device, dtype=evidence.dtype).view(1, self.variants, self.blocks, self.dim)
        return self.norm(h)

    def _context(self, h: torch.Tensor, evidence: torch.Tensor, l: int, s: int) -> torch.Tensor:
        # Matrix attention per variant; still continuous and differentiable.
        w = self.phase_ctx[l, s].to(device=h.device, dtype=h.dtype)
        q = torch.einsum("bvnd,vdh->bvnh", h, w)
        score = torch.einsum("bvnd,bed->bvne", q, evidence) / math.sqrt(h.shape[-1])
        attn = torch.softmax(score.float(), dim=-1).to(h.dtype)
        ctx = torch.einsum("bvne,bed->bvnd", attn, evidence)
        return ctx / float(1 + l + s)

    def forward(self, wav: torch.Tensor) -> Tuple[torch.Tensor, BackboneAux]:
        evidence = self.evidence(wav)
        h = self._block_init(evidence)
        slot_banks = [h]
        names = [f"input.V{v}.B{b}" for v in range(self.variants) for b in range(self.blocks)]
        gates: List[torch.Tensor] = []
        updates: List[torch.Tensor] = []
        transport_entropy: List[torch.Tensor] = []
        for l in range(self.layers):
            h = self.norm(h + self.layer_bias[l].to(device=h.device, dtype=h.dtype))
            for s in range(self.steps):
                h = h + self.step_bias[s].to(device=h.device, dtype=h.dtype)
                ctx = self._context(h, evidence, l, s)
                unit = self.units[l * self.steps + s]
                h, gate, update_norm, v_ent = unit(h, ctx)
                slot_banks.append(h)
                gates.append(gate)
                updates.append(update_norm)
                transport_entropy.append(v_ent)
                phase = PHASES[min(l, len(PHASES) - 1)]
                names.extend([f"L{l}.{phase}.V{v}.B{b}.S{s}" for v in range(self.variants) for b in range(self.blocks)])
        slots = torch.stack(slot_banks, dim=1).reshape(wav.shape[0], -1, self.dim)
        aux = BackboneAux(
            slots=slots,
            slot_names=names,
            write_gates=torch.stack(gates, dim=1) if gates else torch.empty(wav.shape[0], 0, self.blocks, device=wav.device),
            update_norms=torch.stack(updates, dim=1) if updates else torch.empty(wav.shape[0], 0, self.blocks, device=wav.device),
        )
        aux.transport_entropy = torch.stack(transport_entropy).mean() if transport_entropy else torch.tensor(0.0, device=wav.device)
        return slots, aux


class SoftMatrixTransportClassifier(nn.Module):
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
        self.head = MatrixClassificationHead(args.dim, classes, args.head_dropout)

    def forward(self, wav: torch.Tensor):
        slots, baux = self.backbone(wav)
        logits, haux = self.head(slots)
        return logits, baux, haux


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
    model = SoftMatrixTransportClassifier(len(classes), args).to(device)
    print(f"loaded datasets: train={len(train_loader.dataset)} val={len(val_loader.dataset)} classes={classes}", flush=True)
    print(f"SoftMatrixTransport params={sum(p.numel() for p in model.parameters())} V={args.variants} L={args.layers} B={args.blocks} S={args.steps} D={args.dim} device={device} amp={args.amp}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda") and dtype == torch.float16)
    fields = ["epoch", "train_loss", "train_ce", "train_acc", "val_loss", "val_acc", "best_acc", "write_budget", "update_alive", "class_read_div", "slot_div", "logit_norm"]
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
            "slot_div": tr.get("slot_div", 0.0),
            "logit_norm": tr.get("logit_norm", 0.0),
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
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--evidence-cells", type=int, default=36)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--variants", type=int, default=3)
    p.add_argument("--channel-stages", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.04)
    p.add_argument("--head-dropout", type=float, default=0.04)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=0.8)
    p.add_argument("--write-target", type=float, default=0.46)
    p.add_argument("--min-update-norm", type=float, default=0.20)
    p.add_argument("--lambda-write-budget", type=float, default=0.025)
    p.add_argument("--lambda-update-alive", type=float, default=0.005)
    p.add_argument("--lambda-class-read-div", type=float, default=0.020)
    p.add_argument("--lambda-slot-div", type=float, default=0.002)
    p.add_argument("--lambda-logit-norm", type=float, default=0.0004)
    p.add_argument("--amp", choices=["fp16", "bf16", "fp32", "off"], default="fp32")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--max-val-batches", type=int, default=0)
    p.add_argument("--log-every", type=int, default=4)
    p.add_argument("--out-dir", default="./simple_butterfly_matrix_v2/runs/smoke")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
