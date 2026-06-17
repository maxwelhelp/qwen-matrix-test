#!/usr/bin/env python3
"""Simple transferable butterfly-matrix architecture.

The experiment keeps the core deliberately small:

  evidence -> block states -> fixed phase sequence -> all-slot matrix head

There is no internal router. Specialization comes from address, sequence, and
class reads over all intermediate slots. The task head is isolated so the same
backbone can be reused with another objective.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import torchaudio
except Exception:
    torchaudio = None


PHASES = ("extract", "compare", "suppress", "aggregate")
PRIMITIVES = ("channel_butterfly", "block_butterfly", "low_rank", "ctx_matrix", "product_gate", "phase_matrix")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def amp_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def entropy(p: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    p = p.float().clamp_min(eps)
    p = p / p.sum(dim=dim, keepdim=True).clamp_min(eps)
    return -(p * p.log()).sum(dim=dim)


class SyntheticMatrixTask(Dataset):
    """Balanced synthetic sequence task for fast sanity checks.

    Samples are 1-D signals. Classes differ by frequency, phase envelope, and a
    weak onset bump, so the model has to extract/compare/suppress rather than
    memorize a scalar mean.
    """

    def __init__(self, n: int, classes: int, length: int = 512):
        self.n = int(n)
        self.classes = int(classes)
        self.length = int(length)
        self.t = torch.linspace(0.0, 1.0, self.length)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        y = int(idx % self.classes)
        freq = 2.0 + float(y)
        phase = 0.17 * float(idx % 19)
        envelope = 0.55 + 0.45 * torch.sin(2.0 * math.pi * (1 + y % 3) * self.t).abs()
        onset_center = 0.12 + 0.07 * (y % 5)
        onset = torch.exp(-((self.t - onset_center) ** 2) / 0.0025)
        wav = envelope * torch.sin(2.0 * math.pi * freq * self.t + phase)
        wav = wav + 0.35 * torch.cos(2.0 * math.pi * (freq + 0.5) * self.t)
        wav = wav + 0.25 * onset + 0.04 * torch.randn_like(wav)
        return wav.unsqueeze(0).float(), y


class SpeechCommandsBalanced(Dataset):
    def __init__(self, root: str, subset: str, classes: Sequence[str], limit: int = 0, download: bool = False):
        if torchaudio is None:
            raise RuntimeError("torchaudio is not available; use --synthetic for smoke tests")
        self.classes = list(classes)
        self.class_to_id = {c: i for i, c in enumerate(self.classes)}
        self.ds = torchaudio.datasets.SPEECHCOMMANDS(root=root, subset=subset, download=download)
        per_class_limit = None
        if limit and limit > 0:
            per_class_limit = max(1, math.ceil(limit / max(1, len(self.classes))))
        buckets: Dict[str, List[int]] = {c: [] for c in self.classes}
        for i in range(len(self.ds)):
            try:
                label = Path(self.ds._walker[i]).parent.name
            except Exception:
                label = str(self.ds[i][2])
            if label not in self.class_to_id:
                continue
            if per_class_limit is not None and len(buckets[label]) >= per_class_limit:
                continue
            buckets[label].append(i)
            if per_class_limit is not None and all(len(buckets[c]) >= per_class_limit for c in self.classes):
                break
        keep: List[int] = []
        max_len = max((len(v) for v in buckets.values()), default=0)
        for j in range(max_len):
            for c in self.classes:
                if j < len(buckets[c]):
                    keep.append(buckets[c][j])
                    if limit and len(keep) >= limit:
                        break
            if limit and len(keep) >= limit:
                break
        self.keep = keep
        self.counts = {c: len(buckets[c]) for c in self.classes}

    def __len__(self) -> int:
        return len(self.keep)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        wav, _sr, label, *_ = self.ds[self.keep[idx]]
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav.float(), self.class_to_id[str(label)]


def collate_wavs(batch, target_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
    wavs, ys = [], []
    for wav, y in batch:
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if wav.shape[-1] < target_length:
            wav = F.pad(wav, (0, target_length - wav.shape[-1]))
        elif wav.shape[-1] > target_length:
            wav = wav[..., :target_length]
        wavs.append(wav)
        ys.append(int(y))
    return torch.stack(wavs, dim=0), torch.tensor(ys, dtype=torch.long)


def make_loaders(args):
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    if args.synthetic:
        train_ds = SyntheticMatrixTask(args.train_limit or 1024, len(classes), args.synthetic_length)
        val_ds = SyntheticMatrixTask(args.val_limit or 256, len(classes), args.synthetic_length)
        counts = {c: (args.train_limit or 1024) // len(classes) for c in classes}
        train_counts, val_counts = counts, {c: (args.val_limit or 256) // len(classes) for c in classes}
        target_length = args.synthetic_length
    else:
        train_ds = SpeechCommandsBalanced(args.data_root, "training", classes, args.train_limit, args.download)
        val_ds = SpeechCommandsBalanced(args.data_root, "validation", classes, args.val_limit, args.download)
        train_counts = getattr(train_ds, "counts", {})
        val_counts = getattr(val_ds, "counts", {})
        target_length = int(args.sample_rate * args.seconds)
    collate = lambda b: collate_wavs(b, target_length)
    train = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=args.pin_memory, collate_fn=collate)
    val = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.workers, pin_memory=args.pin_memory, collate_fn=collate)
    return train, val, classes, train_counts, val_counts


class MatrixEvidence(nn.Module):
    """Evidence adapter that outputs `[batch, evidence_cells, dim]`.

    The trainable part is a scalar-to-vector matrix projection. Mel/pooling is a
    fixed measurement step, not a learned route.
    """

    def __init__(self, sample_rate: int, n_mels: int, hop_length: int, evidence_cells: int, dim: int):
        super().__init__()
        self.evidence_cells = int(evidence_cells)
        if torchaudio is not None:
            self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=400, hop_length=hop_length, n_mels=n_mels, power=2.0)
        else:
            self.mel = None
        self.scalar_weight = nn.Parameter(torch.randn(1, dim) * 0.04)
        self.scalar_bias = nn.Parameter(torch.zeros(dim))
        self.pos = nn.Parameter(torch.randn(evidence_cells, dim) * 0.02)
        self.norm = nn.LayerNorm(dim)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=wav.device.type, enabled=False):
            wav32 = wav.float()
            if self.mel is not None and wav32.shape[-1] >= 1024:
                x = self.mel(wav32.squeeze(1)).float().clamp_min(1e-5).log()
            else:
                frame = min(320, max(16, wav32.shape[-1] // 8))
                hop = max(8, frame // 2)
                unfolded = wav32.squeeze(1).unfold(-1, frame, hop)
                x = unfolded.abs().mean(dim=-1).unsqueeze(1).repeat(1, 16, 1)
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            x = (x - x.mean(dim=(-2, -1), keepdim=True)) / x.std(dim=(-2, -1), keepdim=True).clamp_min(1e-4)
            grid_side = max(1, int(math.sqrt(self.evidence_cells)))
            pooled = F.adaptive_avg_pool2d(x.unsqueeze(1), (grid_side, math.ceil(self.evidence_cells / grid_side))).flatten(1)
            if pooled.shape[1] < self.evidence_cells:
                pooled = pooled.repeat(1, math.ceil(self.evidence_cells / pooled.shape[1]))
            feats = pooled[:, : self.evidence_cells]
        out = feats.unsqueeze(-1).to(self.scalar_weight.dtype) @ self.scalar_weight
        out = out + self.scalar_bias.view(1, 1, -1) + self.pos.view(1, self.evidence_cells, -1)
        return self.norm(out.to(device=wav.device))


class ChannelButterfly(nn.Module):
    def __init__(self, dim: int, stages: int = 4):
        super().__init__()
        self.dim = int(dim)
        self.padded_dim = self.dim if self.dim % 2 == 0 else self.dim + 1
        self.pairs = self.padded_dim // 2
        self.stages = int(stages)
        eye = torch.eye(2).view(1, 1, 2, 2).repeat(self.stages, self.pairs, 1, 1)
        self.weight = nn.Parameter(eye + 0.03 * torch.randn_like(eye))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        if self.padded_dim != self.dim:
            y = F.pad(y, (0, self.padded_dim - self.dim))
        for si in range(self.stages):
            shift = 0 if si == 0 else 2 ** (si - 1)
            if shift:
                y = torch.roll(y, shifts=shift, dims=-1)
            yp = y.reshape(*y.shape[:-1], self.pairs, 2)
            w = self.weight[si].to(device=x.device, dtype=x.dtype)
            y = torch.einsum("...pi,pij->...pj", yp, w).reshape(*y.shape[:-1], self.padded_dim)
            if shift:
                y = torch.roll(y, shifts=-shift, dims=-1)
        return y[..., : self.dim]


class BlockButterfly(nn.Module):
    def __init__(self, blocks: int):
        super().__init__()
        self.blocks = int(blocks)
        stages = max(1, math.ceil(math.log2(max(2, self.blocks))))
        self.stages = stages
        self.weight = nn.Parameter(torch.eye(2).view(1, 1, 2, 2).repeat(stages, max(1, blocks // 2), 1, 1) + 0.03 * torch.randn(stages, max(1, blocks // 2), 2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, blocks, dim]
        if self.blocks < 2:
            return x
        y = x
        for si in range(self.stages):
            stride = 2 ** si
            order = self._stage_order(stride, x.device)
            inv = torch.empty_like(order)
            inv[order] = torch.arange(order.numel(), device=x.device)
            yp = y.index_select(1, order)
            if yp.shape[1] % 2 != 0:
                yp = torch.cat([yp, yp[:, -1:, :]], dim=1)
            pairs = yp.shape[1] // 2
            w = self.weight[si, :pairs].to(device=x.device, dtype=x.dtype)
            mixed = torch.einsum("bpjd,pij->bpid", yp.reshape(x.shape[0], pairs, 2, x.shape[-1]), w)
            y = mixed.reshape(x.shape[0], pairs * 2, x.shape[-1])[:, : self.blocks, :].index_select(1, inv)
        return y

    def _stage_order(self, stride: int, device: torch.device) -> torch.Tensor:
        idx = torch.arange(self.blocks, device=device)
        if stride <= 1:
            return idx
        groups = []
        used = torch.zeros(self.blocks, dtype=torch.bool, device=device)
        for i in range(self.blocks):
            if used[i]:
                continue
            j = (i + stride) % self.blocks
            groups.extend([i, int(j)])
            used[i] = True
            used[j] = True
        rest = [int(i) for i in idx.tolist() if i not in groups]
        return torch.tensor(groups + rest, dtype=torch.long, device=device)


class ButterflyMatrixStep(nn.Module):
    """One fixed matrix step. No operation selection happens here."""

    def __init__(self, dim: int, blocks: int, phase: str, channel_stages: int, dropout: float):
        super().__init__()
        self.phase = str(phase)
        self.dim = int(dim)
        self.channel = ChannelButterfly(dim, channel_stages)
        self.block = BlockButterfly(blocks)
        rank = max(8, dim // 4)
        self.low_a = nn.Parameter(torch.randn(dim, rank) * 0.04)
        self.low_b = nn.Parameter(torch.randn(rank, dim) * 0.04)
        self.ctx_w = nn.Parameter(torch.randn(dim, dim) * 0.04)
        self.gate_h = nn.Parameter(torch.randn(dim, dim) * 0.02)
        self.gate_c = nn.Parameter(torch.randn(dim, dim) * 0.02)
        self.gate_bias = nn.Parameter(torch.full((dim,), -0.4))
        self.write_logit = nn.Parameter(torch.tensor(-0.2))
        self.primitive_gain = nn.Parameter(torch.full((len(PRIMITIVES),), -0.15))
        self.op_transition = nn.Parameter(self._transition_prior(phase))
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    @staticmethod
    def _transition_prior(phase: str) -> torch.Tensor:
        p = len(PRIMITIVES)
        m = torch.eye(p) * 0.35
        idx = {name: i for i, name in enumerate(PRIMITIVES)}
        if phase == "extract":
            for name, val in (("channel_butterfly", 0.75), ("ctx_matrix", 0.45), ("phase_matrix", 0.40)):
                m[idx[name], idx[name]] = val
            m[idx["phase_matrix"], idx["ctx_matrix"]] = 0.25
        elif phase == "compare":
            for name, val in (("channel_butterfly", 0.50), ("low_rank", 0.55), ("product_gate", 0.45), ("phase_matrix", 0.60)):
                m[idx[name], idx[name]] = val
            m[idx["phase_matrix"], idx["low_rank"]] = 0.20
        elif phase == "suppress":
            for name, val in (("block_butterfly", 0.45), ("product_gate", 0.40), ("phase_matrix", 0.85)):
                m[idx[name], idx[name]] = val
            m[idx["phase_matrix"], idx["block_butterfly"]] = 0.25
        elif phase == "aggregate":
            for name, val in (("block_butterfly", 0.75), ("channel_butterfly", 0.45), ("low_rank", 0.35), ("phase_matrix", 0.65)):
                m[idx[name], idx[name]] = val
            m[idx["phase_matrix"], idx["block_butterfly"]] = 0.30
        return m + 0.01 * torch.randn(p, p)

    def forward(self, h: torch.Tensor, ctx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx_m = ctx @ self.ctx_w.to(device=h.device, dtype=h.dtype)
        channel = self.channel(h + ctx_m)
        block = self.block(h)
        low = (h @ self.low_a.to(device=h.device, dtype=h.dtype)) @ self.low_b.to(device=h.device, dtype=h.dtype)
        gate = torch.sigmoid(h @ self.gate_h.to(device=h.device, dtype=h.dtype) + ctx_m @ self.gate_c.to(device=h.device, dtype=h.dtype) + self.gate_bias.to(device=h.device, dtype=h.dtype))

        if self.phase == "extract":
            phase = channel + 0.50 * ctx_m
        elif self.phase == "compare":
            phase = self.channel(h - ctx_m)
        elif self.phase == "suppress":
            common = block.mean(dim=1, keepdim=True)
            phase = -gate * common
        elif self.phase == "aggregate":
            mean = h.mean(dim=1, keepdim=True).expand_as(h)
            phase = self.block(h + mean)
        else:
            phase = channel

        cands = torch.stack([channel, block, low, ctx_m, h * torch.tanh(ctx_m), phase], dim=2)
        trans = self.op_transition.to(device=h.device, dtype=h.dtype)
        mixed = torch.einsum("pq,bnqd->bnpd", trans, cands)
        primitive_gain = torch.sigmoid(self.primitive_gain.to(device=h.device, dtype=h.dtype)).view(1, 1, len(PRIMITIVES), 1)
        update = (primitive_gain * mixed).sum(dim=2) / primitive_gain.sum(dim=2).clamp_min(1e-4)

        dynamic_write = torch.sigmoid(self.write_logit.to(device=h.device, dtype=h.dtype))
        h_next = self.norm(h + dynamic_write * self.drop(update * gate))
        update_norm = update.detach().float().norm(dim=-1)
        return h_next, dynamic_write.expand(h.shape[0], h.shape[1]), update_norm


@dataclass
class BackboneAux:
    slots: torch.Tensor
    slot_names: List[str]
    write_gates: torch.Tensor
    update_norms: torch.Tensor


class ButterflyMatrixBackbone(nn.Module):
    """Task-transferable matrix backbone.

    The output is a bank of named slots. A task can replace only the head while
    keeping this backbone.
    """

    def __init__(self, dim: int, evidence_cells: int, layers: int, blocks: int, steps: int, sample_rate: int, n_mels: int, hop_length: int, channel_stages: int, dropout: float):
        super().__init__()
        self.dim = int(dim)
        self.layers = int(layers)
        self.blocks = int(blocks)
        self.steps = int(steps)
        self.evidence = MatrixEvidence(sample_rate, n_mels, hop_length, evidence_cells, dim)
        self.block_query = nn.Parameter(torch.randn(blocks, dim) * 0.04)
        self.block_bias = nn.Parameter(torch.randn(blocks, dim) * 0.03)
        self.phase_ctx = nn.Parameter(torch.randn(layers, steps, dim, dim) * 0.025)
        self.layer_bias = nn.Parameter(torch.randn(layers, dim) * 0.02)
        self.step_bias = nn.Parameter(torch.randn(steps, dim) * 0.02)
        self.norm = nn.LayerNorm(dim)
        phase_names = [PHASES[min(i, len(PHASES) - 1)] for i in range(layers)]
        self.units = nn.ModuleList([
            ButterflyMatrixStep(dim, blocks, phase_names[l], channel_stages, dropout)
            for l in range(layers)
            for _s in range(steps)
        ])

    def _block_init(self, evidence: torch.Tensor) -> torch.Tensor:
        q = self.block_query.to(device=evidence.device, dtype=evidence.dtype)
        score = torch.einsum("nd,bed->bne", q, evidence) / math.sqrt(evidence.shape[-1])
        attn = torch.softmax(score.float(), dim=-1).to(evidence.dtype)
        h = torch.einsum("bne,bed->bnd", attn, evidence)
        return self.norm(h + self.block_bias.to(device=evidence.device, dtype=evidence.dtype).view(1, self.blocks, self.dim))

    def _evidence_context(self, h: torch.Tensor, evidence: torch.Tensor, l: int, s: int) -> torch.Tensor:
        w = self.phase_ctx[l, s].to(device=h.device, dtype=h.dtype)
        q = h @ w
        score = torch.einsum("bnd,bed->bne", q, evidence) / math.sqrt(h.shape[-1])
        attn = torch.softmax(score.float(), dim=-1).to(h.dtype)
        ctx = torch.einsum("bne,bed->bnd", attn, evidence)
        phase_gain = 1.0 / float(1 + l + s)
        return phase_gain * ctx

    def forward(self, wav: torch.Tensor) -> Tuple[torch.Tensor, BackboneAux]:
        evidence = self.evidence(wav)
        h = self._block_init(evidence)
        slots = [h]
        names = [f"input.B{b}" for b in range(self.blocks)]
        gates, updates = [], []
        for l in range(self.layers):
            h = self.norm(h + self.layer_bias[l].to(device=h.device, dtype=h.dtype).view(1, 1, -1))
            for s in range(self.steps):
                h = h + self.step_bias[s].to(device=h.device, dtype=h.dtype).view(1, 1, -1)
                ctx = self._evidence_context(h, evidence, l, s)
                unit = self.units[l * self.steps + s]
                h, gate, update_norm = unit(h, ctx)
                slots.append(h)
                gates.append(gate)
                updates.append(update_norm)
                for b in range(self.blocks):
                    names.append(f"L{l}.{PHASES[min(l, len(PHASES)-1)]}.B{b}.S{s}")
        slot_tensor = torch.stack(slots, dim=1)  # [batch, slot_steps, blocks, dim]
        flat_slots = slot_tensor.reshape(wav.shape[0], -1, self.dim)
        aux = BackboneAux(
            slots=flat_slots,
            slot_names=names,
            write_gates=torch.stack(gates, dim=1) if gates else torch.empty(wav.shape[0], 0, self.blocks, device=wav.device),
            update_norms=torch.stack(updates, dim=1) if updates else torch.empty(wav.shape[0], 0, self.blocks, device=wav.device),
        )
        return flat_slots, aux


class MatrixClassificationHead(nn.Module):
    """Replaceable matrix head for classification tasks."""

    def __init__(self, dim: int, classes: int, dropout: float):
        super().__init__()
        self.classes = int(classes)
        self.class_query = nn.Parameter(torch.randn(classes, dim) * 0.04)
        self.key_w = nn.Parameter(torch.randn(dim, dim) * 0.04)
        self.value_w = nn.Parameter(torch.randn(dim, dim) * 0.04)
        self.out_w = nn.Parameter(torch.randn(classes, dim) * 0.04)
        self.bias = nn.Parameter(torch.zeros(classes))
        self.drop = nn.Dropout(dropout)

    def forward(self, slots: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        k = slots @ self.key_w.to(device=slots.device, dtype=slots.dtype)
        v = self.drop(slots @ self.value_w.to(device=slots.device, dtype=slots.dtype))
        q = self.class_query.to(device=slots.device, dtype=slots.dtype)
        score = torch.einsum("cd,bsd->bcs", q, k) / math.sqrt(slots.shape[-1])
        attn = torch.softmax(score.float(), dim=-1).to(slots.dtype)
        read = torch.einsum("bcs,bsd->bcd", attn, v)
        logits = torch.einsum("bcd,cd->bc", read, self.out_w.to(device=slots.device, dtype=slots.dtype)) + self.bias.to(device=slots.device, dtype=slots.dtype)
        return logits, {"class_slot_attention": attn.detach(), "class_read": read.detach()}


class SimpleButterflyMatrixClassifier(nn.Module):
    def __init__(self, classes: int, args):
        super().__init__()
        self.backbone = ButterflyMatrixBackbone(
            dim=args.dim,
            evidence_cells=args.evidence_cells,
            layers=args.layers,
            blocks=args.blocks,
            steps=args.steps,
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


def class_read_diversity_loss(attn: torch.Tensor) -> torch.Tensor:
    # attn [B,C,S]. Penalize classes that read the same slot distribution.
    a = attn.float().mean(dim=0)
    a = F.normalize(a, dim=-1)
    sim = a @ a.t()
    offdiag = sim - torch.eye(sim.shape[0], device=sim.device)
    return F.relu(offdiag - 0.25).mean()


def slot_diversity_loss(slots: torch.Tensor) -> torch.Tensor:
    s = slots.float().mean(dim=0)
    s = F.normalize(s, dim=-1)
    sim = s @ s.t()
    offdiag = sim - torch.eye(sim.shape[0], device=sim.device)
    return F.relu(offdiag - 0.55).mean()


def aux_losses(logits: torch.Tensor, baux: BackboneAux, haux: Dict[str, torch.Tensor], args) -> Dict[str, torch.Tensor]:
    gate = baux.write_gates.float()
    upd = baux.update_norms.float()
    target = torch.tensor(float(args.write_target), device=logits.device)
    return {
        "write_budget": (gate.mean() - target).pow(2) if gate.numel() else torch.zeros((), device=logits.device),
        "update_alive": F.relu(torch.tensor(float(args.min_update_norm), device=logits.device) - upd.mean()).pow(2) if upd.numel() else torch.zeros((), device=logits.device),
        "class_read_div": class_read_diversity_loss(haux["class_slot_attention"]),
        "slot_div": slot_diversity_loss(baux.slots),
        "logit_norm": logits.float().pow(2).mean(),
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
    model = SimpleButterflyMatrixClassifier(len(classes), args).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"loaded datasets: train={len(train_loader.dataset)} val={len(val_loader.dataset)} classes={classes}", flush=True)
    print(f"SimpleButterflyMatrix params={params} L={args.layers} B={args.blocks} S={args.steps} D={args.dim} device={device} amp={args.amp}", flush=True)

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "bias" in name or "norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    opt = torch.optim.AdamW([
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=args.lr, betas=(0.9, 0.95))
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
        analysis = {
            "epoch": epoch,
            "train": tr,
            "val": {"loss": va["loss"], "acc": va["acc"], "n": va["n"], "confusion": va["confusion"]},
            "best_acc": best,
            "best_epoch": best_epoch,
            "classes": classes,
            "train_counts": train_counts,
            "val_counts": val_counts,
            "matrix_report": va["report"],
        }
        write_json(out_dir / f"analysis_epoch_{epoch:03d}.json", analysis)
        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train={tr['loss']:.4f}/{100*tr['acc']:.2f}% "
            f"val={va['loss']:.4f}/{100*va['acc']:.2f}% "
            f"best={100*best:.2f}%@{best_epoch}",
            flush=True,
        )
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
    p.add_argument("--channel-stages", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.04)
    p.add_argument("--head-dropout", type=float, default=0.04)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=0.8)
    p.add_argument("--write-target", type=float, default=0.45)
    p.add_argument("--min-update-norm", type=float, default=0.20)
    p.add_argument("--lambda-write-budget", type=float, default=0.025)
    p.add_argument("--lambda-update-alive", type=float, default=0.005)
    p.add_argument("--lambda-class-read-div", type=float, default=0.020)
    p.add_argument("--lambda-slot-div", type=float, default=0.002)
    p.add_argument("--lambda-logit-norm", type=float, default=0.0004)
    p.add_argument("--amp", choices=["fp16", "bf16", "fp32", "off"], default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--max-val-batches", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--out-dir", default="./runs/simple_butterfly_matrix")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
