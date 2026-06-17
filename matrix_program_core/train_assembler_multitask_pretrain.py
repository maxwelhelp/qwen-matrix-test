#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multitask synthetic pretrain for MatrixProgramAssemblerCore.

This is the next pretrain after the single-task assembler debug.
It deliberately mirrors what the real system sees:

    input/evidence + task/head key + solution family key
        -> MatrixProgramAssemblerCore
        -> synthetic task head

The checkpoint still contains the same `assembler_core` that audio transfer loads.
No routers are used. Context changes soft-flow logits through dense projections.
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
from typing import Dict, List, Sequence, Tuple

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

DEFAULT_TASK_FAMILIES = (
    "local_pattern",
    "global_summary",
    "selective_copy",
    "count_matching",
    "rare_event",
    "segment_summary",
    "denoise_reconstruct",
    "program_composition",
)

DEFAULT_SOLUTION_FAMILIES = (
    "short",
    "medium",
    "deep",
    "memory_heavy",
    "global_heavy",
    "local_only",
    "mixed",
)


def parse_csv_names(s: str, default: Sequence[str]) -> List[str]:
    if not s:
        return list(default)
    return [x.strip() for x in s.split(",") if x.strip()]


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


class MultiTaskAssemblerDataset(Dataset):
    """Synthetic multitask data that includes task/head context in evidence.

    The first evidence tokens are explicit keys:
      token 0: task key
      token 1: head/class query
      token 2: solution-family key
      token 3: task+class mixed request

    The rest is structured input evidence. This matches the real plan where the
    assembler sees input state plus task/head context.
    """

    def __init__(
        self,
        n: int,
        classes: int,
        evidence_cells: int,
        dim: int,
        task_families: Sequence[str],
        solution_families: Sequence[str],
        seed: int = 0,
        noise: float = 0.08,
    ):
        self.n = int(n)
        self.classes = int(classes)
        self.evidence_cells = int(evidence_cells)
        self.dim = int(dim)
        self.task_families = list(task_families)
        self.solution_families = list(solution_families)
        self.seed = int(seed)
        self.noise = float(noise)
        g = torch.Generator().manual_seed(seed + 12345)
        self.class_proto = F.normalize(torch.randn(classes, dim, generator=g), dim=-1)
        self.task_proto = F.normalize(torch.randn(len(self.task_families), dim, generator=g), dim=-1)
        self.solution_proto = F.normalize(torch.randn(len(self.solution_families), dim, generator=g), dim=-1)
        self.head_proto = F.normalize(torch.randn(classes, dim, generator=g), dim=-1)
        self.pos = torch.linspace(0.0, 1.0, evidence_cells).view(evidence_cells, 1)
        self.freq = torch.linspace(1.0, 6.0, dim).view(1, dim)

    def __len__(self) -> int:
        return self.n

    def _ids(self, idx: int) -> Tuple[int, int, int]:
        y = int(idx % self.classes)
        task = int((idx // self.classes) % len(self.task_families))
        sol = int((idx // (self.classes * len(self.task_families))) % len(self.solution_families))
        return y, task, sol

    def __getitem__(self, idx: int):
        y, task_id, sol_id = self._ids(idx)
        task = self.task_families[task_id]
        sol = self.solution_families[sol_id]
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + int(idx))
        proto = self.class_proto[y].view(1, self.dim)
        taskp = self.task_proto[task_id].view(1, self.dim)
        solp = self.solution_proto[sol_id].view(1, self.dim)
        headq = self.head_proto[y].view(1, self.dim)
        phase = 0.11 * float(idx % 37)
        center = (y + 1) / float(self.classes + 1)
        bump = torch.exp(-((self.pos - center) ** 2) / 0.012)
        wave_fast = torch.sin(2 * math.pi * (1 + y % 5) * self.pos * self.freq + phase)
        wave_slow = torch.cos(2 * math.pi * (1 + task_id % 4) * self.pos + phase * self.freq)
        segment = ((self.pos > (task_id % 4) * 0.20) & (self.pos < 0.35 + (task_id % 4) * 0.15)).float()
        evidence = 0.10 * torch.randn(self.evidence_cells, self.dim, generator=g)

        if task == "local_pattern":
            evidence = evidence + 0.55 * bump * proto + 0.25 * wave_fast + 0.25 * taskp
            target_vec = F.normalize(proto.squeeze(0) + 0.25 * wave_fast.mean(dim=0), dim=0)
        elif task == "global_summary":
            evidence = evidence + 0.35 * proto + 0.30 * wave_slow + 0.30 * taskp
            target_vec = F.normalize(evidence.mean(dim=0) + proto.squeeze(0), dim=0)
        elif task == "selective_copy":
            key_pos = int(4 + (y * 3 + task_id) % max(4, self.evidence_cells - 4))
            evidence = evidence + 0.20 * wave_slow + 0.20 * taskp
            evidence[key_pos : key_pos + 1] += 1.10 * proto + 0.50 * headq
            target_vec = F.normalize(evidence[key_pos] + 0.25 * proto.squeeze(0), dim=0)
        elif task == "count_matching":
            evidence = evidence + 0.20 * taskp + 0.15 * wave_fast
            for j in range(1 + (y % 4)):
                c = ((j + 1) * (y + 2)) % self.evidence_cells
                evidence[c : c + 1] += 0.80 * proto + 0.20 * headq
            target_vec = F.normalize(proto.squeeze(0) * float(1 + (y % 4)) + 0.10 * evidence.mean(dim=0), dim=0)
        elif task == "rare_event":
            evidence = evidence + 0.40 * wave_slow + 0.25 * taskp
            c = int((y * 7 + sol_id * 3) % self.evidence_cells)
            evidence[c : c + 1] += 1.40 * proto - 0.30 * wave_slow[c : c + 1]
            target_vec = F.normalize(evidence[c] + proto.squeeze(0), dim=0)
        elif task == "segment_summary":
            evidence = evidence + 0.20 * wave_fast + 0.20 * taskp + 0.35 * segment * proto
            target_vec = F.normalize((segment * evidence).sum(dim=0) / segment.sum().clamp_min(1.0) + proto.squeeze(0), dim=0)
        elif task == "denoise_reconstruct":
            clean = 0.45 * proto + 0.25 * wave_slow + 0.15 * wave_fast
            evidence = clean + 0.25 * torch.randn(self.evidence_cells, self.dim, generator=g) + 0.20 * taskp
            target_vec = F.normalize(clean.mean(dim=0) + proto.squeeze(0), dim=0)
        elif task == "program_composition":
            evidence = evidence + 0.30 * bump * proto + 0.25 * wave_fast + 0.25 * wave_slow + 0.25 * taskp
            target_vec = F.normalize((bump * evidence).mean(dim=0) + evidence.mean(dim=0) + proto.squeeze(0), dim=0)
        else:
            evidence = evidence + 0.35 * proto + 0.30 * wave_fast + 0.20 * taskp
            target_vec = F.normalize(evidence.mean(dim=0) + proto.squeeze(0), dim=0)

        # Solution family changes the path style without changing the target.
        if sol == "memory_heavy":
            evidence = evidence + 0.18 * solp
        elif sol == "global_heavy":
            evidence = evidence + 0.12 * evidence.mean(dim=0, keepdim=True) + 0.15 * solp
        elif sol == "local_only":
            evidence = evidence + 0.10 * wave_fast
        elif sol == "deep":
            evidence = evidence + 0.08 * torch.roll(evidence, shifts=1, dims=0) + 0.08 * solp
        elif sol == "mixed":
            evidence = evidence + 0.08 * wave_fast * torch.tanh(wave_slow) + 0.10 * solp
        else:
            evidence = evidence + 0.06 * solp

        # Explicit context tokens.
        evidence[0] = taskp.squeeze(0)
        evidence[1] = headq.squeeze(0)
        evidence[2] = solp.squeeze(0)
        evidence[3] = F.normalize(taskp.squeeze(0) + headq.squeeze(0) + solp.squeeze(0), dim=0)
        evidence = (evidence - evidence.mean(dim=0, keepdim=True)) / evidence.std(dim=0, keepdim=True).clamp_min(1e-4)
        evidence = evidence + self.noise * torch.randn(self.evidence_cells, self.dim, generator=g)
        return (
            evidence.float(),
            torch.tensor(y, dtype=torch.long),
            torch.tensor(task_id, dtype=torch.long),
            torch.tensor(sol_id, dtype=torch.long),
            target_vec.float(),
        )


class MultiTaskHead(nn.Module):
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
        self.classifier = nn.Linear(dim, classes)
        self.recon = nn.Linear(dim, dim)

    def forward(self, slots: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.query.to(device=slots.device, dtype=slots.dtype)
        k = self.key(slots)
        v = self.value(slots)
        score = torch.einsum("cd,nsd->ncs", q, k) / math.sqrt(slots.shape[-1])
        attn = torch.softmax(score.float(), dim=-1).to(slots.dtype)
        read = torch.einsum("ncs,nsd->ncd", attn, v)
        global_read = slots.mean(dim=1, keepdim=True).expand_as(read)
        cls_state = q.view(1, self.classes, -1).expand(slots.shape[0], -1, -1)
        h = self.norm(cls_state + self.mix(torch.cat([cls_state, read, global_read], dim=-1)))
        pooled = h.mean(dim=1)
        return self.classifier(pooled), self.recon(pooled)


def _add_prim(x: torch.Tensor, idx: Dict[str, int], names: Sequence[str], amount: float) -> None:
    for name in names:
        if name in idx:
            x[idx[name]] += amount


def _link(x: torch.Tensor, idx: Dict[str, int], a: str, b: str, amount: float) -> None:
    if a in idx and b in idx:
        x[idx[a], idx[b]] += amount


def build_multitask_flow_targets(
    cfg: AssemblerConfig,
    task_ids: torch.Tensor,
    solution_ids: torch.Tensor,
    task_families: Sequence[str],
    solution_families: Sequence[str],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    T = cfg.layers * cfg.steps
    N = int(task_ids.numel())
    B, K, A = cfg.blocks, cfg.primitive_slots, cfg.address_cells
    P = len(PRIMITIVES)
    mem0 = cfg.blocks
    glob0 = cfg.blocks + cfg.memory_cells
    idx = {n: i for i, n in enumerate(PRIMITIVES)}

    read = torch.full((T, N, B, K, A), 0.012, device=device)
    prim = torch.full((T, N, B, K, P), 0.018, device=device)
    slot_tr = torch.full((T, N, B, K, K), 0.018, device=device)
    prim_tr = torch.full((T, N, P, P), 0.010, device=device)
    comp = torch.full((T, N, B, K), 0.035, device=device)
    write = torch.full((T, N, B, A), 0.012, device=device)

    phase_prims = {
        "extract": ["ctx_matrix", "channel_butterfly", "phase_matrix"],
        "compare": ["low_rank", "product_gate", "phase_matrix"],
        "suppress": ["block_butterfly", "product_gate", "phase_matrix"],
        "aggregate": ["block_butterfly", "channel_butterfly", "phase_matrix"],
    }
    task_prims = {
        "local_pattern": ["ctx_matrix", "channel_butterfly", "phase_matrix"],
        "global_summary": ["block_butterfly", "channel_butterfly", "phase_matrix"],
        "selective_copy": ["ctx_matrix", "product_gate", "low_rank"],
        "count_matching": ["low_rank", "product_gate", "phase_matrix"],
        "rare_event": ["product_gate", "phase_matrix", "block_butterfly"],
        "segment_summary": ["ctx_matrix", "block_butterfly", "channel_butterfly"],
        "denoise_reconstruct": ["channel_butterfly", "low_rank", "ctx_matrix"],
        "program_composition": list(PRIMITIVES),
    }

    for n in range(N):
        task = task_families[int(task_ids[n].item())]
        sol = solution_families[int(solution_ids[n].item())]
        for t in range(T):
            layer = t // cfg.steps
            phase = PHASES[min(layer, len(PHASES) - 1)]
            names = list(dict.fromkeys(phase_prims.get(phase, []) + task_prims.get(task, [])))
            for b in range(B):
                for k in range(K):
                    # Read target.
                    read[t, n, b, k, b] += 1.0
                    if b > 0:
                        read[t, n, b, k, b - 1] += 0.12
                    if b + 1 < B:
                        read[t, n, b, k, b + 1] += 0.12
                    if cfg.memory_cells > 0 and sol != "local_only":
                        read[t, n, b, k, mem0 + (b + k + t) % cfg.memory_cells] += 0.35
                    if cfg.global_cells > 0 and sol in ("global_heavy", "mixed", "medium", "deep"):
                        read[t, n, b, k, glob0 + (k + t) % cfg.global_cells] += 0.30
                    if task in ("global_summary", "segment_summary") and cfg.global_cells > 0:
                        read[t, n, b, k, glob0 + k % cfg.global_cells] += 0.45
                    if task in ("selective_copy", "count_matching") and cfg.memory_cells > 0:
                        read[t, n, b, k, mem0 + k % cfg.memory_cells] += 0.45

                    # Primitive placement target.
                    for j, name in enumerate(names):
                        if name in idx:
                            prim[t, n, b, k, idx[name]] += 0.70 / (1 + abs(k - (j % K)))
                    if sol == "mixed":
                        prim[t, n, b, k, :] += 0.06
                    if sol == "short" and k > max(1, K // 2):
                        prim[t, n, b, k, idx.get("ctx_matrix", 0)] += 0.20
                    if sol == "deep":
                        prim[t, n, b, k, idx.get("phase_matrix", 0)] += 0.18

                    # Slot transitions.
                    slot_tr[t, n, b, k, k] += 0.80
                    if sol in ("deep", "mixed", "medium"):
                        slot_tr[t, n, b, k, (k - 1) % K] += 0.30
                        slot_tr[t, n, b, k, (k + 1) % K] += 0.22
                    if sol == "short":
                        slot_tr[t, n, b, k, 0] += 0.20

                    # Composition target.
                    if sol == "short":
                        comp[t, n, b, k] += 1.00 / (1 + k)
                    elif sol == "deep":
                        comp[t, n, b, k] += 0.60 + 0.25 * k
                    else:
                        comp[t, n, b, k] += 1.0 / (1 + abs(k - ((t + b) % K)))

                # Write target.
                write[t, n, b, b] += 1.0
                if cfg.memory_cells > 0 and sol in ("memory_heavy", "deep", "mixed", "medium"):
                    write[t, n, b, mem0 + (b + t) % cfg.memory_cells] += 0.45
                if cfg.global_cells > 0 and sol in ("global_heavy", "mixed"):
                    write[t, n, b, glob0 + b % cfg.global_cells] += 0.45
                if task in ("global_summary", "segment_summary", "program_composition") and cfg.global_cells > 0:
                    write[t, n, b, glob0 + (b + t) % cfg.global_cells] += 0.30

            # Primitive transition target.
            prim_tr[t, n] += torch.eye(P, device=device) * 0.18
            if task == "local_pattern":
                _link(prim_tr[t, n], idx, "ctx_matrix", "channel_butterfly", 0.9)
                _link(prim_tr[t, n], idx, "channel_butterfly", "phase_matrix", 0.8)
            elif task == "global_summary":
                _link(prim_tr[t, n], idx, "block_butterfly", "channel_butterfly", 0.9)
                _link(prim_tr[t, n], idx, "phase_matrix", "block_butterfly", 0.7)
            elif task == "selective_copy":
                _link(prim_tr[t, n], idx, "ctx_matrix", "product_gate", 0.9)
                _link(prim_tr[t, n], idx, "product_gate", "low_rank", 0.7)
            elif task == "count_matching":
                _link(prim_tr[t, n], idx, "low_rank", "product_gate", 0.9)
                _link(prim_tr[t, n], idx, "product_gate", "phase_matrix", 0.7)
            elif task == "rare_event":
                _link(prim_tr[t, n], idx, "product_gate", "phase_matrix", 0.9)
                _link(prim_tr[t, n], idx, "block_butterfly", "phase_matrix", 0.7)
            elif task == "denoise_reconstruct":
                _link(prim_tr[t, n], idx, "channel_butterfly", "low_rank", 0.8)
                _link(prim_tr[t, n], idx, "low_rank", "ctx_matrix", 0.6)
            else:
                _link(prim_tr[t, n], idx, "ctx_matrix", "channel_butterfly", 0.6)
                _link(prim_tr[t, n], idx, "low_rank", "product_gate", 0.6)
                _link(prim_tr[t, n], idx, "phase_matrix", "block_butterfly", 0.6)

    def norm(x, dim):
        return x / x.sum(dim=dim, keepdim=True).clamp_min(1e-8)

    return {
        "read_flow": norm(read, -1),
        "primitive_slot_flow": norm(prim, -1),
        "slot_transition_flow": norm(slot_tr, -1),
        "primitive_transition_flow": norm(prim_tr, -1),
        "slot_composition_flow": norm(comp, -1),
        "write_flow": norm(write, -1),
    }


def entropy_report(aux) -> Dict[str, float]:
    return {f"entropy_{k}": float(v.detach().cpu()) for k, v in aux.entropies.items()}


def evaluate(core, head, loader, device, dtype, args, task_families, solution_families, skill_weights):
    core.eval(); head.eval()
    use_amp = device.startswith("cuda") and dtype != torch.float32
    total = ce_total = rec_total = skill_total = 0.0
    correct = n = 0
    last_aux = None
    with torch.no_grad():
        for evidence, y, task_id, sol_id, target_vec in loader:
            evidence = evidence.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            task_id = task_id.to(device, non_blocking=True)
            sol_id = sol_id.to(device, non_blocking=True)
            target_vec = target_vec.to(device, non_blocking=True)
            flow_targets = build_multitask_flow_targets(core.cfg, task_id, sol_id, task_families, solution_families, torch.device(device))
            with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=use_amp):
                _cells, aux = core(evidence)
                logits, vec = head(aux.slots)
                ce = F.cross_entropy(logits.float(), y)
                rec = F.mse_loss(F.normalize(vec.float(), dim=-1), target_vec.float())
                skill, flow_losses = assembler_skill_loss(aux, flow_targets, skill_weights)
                loss = ce + args.lambda_recon * rec + args.lambda_skill * skill
            bs = y.numel()
            total += float(loss.detach().cpu()) * bs
            ce_total += float(ce.detach().cpu()) * bs
            rec_total += float(rec.detach().cpu()) * bs
            skill_total += float(skill.detach().cpu()) * bs
            correct += int((logits.argmax(-1) == y).sum().detach().cpu())
            n += bs
            last_aux = aux
    out = {
        "loss": total / max(1, n),
        "ce": ce_total / max(1, n),
        "recon": rec_total / max(1, n),
        "skill": skill_total / max(1, n),
        "acc": correct / max(1, n),
    }
    if last_aux is not None:
        out.update(entropy_report(last_aux))
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
    task_families = parse_csv_names(args.task_families, DEFAULT_TASK_FAMILIES)
    solution_families = parse_csv_names(args.solution_families, DEFAULT_SOLUTION_FAMILIES)

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
    head = MultiTaskHead(args.dim, args.classes, args.head_dropout).to(device)
    if args.init_assembler:
        ckpt = torch.load(args.init_assembler, map_location=device)
        state = ckpt.get("assembler_core", ckpt.get("core", ckpt))
        missing, unexpected = core.load_state_dict(state, strict=False)
        print(f"loaded init assembler: {args.init_assembler} missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    train_ds = MultiTaskAssemblerDataset(args.train_n, args.classes, args.evidence_cells, args.dim, task_families, solution_families, args.seed, args.noise)
    val_ds = MultiTaskAssemblerDataset(args.val_n, args.classes, args.evidence_cells, args.dim, task_families, solution_families, args.seed + 1000, args.noise)
    # Critical: validation must test new examples/noise, not a different label universe.
    # Keep the same class/task/head/solution prototypes across train/val so class ids
    # and task ids mean the same thing. Otherwise validation is effectively impossible.
    val_ds.class_proto = train_ds.class_proto.clone()
    val_ds.task_proto = train_ds.task_proto.clone()
    val_ds.solution_proto = train_ds.solution_proto.clone()
    val_ds.head_proto = train_ds.head_proto.clone()
    train = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=args.pin_memory)
    val = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.workers, pin_memory=args.pin_memory)
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
        "memory_usage", "global_usage", "lambda_skill_eff",
    ]
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    best = -1.0
    best_epoch = 0
    t0 = time.time()
    print(f"multitask pretrain tasks={task_families} solutions={solution_families} cfg={cfg}", flush=True)
    for ep in range(1, args.epochs + 1):
        core.train(); head.train()
        lambda_skill_eff = args.lambda_skill * min(1.0, ep / max(1, args.skill_warmup_epochs))
        total = ce_total = rec_total = skill_total = 0.0
        correct = n = 0
        for step, (evidence, y, task_id, sol_id, target_vec) in enumerate(train, 1):
            evidence = evidence.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            task_id = task_id.to(device, non_blocking=True)
            sol_id = sol_id.to(device, non_blocking=True)
            target_vec = target_vec.to(device, non_blocking=True)
            flow_targets = build_multitask_flow_targets(cfg, task_id, sol_id, task_families, solution_families, torch.device(device))
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=use_amp):
                _cells, aux = core(evidence)
                logits, vec = head(aux.slots)
                ce = F.cross_entropy(logits.float(), y)
                rec = F.mse_loss(F.normalize(vec.float(), dim=-1), target_vec.float())
                skill, _flow_losses = assembler_skill_loss(aux, flow_targets, skill_weights)
                entropy_keep = torch.zeros((), device=evidence.device)
                if args.lambda_entropy_keep > 0:
                    for ent in aux.entropies.values():
                        entropy_keep = entropy_keep + F.relu(torch.tensor(args.min_entropy, device=evidence.device) - ent.float()).pow(2)
                loss = ce + args.lambda_recon * rec + lambda_skill_eff * skill + args.lambda_entropy_keep * entropy_keep
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
                print(f"epoch {ep:03d} step {step:05d} loss={total/max(1,n):.4f} ce={ce_total/max(1,n):.4f} acc={100*correct/max(1,n):.2f}% skill={skill_total/max(1,n):.4f} lskill={lambda_skill_eff:.4f}", flush=True)

        va = evaluate(core, head, val, device, dtype, args, task_families, solution_families, skill_weights)
        train_acc = correct / max(1, n)
        if va["acc"] > best:
            best = va["acc"]
            best_epoch = ep
            torch.save({
                "assembler_core": core.state_dict(),
                "head": head.state_dict(),
                "config": cfg.__dict__,
                "args": vars(args),
                "task_families": task_families,
                "solution_families": solution_families,
                "best_acc": best,
                "epoch": ep,
                "primitives": list(PRIMITIVES),
            }, out / "assembler_multitask_best.pt")
        torch.save({
            "assembler_core": core.state_dict(),
            "head": head.state_dict(),
            "config": cfg.__dict__,
            "args": vars(args),
            "task_families": task_families,
            "solution_families": solution_families,
            "best_acc": best,
            "epoch": ep,
            "primitives": list(PRIMITIVES),
        }, out / "assembler_multitask_last.pt")
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
            "lambda_skill_eff": lambda_skill_eff,
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
        "task_families": task_families,
        "solution_families": solution_families,
        "checkpoint": str(out / "assembler_multitask_best.pt"),
    })


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="matrix_program_core/runs/assembler_multitask_debug")
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
    p.add_argument("--primitive-slots", type=int, default=4)
    p.add_argument("--memory-cells", type=int, default=4)
    p.add_argument("--global-cells", type=int, default=2)
    p.add_argument("--channel-stages", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.04)
    p.add_argument("--head-dropout", type=float, default=0.05)
    p.add_argument("--noise", type=float, default=0.08)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=0.75)
    p.add_argument("--lambda-skill", type=float, default=0.10)
    p.add_argument("--skill-warmup-epochs", type=int, default=3)
    p.add_argument("--lambda-recon", type=float, default=0.20)
    p.add_argument("--lambda-entropy-keep", type=float, default=0.005)
    p.add_argument("--min-entropy", type=float, default=0.55)
    p.add_argument("--w-read", type=float, default=0.25)
    p.add_argument("--w-primitive", type=float, default=0.35)
    p.add_argument("--w-slot-transition", type=float, default=0.20)
    p.add_argument("--w-primitive-transition", type=float, default=0.25)
    p.add_argument("--w-composition", type=float, default=0.12)
    p.add_argument("--w-write", type=float, default=0.25)
    p.add_argument("--task-families", default=",".join(DEFAULT_TASK_FAMILIES))
    p.add_argument("--solution-families", default=",".join(DEFAULT_SOLUTION_FAMILIES))
    p.add_argument("--init-assembler", default="")
    p.add_argument("--log-every", type=int, default=25)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
