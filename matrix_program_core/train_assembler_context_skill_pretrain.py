#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Context-only assembly-skill pretrain.

Purpose:
  Learn the SAME assembly mechanism used by the live model, but train it from
  context -> flow targets, not from a task head and not from raw audio data.

This is different from multitask execution pretrain:
  - no classifier head CE;
  - no raw audio/task data;
  - evidence is compact structured context tokens;
  - loss directly teaches read/primitive/transition/composition/write assembly;
  - optional holdout task/solution combinations test whether it generalizes by
    context similarity instead of memorizing one fixed recipe.

Saved checkpoint contains both:
  - assembler_core
  - task_context

because context embeddings/projections are part of the transferable skill.
"""

from __future__ import annotations

import argparse
import csv
import json
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

from matrix_program_core.assembler_core import AssemblerConfig, MatrixProgramAssemblerCore, assembler_skill_loss  # noqa: E402
from matrix_program_core.task_context_v2 import TaskContextV2Config, TaskIOContextEncoder  # noqa: E402
from matrix_program_core.train_assembler_multitask_pretrain import (  # noqa: E402
    DEFAULT_SOLUTION_FAMILIES,
    DEFAULT_TASK_FAMILIES,
    build_multitask_flow_targets,
    parse_csv_names,
)
from simple_butterfly_matrix.simple_butterfly_matrix import PRIMITIVES  # noqa: E402

ROLE_IDS = {
    "task_head_core": 0,
    "sequence_mixer": 1,
    "attention_replacement": 2,
    "matrix_decompiler": 3,
    "generic_program_core": 4,
}
INPUT_KIND_IDS = {
    "audio_features": 0,
    "token_states": 1,
    "matrix_features": 2,
    "generic_evidence": 3,
    "synthetic_context": 4,
}
OUTPUT_KIND_IDS = {
    "class_logits": 0,
    "token_states": 1,
    "matrix_recon": 2,
    "program_slots": 3,
    "generic_state": 4,
}
LOSS_KIND_IDS = {
    "cross_entropy": 0,
    "reconstruction": 1,
    "distillation": 2,
    "downstream": 3,
    "flow_skill": 4,
}
READOUT_KIND_IDS = {
    "class_query": 0,
    "residual_writeback": 1,
    "program_slot": 2,
    "generic_head": 3,
    "state_writeback": 4,
}


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


class ContextSkillDataset(Dataset):
    """Only context descriptors, no raw task input and no labels for a head."""

    def __init__(
        self,
        n: int,
        task_families: Sequence[str],
        solution_families: Sequence[str],
        holdout_mod: int = 0,
        holdout_rem: int = 0,
        split: str = "train",
        seed: int = 0,
    ):
        self.rows: List[Tuple[int, int, int, int, int, int, int, float, float, float, float]] = []
        self.task_families = list(task_families)
        self.solution_families = list(solution_families)
        rng = random.Random(seed + (0 if split == "train" else 10000))
        roles = list(ROLE_IDS.values())
        inputs = list(INPUT_KIND_IDS.values())
        outputs = list(OUTPUT_KIND_IDS.values())
        losses = list(LOSS_KIND_IDS.values())
        readouts = list(READOUT_KIND_IDS.values())
        combos = []
        for ti in range(len(task_families)):
            for si in range(len(solution_families)):
                is_holdout = holdout_mod > 0 and ((ti * 17 + si * 31) % holdout_mod == holdout_rem)
                if (split == "train" and is_holdout) or (split != "train" and not is_holdout and holdout_mod > 0):
                    continue
                combos.append((ti, si))
        if not combos:
            combos = [(ti, si) for ti in range(len(task_families)) for si in range(len(solution_families))]
        for i in range(int(n)):
            ti, si = combos[i % len(combos)]
            # Generic I/O contracts are intentionally varied so the skill is not
            # a classification recipe. The task/solution IDs still decide target flows.
            role = roles[(ti + si + i) % len(roles)]
            inp = inputs[(ti * 3 + si + i) % len(inputs)]
            out = outputs[(ti + si * 2 + i) % len(outputs)]
            loss = losses[(ti * 5 + si + i) % len(losses)]
            readout = readouts[(ti + si * 7 + i) % len(readouts)]
            num_outputs = float(2 + ((ti + si + i) % 64))
            seq_len = float(16 + ((ti * 11 + si * 5 + i) % 512))
            hidden = float(32 + ((ti * 13 + si * 3 + i) % 512))
            extra = rng.random()
            self.rows.append((ti, si, role, inp, out, loss, readout, num_outputs, seq_len, hidden, extra))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        ti, si, role, inp, out, loss, readout, num_outputs, seq_len, hidden, extra = self.rows[idx]
        return {
            "task_id": torch.tensor(ti, dtype=torch.long),
            "solution_id": torch.tensor(si, dtype=torch.long),
            "role_id": torch.tensor(role, dtype=torch.long),
            "input_kind_id": torch.tensor(inp, dtype=torch.long),
            "output_kind_id": torch.tensor(out, dtype=torch.long),
            "loss_kind_id": torch.tensor(loss, dtype=torch.long),
            "readout_kind_id": torch.tensor(readout, dtype=torch.long),
            "num_outputs": torch.tensor(num_outputs, dtype=torch.float32),
            "sequence_length": torch.tensor(seq_len, dtype=torch.float32),
            "hidden_dim": torch.tensor(hidden, dtype=torch.float32),
            "extra_scalar": torch.tensor(extra, dtype=torch.float32),
        }


class ContextEvidenceBuilder(nn.Module):
    """Build compact evidence from context tokens only.

    It adds task/solution tokens and a few learned/noisy evidence seed tokens so
    the assembler sees the same interface shape as live mode, but no raw audio.
    """

    def __init__(self, dim: int, task_count: int, solution_count: int, extra_tokens: int = 8, dropout: float = 0.02):
        super().__init__()
        self.task = nn.Embedding(task_count, dim)
        self.solution = nn.Embedding(solution_count, dim)
        self.extra = nn.Parameter(torch.randn(max(1, extra_tokens), dim) * 0.02)
        self.mix = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, context: torch.Tensor, task_id: torch.Tensor, solution_id: torch.Tensor) -> torch.Tensor:
        B, _, D = context.shape
        t = self.task(task_id).view(B, 1, D)
        s = self.solution(solution_id).view(B, 1, D)
        ts = self.mix(t + s)
        extra = self.extra.to(device=context.device, dtype=context.dtype).view(1, -1, D).expand(B, -1, -1)
        # Modulate extra evidence seed tokens by task/solution, so the core sees
        # structured context but not a raw input dataset.
        extra = extra + 0.25 * ts
        return self.drop(self.norm(torch.cat([context, t, s, ts, extra], dim=1)))


def _maybe_vector(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return x.to(device=device, non_blocking=True)


def make_context_tokens(task_context: TaskIOContextEncoder, batch: Dict[str, torch.Tensor], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # TaskIOContextEncoder was patched to accept tensors; if local file was not patched,
    # this script will fail loudly so the runner applies the patch first.
    B = int(batch["task_id"].shape[0])
    return task_context(
        B,
        device,
        dtype,
        role_id=_maybe_vector(batch["role_id"], device),
        input_kind_id=_maybe_vector(batch["input_kind_id"], device),
        output_kind_id=_maybe_vector(batch["output_kind_id"], device),
        loss_kind_id=_maybe_vector(batch["loss_kind_id"], device),
        readout_kind_id=_maybe_vector(batch["readout_kind_id"], device),
        num_outputs=_maybe_vector(batch["num_outputs"], device),
        sequence_length=_maybe_vector(batch["sequence_length"], device),
        hidden_dim=_maybe_vector(batch["hidden_dim"], device),
        extra_scalar=_maybe_vector(batch["extra_scalar"], device),
        head_query=None,
    )


def train_or_eval(core, task_context, evidence_builder, loader, cfg, task_families, solution_families, skill_weights, args, device, dtype, train: bool, opt=None, scaler=None):
    core.train(train); task_context.train(train); evidence_builder.train(train)
    use_amp = device.startswith("cuda") and dtype != torch.float32
    total = 0.0
    n = 0
    aux_last = None
    losses_acc: Dict[str, float] = {}
    for step, batch in enumerate(loader, 1):
        task_id = _maybe_vector(batch["task_id"], device)
        sol_id = _maybe_vector(batch["solution_id"], device)
        if train:
            opt.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=use_amp):
                ctx = make_context_tokens(task_context, batch, torch.device(device), dtype)
                evidence = evidence_builder(ctx, task_id, sol_id)
                _cells, aux = core(evidence)
                targets = build_multitask_flow_targets(cfg, task_id, sol_id, task_families, solution_families, torch.device(device))
                skill, flow_losses = assembler_skill_loss(aux, targets, skill_weights)
                entropy_keep = torch.zeros((), device=evidence.device)
                if args.lambda_entropy_keep > 0:
                    for ent in aux.entropies.values():
                        entropy_keep = entropy_keep + F.relu(torch.tensor(args.min_entropy, device=evidence.device) - ent.float()).pow(2)
                # Anchor deltas lightly during base skill pretrain. This is context -> base assembly skill.
                delta_l2 = torch.zeros((), device=evidence.device)
                if args.lambda_delta_l2 > 0:
                    for name, p in core.named_parameters():
                        if "_delta" in name:
                            delta_l2 = delta_l2 + p.float().pow(2).mean()
                loss = skill + args.lambda_entropy_keep * entropy_keep + args.lambda_delta_l2 * delta_l2
        if train:
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                params = list(core.parameters()) + list(task_context.parameters()) + list(evidence_builder.parameters())
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            scaler.step(opt); scaler.update()
        bs = int(task_id.numel())
        total += float(loss.detach().cpu()) * bs
        n += bs
        for k, v in flow_losses.items():
            losses_acc[k] = losses_acc.get(k, 0.0) + float(v.detach().cpu()) * bs
        aux_last = aux
        if train and args.log_every and step % args.log_every == 0:
            print(f"step {step:05d} context_skill={total/max(1,n):.4f} read={losses_acc.get('read_flow_kl',0)/max(1,n):.3f} prim={losses_acc.get('primitive_slot_kl',0)/max(1,n):.3f} write={losses_acc.get('write_flow_kl',0)/max(1,n):.3f}", flush=True)
    out = {"loss": total / max(1, n)}
    for k, v in losses_acc.items():
        out[k] = v / max(1, n)
    if aux_last is not None:
        for k, v in aux_last.entropies.items():
            out[f"entropy_{k}"] = float(v.detach().cpu())
        out["memory_usage"] = float(aux_last.memory_usage.detach().cpu())
        out["global_usage"] = float(aux_last.global_usage.detach().cpu())
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
    task_context = TaskIOContextEncoder(TaskContextV2Config(
        dim=args.dim,
        free_tokens=args.task_context_tokens,
        max_head_tokens=args.head_context_tokens,
        dropout=args.dropout,
    )).to(device)
    evidence_builder = ContextEvidenceBuilder(args.dim, len(task_families), len(solution_families), args.extra_context_tokens, args.dropout).to(device)

    if args.init_assembler:
        ckpt = torch.load(args.init_assembler, map_location=device)
        state = ckpt.get("assembler_core", ckpt.get("core", ckpt.get("assembler_skill_base", ckpt)))
        missing, unexpected = core.load_state_dict(state, strict=False)
        print(f"loaded init assembler: {args.init_assembler} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        if "task_context" in ckpt:
            miss, unexp = task_context.load_state_dict(ckpt["task_context"], strict=False)
            print(f"loaded init task_context missing={len(miss)} unexpected={len(unexp)}", flush=True)

    train_ds = ContextSkillDataset(args.train_n, task_families, solution_families, args.holdout_mod, args.holdout_rem, "train", args.seed)
    val_ds = ContextSkillDataset(args.val_n, task_families, solution_families, args.holdout_mod, args.holdout_rem, "val", args.seed)
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
    params = list(core.parameters()) + list(task_context.parameters()) + list(evidence_builder.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda") and dtype == torch.float16)

    fields = ["epoch", "train_loss", "val_loss", "best_val", "read_flow_kl", "primitive_slot_kl", "slot_transition_kl", "primitive_transition_kl", "slot_composition_kl", "write_flow_kl", "entropy_read", "entropy_primitive", "entropy_write", "memory_usage", "global_usage"]
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()
    best = 1e18
    best_epoch = 0
    t0 = time.time()
    print(f"context-skill pretrain train={len(train_ds)} val={len(val_ds)} tasks={task_families} solutions={solution_families}", flush=True)
    for ep in range(1, args.epochs + 1):
        tr = train_or_eval(core, task_context, evidence_builder, train, cfg, task_families, solution_families, skill_weights, args, device, dtype, True, opt, scaler)
        with torch.no_grad():
            va = train_or_eval(core, task_context, evidence_builder, val, cfg, task_families, solution_families, skill_weights, args, device, dtype, False)
        if va["loss"] < best:
            best = va["loss"]
            best_epoch = ep
            torch.save({
                "assembler_core": core.state_dict(),
                "task_context": task_context.state_dict(),
                "context_evidence_builder": evidence_builder.state_dict(),
                "config": cfg.__dict__,
                "args": vars(args),
                "task_families": task_families,
                "solution_families": solution_families,
                "best_val_loss": best,
                "epoch": ep,
                "primitives": list(PRIMITIVES),
            }, out / "assembler_context_skill_best.pt")
        torch.save({
            "assembler_core": core.state_dict(),
            "task_context": task_context.state_dict(),
            "context_evidence_builder": evidence_builder.state_dict(),
            "config": cfg.__dict__,
            "args": vars(args),
            "task_families": task_families,
            "solution_families": solution_families,
            "best_val_loss": best,
            "epoch": ep,
            "primitives": list(PRIMITIVES),
        }, out / "assembler_context_skill_last.pt")
        row = {"epoch": ep, "train_loss": tr["loss"], "val_loss": va["loss"], "best_val": best}
        for k in fields:
            if k in va:
                row[k] = va[k]
        with (out / "metrics.csv").open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writerow({k: row.get(k, 0.0) for k in fields})
        print(f"epoch {ep:03d}/{args.epochs} train_skill={tr['loss']:.4f} val_skill={va['loss']:.4f} best={best:.4f}@{best_epoch}", flush=True)
    write_json(out / "final_report.json", {
        "best_val_loss": best,
        "best_epoch": best_epoch,
        "elapsed_sec": time.time() - t0,
        "config": cfg.__dict__,
        "args": vars(args),
        "task_families": task_families,
        "solution_families": solution_families,
        "checkpoint": str(out / "assembler_context_skill_best.pt"),
    })


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="matrix_program_core/runs/context_skill_debug")
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", choices=["fp16", "bf16", "fp32", "off"], default="bf16")
    p.add_argument("--seed", type=int, default=42)
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
    p.add_argument("--task-context-tokens", type=int, default=4)
    p.add_argument("--head-context-tokens", type=int, default=10)
    p.add_argument("--extra-context-tokens", type=int, default=8)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=0.75)
    p.add_argument("--lambda-entropy-keep", type=float, default=0.01)
    p.add_argument("--min-entropy", type=float, default=0.55)
    p.add_argument("--lambda-delta-l2", type=float, default=0.01)
    p.add_argument("--w-read", type=float, default=0.25)
    p.add_argument("--w-primitive", type=float, default=0.35)
    p.add_argument("--w-slot-transition", type=float, default=0.20)
    p.add_argument("--w-primitive-transition", type=float, default=0.25)
    p.add_argument("--w-composition", type=float, default=0.12)
    p.add_argument("--w-write", type=float, default=0.25)
    p.add_argument("--task-families", default=",".join(DEFAULT_TASK_FAMILIES))
    p.add_argument("--solution-families", default=",".join(DEFAULT_SOLUTION_FAMILIES))
    p.add_argument("--holdout-mod", type=int, default=5)
    p.add_argument("--holdout-rem", type=int, default=0)
    p.add_argument("--init-assembler", default="")
    p.add_argument("--log-every", type=int, default=25)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
