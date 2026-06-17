#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stream-style MatrixProgramAssembler skill pretrain.

This trains the same assembler core on small differentiable repair trajectories:

    current soft flow matrices + quality/loss feedback -> next better flow matrices

There is no router, top-k, argmax, or hard best candidate. Current flows are
continuous corruptions of target flows, feedback is numeric tensor context, and
the target is a soft next step toward the clean flow. This teaches the core's
matrix editor how to move primitive/operator/write distributions when a program
is weak, matching live transfer more closely than static flow reconstruction.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from matrix_program_core.assembler_core import MatrixProgramAssemblerCore, flow_kl  # noqa: E402
from matrix_program_core.task_context_v2 import TaskContextV2Config, TaskIOContextEncoder  # noqa: E402
from matrix_program_core.train_assembler_mechanism_skill_pretrain import (  # noqa: E402
    FLOW_KEYS,
    TASK_FAMILIES,
    MechanismEvidenceBuilder,
    MechanismFlowDataset,
    ensure_dir,
    flow_kl_per_sample,
    flow_summaries,
    infer_cfg_from_pack,
    make_context_tokens,
    set_seed,
    task_masks,
    write_json,
)


def _tensor_state(obj) -> bool:
    return isinstance(obj, dict) and bool(obj) and all(torch.is_tensor(v) for v in obj.values())


def _strip_prefix(state: Dict[str, torch.Tensor], prefixes: Sequence[str]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        new_key = key
        for prefix in prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
                break
        out[new_key] = value
    return out


def normalize_last(x: torch.Tensor) -> torch.Tensor:
    x = x.float().clamp_min(1e-7)
    return x / x.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def uniform_like_flow(x: torch.Tensor) -> torch.Tensor:
    return torch.full_like(x, 1.0 / float(x.shape[-1]))


def flow_summaries_from_full(flows: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return flow_summaries(flows)


def targets_to_time_first(flows: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: flows[key].transpose(0, 1).contiguous() for key in FLOW_KEYS}


def make_stream_flows(batch: Dict[str, torch.Tensor], epoch: int, train: bool, args, device: torch.device):
    """Return current summaries, next-step targets, and numeric feedback.

    Clean flow is the fixed code/weight-derived matrix program. Current flow is
    a soft low-quality variant. Next flow is a small differentiable move toward
    clean, not a hard replacement.
    """

    clean = {key: batch[key].to(device, non_blocking=True).float() for key in FLOW_KEYS}
    first = clean[FLOW_KEYS[0]]
    N = int(first.shape[0])

    gen = torch.Generator(device="cpu")
    seed = int(args.seed + epoch * 977 + (0 if train else 444_123))
    if "idx" in batch:
        seed += int(batch["idx"].detach().cpu().sum().item()) % 100_000
    gen.manual_seed(seed)

    noise = torch.empty(N, device="cpu").uniform_(float(args.noise_min), float(args.noise_max), generator=gen).to(device)
    quality = (1.0 - noise).clamp(0.01, 1.0)
    repair = torch.empty(N, device="cpu").uniform_(float(args.repair_min), float(args.repair_max), generator=gen).to(device)
    next_quality = (quality + repair * (1.0 - quality)).clamp(0.0, 1.0)

    current: Dict[str, torch.Tensor] = {}
    target: Dict[str, torch.Tensor] = {}
    for key, x in clean.items():
        q_shape = [N] + [1] * (x.ndim - 1)
        q0 = quality.view(*q_shape)
        q1 = next_quality.view(*q_shape)
        perm = torch.randperm(N, generator=gen, device="cpu").to(device)
        other = x.index_select(0, perm)
        uniform = uniform_like_flow(x)
        eps = torch.randn(x.shape, generator=gen, device="cpu").to(device=device, dtype=x.dtype) * float(args.noise_std)
        bad = normalize_last(float(args.uniform_mix) * uniform + float(args.permute_mix) * other + float(args.local_mix) * (x + eps))
        cur = normalize_last(q0 * x + (1.0 - q0) * bad)
        nxt = normalize_last(q1 * x + (1.0 - q1) * bad)
        current[key] = cur
        target[key] = nxt

    loss_proxy = (1.0 - quality).clamp_min(0.0)
    improve_pressure = (next_quality - quality).clamp_min(0.0)
    epoch_frac = torch.full_like(quality, float(epoch) / max(1.0, float(args.epochs)))
    feedback = torch.stack([quality, loss_proxy, improve_pressure, epoch_frac], dim=-1)
    sample_weight = 1.0 + float(args.feedback_importance) * loss_proxy
    return flow_summaries_from_full(current), targets_to_time_first(target), feedback, sample_weight


def stream_loss(aux, targets: Dict[str, torch.Tensor], sample_weight: torch.Tensor, args):
    total = torch.zeros((), device=aux.cells.device)
    parts: Dict[str, torch.Tensor] = {}
    preds = {
        "read_flow": aux.read_flow,
        "primitive_slot_flow": aux.primitive_slot_flow,
        "slot_transition_flow": aux.slot_transition_flow,
        "primitive_transition_flow": aux.primitive_transition_flow,
        "slot_composition_flow": aux.slot_composition_flow,
        "write_flow": aux.write_flow,
    }
    weights = {
        "read_flow": args.w_read,
        "primitive_slot_flow": args.w_primitive,
        "slot_transition_flow": args.w_slot_transition,
        "primitive_transition_flow": args.w_primitive_transition,
        "slot_composition_flow": args.w_composition,
        "write_flow": args.w_write,
    }
    sw = sample_weight.to(device=aux.cells.device, dtype=torch.float32)
    for key in FLOW_KEYS:
        pred, target = preds[key], targets[key]
        per_sample = flow_kl_per_sample(pred, target, dim=-1)
        weighted = (per_sample * sw).sum() / sw.sum().clamp_min(1.0)
        total = total + float(weights[key]) * weighted
        parts[f"{key}_kl"] = flow_kl(pred, target, dim=-1).detach()
        parts[f"{key}_weighted"] = weighted.detach()

    if args.lambda_entropy_keep > 0:
        ent_pen = torch.zeros((), device=aux.cells.device)
        for ent in aux.entropies.values():
            ent_pen = ent_pen + F.relu(torch.tensor(float(args.min_entropy), device=aux.cells.device) - ent.float()).pow(2)
        total = total + float(args.lambda_entropy_keep) * ent_pen
        parts["entropy_keep"] = ent_pen.detach()
    return total, parts


def load_init(core: MatrixProgramAssemblerCore, task_context: TaskIOContextEncoder, evidence_builder: MechanismEvidenceBuilder, path: str, device: torch.device) -> None:
    if not path:
        return
    ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict):
        raise TypeError("init checkpoint must be a dict")
    for key in ("assembler_core", "assembler_skill_base", "core"):
        state = ckpt.get(key)
        if _tensor_state(state):
            missing, unexpected = core.load_state_dict(_strip_prefix(state, ("assembler_core.", "core.")), strict=False)
            print(f"loaded init core {path} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
            break
    ctx = ckpt.get("task_context", ckpt.get("task_context_base"))
    if _tensor_state(ctx):
        missing, unexpected = task_context.load_state_dict(_strip_prefix(ctx, ("task_context.",)), strict=False)
        print(f"loaded init task_context missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    mech = ckpt.get("mechanism_evidence_builder", ckpt.get("mechanism_context_base"))
    if _tensor_state(mech):
        missing, unexpected = evidence_builder.load_state_dict(_strip_prefix(mech, ("mechanism_evidence_builder.", "mechanism_builder.")), strict=False)
        print(f"loaded init mechanism_evidence_builder missing={len(missing)} unexpected={len(unexpected)}", flush=True)


def configure(core: MatrixProgramAssemblerCore, task_context: TaskIOContextEncoder, evidence_builder: MechanismEvidenceBuilder, args) -> None:
    core.freeze_for_mode(args.train_mode)
    for p in task_context.parameters():
        p.requires_grad = bool(args.train_task_context)
    for p in evidence_builder.parameters():
        p.requires_grad = bool(args.train_evidence_builder)


def run_epoch(core, task_context, evidence_builder, loader, device, dtype, args, train: bool, opt=None, scaler=None, epoch: int = 0):
    core.train(train)
    task_context.train(train and bool(args.train_task_context))
    evidence_builder.train(train and bool(args.train_evidence_builder))
    use_amp = str(device).startswith("cuda") and dtype != torch.float32
    total, n = 0.0, 0
    acc: Dict[str, float] = {}
    task_id = torch.full((1,), TASK_FAMILIES.index("full_denoise"), dtype=torch.long)
    for step, batch in enumerate(loader, 1):
        if train and args.max_train_batches > 0 and step > args.max_train_batches:
            break
        if (not train) and args.max_val_batches > 0 and step > args.max_val_batches:
            break
        bs = int(batch["role_id"].shape[0])
        task_ids = task_id.expand(bs)
        visible, _target_mask = task_masks(task_ids, device)
        current_summaries, next_targets, feedback, sample_weight = make_stream_flows(batch, epoch, train, args, device)
        if train:
            opt.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=str(device).split(":")[0], dtype=dtype, enabled=use_amp):
                ctx = make_context_tokens(task_context, batch, device, dtype)
                mech = evidence_builder(current_summaries, visible, task_ids.to(device), feedback=feedback).to(dtype=dtype)
                evidence = torch.cat([ctx, mech], dim=1)
                _cells, aux = core(evidence)
                loss, parts = stream_loss(aux, next_targets, sample_weight, args)
        if train:
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                params = [p for p in list(core.parameters()) + list(task_context.parameters()) + list(evidence_builder.parameters()) if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            scaler.step(opt)
            scaler.update()
        total += float(loss.detach().cpu()) * bs
        n += bs
        for k, v in parts.items():
            acc[k] = acc.get(k, 0.0) + float(v.detach().cpu()) * bs
        if train and args.log_every and step % args.log_every == 0:
            print(
                f"epoch {epoch:03d} step {step:05d} loss={total/max(1,n):.4f} "
                f"prim={acc.get('primitive_slot_flow_kl',0)/max(1,n):.3f} "
                f"op={acc.get('primitive_transition_flow_kl',0)/max(1,n):.3f} "
                f"write={acc.get('write_flow_kl',0)/max(1,n):.3f}",
                flush=True,
            )
    out = {"loss": total / max(1, n)}
    for k, v in acc.items():
        out[k] = v / max(1, n)
    return out


def save_ckpt(path: Path, core, task_context, evidence_builder, cfg, args, best: float, epoch: int, pack_meta) -> None:
    torch.save({
        "assembler_core": core.state_dict(),
        "task_context": task_context.state_dict(),
        "mechanism_evidence_builder": evidence_builder.state_dict(),
        "config": cfg.__dict__,
        "args": vars(args),
        "best_val_loss": best,
        "epoch": epoch,
        "meta": {
            "source_dataset": args.dataset,
            "skill": "mechanism_flow_stream_repair",
            "trajectory": {
                "noise_min": args.noise_min,
                "noise_max": args.noise_max,
                "repair_min": args.repair_min,
                "repair_max": args.repair_max,
            },
            "task_families": list(TASK_FAMILIES),
            "pack_meta": pack_meta,
        },
    }, path)


def run(args) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    if str(device).startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(args.amp, torch.float32)
    out = ensure_dir(args.out_dir)
    pack = torch.load(args.dataset, map_location="cpu")
    cfg = infer_cfg_from_pack(pack, args)
    N = int(pack["read_flow"].shape[0])
    idx = torch.randperm(N)
    n_train = max(1, int(N * args.train_frac))
    train = DataLoader(MechanismFlowDataset(pack, idx[:n_train].tolist()), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=args.pin_memory)
    val_idx = idx[n_train:].tolist() or idx[:n_train].tolist()
    val = DataLoader(MechanismFlowDataset(pack, val_idx), batch_size=args.eval_batch_size, shuffle=False, num_workers=args.workers, pin_memory=args.pin_memory)

    core = MatrixProgramAssemblerCore(cfg).to(device)
    task_context = TaskIOContextEncoder(TaskContextV2Config(dim=args.dim, free_tokens=args.task_context_tokens, max_head_tokens=args.head_context_tokens, dropout=args.dropout)).to(device)
    evidence_builder = MechanismEvidenceBuilder(cfg, args.dim, dropout=args.dropout).to(device)
    load_init(core, task_context, evidence_builder, args.init_assembler, device)
    configure(core, task_context, evidence_builder, args)

    params = [p for p in list(core.parameters()) + list(task_context.parameters()) + list(evidence_builder.parameters()) if p.requires_grad]
    if not params:
        raise ValueError("no trainable parameters; enable a train mode or builder/context training")
    print(
        f"StreamSkill params total={sum(p.numel() for p in list(core.parameters()) + list(task_context.parameters()) + list(evidence_builder.parameters()))} "
        f"trainable={sum(p.numel() for p in params)} core_trainable={sum(p.numel() for p in core.parameters() if p.requires_grad)} "
        f"mode={args.train_mode}",
        flush=True,
    )
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=str(device).startswith("cuda") and dtype == torch.float16)

    fields = [
        "epoch", "train_loss", "val_loss", "best_val",
        "read_flow_kl", "primitive_slot_flow_kl", "slot_transition_flow_kl",
        "primitive_transition_flow_kl", "slot_composition_flow_kl", "write_flow_kl",
    ]
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()
    best, best_epoch = 1e18, 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(core, task_context, evidence_builder, train, device, dtype, args, True, opt, scaler, epoch)
        with torch.no_grad():
            va = run_epoch(core, task_context, evidence_builder, val, device, dtype, args, False, epoch=epoch)
        if va["loss"] < best:
            best, best_epoch = va["loss"], epoch
            save_ckpt(out / "assembler_stream_skill_best.pt", core, task_context, evidence_builder, cfg, args, best, epoch, pack.get("meta", {}))
        save_ckpt(out / "assembler_stream_skill_last.pt", core, task_context, evidence_builder, cfg, args, best, epoch, pack.get("meta", {}))
        row = {"epoch": epoch, "train_loss": tr["loss"], "val_loss": va["loss"], "best_val": best}
        for k in fields:
            if k in va:
                row[k] = va[k]
        with (out / "metrics.csv").open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writerow({k: row.get(k, 0.0) for k in fields})
        print(f"epoch {epoch:03d}/{args.epochs} train={tr['loss']:.4f} val={va['loss']:.4f} best={best:.4f}@{best_epoch}", flush=True)

    write_json(out / "final_report.json", {
        "best_val_loss": best,
        "best_epoch": best_epoch,
        "elapsed_sec": time.time() - t0,
        "checkpoint": str(out / "assembler_stream_skill_best.pt"),
        "dataset": args.dataset,
        "config": cfg.__dict__,
        "train_mode": args.train_mode,
        "trajectory": {
            "noise_min": args.noise_min,
            "noise_max": args.noise_max,
            "repair_min": args.repair_min,
            "repair_max": args.repair_max,
        },
    })


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--out-dir", default="matrix_program_core/runs/stream_skill")
    p.add_argument("--init-assembler", default="")
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", choices=["fp16", "bf16", "fp32", "off"], default="fp32")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dim", type=int, default=96)
    p.add_argument("--evidence-cells", type=int, default=48)
    p.add_argument("--steps-core", type=int, default=2)
    p.add_argument("--memory-cells", type=int, default=4)
    p.add_argument("--global-cells", type=int, default=2)
    p.add_argument("--channel-stages", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.04)
    p.add_argument("--task-context-tokens", type=int, default=4)
    p.add_argument("--head-context-tokens", type=int, default=10)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--train-frac", type=float, default=0.85)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=0.75)
    p.add_argument("--train-mode", choices=["editor_delta", "delta", "full"], default="editor_delta")
    p.add_argument("--train-task-context", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--train-evidence-builder", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--noise-min", type=float, default=0.15)
    p.add_argument("--noise-max", type=float, default=0.85)
    p.add_argument("--repair-min", type=float, default=0.20)
    p.add_argument("--repair-max", type=float, default=0.55)
    p.add_argument("--noise-std", type=float, default=0.03)
    p.add_argument("--uniform-mix", type=float, default=0.45)
    p.add_argument("--permute-mix", type=float, default=0.40)
    p.add_argument("--local-mix", type=float, default=0.15)
    p.add_argument("--feedback-importance", type=float, default=1.50)
    p.add_argument("--lambda-entropy-keep", type=float, default=0.01)
    p.add_argument("--min-entropy", type=float, default=0.55)
    p.add_argument("--w-read", type=float, default=0.25)
    p.add_argument("--w-primitive", type=float, default=0.35)
    p.add_argument("--w-slot-transition", type=float, default=0.20)
    p.add_argument("--w-primitive-transition", type=float, default=0.30)
    p.add_argument("--w-composition", type=float, default=0.12)
    p.add_argument("--w-write", type=float, default=0.25)
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--max-val-batches", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
