#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Checkpoint split utilities for MatrixProgramAssemblerCore experiments.

This file enforces the separation we want:

1. assembler_skill_base.pt
   Transferable base assembly skill. No task input adapter, no task head.
   It may include `task_context_base` when the checkpoint was produced by a
   context-skill pretrain, because the context encoder is part of the skill.

2. assembler_skill_delta.pt
   LoRA-like task adaptation: only *_delta and context-delta/free context.

3. task_adapter_head.pt
   Task-specific input adapter + task head/consumer + task metadata.

It also includes a small CLI for exporting split packs from existing experiment
checkpoints produced by train_assembler_* and transfer_audio_assembler.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch


DELTA_MARKERS = (
    "_delta",
    ".ctx_flow_delta.",
    "ctx_flow_delta",
)

# Task adapter/head pack should contain the real task I/O modules.
# Top-level task_context from context-skill pretrain is handled separately as base skill.
TASK_SPECIFIC_PREFIXES = (
    "input_adapter.",
    "head.",
    "mechanism_adapter.",
    "context_norm.",
    "task_context_tokens",
    "head_context_tokens",
)

CORE_KEYS = (
    "assembler_core",
    "core",
)


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def is_delta_key(name: str) -> bool:
    return any(m in name for m in DELTA_MARKERS)


def is_task_specific_model_key(name: str) -> bool:
    return any(name.startswith(p) for p in TASK_SPECIFIC_PREFIXES)


def strip_prefix(name: str, prefixes: Iterable[str]) -> str:
    for p in prefixes:
        if name.startswith(p):
            return name[len(p) :]
    return name


def count_params(state: Dict[str, torch.Tensor]) -> int:
    total = 0
    for v in state.values():
        if torch.is_tensor(v):
            total += int(v.numel())
    return total


def prefixed_state(prefix: str, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {f"{prefix}.{k}": v for k, v in state.items()}


def get_core_state(ckpt: Dict) -> Dict[str, torch.Tensor]:
    for k in CORE_KEYS:
        if k in ckpt and isinstance(ckpt[k], dict):
            return ckpt[k]
    model = ckpt.get("model")
    if isinstance(model, dict):
        return {
            strip_prefix(k, ("assembler_core.", "core.")): v
            for k, v in model.items()
            if k.startswith("assembler_core.") or k.startswith("core.")
        }
    raise KeyError("checkpoint has no assembler_core/core/model core state")


def get_context_base_state(ckpt: Dict) -> Dict[str, torch.Tensor]:
    """Return portable task/context encoder state if present.

    Context-skill pretrain saves top-level `task_context`. That is a learned
    context interpreter, not the downstream task head, so we keep it with the
    transferable skill base. If a live task model has `model.task_context.*`, it
    is not added here because it may be task-adapted.
    """
    ctx = ckpt.get("task_context")
    if isinstance(ctx, dict):
        return prefixed_state("task_context", ctx)
    return {}


def get_mechanism_base_state(ckpt: Dict) -> Dict[str, torch.Tensor]:
    mech = ckpt.get("mechanism_evidence_builder")
    if isinstance(mech, dict):
        return prefixed_state("mechanism_evidence_builder", mech)
    return {}


def get_context_task_state(ckpt: Dict) -> Dict[str, torch.Tensor]:
    """Return live task context state from full model checkpoints.

    This is task-side state because it was trained together with the adapter/head.
    """
    model = ckpt.get("model")
    if isinstance(model, dict):
        return {k: v for k, v in model.items() if k.startswith("task_context.")}
    return {}


def split_core_base_delta(core_state: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    base = {k: v for k, v in core_state.items() if not is_delta_key(k)}
    delta = {k: v for k, v in core_state.items() if is_delta_key(k)}
    return base, delta


def split_model_task_state(ckpt: Dict) -> Dict[str, torch.Tensor]:
    model = ckpt.get("model")
    if not isinstance(model, dict):
        return {}
    task = {k: v for k, v in model.items() if is_task_specific_model_key(k)}
    task.update(get_context_task_state(ckpt))
    return task


def meta_from_ckpt(ckpt: Dict) -> Dict:
    meta = {
        "args": ckpt.get("args", {}),
        "classes": ckpt.get("classes"),
        "epoch": ckpt.get("epoch"),
        "best_acc": ckpt.get("best_acc"),
        "best_val_loss": ckpt.get("best_val_loss"),
        "config": ckpt.get("config") or ckpt.get("assembler_config"),
        "task_families": ckpt.get("task_families"),
        "solution_families": ckpt.get("solution_families"),
        "primitives": ckpt.get("primitives"),
        "trainable_summary": ckpt.get("trainable_summary"),
        "meta": ckpt.get("meta"),
    }
    return {k: v for k, v in meta.items() if v is not None}


def export_split(input_path: str | Path, out_dir: str | Path, tag: str = "") -> Dict[str, str | int]:
    input_path = Path(input_path)
    out = ensure_dir(out_dir)
    ckpt = torch.load(input_path, map_location="cpu")
    core = get_core_state(ckpt)
    base, delta = split_core_base_delta(core)
    context_base = get_context_base_state(ckpt)
    mechanism_base = get_mechanism_base_state(ckpt)
    base_with_context = dict(base)
    base_with_context.update(context_base)
    base_with_context.update(mechanism_base)
    task_state = split_model_task_state(ckpt)
    meta = meta_from_ckpt(ckpt)
    meta["source_checkpoint"] = str(input_path)
    meta["tag"] = tag

    prefix = f"{tag}_" if tag else ""
    base_path = out / f"{prefix}assembler_skill_base.pt"
    delta_path = out / f"{prefix}assembler_skill_delta.pt"
    task_path = out / f"{prefix}task_adapter_head.pt"
    report_path = out / f"{prefix}split_report.json"

    torch.save({
        "assembler_skill_base": base,
        "task_context_base": context_base,
        "mechanism_context_base": mechanism_base,
        "meta": meta,
        "param_count": count_params(base_with_context),
    }, base_path)
    torch.save({
        "assembler_skill_delta": delta,
        "meta": meta,
        "param_count": count_params(delta),
    }, delta_path)
    torch.save({
        "task_adapter_head": task_state,
        "meta": meta,
        "param_count": count_params(task_state),
    }, task_path)

    report = {
        "source_checkpoint": str(input_path),
        "tag": tag,
        "base_path": str(base_path),
        "delta_path": str(delta_path),
        "task_path": str(task_path),
        "base_params": count_params(base),
        "context_base_params": count_params(context_base),
        "mechanism_base_params": count_params(mechanism_base),
        "base_plus_context_params": count_params(base_with_context),
        "delta_params": count_params(delta),
        "task_params": count_params(task_state),
        "base_keys": len(base),
        "context_base_keys": len(context_base),
        "mechanism_base_keys": len(mechanism_base),
        "delta_keys": len(delta),
        "task_keys": len(task_state),
        "meta": meta,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="checkpoint .pt to split")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tag", default="")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    report = export_split(args.input, args.out_dir, args.tag)
    print(json.dumps(report, ensure_ascii=False, indent=2))
