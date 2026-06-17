#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fast wrapper for multitask assembler pretrain.

The first multitask version built flow targets inside every batch with Python
loops and CUDA scalar writes. This wrapper keeps the same training code, but
replaces `build_multitask_flow_targets` with a cached target-bank version:

    bank[task_id, solution_id] -> flow targets

Then each batch only does tensor indexing. No weights/checkpoints format changes.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matrix_program_core.train_assembler_multitask_pretrain as mt  # noqa: E402

_slow_build = mt.build_multitask_flow_targets
_BANK_CACHE: Dict[Tuple, Dict[str, torch.Tensor]] = {}


def _cfg_key(cfg, task_families: Sequence[str], solution_families: Sequence[str], device: torch.device) -> Tuple:
    return (
        str(device),
        int(cfg.layers), int(cfg.blocks), int(cfg.steps), int(cfg.primitive_slots),
        int(cfg.memory_cells), int(cfg.global_cells), int(cfg.address_cells),
        tuple(task_families), tuple(solution_families),
    )


def _build_bank(cfg, task_families: Sequence[str], solution_families: Sequence[str], device: torch.device) -> Dict[str, torch.Tensor]:
    key = _cfg_key(cfg, task_families, solution_families, device)
    if key in _BANK_CACHE:
        return _BANK_CACHE[key]
    chunks = []
    for task_id in range(len(task_families)):
        for sol_id in range(len(solution_families)):
            t = torch.tensor([task_id], device=device, dtype=torch.long)
            s = torch.tensor([sol_id], device=device, dtype=torch.long)
            chunks.append(_slow_build(cfg, t, s, task_families, solution_families, device))
    bank: Dict[str, torch.Tensor] = {}
    for name in chunks[0].keys():
        # slow target shape: [T,1,...], bank shape: [T,C,...]
        bank[name] = torch.cat([c[name] for c in chunks], dim=1).contiguous()
    _BANK_CACHE[key] = bank
    print(
        f"[fast-target-bank] built combos={len(chunks)} "
        f"tasks={len(task_families)} solutions={len(solution_families)} device={device}",
        flush=True,
    )
    return bank


def build_multitask_flow_targets_fast(cfg, task_ids, solution_ids, task_families, solution_families, device):
    bank = _build_bank(cfg, task_families, solution_families, device)
    sol_n = len(solution_families)
    combo = (task_ids.to(device=device, dtype=torch.long) * sol_n + solution_ids.to(device=device, dtype=torch.long)).view(-1)
    return {name: value.index_select(1, combo) for name, value in bank.items()}


mt.build_multitask_flow_targets = build_multitask_flow_targets_fast

if __name__ == "__main__":
    mt.run(mt.parser().parse_args())
