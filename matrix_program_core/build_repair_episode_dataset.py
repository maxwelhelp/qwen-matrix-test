#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build cached repair/from-scratch episodes for mechanism skill pretrain.

Input is a normal flow pack with full target matrix programs. Output repeats
each program as a sequence of cached input states:

  0. empty/basis program
  1. read/write/composition scaffold
  2. scaffold + primitives
  3. scaffold + primitives + slot operators
  4. full program
  5. full program with primitive/operator removed
  6. full program with extra sharp primitive/operator
  7. full program shifted to wrong step

The target remains the full correct program. This creates long recipe examples
that teach assembly and repair without regenerating corruption during training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from matrix_program_core import train_assembler_mechanism_skill_pretrain as ms  # noqa: E402


def _summary_one(pack: Dict[str, torch.Tensor], i: int) -> Dict[str, torch.Tensor]:
    return {
        "read_flow": pack["read_flow"][i].float().mean(dim=(1, 2)),
        "primitive_slot_flow": pack["primitive_slot_flow"][i].float().mean(dim=(1, 2)),
        "slot_transition_flow": pack["slot_transition_flow"][i].float().mean(dim=1).flatten(start_dim=1),
        "primitive_transition_flow": pack["primitive_transition_flow"][i].float().flatten(start_dim=1),
        "slot_composition_flow": pack["slot_composition_flow"][i].float().mean(dim=1),
        "write_flow": pack["write_flow"][i].float().mean(dim=1),
    }


def _target_mask_for_visible(visible: torch.Tensor) -> torch.Tensor:
    # Even visible mechanisms get a weak consistency loss in the trainer, but
    # missing parts are the primary target.
    return torch.ones_like(visible)


def _phase_pressure(delta_by_step: List[torch.Tensor], steps: int) -> torch.Tensor:
    if not delta_by_step:
        return torch.zeros(len(ms.PHASES))
    stacked = torch.stack(delta_by_step, dim=0).mean(dim=0).view(1, -1)
    return ms._phase_pressure_from_steps(stacked, steps).squeeze(0).cpu()


def _episode_feedback(
    clean: Dict[str, torch.Tensor],
    inp: Dict[str, torch.Tensor],
    visible: torch.Tensor,
    target: torch.Tensor,
    mode: int,
    cycle_pos: float,
    steps: int,
) -> Dict[str, torch.Tensor]:
    pressures = []
    step_deltas = []
    for key in ms.FLOW_KEYS:
        delta = (inp[key].float() - clean[key].float()).abs().mean(dim=-1)
        pressures.append(delta.mean())
        step_deltas.append(delta)
    flow_pressure = torch.stack(pressures).float()
    loss_proxy = flow_pressure.mean().clamp(0.0, 1.0)
    quality = 1.0 - loss_proxy
    flow = torch.stack([
        flow_pressure,
        visible.float(),
        target.float(),
        torch.full_like(flow_pressure, float(mode) / 5.0),
        torch.full_like(flow_pressure, float(cycle_pos)),
    ], dim=-1)
    return {
        "global": torch.tensor([quality, loss_proxy, loss_proxy.sqrt(), float(cycle_pos)], dtype=torch.float32),
        "flow": flow,
        "phase": _phase_pressure(step_deltas, steps).float(),
    }


def _copy_selected(clean: Dict[str, torch.Tensor], base: Dict[str, torch.Tensor], selected: List[str]) -> Dict[str, torch.Tensor]:
    return {key: (clean[key].clone() if key in selected else base[key].clone()) for key in ms.FLOW_KEYS}


def _make_episodes_for_one(pack: Dict[str, torch.Tensor], i: int, steps: int, generator: torch.Generator, episode_mode: str):
    clean = _summary_one(pack, i)
    base = {key: ms._base_repair_distribution(key, clean[key].unsqueeze(0)).squeeze(0).cpu() for key in ms.FLOW_KEYS}
    episodes = []
    stage_defs = [
        ("empty_basis", [], 0),
        ("scaffold_read_write", ["read_flow", "write_flow", "slot_composition_flow"], 0),
        ("add_primitives", ["read_flow", "write_flow", "slot_composition_flow", "primitive_slot_flow"], 0),
        ("add_slot_ops", ["read_flow", "write_flow", "slot_composition_flow", "primitive_slot_flow", "slot_transition_flow"], 0),
        ("full_recipe", list(ms.FLOW_KEYS), 0),
    ]
    if episode_mode in ("all", "from_scratch"):
        denom = max(1, len(stage_defs) - 1)
        for stage_idx, (_name, selected, mode) in enumerate(stage_defs):
            inp = _copy_selected(clean, base, selected)
            visible = torch.tensor([1.0 if key in selected else 0.0 for key in ms.FLOW_KEYS])
            target = _target_mask_for_visible(visible)
            cycle = float(stage_idx) / float(denom)
            feedback = _episode_feedback(clean, inp, visible, target, mode, cycle, steps)
            episodes.append((inp, visible, target, feedback, stage_idx % len(ms.TASK_FAMILIES)))

    repair_defs = [
        ("repair_drop", 0),
        ("repair_extra", 1),
        ("repair_wrong_step", 3),
    ]
    if episode_mode in ("all", "repair"):
        denom = max(1, len(repair_defs) - 1)
        for local_idx, (_name, mode) in enumerate(repair_defs):
            offset = local_idx + (len(stage_defs) if episode_mode == "all" else 0)
            inp = {key: value.clone() for key, value in clean.items()}
            if mode == 0:
                for key in ("primitive_slot_flow", "primitive_transition_flow", "slot_transition_flow"):
                    inp[key] = base[key].clone()
            elif mode == 1:
                for key in ("primitive_slot_flow", "primitive_transition_flow"):
                    inp[key] = ms._random_peak_like(clean[key].unsqueeze(0), generator).squeeze(0).cpu()
            elif mode == 3:
                for key in ms.FLOW_KEYS:
                    inp[key] = torch.roll(inp[key], shifts=1, dims=0)
            visible = torch.ones(len(ms.FLOW_KEYS))
            target = torch.ones(len(ms.FLOW_KEYS))
            cycle = float(local_idx) / float(denom)
            feedback = _episode_feedback(clean, inp, visible, target, mode, cycle, steps)
            episodes.append((inp, visible, target, feedback, offset % len(ms.TASK_FAMILIES)))
    return episodes


def _candidate_inputs(root: Path) -> Sequence[Path]:
    patterns = (
        "matrix_program_core/runs/**/code_context_dataset.pt",
        "matrix_program_core/runs/**/merged_flow_pack.pt",
        "matrix_program_core/cache/**/code_context_dataset.pt",
    )
    out: List[Path] = []
    for pat in patterns:
        out.extend(root.glob(pat))
    out = [p for p in out if p.is_file()]
    return sorted(out, key=lambda p: p.stat().st_mtime, reverse=True)


def _resolve_input(path: str) -> Path:
    p = Path(path)
    if p.exists():
        return p
    candidates = _candidate_inputs(ROOT)
    if not candidates:
        raise FileNotFoundError(
            f"input flow pack not found: {path}\n"
            "No fallback code_context_dataset.pt/merged_flow_pack.pt was found under matrix_program_core/runs."
        )
    print(
        json.dumps(
            {
                "warning": "input flow pack not found; using newest available fallback",
                "requested": path,
                "fallback": str(candidates[0]),
                "other_candidates": [str(x) for x in candidates[1:6]],
            },
            ensure_ascii=False,
            indent=2,
        ),
        file=sys.stderr,
        flush=True,
    )
    return candidates[0]


def build(args) -> Dict[str, torch.Tensor]:
    input_path = _resolve_input(args.input)
    pack = torch.load(input_path, map_location="cpu")
    n = int(pack["read_flow"].shape[0])
    limit = n if args.max_programs <= 0 else min(n, int(args.max_programs))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed))

    out: Dict[str, List[torch.Tensor]] = {key: [] for key in ms.FLOW_KEYS}
    out.update({f"input_{key}": [] for key in ms.FLOW_KEYS})
    for key in ("role_id", "input_kind_id", "output_kind_id", "loss_kind_id", "readout_kind_id", "num_outputs", "sequence_length", "hidden_dim", "extra_scalar"):
        out[key] = []
    for key in ms.SKETCH_KEYS:
        if key in pack:
            out[key] = []
    out.update({
        "episode_task_id": [],
        "episode_visible": [],
        "episode_target_mask": [],
        "episode_feedback_global": [],
        "episode_feedback_flow": [],
        "episode_feedback_phase": [],
        "episode_program_index": [],
        "episode_stage": [],
    })

    for i in range(limit):
        episodes = _make_episodes_for_one(pack, i, int(args.steps_core), generator, args.episode_mode)
        for stage_idx, (inp, visible, target, feedback, task_id) in enumerate(episodes):
            for key in ms.FLOW_KEYS:
                out[key].append(pack[key][i].float())
                out[f"input_{key}"].append(inp[key].float())
            for key in ("role_id", "input_kind_id", "output_kind_id", "loss_kind_id", "readout_kind_id", "num_outputs", "sequence_length", "hidden_dim", "extra_scalar"):
                out[key].append(pack[key][i])
            for key in ms.SKETCH_KEYS:
                if key in pack:
                    out[key].append(pack[key][i].float())
            out["episode_task_id"].append(torch.tensor(task_id, dtype=torch.long))
            out["episode_visible"].append(visible.float())
            out["episode_target_mask"].append(target.float())
            out["episode_feedback_global"].append(feedback["global"].float())
            out["episode_feedback_flow"].append(feedback["flow"].float())
            out["episode_feedback_phase"].append(feedback["phase"].float())
            out["episode_program_index"].append(torch.tensor(i, dtype=torch.long))
            out["episode_stage"].append(torch.tensor(stage_idx, dtype=torch.long))
        if args.log_every and (i + 1) % args.log_every == 0:
            print(f"[episode-build] {i+1}/{limit}", flush=True)

    final: Dict[str, torch.Tensor] = {}
    for key, values in out.items():
        if values:
            final[key] = torch.stack(values, dim=0)
    final["meta"] = {
        "truth_level": "cached_from_scratch_and_repair_episodes",
        "source": str(input_path),
        "requested_source": str(args.input),
        "source_n": n,
        "programs": limit,
        "episode_mode": args.episode_mode,
        "episodes_per_program": int(final["read_flow"].shape[0]) // max(1, limit),
        "episodes": int(final["read_flow"].shape[0]),
        "stage_names": [
            "empty_basis",
            "scaffold_read_write",
            "add_primitives",
            "add_slot_ops",
            "full_recipe",
            "repair_drop",
            "repair_extra",
            "repair_wrong_step",
        ],
        "program_sketch_keys": [key for key in ms.SKETCH_KEYS if key in final],
    }
    return final


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps-core", type=int, default=2)
    p.add_argument("--episode-mode", choices=["all", "from_scratch", "repair"], default="all")
    p.add_argument("--max-programs", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=500)
    args = p.parse_args()
    pack = build(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pack, out)
    print(json.dumps({"out": str(out), "meta": pack["meta"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
