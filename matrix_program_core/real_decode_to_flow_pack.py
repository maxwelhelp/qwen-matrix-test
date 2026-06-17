#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert old real_matrix_decodes.jsonl into AssemblerCore flow-pack rows.

Input: old v3 real_decode/real_matrix_decodes.jsonl records:
  role_guess, selected_terms, formula, original_shape, metrics

Output: .pt pack compatible with CodeContextDataset:
  read_flow, primitive_slot_flow, slot_transition_flow,
  primitive_transition_flow, slot_composition_flow, write_flow,
  role_id/input_kind_id/output_kind_id/loss_kind_id/readout_kind_id,
  numeric context fields.

This uses the old real matrix decoder as source of real matrix-program sketches,
not as W->label supervision.
"""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matrix_program_core.train_assembler_code_context_pretrain as cc
from matrix_program_core.assembler_core import AssemblerConfig
from simple_butterfly_matrix.simple_butterfly_matrix import PRIMITIVES

P = list(PRIMITIVES)
P_IDX = {n: i for i, n in enumerate(P)}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def terms_to_prims(terms: Sequence[Dict[str, Any]]) -> List[List[str]]:
    groups: List[List[str]] = []
    for t in list(terms)[:16]:
        s = (str(t.get("name", "")) + " " + str(t.get("family", ""))).lower()
        g: List[str] = []
        if "low" in s or "svd" in s or "rank" in s: g += ["low_rank", "ctx_matrix"]
        if "project" in s or "const" in s or "ramp" in s: g += ["ctx_matrix", "low_rank"]
        if "diag" in s or "gate" in s: g += ["product_gate", "phase_matrix"]
        if "dct" in s or "blur" in s or "shift" in s or "toeplitz" in s: g += ["channel_butterfly", "phase_matrix"]
        if "block" in s or "graph" in s or "ring" in s or "diffusion" in s: g += ["block_butterfly", "phase_matrix"]
        if "diff" in s or "laplacian" in s or "edge" in s: g += ["phase_matrix", "channel_butterfly"]
        if not g: g = ["ctx_matrix", "phase_matrix"]
        out, seen = [], set()
        for x in g:
            if x not in seen:
                seen.add(x); out.append(x)
        groups.append(out)
    return groups or [["ctx_matrix", "phase_matrix"]]


def role_contract(role: str, shape: Sequence[int] | None) -> Dict[str, float | int]:
    r = str(role).lower()
    out_dim = float(shape[0] if shape else 1)
    hid = float(shape[-1] if shape else 96)
    if r.startswith("attention"):
        return dict(role_id=cc.ROLE_IDS["attention_replacement"], input_kind_id=cc.INPUT_KIND_IDS["token_states"], output_kind_id=cc.OUTPUT_KIND_IDS["token_states"], loss_kind_id=cc.LOSS_KIND_IDS["distillation"], readout_kind_id=cc.READOUT_KIND_IDS["residual_writeback"], num_outputs=out_dim, sequence_length=128.0, hidden_dim=hid, extra_scalar=1.0)
    if r == "head":
        return dict(role_id=cc.ROLE_IDS["task_head_core"], input_kind_id=cc.INPUT_KIND_IDS["generic_evidence"], output_kind_id=cc.OUTPUT_KIND_IDS["class_logits"], loss_kind_id=cc.LOSS_KIND_IDS["cross_entropy"], readout_kind_id=cc.READOUT_KIND_IDS["class_query"], num_outputs=out_dim, sequence_length=1.0, hidden_dim=hid, extra_scalar=2.0)
    if r in ("mlp", "linear", "norm", "conv", "embedding"):
        return dict(role_id=cc.ROLE_IDS["sequence_mixer"], input_kind_id=cc.INPUT_KIND_IDS["token_states"], output_kind_id=cc.OUTPUT_KIND_IDS["token_states"], loss_kind_id=cc.LOSS_KIND_IDS["downstream"], readout_kind_id=cc.READOUT_KIND_IDS["state_writeback"], num_outputs=out_dim, sequence_length=64.0, hidden_dim=hid, extra_scalar=3.0)
    return dict(role_id=cc.ROLE_IDS["matrix_decompiler"], input_kind_id=cc.INPUT_KIND_IDS["matrix_features"], output_kind_id=cc.OUTPUT_KIND_IDS["program_slots"], loss_kind_id=cc.LOSS_KIND_IDS["flow_skill"], readout_kind_id=cc.READOUT_KIND_IDS["program_slot"], num_outputs=out_dim, sequence_length=1.0, hidden_dim=hid, extra_scalar=4.0)


def rec_to_flows(rec: Dict[str, Any], cfg: AssemblerConfig) -> Dict[str, torch.Tensor]:
    T, B, K, A, NP = cfg.layers * cfg.steps, cfg.blocks, cfg.primitive_slots, cfg.address_cells, len(P)
    read = torch.full((T, B, K, A), 0.012)
    prim = torch.full((T, B, K, NP), 0.018)
    slot = torch.full((T, B, K, K), 0.018)
    ptr = torch.full((T, NP, NP), 0.010)
    comp = torch.full((T, B, K), 0.035)
    write = torch.full((T, B, A), 0.012)
    role = str(rec.get("role_guess", "linear"))
    groups = terms_to_prims(rec.get("selected_terms") or [])
    if role.startswith("attention"):
        rn, wn, tr = "global", "state", ["compare_mix", "product_gate"]
    elif role == "head":
        rn, wn, tr = "global", "class", ["replace", "norm_residual"]
    elif role == "mlp":
        rn, wn, tr = "state", "state", ["residual", "product_gate"]
    elif role == "conv":
        rn, wn, tr = "input", "state", ["residual"]
    else:
        rn, wn, tr = "state", "state", ["keep", "residual"]
    for t in range(T):
        for b in range(B):
            for k in range(K):
                read[t, b, k] = cc.read_distribution(rn, b, k, t, cfg)
                d = torch.full((NP,), 0.018)
                for j, name in enumerate(groups[(t + b + k) % len(groups)]):
                    if name in P_IDX: d[P_IDX[name]] += 0.85 / (1 + j)
                prim[t, b, k] = d / d.sum().clamp_min(1e-8)
                comp[t, b, k] += 0.85 / (1 + abs(k - (t % K)))
            slot[t, b] = cc.slot_transition_distribution(tr, K)
            write[t, b] = cc.write_distribution(wn, b, t, cfg)
        ptr[t] += cc.primitive_transition_distribution(tr)
        for a, bg in zip(groups, groups[1:]):
            for x in a:
                for y in bg:
                    if x in P_IDX and y in P_IDX: ptr[t, P_IDX[x], P_IDX[y]] += 0.35
    return dict(read_flow=cc.normalize_last(read), primitive_slot_flow=cc.normalize_last(prim), slot_transition_flow=cc.normalize_last(slot), primitive_transition_flow=cc.normalize_last(ptr), slot_composition_flow=cc.normalize_last(comp), write_flow=cc.normalize_last(write))


def build_pack(records: Sequence[Dict[str, Any]], cfg: AssemblerConfig) -> Dict[str, Any]:
    flows = {k: [] for k in ("read_flow", "primitive_slot_flow", "slot_transition_flow", "primitive_transition_flow", "slot_composition_flow", "write_flow")}
    ctx = {k: [] for k in ("role_id", "input_kind_id", "output_kind_id", "loss_kind_id", "readout_kind_id", "num_outputs", "sequence_length", "hidden_dim", "extra_scalar")}
    for rec in records:
        ft = rec_to_flows(rec, cfg)
        for k, v in ft.items(): flows[k].append(v)
        c = role_contract(rec.get("role_guess", "linear"), rec.get("original_shape"))
        met = rec.get("metrics", {}) or {}
        c["extra_scalar"] = float(c["extra_scalar"]) + float(met.get("rec_err", 0.0)) + float(met.get("functional_err_gaussian", 0.0))
        for k, v in c.items(): ctx[k].append(torch.tensor(v, dtype=torch.long if k.endswith("_id") else torch.float32))
    pack: Dict[str, Any] = {k: torch.stack(v) for k, v in flows.items()}
    pack.update({k: torch.stack(v) for k, v in ctx.items()})
    pack["meta"] = {"truth_level": "real_weight_program_decode_to_assembler_flow", "n": len(records), "primitives": P}
    return pack


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--primitive-slots", type=int, default=4)
    ap.add_argument("--memory-cells", type=int, default=4)
    ap.add_argument("--global-cells", type=int, default=2)
    args = ap.parse_args()
    cfg = AssemblerConfig(dim=args.dim, layers=args.layers, blocks=args.blocks, steps=args.steps, primitive_slots=args.primitive_slots, memory_cells=args.memory_cells, global_cells=args.global_cells)
    records = read_jsonl(Path(args.input_jsonl))
    pack = build_pack(records, cfg)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pack, out)
    print(json.dumps({"out": str(out), "n": len(records), "meta": pack["meta"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
