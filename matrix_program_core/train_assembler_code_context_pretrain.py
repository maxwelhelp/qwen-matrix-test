#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Code-context assembly-skill pretrain.

This is the corrected bridge from parsed neural code to the transferable
MatrixProgramAssemblerCore.

It uses the old v3 parser/synthesizer only for:
  - parse Python/PyTorch code into CodeSkeleton + AST interactions;
  - sample logical step programs from real code structure.

It does NOT train a W->labels decoder. Instead it converts parsed/sampled code
program steps into the same flow targets used by MatrixProgramAssemblerCore:
  read_flow, primitive_slot_flow, slot_transition_flow,
  primitive_transition_flow, slot_composition_flow, write_flow.

The input to the assembler is compact context, not raw audio:
  TaskIOContextEncoder contract tokens
  + code skeleton tokens
  + input/head/consumer context tokens
  + task/solution/program hints.

This trains the skill: context -> assemble soft matrix program.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from matrix_program_core.assembler_core import AssemblerConfig, MatrixProgramAssemblerCore, assembler_skill_loss  # noqa: E402
from matrix_program_core.task_context_v2 import TaskContextV2Config, TaskIOContextEncoder  # noqa: E402
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
    "code_context": 5,
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

CORE_PRIM = list(PRIMITIVES)
P_IDX = {n: i for i, n in enumerate(CORE_PRIM)}
FLOW_KEYS = (
    "read_flow",
    "primitive_slot_flow",
    "slot_transition_flow",
    "primitive_transition_flow",
    "slot_composition_flow",
    "write_flow",
)
SKETCH_KEYS = (
    "primitive_hist",
    "primitive_transition_hist",
    "read_hist",
    "write_hist",
    "slot_transition_hist",
    "composition_hist",
)

OLD_TO_CORE_PRIMS = {
    "noop": ["phase_matrix"],
    "identity": ["ctx_matrix", "phase_matrix"],
    "keep_state": ["ctx_matrix", "phase_matrix"],
    "small_refine": ["channel_butterfly", "phase_matrix"],
    "mlp": ["low_rank", "ctx_matrix"],
    "matrix_mlp": ["block_butterfly", "low_rank", "ctx_matrix"],
    "compare": ["low_rank", "product_gate", "phase_matrix"],
    "memory_read": ["ctx_matrix", "low_rank"],
    "global_read": ["block_butterfly", "ctx_matrix"],
    "normalize": ["phase_matrix", "channel_butterfly"],
    "suppress": ["product_gate", "phase_matrix"],
    "attention": ["block_butterfly", "product_gate", "ctx_matrix"],
    "conv": ["channel_butterfly", "ctx_matrix"],
    "gate": ["product_gate", "phase_matrix"],
}
TRANS_TO_CORE_LINKS = {
    "keep": [("ctx_matrix", "ctx_matrix", 1.0), ("phase_matrix", "phase_matrix", 0.7)],
    "replace": [("ctx_matrix", "low_rank", 0.8), ("low_rank", "ctx_matrix", 0.5)],
    "residual": [("ctx_matrix", "channel_butterfly", 0.8), ("channel_butterfly", "phase_matrix", 0.5)],
    "norm_residual": [("channel_butterfly", "phase_matrix", 0.8), ("ctx_matrix", "phase_matrix", 0.5)],
    "product_gate": [("ctx_matrix", "product_gate", 0.9), ("product_gate", "phase_matrix", 0.7)],
    "compare_mix": [("low_rank", "product_gate", 0.9), ("product_gate", "phase_matrix", 0.6)],
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


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_old_v3_module(path: str | Path):
    p = Path(path)
    if p.is_dir():
        p = p / "neural_matrix_program_dataset_v3.py"
    if not p.exists():
        raise FileNotFoundError(f"old v3 module not found: {p}")
    spec = importlib.util.spec_from_file_location("old_neural_matrix_program_dataset_v3", str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import old v3 module: {p}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["old_neural_matrix_program_dataset_v3"] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_old_code(old, parse_files: Sequence[str], parse_dirs: Sequence[str], max_parse_files: int):
    files = old.resolve_parse_files(parse_files, parse_dirs, max_parse_files)
    sk, records = old.parse_code_files(files)
    return files, sk, records


def soft_one(size: int, hot: int | None = None, floor: float = 0.01, value: float = 1.0, device=None) -> torch.Tensor:
    x = torch.full((size,), floor, device=device)
    if hot is not None and 0 <= int(hot) < size:
        x[int(hot)] += value
    return x / x.sum().clamp_min(1e-8)


def add_to_dist(x: torch.Tensor, idx: int, value: float) -> None:
    if 0 <= int(idx) < x.numel():
        x[int(idx)] += float(value)


def normalize_last(x: torch.Tensor) -> torch.Tensor:
    return x / x.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def read_distribution(read: str, b: int, k: int, t: int, cfg: AssemblerConfig) -> torch.Tensor:
    A = cfg.address_cells
    out = torch.full((A,), 0.012)
    mem0 = cfg.blocks
    glob0 = cfg.blocks + cfg.memory_cells
    b = int(b) % max(1, cfg.blocks)
    r = str(read).lower()
    if r in ("state", "prev_step", "prev_same_block", "keep_state"):
        add_to_dist(out, b, 1.00)
        add_to_dist(out, (b - 1) % cfg.blocks, 0.15)
        add_to_dist(out, (b + 1) % cfg.blocks, 0.15)
    elif r in ("input", "task", "layer_route"):
        add_to_dist(out, glob0 + (k % max(1, cfg.global_cells)), 0.65 if cfg.global_cells else 0.0)
        add_to_dist(out, b, 0.45)
    elif r in ("global", "global_mean", "all_prev_steps", "prev_layer_mean"):
        for g in range(cfg.global_cells):
            add_to_dist(out, glob0 + g, 0.60)
        add_to_dist(out, b, 0.20)
    elif r in ("memory", "memory_mean"):
        for m in range(cfg.memory_cells):
            add_to_dist(out, mem0 + m, 0.55 / max(1, cfg.memory_cells))
        add_to_dist(out, mem0 + ((b + k + t) % max(1, cfg.memory_cells)), 0.55 if cfg.memory_cells else 0.0)
        add_to_dist(out, b, 0.20)
    else:
        add_to_dist(out, b, 0.75)
        if cfg.global_cells:
            add_to_dist(out, glob0, 0.25)
    return out / out.sum().clamp_min(1e-8)


def primitive_distribution(old_prim: str, slot_k: int, slot_count: int) -> torch.Tensor:
    out = torch.full((len(CORE_PRIM),), 0.018)
    names = OLD_TO_CORE_PRIMS.get(str(old_prim), None)
    if names is None:
        p = str(old_prim).lower()
        if "attn" in p:
            names = OLD_TO_CORE_PRIMS["attention"]
        elif "conv" in p:
            names = OLD_TO_CORE_PRIMS["conv"]
        elif "gate" in p:
            names = OLD_TO_CORE_PRIMS["gate"]
        elif "norm" in p:
            names = OLD_TO_CORE_PRIMS["normalize"]
        elif "mlp" in p or "linear" in p or "proj" in p:
            names = OLD_TO_CORE_PRIMS["mlp"]
        else:
            names = ["ctx_matrix", "phase_matrix"]
    for j, name in enumerate(names):
        add_to_dist(out, P_IDX.get(name, 0), 0.9 / (1 + abs(slot_k - (j % max(1, slot_count)))))
    return out / out.sum().clamp_min(1e-8)


def primitive_transition_distribution(trans_names: Sequence[str]) -> torch.Tensor:
    P = len(CORE_PRIM)
    out = torch.full((P, P), 0.01)
    out += torch.eye(P) * 0.20
    for tr in trans_names or ["keep"]:
        for a, b, w in TRANS_TO_CORE_LINKS.get(str(tr), [("ctx_matrix", "phase_matrix", 0.3)]):
            if a in P_IDX and b in P_IDX:
                out[P_IDX[a], P_IDX[b]] += float(w)
    return normalize_last(out)


def slot_transition_distribution(trans_names: Sequence[str], K: int) -> torch.Tensor:
    out = torch.full((K, K), 0.018)
    for k in range(K):
        out[k, k] += 0.70
        if trans_names:
            out[k, (k + 1) % K] += 0.30
            out[k, (k - 1) % K] += 0.12
        else:
            out[k, 0] += 0.12
    return normalize_last(out)


def write_distribution(write_target: str, b: int, t: int, cfg: AssemblerConfig) -> torch.Tensor:
    A = cfg.address_cells
    out = torch.full((A,), 0.012)
    mem0 = cfg.blocks
    glob0 = cfg.blocks + cfg.memory_cells
    b = int(b) % max(1, cfg.blocks)
    w = str(write_target).lower()
    if w == "memory" and cfg.memory_cells:
        add_to_dist(out, mem0 + ((b + t) % cfg.memory_cells), 1.0)
        add_to_dist(out, b, 0.25)
    elif w == "global" and cfg.global_cells:
        for g in range(cfg.global_cells):
            add_to_dist(out, glob0 + g, 0.55)
        add_to_dist(out, b, 0.20)
    elif w == "class" and cfg.global_cells:
        add_to_dist(out, glob0, 0.90)
        add_to_dist(out, b, 0.20)
    else:
        add_to_dist(out, b, 1.0)
        if cfg.memory_cells:
            add_to_dist(out, mem0 + (b % cfg.memory_cells), 0.18)
    return out / out.sum().clamp_min(1e-8)


def record_to_flow_targets(rec: Dict[str, Any], cfg: AssemblerConfig) -> Dict[str, torch.Tensor]:
    T = cfg.layers * cfg.steps
    B, K, A, P = cfg.blocks, cfg.primitive_slots, cfg.address_cells, len(CORE_PRIM)
    read = torch.full((T, B, K, A), 0.012)
    prim = torch.full((T, B, K, P), 0.018)
    slot_tr = torch.full((T, B, K, K), 0.018)
    prim_tr = torch.full((T, P, P), 0.010)
    comp = torch.full((T, B, K), 0.035)
    write = torch.full((T, B, A), 0.012)

    steps = rec.get("steps") or []
    if not steps:
        for t in range(T):
            for b in range(B):
                for k in range(K):
                    read[t, b, k] = read_distribution("state", b, k, t, cfg)
                    prim[t, b, k] = primitive_distribution("identity", k, K)
                slot_tr[t, b] = slot_transition_distribution(["keep"], K)
                write[t, b] = write_distribution("state", b, t, cfg)
            prim_tr[t] = primitive_transition_distribution(["keep"])
        return {"read_flow": read, "primitive_slot_flow": prim, "slot_transition_flow": slot_tr, "primitive_transition_flow": prim_tr, "slot_composition_flow": normalize_last(comp), "write_flow": write}

    for i, st in enumerate(steps):
        t = (int(st.get("layer", 0)) * cfg.steps + int(st.get("step", 0))) % T
        b = int(st.get("block", 0)) % B
        read_name = st.get("read", "state")
        prim_names = list(st.get("primitive_names") or ["identity"])
        trans_names = list(st.get("transition_names") or ["keep"])
        write_target = st.get("write_target", "state")
        for k in range(K):
            read[t, b, k] = read_distribution(read_name, b, k, t, cfg)
            old_p = prim_names[k % len(prim_names)]
            prim[t, b, k] = primitive_distribution(old_p, k, K)
            comp[t, b, k] += 0.85 / (1 + abs(k - (i % K)))
        slot_tr[t, b] = slot_transition_distribution(trans_names, K)
        prim_tr[t] += primitive_transition_distribution(trans_names)
        write[t, b] = write_distribution(write_target, b, t, cfg)

    return {
        "read_flow": normalize_last(read),
        "primitive_slot_flow": normalize_last(prim),
        "slot_transition_flow": normalize_last(slot_tr),
        "primitive_transition_flow": normalize_last(prim_tr),
        "slot_composition_flow": normalize_last(comp),
        "write_flow": normalize_last(write),
    }


def flow_sketch(ft: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Coarse program sketch tokens.

    These are not per-step labels. They are low-bandwidth histograms that tell
    the assembler what kind of program family the decoded matrix resembles.
    """

    return {
        "primitive_hist": ft["primitive_slot_flow"].float().mean(dim=(0, 1, 2)),
        "primitive_transition_hist": ft["primitive_transition_flow"].float().mean(dim=0).flatten(),
        "read_hist": ft["read_flow"].float().mean(dim=(0, 1, 2)),
        "write_hist": ft["write_flow"].float().mean(dim=(0, 1)),
        "slot_transition_hist": ft["slot_transition_flow"].float().mean(dim=(0, 1)).flatten(),
        "composition_hist": ft["slot_composition_flow"].float().mean(dim=(0, 1)),
    }


def infer_contract(sk: Any, records: Sequence[Dict[str, Any]], sample_record: Dict[str, Any] | None = None) -> Dict[str, int | float]:
    calls = getattr(sk, "call_counts", {}) or {}
    classes = " ".join(getattr(sk, "detected_classes", []) or []).lower()
    funcs = " ".join(getattr(sk, "detected_functions", []) or []).lower()
    text = (classes + " " + funcs + " " + " ".join(str(k) for k in calls.keys())).lower()
    if "attention" in text or "attn" in text or calls.get("softmax", 0) > 0:
        role = ROLE_IDS["attention_replacement"]
        inp = INPUT_KIND_IDS["token_states"]
        out = OUTPUT_KIND_IDS["token_states"]
        loss = LOSS_KIND_IDS["distillation"]
        readout = READOUT_KIND_IDS["residual_writeback"]
    elif "classifier" in text or "class" in text or "head" in text:
        role = ROLE_IDS["task_head_core"]
        inp = INPUT_KIND_IDS["generic_evidence"]
        out = OUTPUT_KIND_IDS["class_logits"]
        loss = LOSS_KIND_IDS["cross_entropy"]
        readout = READOUT_KIND_IDS["class_query"]
    elif "matrix" in text or "program" in text:
        role = ROLE_IDS["matrix_decompiler"]
        inp = INPUT_KIND_IDS["matrix_features"]
        out = OUTPUT_KIND_IDS["program_slots"]
        loss = LOSS_KIND_IDS["flow_skill"]
        readout = READOUT_KIND_IDS["program_slot"]
    else:
        role = ROLE_IDS["generic_program_core"]
        inp = INPUT_KIND_IDS["code_context"]
        out = OUTPUT_KIND_IDS["generic_state"]
        loss = LOSS_KIND_IDS["flow_skill"]
        readout = READOUT_KIND_IDS["state_writeback"]
    return {
        "role_id": role,
        "input_kind_id": inp,
        "output_kind_id": out,
        "loss_kind_id": loss,
        "readout_kind_id": readout,
        "num_outputs": float(max(1, len(getattr(sk, "primitive_names", []) or []))),
        "sequence_length": float(max(1, int(getattr(sk, "L", 4)) * int(getattr(sk, "S", 2)) * int(getattr(sk, "N", 4)))),
        "hidden_dim": float(96),
        "extra_scalar": float(sum(calls.values()) if calls else 0),
    }


class CodeContextDataset(Dataset):
    def __init__(self, pack: Dict[str, Any], indices: Sequence[int]):
        self.pack = pack
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        j = self.indices[i]
        out = {"idx": torch.tensor(j, dtype=torch.long)}
        for k in ("role_id", "input_kind_id", "output_kind_id", "loss_kind_id", "readout_kind_id"):
            out[k] = self.pack[k][j].long()
        for k in ("num_outputs", "sequence_length", "hidden_dim", "extra_scalar"):
            out[k] = self.pack[k][j].float()
        for k in FLOW_KEYS:
            out[k] = self.pack[k][j].float()
        for k in SKETCH_KEYS:
            if k in self.pack:
                out[k] = self.pack[k][j].float()
        return out


class CodeContextEvidenceBuilder(nn.Module):
    def __init__(
        self,
        dim: int,
        cfg: AssemblerConfig | None = None,
        skeleton_kinds: int = 8,
        extra_tokens: int = 8,
        dropout: float = 0.02,
    ):
        super().__init__()
        P = len(CORE_PRIM)
        A = int(cfg.address_cells) if cfg is not None else 0
        K = int(cfg.primitive_slots) if cfg is not None else 0
        self.skeleton_kind = nn.Embedding(skeleton_kinds, dim)
        self.sketch_proj = nn.ModuleDict()
        if cfg is not None:
            self.sketch_proj = nn.ModuleDict({
                "primitive_hist": nn.Linear(P, dim),
                "primitive_transition_hist": nn.Linear(P * P, dim),
                "read_hist": nn.Linear(A, dim),
                "write_hist": nn.Linear(A, dim),
                "slot_transition_hist": nn.Linear(K * K, dim),
                "composition_hist": nn.Linear(K, dim),
            })
        self.sketch_type_embed = nn.Parameter(torch.randn(len(SKETCH_KEYS), dim) * 0.02)
        self.extra = nn.Parameter(torch.randn(max(1, extra_tokens), dim) * 0.02)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        context: torch.Tensor,
        skeleton_kind_id: torch.Tensor | None = None,
        sketch: Dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        B, _, D = context.shape
        extra = self.extra.to(context.device, context.dtype).view(1, -1, D).expand(B, -1, -1)
        sketch_tokens: List[torch.Tensor] = []
        if sketch and self.sketch_proj:
            for i, key in enumerate(SKETCH_KEYS):
                value = sketch.get(key)
                proj = self.sketch_proj[key] if key in self.sketch_proj else None
                if value is None or proj is None:
                    continue
                token = proj(value.to(device=context.device, dtype=torch.float32).view(B, -1)).to(context.dtype)
                token = token + self.sketch_type_embed[i].to(device=context.device, dtype=context.dtype).view(1, -1)
                sketch_tokens.append(token.view(B, 1, D))
        if skeleton_kind_id is not None:
            sk = self.skeleton_kind(skeleton_kind_id.to(context.device).long()).view(B, 1, D).to(context.dtype)
            extra = extra + 0.25 * sk
            context = torch.cat([context, sk], dim=1)
        if sketch_tokens:
            context = torch.cat([context] + sketch_tokens, dim=1)
        return self.drop(self.norm(torch.cat([context, extra], dim=1)))


def build_code_context_pack(args, cfg: AssemblerConfig) -> Dict[str, Any]:
    old = load_old_v3_module(args.old_v3_path)
    files, sk, ast_records = parse_old_code(old, args.parse_files, args.parse_dirs, args.max_parse_files)
    if args.layers is not None:
        sk.L = args.layers
    if args.blocks is not None:
        sk.N = args.blocks
    if args.steps is not None:
        sk.S = args.steps
    if args.primitive_slots is not None:
        sk.K = args.primitive_slots
    lib = old.MatrixOpLibrary(args.matrix_D, device="cpu")
    synth = old.StructuredSynthesizer(sk, lib, seed=args.seed, step_scale=args.step_scale, route_scale=args.route_scale, max_program_steps=args.max_program_steps)
    records: List[Dict[str, Any]] = []
    flows: Dict[str, List[torch.Tensor]] = {k: [] for k in ("read_flow", "primitive_slot_flow", "slot_transition_flow", "primitive_transition_flow", "slot_composition_flow", "write_flow")}
    sketches: Dict[str, List[torch.Tensor]] = {k: [] for k in ("primitive_hist", "primitive_transition_hist", "read_hist", "write_hist", "slot_transition_hist", "composition_hist")}
    contracts: Dict[str, List[torch.Tensor]] = {k: [] for k in ("role_id", "input_kind_id", "output_kind_id", "loss_kind_id", "readout_kind_id", "num_outputs", "sequence_length", "hidden_dim", "extra_scalar")}
    t0 = time.time()
    for i in range(args.n):
        _row, rec = synth.sample_one(i)
        contract = infer_contract(sk, ast_records, rec)
        ft = record_to_flow_targets(rec, cfg)
        for k, v in ft.items():
            flows[k].append(v.cpu())
        for k, v in flow_sketch(ft).items():
            sketches[k].append(v.cpu())
        for k, v in contract.items():
            dtype = torch.long if k.endswith("_id") else torch.float32
            contracts[k].append(torch.tensor(v, dtype=dtype))
        records.append(rec)
        if args.log_every and (i + 1) % args.log_every == 0:
            print(f"[code-context-build] {i+1}/{args.n} t={time.time()-t0:.1f}s", flush=True)
    pack: Dict[str, Any] = {}
    for k, lst in flows.items():
        pack[k] = torch.stack(lst, dim=0)
    for k, lst in contracts.items():
        pack[k] = torch.stack(lst, dim=0)
    for k, lst in sketches.items():
        pack[k] = torch.stack(lst, dim=0)
    pack["meta"] = {
        "n": args.n,
        "parse_files": files,
        "skeleton_kind": getattr(sk, "skeleton_kind", "unknown"),
        "skeleton": sk.__dict__ if hasattr(sk, "__dict__") else {},
        "ast_records": len(ast_records),
        "core_primitives": CORE_PRIM,
        "truth_level": "code_context_matrix_program_flow_targets",
        "program_sketch_keys": list(sketches.keys()),
    }
    return pack, records, ast_records


def make_context_tokens(task_context: TaskIOContextEncoder, batch: Dict[str, torch.Tensor], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    B = int(batch["role_id"].shape[0])
    return task_context(
        B, device, dtype,
        role_id=batch["role_id"].to(device),
        input_kind_id=batch["input_kind_id"].to(device),
        output_kind_id=batch["output_kind_id"].to(device),
        loss_kind_id=batch["loss_kind_id"].to(device),
        readout_kind_id=batch["readout_kind_id"].to(device),
        num_outputs=batch["num_outputs"].to(device),
        sequence_length=batch["sequence_length"].to(device),
        hidden_dim=batch["hidden_dim"].to(device),
        extra_scalar=batch["extra_scalar"].to(device),
        head_query=None,
    )


def targets_from_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    # Dataset gives [N,T,...]; assembler_skill_loss expects [T,N,...].
    out = {}
    for k in FLOW_KEYS:
        out[k] = batch[k].to(device, non_blocking=True).transpose(0, 1).contiguous()
    return out


def run_epoch(core, task_context, evidence_builder, loader, device, dtype, args, train: bool, opt=None, scaler=None):
    core.train(train); task_context.train(train); evidence_builder.train(train)
    use_amp = str(device).startswith("cuda") and dtype != torch.float32
    total, n = 0.0, 0
    loss_acc: Dict[str, float] = {}
    last_aux = None
    skill_weights = {
        "read_flow_kl": args.w_read,
        "primitive_slot_kl": args.w_primitive,
        "slot_transition_kl": args.w_slot_transition,
        "primitive_transition_kl": args.w_primitive_transition,
        "slot_composition_kl": args.w_composition,
        "write_flow_kl": args.w_write,
    }
    for step, batch in enumerate(loader, 1):
        if train:
            opt.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=str(device).split(":")[0], dtype=dtype, enabled=use_amp):
                ctx = make_context_tokens(task_context, batch, device, dtype)
                sketch = {k: batch[k].to(device, non_blocking=True) for k in SKETCH_KEYS if k in batch}
                evidence = evidence_builder(ctx, sketch=sketch)
                _cells, aux = core(evidence)
                skill, flow_losses = assembler_skill_loss(aux, targets_from_batch(batch, device), skill_weights)
                entropy_keep = torch.zeros((), device=device)
                if args.lambda_entropy_keep > 0:
                    for ent in aux.entropies.values():
                        entropy_keep = entropy_keep + F.relu(torch.tensor(args.min_entropy, device=device) - ent.float()).pow(2)
                delta_l2 = torch.zeros((), device=device)
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
        bs = int(batch["role_id"].shape[0])
        total += float(loss.detach().cpu()) * bs
        n += bs
        for k, v in flow_losses.items():
            loss_acc[k] = loss_acc.get(k, 0.0) + float(v.detach().cpu()) * bs
        last_aux = aux
        if train and args.log_every and step % args.log_every == 0:
            print(f"step {step:05d} loss={total/max(1,n):.4f} read={loss_acc.get('read_flow_kl',0)/max(1,n):.3f} prim={loss_acc.get('primitive_slot_kl',0)/max(1,n):.3f} write={loss_acc.get('write_flow_kl',0)/max(1,n):.3f}", flush=True)
    out = {"loss": total / max(1, n)}
    for k, v in loss_acc.items():
        out[k] = v / max(1, n)
    if last_aux is not None:
        for k, v in last_aux.entropies.items():
            out[f"entropy_{k}"] = float(v.detach().cpu())
        out["memory_usage"] = float(last_aux.memory_usage.detach().cpu())
        out["global_usage"] = float(last_aux.global_usage.detach().cpu())
    return out


def run(args) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    if str(device).startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(args.amp, torch.float32)
    out = ensure_dir(args.out_dir)
    cfg = AssemblerConfig(dim=args.dim, evidence_cells=args.evidence_cells, layers=args.layers_core, blocks=args.blocks_core, steps=args.steps_core, primitive_slots=args.primitive_slots_core, memory_cells=args.memory_cells, global_cells=args.global_cells, channel_stages=args.channel_stages, dropout=args.dropout, use_deltas=True)

    if args.dataset and Path(args.dataset).exists():
        pack = torch.load(args.dataset, map_location="cpu")
        records, ast_records = [], []
        print(f"[code-context] loaded dataset {args.dataset}", flush=True)
    else:
        pack, records, ast_records = build_code_context_pack(args, cfg)
        torch.save(pack, out / "code_context_dataset.pt")
        write_jsonl(out / "programs.jsonl", records[: min(len(records), args.max_json_records)])
        write_json(out / "ast_summary.json", {"ast_records": len(ast_records), "meta": pack.get("meta", {})})
        print(f"[code-context] saved dataset {out/'code_context_dataset.pt'}", flush=True)

    N = int(pack["read_flow"].shape[0])
    idx = torch.randperm(N)
    n_train = max(1, int(N * args.train_frac))
    tr_idx = idx[:n_train].tolist()
    va_idx = idx[n_train:].tolist() or tr_idx
    train = DataLoader(CodeContextDataset(pack, tr_idx), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=args.pin_memory)
    val = DataLoader(CodeContextDataset(pack, va_idx), batch_size=args.eval_batch_size, shuffle=False, num_workers=args.workers, pin_memory=args.pin_memory)

    core = MatrixProgramAssemblerCore(cfg).to(device)
    task_context = TaskIOContextEncoder(TaskContextV2Config(dim=args.dim, free_tokens=args.task_context_tokens, max_head_tokens=args.head_context_tokens, dropout=args.dropout)).to(device)
    evidence_builder = CodeContextEvidenceBuilder(args.dim, cfg=cfg, extra_tokens=args.extra_context_tokens, dropout=args.dropout).to(device)
    if args.init_assembler:
        ckpt = torch.load(args.init_assembler, map_location=device)
        state = ckpt.get("assembler_core", ckpt.get("assembler_skill_base", ckpt.get("core", ckpt)))
        missing, unexpected = core.load_state_dict(state, strict=False)
        print(f"loaded init assembler missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        if "task_context" in ckpt:
            task_context.load_state_dict(ckpt["task_context"], strict=False)

    params = list(core.parameters()) + list(task_context.parameters()) + list(evidence_builder.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=str(device).startswith("cuda") and dtype == torch.float16)
    fields = ["epoch", "train_loss", "val_loss", "best_val", "read_flow_kl", "primitive_slot_kl", "slot_transition_kl", "primitive_transition_kl", "slot_composition_kl", "write_flow_kl", "entropy_read", "entropy_primitive", "entropy_write", "memory_usage", "global_usage"]
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()
    best, best_epoch = 1e18, 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        tr = run_epoch(core, task_context, evidence_builder, train, device, dtype, args, True, opt, scaler)
        with torch.no_grad():
            va = run_epoch(core, task_context, evidence_builder, val, device, dtype, args, False)
        if va["loss"] < best:
            best, best_epoch = va["loss"], ep
            torch.save({"assembler_core": core.state_dict(), "task_context": task_context.state_dict(), "code_context_evidence_builder": evidence_builder.state_dict(), "config": cfg.__dict__, "args": vars(args), "best_val_loss": best, "epoch": ep, "meta": pack.get("meta", {}), "primitives": CORE_PRIM}, out / "assembler_code_context_best.pt")
        torch.save({"assembler_core": core.state_dict(), "task_context": task_context.state_dict(), "code_context_evidence_builder": evidence_builder.state_dict(), "config": cfg.__dict__, "args": vars(args), "best_val_loss": best, "epoch": ep, "meta": pack.get("meta", {}), "primitives": CORE_PRIM}, out / "assembler_code_context_last.pt")
        row = {"epoch": ep, "train_loss": tr["loss"], "val_loss": va["loss"], "best_val": best}
        for k in fields:
            if k in va:
                row[k] = va[k]
        with (out / "metrics.csv").open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writerow({k: row.get(k, 0.0) for k in fields})
        print(f"epoch {ep:03d}/{args.epochs} train={tr['loss']:.4f} val={va['loss']:.4f} best={best:.4f}@{best_epoch}", flush=True)
    write_json(out / "final_report.json", {"best_val_loss": best, "best_epoch": best_epoch, "elapsed_sec": time.time() - t0, "checkpoint": str(out / "assembler_code_context_best.pt"), "dataset": str(out / "code_context_dataset.pt"), "meta": pack.get("meta", {})})


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--old-v3-path", default="neural_matrix_program_dataset_v3/neural_matrix_program_dataset_v3.py")
    p.add_argument("--parse-files", nargs="*", default=[])
    p.add_argument("--parse-dirs", nargs="*", default=["."])
    p.add_argument("--max-parse-files", type=int, default=128)
    p.add_argument("--dataset", default="")
    p.add_argument("--out-dir", default="matrix_program_core/runs/code_context_skill")
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", choices=["fp16", "bf16", "fp32", "off"], default="bf16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n", type=int, default=6000)
    p.add_argument("--matrix-D", type=int, default=32)
    p.add_argument("--dim", type=int, default=96)
    p.add_argument("--evidence-cells", type=int, default=48)
    p.add_argument("--layers-core", type=int, default=4)
    p.add_argument("--blocks-core", type=int, default=4)
    p.add_argument("--steps-core", type=int, default=2)
    p.add_argument("--primitive-slots-core", type=int, default=4)
    p.add_argument("--layers", type=int, default=None)
    p.add_argument("--blocks", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--primitive-slots", type=int, default=None)
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
    p.add_argument("--train-frac", type=float, default=0.85)
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
    p.add_argument("--step-scale", type=float, default=0.23)
    p.add_argument("--route-scale", type=float, default=0.10)
    p.add_argument("--max-program-steps", type=int, default=0)
    p.add_argument("--init-assembler", default="")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--max-json-records", type=int, default=500)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if args.max_program_steps <= 0:
        args.max_program_steps = None
    run(args)
