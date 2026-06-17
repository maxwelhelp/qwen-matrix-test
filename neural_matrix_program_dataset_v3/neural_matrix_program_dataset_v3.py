#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
neural_matrix_program_dataset_v3.py

Final unified tool for building a logical dataset of matrix programs from neural code.

It has four honest data levels:

1) static_pseudocode
   Python/PyTorch code -> AST structure and interactions.
   No claim of exact matrix behavior.

2) real_structure_synthetic_matrix
   Real code structure -> logical synthetic matrix programs.
   This cheaply teaches syntax/grammar/logic: layer/block/step/read/primitive/transition/write.

3) real_weight_program_decode
   Real checkpoint/state_dict matrices -> operator dictionary -> readable matrix program.
   This is not fake: the target W comes from real model weights.

4) runtime_jacobian_program_decode
   Placeholder protocol in README. This script prepares the format; true runtime tracing
   needs model-specific forward hooks and inputs.

Main commands:

  python neural_matrix_program_dataset_v3.py parse --parse-dir ./project --out ./runs/proj

  python neural_matrix_program_dataset_v3.py build-synth --parse-dir ./project --out ./runs/proj --n 5000 --D 32

  python neural_matrix_program_dataset_v3.py decode-real --checkpoint ./model.pt --out ./runs/proj --D 64

  python neural_matrix_program_dataset_v3.py train-synth --dataset ./runs/proj/synthetic/dataset.pt --out ./runs/proj

  python neural_matrix_program_dataset_v3.py all --parse-dir ./project --checkpoint ./model.pt --out ./runs/full

Outputs:
  ast/skeleton.json
  ast/ast_interactions.jsonl
  synthetic/dataset.pt
  synthetic/programs.jsonl
  synthetic/baseline_metrics.json
  real_decode/real_matrix_decodes.jsonl
  real_decode/real_matrix_dataset.pt
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Utilities
# =============================================================================

def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    ensure_dir(path.parent)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return type(node).__name__


def rel_err(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    a = a.detach().float()
    b = b.detach().float()
    return float(torch.norm(a - b) / (torch.norm(a) + eps))


def mat_norm(M: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.linalg.matrix_norm(M.float()).clamp_min(eps)


def safe_scale(M: torch.Tensor, max_norm: float = 1.0) -> torch.Tensor:
    n = mat_norm(M)
    if float(n) > max_norm:
        return M * (max_norm / n)
    return M


def sanitize_tensor(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)


def resample_matrix(W: torch.Tensor, D: int) -> torch.Tensor:
    """Resize any 2D matrix to D x D for dictionary decoding.

    This keeps the pattern but changes exact semantics for rectangular matrices.
    The record stores original_shape and truth_level so we don't pretend otherwise.
    """
    W = sanitize_tensor(W)
    if W.ndim != 2:
        W = W.reshape(W.shape[0], -1)
    if W.shape == (D, D):
        return W
    X = W[None, None, :, :]
    Y = F.interpolate(X, size=(D, D), mode="bilinear", align_corners=False)[0, 0]
    return Y


def functional_error_square(W: torch.Tensor, Wh: torch.Tensor, batch: int = 512, mode: str = "gaussian") -> float:
    W = W.float()
    Wh = Wh.float()
    D = W.shape[0]
    g = torch.Generator(device=W.device)
    g.manual_seed(123)
    if mode == "smooth":
        z = torch.randn(batch, D, generator=g, device=W.device)
        C = dct_matrix(D, W.device, W.dtype)
        keep = max(1, D // 4)
        zz = torch.zeros_like(z)
        zz[:, :keep] = z[:, :keep]
        x = zz @ C
    elif mode == "spiky":
        x = 0.05 * torch.randn(batch, D, generator=g, device=W.device)
        idx = torch.randint(0, D, (batch, 2), generator=g, device=W.device)
        x.scatter_add_(1, idx, torch.randn(batch, 2, generator=g, device=W.device))
    else:
        x = torch.randn(batch, D, generator=g, device=W.device)
    y = x @ W.T
    yh = x @ Wh.T
    return rel_err(y, yh)


# =============================================================================
# AST parser / real code skeleton
# =============================================================================

IMPORTANT_CALLS = {
    "softmax": "gate_softmax",
    "sigmoid": "gate_sigmoid",
    "einsum": "weighted_mix_or_tensor_interaction",
    "matmul": "matrix_product",
    "bmm": "batched_matrix_product",
    "mm": "matrix_product",
    "cat": "concat_context",
    "stack": "stack_candidates",
    "Linear": "linear_projection",
    "Conv1d": "conv1d_projection",
    "Conv2d": "conv2d_projection",
    "LayerNorm": "normalization",
    "BatchNorm1d": "normalization",
    "BatchNorm2d": "normalization",
    "RMSNorm": "normalization",
    "GELU": "nonlinear_activation",
    "ReLU": "nonlinear_activation",
    "SiLU": "nonlinear_activation",
    "Dropout": "regularization",
    "ModuleList": "module_container",
    "Sequential": "module_container",
    "Parameter": "learned_parameter",
    "adaptive_avg_pool2d": "pooling",
}

STATE_WORDS = [
    "state", "hidden", "h", "x", "prev_layer", "prev_step", "all_prev",
    "read_ctx", "input_ctx", "global_ctx", "memory_ctx", "task_ctx",
    "global_cells", "memory_cells", "route_gates", "read_gates",
    "prim_gates", "trans_gates", "write_gate", "logits", "slot_states",
    "class_read", "attn", "attention", "residual",
]

DEFAULT_PRIMITIVE_NAMES = [
    "noop", "identity", "keep_state", "small_refine",
    "mlp", "matrix_mlp", "compare", "memory_read", "global_read", "normalize", "suppress",
]
DEFAULT_PRIM_TRANSITION_NAMES = ["keep", "replace", "residual", "norm_residual", "product_gate", "compare_mix"]
DEFAULT_STEP_READ_NAMES = ["state", "input", "prev_step", "all_prev_steps", "layer_route", "global", "memory", "task"]
DEFAULT_LAYER_ROUTE_NAMES = ["input", "prev_same_block", "prev_layer_mean", "global_mean", "memory_mean"]


def name_of(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = name_of(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Call):
        return name_of(node.func)
    if isinstance(node, ast.Subscript):
        return name_of(node.value)
    return ""


def target_names(target: ast.AST) -> List[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, ast.Attribute):
        return [name_of(target)]
    if isinstance(target, ast.Subscript):
        return [name_of(target)]
    if isinstance(target, ast.Tuple):
        out: List[str] = []
        for e in target.elts:
            out.extend(target_names(e))
        return out
    return [unparse(target)]


def literal_str_list(node: ast.AST) -> Optional[List[str]]:
    if isinstance(node, (ast.List, ast.Tuple)):
        vals = []
        for e in node.elts:
            if isinstance(e, ast.Constant) and isinstance(e.value, str):
                vals.append(e.value)
            else:
                return None
        return vals
    return None


def literal_int(node: ast.AST) -> Optional[int]:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    return None


def merge_unique_names(*seqs: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for seq in seqs:
        for name in seq or []:
            s = str(name)
            if s and s not in seen:
                seen.add(s)
                out.append(s)
    return out


@dataclass
class CodeSkeleton:
    source_files: List[str] = field(default_factory=list)
    detected_classes: List[str] = field(default_factory=list)
    detected_functions: List[str] = field(default_factory=list)
    constants: Dict[str, List[str]] = field(default_factory=dict)
    init_defaults: Dict[str, int] = field(default_factory=dict)
    call_counts: Dict[str, int] = field(default_factory=dict)
    L: int = 4
    N: int = 3
    S: int = 3
    K: int = 3
    primitive_names: List[str] = field(default_factory=lambda: list(DEFAULT_PRIMITIVE_NAMES))
    transition_names: List[str] = field(default_factory=lambda: list(DEFAULT_PRIM_TRANSITION_NAMES))
    read_names: List[str] = field(default_factory=lambda: list(DEFAULT_STEP_READ_NAMES))
    route_names: List[str] = field(default_factory=lambda: list(DEFAULT_LAYER_ROUTE_NAMES))
    skeleton_kind: str = "generic"
    notes: List[str] = field(default_factory=list)


class ASTCollector(ast.NodeVisitor):
    def __init__(self, filename: str):
        self.filename = filename
        self.class_stack: List[str] = []
        self.func_stack: List[str] = []
        self.records: List[Dict[str, Any]] = []
        self.constants: Dict[str, List[str]] = {}
        self.init_defaults: Dict[str, int] = {}
        self.classes: List[str] = []
        self.functions: List[str] = []
        self.call_counts: Dict[str, int] = {}

    @property
    def scope(self) -> str:
        parts = []
        if self.class_stack:
            parts.append(".".join(self.class_stack))
        if self.func_stack:
            parts.append(".".join(self.func_stack))
        return ".".join(parts) if parts else "<module>"

    def add(self, kind: str, node: ast.AST, **kw: Any) -> None:
        rec = {
            "file": self.filename,
            "scope": self.scope,
            "kind": kind,
            "lineno": getattr(node, "lineno", None),
            "code": unparse(node),
        }
        rec.update(kw)
        self.records.append(rec)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.classes.append(node.name)
        self.add("class_def", node, name=node.name)
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for t in stmt.targets:
                    if isinstance(t, ast.Name) and t.id.endswith("_NAMES"):
                        vals = literal_str_list(stmt.value)
                        if vals:
                            self.constants[t.id] = vals
                            self.add("name_list_constant", stmt, name=t.id, values=vals)
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.functions.append(f"{self.scope}.{node.name}")
        self.add("function_def", node, name=node.name, args=[a.arg for a in node.args.args])
        if node.name == "__init__":
            args = node.args.args
            defaults = list(node.args.defaults)
            aligned = [None] * (len(args) - len(defaults)) + defaults
            for a, d in zip(args, aligned):
                if d is not None:
                    vi = literal_int(d)
                    if vi is not None:
                        self.init_defaults[a.arg] = vi
                        self.add("init_int_default", d, name=a.arg, value=vi)
        self.func_stack.append(node.name)
        self.generic_visit(node)
        self.func_stack.pop()

    def visit_For(self, node: ast.For) -> None:
        self.add("loop", node, target=unparse(node.target), iterator=unparse(node.iter))
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        targets = []
        for t in node.targets:
            targets.extend(target_names(t))
            if isinstance(t, ast.Name) and (t.id.endswith("_NAMES") or t.id in {"PRIMITIVES", "TRANSITIONS", "READS", "ROUTES", "PHASES"}):
                vals = literal_str_list(node.value)
                if vals:
                    self.constants[t.id] = vals
                    self.add("name_list_constant", node, name=t.id, values=vals)
        reads = sorted({n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)})
        kind = "assign"
        joined = " ".join(targets + reads)
        if any(w in joined for w in STATE_WORDS):
            kind = "state_or_gate_assign"
        self.add(kind, node, targets=targets, reads=reads, value=unparse(node.value))
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.add(
            "state_update",
            node,
            targets=[unparse(node.target)],
            reads=sorted({n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}),
            op=type(node.op).__name__,
            value=unparse(node.value),
        )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        cname = name_of(node.func)
        short = cname.split(".")[-1]
        self.call_counts[short] = self.call_counts.get(short, 0) + 1
        if short in IMPORTANT_CALLS:
            self.add(
                "important_call",
                node,
                call=cname,
                op_family=IMPORTANT_CALLS[short],
                args=[unparse(a) for a in node.args],
                keywords={kw.arg: unparse(kw.value) for kw in node.keywords if kw.arg},
            )
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.MatMult):
            self.add("matrix_matmul", node, left=unparse(node.left), right=unparse(node.right))
        elif isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
            left, right = unparse(node.left), unparse(node.right)
            if any(w in (left + " " + right) for w in STATE_WORDS):
                self.add("state_binary_interaction", node, op=type(node.op).__name__, left=left, right=right)
        self.generic_visit(node)


def resolve_parse_files(parse_files: Sequence[str], parse_dirs: Sequence[str], max_parse_files: int) -> List[str]:
    files: List[str] = []
    for f in parse_files or []:
        files.append(str(f))
    skip_parts = {".git", "__pycache__", ".venv", "venv", "env", "site-packages", "node_modules"}
    for d in parse_dirs or []:
        root = Path(d)
        if root.is_file() and root.suffix == ".py":
            files.append(str(root))
            continue
        if not root.exists():
            print(f"[WARN] parse dir does not exist: {root}", flush=True)
            continue
        for p in sorted(root.rglob("*.py")):
            if any(part in skip_parts for part in p.parts):
                continue
            files.append(str(p))
    seen = set()
    out = []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    if max_parse_files and max_parse_files > 0:
        out = out[:max_parse_files]
    return out


def parse_code_files(files: Sequence[str]) -> Tuple[CodeSkeleton, List[Dict[str, Any]]]:
    sk = CodeSkeleton(source_files=[str(f) for f in files])
    records: List[Dict[str, Any]] = []
    constants: Dict[str, List[str]] = {}
    init_defaults: Dict[str, int] = {}
    call_counts: Dict[str, int] = {}

    for fp in files:
        path = Path(fp)
        if not path.exists():
            print(f"[WARN] parse file does not exist: {path}", flush=True)
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError as e:
            print(f"[WARN] syntax error in {path}: {e}", flush=True)
            continue
        c = ASTCollector(str(path))
        c.visit(tree)
        records.extend(c.records)
        sk.detected_classes.extend(c.classes)
        sk.detected_functions.extend(c.functions)
        constants.update(c.constants)
        init_defaults.update(c.init_defaults)
        for k, v in c.call_counts.items():
            call_counts[k] = call_counts.get(k, 0) + v

    sk.constants = constants
    sk.init_defaults = init_defaults
    sk.call_counts = call_counts

    primitive_sources = []
    if "PRIMITIVE_NAMES" in constants:
        primitive_sources.append(constants["PRIMITIVE_NAMES"])
        sk.skeleton_kind = "step_program_like"
    if "PRIMITIVES" in constants:
        primitive_sources.append(constants["PRIMITIVES"])
        sk.notes.append(f"Auto-expanded primitives from PRIMITIVES: {len(constants['PRIMITIVES'])}")
    if primitive_sources:
        sk.primitive_names = merge_unique_names(DEFAULT_PRIMITIVE_NAMES, *primitive_sources)

    transition_sources = []
    for key in ("PRIM_TRANSITION_NAMES", "TRANSITION_NAMES", "TRANSITIONS"):
        if key in constants:
            transition_sources.append(constants[key])
    if transition_sources:
        sk.transition_names = merge_unique_names(DEFAULT_PRIM_TRANSITION_NAMES, *transition_sources)

    read_sources = []
    for key in ("STEP_READ_NAMES", "READ_NAMES", "READS"):
        if key in constants:
            read_sources.append(constants[key])
    if read_sources:
        sk.read_names = merge_unique_names(DEFAULT_STEP_READ_NAMES, *read_sources)

    route_sources = []
    for key in ("LAYER_ROUTE_NAMES", "ROUTE_NAMES", "ROUTES"):
        if key in constants:
            route_sources.append(constants[key])
    if route_sources:
        sk.route_names = merge_unique_names(DEFAULT_LAYER_ROUTE_NAMES, *route_sources)

    mapping = [
        ("num_layers", "L"),
        ("layers", "L"),
        ("n_layers", "L"),
        ("blocks_per_layer", "N"),
        ("blocks", "N"),
        ("n_blocks", "N"),
        ("steps_per_block", "S"),
        ("steps", "S"),
        ("n_steps", "S"),
        ("primitive_slots", "K"),
        ("n_slots", "K"),
    ]
    for key, attr in mapping:
        if key in init_defaults:
            setattr(sk, attr, int(init_defaults[key]))

    class_join = " ".join(sk.detected_classes)
    if "StepProgramNet" in class_join or "StepProgram" in class_join:
        sk.skeleton_kind = "step_program_like"
        sk.notes.append("Detected StepProgram-like class; using L/B/S/K skeleton.")
    elif any(k in call_counts for k in ("softmax", "einsum", "Linear", "LayerNorm", "Conv1d", "Conv2d")):
        sk.skeleton_kind = "pytorch_like"
        sk.notes.append("Detected PyTorch-like operations; using generic neural skeleton.")
    else:
        sk.notes.append("No strong neural pattern detected; using fallback generic L/B/S skeleton.")
    return sk, records


def run_parse(args: argparse.Namespace) -> Tuple[CodeSkeleton, List[Dict[str, Any]]]:
    out = ensure_dir(Path(args.out) / "ast")
    files = resolve_parse_files(args.parse_files, args.parse_dirs, args.max_parse_files)
    sk, records = parse_code_files(files)
    write_json(out / "skeleton.json", asdict(sk))
    write_jsonl(out / "ast_interactions.jsonl", records)
    write_json(out / "ast_summary.json", {
        "parsed_files": len(files),
        "records": len(records),
        "skeleton_kind": sk.skeleton_kind,
        "L": sk.L, "N": sk.N, "S": sk.S, "K": sk.K,
        "call_counts": sk.call_counts,
        "notes": sk.notes,
    })
    print(f"[parse] files={len(files)} records={len(records)} skeleton={sk.skeleton_kind} -> {out}", flush=True)
    return sk, records


# =============================================================================
# Matrix operator library
# =============================================================================

@dataclass
class OpSpec:
    name: str
    family: str
    matrix: torch.Tensor
    desc: str = ""
    origin: str = "base"


def dct_matrix(D: int, device=None, dtype=torch.float32) -> torch.Tensor:
    device = device or torch.device("cpu")
    k = torch.arange(D, device=device, dtype=dtype).view(-1, 1)
    i = torch.arange(D, device=device, dtype=dtype).view(1, -1)
    C = torch.cos(math.pi / D * (i + 0.5) * k)
    C[0, :] *= math.sqrt(1.0 / D)
    if D > 1:
        C[1:, :] *= math.sqrt(2.0 / D)
    return C


class MatrixOpLibrary:
    def __init__(
        self,
        D: int,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        include_base: bool = True,
    ):
        self.D = int(D)
        self.device = torch.device(device)
        self.dtype = dtype
        self.ops: List[OpSpec] = []
        if include_base:
            self._build_base()
        self._refresh()

    def _refresh(self) -> None:
        self.names = [o.name for o in self.ops]
        self.name_to_idx = {n: i for i, n in enumerate(self.names)}
        self.families: Dict[str, int] = {}
        for o in self.ops:
            self.families[o.family] = self.families.get(o.family, 0) + 1
        if self.ops:
            self.M = torch.stack([o.matrix.to(self.device, self.dtype) for o in self.ops], dim=0)
        else:
            self.M = torch.empty(0, self.D, self.D, device=self.device, dtype=self.dtype)

    def add(self, name: str, family: str, M: torch.Tensor, desc: str = "", origin: str = "base") -> None:
        base = name
        n = base
        j = 2
        existing = {o.name for o in self.ops}
        while n in existing:
            n = f"{base}#{j}"
            j += 1
        M = torch.nan_to_num(M.to(self.device, self.dtype), nan=0.0, posinf=0.0, neginf=0.0)
        self.ops.append(OpSpec(n, family, M, desc, origin))
        self._refresh()

    def shift(self, s: int) -> torch.Tensor:
        D = self.D
        M = torch.zeros(D, D, device=self.device, dtype=self.dtype)
        for i in range(D):
            j = i - s
            if 0 <= j < D:
                M[i, j] = 1.0
        return M

    def conv(self, kernel: Sequence[float], center: int) -> torch.Tensor:
        D = self.D
        M = torch.zeros(D, D, device=self.device, dtype=self.dtype)
        for i in range(D):
            for k, w in enumerate(kernel):
                j = i + (k - center)
                if 0 <= j < D:
                    M[i, j] += float(w)
        return M

    def lowrank_fixed(self, seed: int, rank: int = 2) -> torch.Tensor:
        g = torch.Generator(device=self.device)
        g.manual_seed(seed)
        U = torch.randn(self.D, rank, generator=g, device=self.device, dtype=self.dtype)
        V = torch.randn(self.D, rank, generator=g, device=self.device, dtype=self.dtype)
        return safe_scale(U @ V.T / math.sqrt(max(1, self.D * rank)), 1.0)

    def _build_base(self) -> None:
        D = self.D
        I = torch.eye(D, device=self.device, dtype=self.dtype)
        Z = torch.zeros(D, D, device=self.device, dtype=self.dtype)
        self.add("Zero", "basic", Z, "zero/noop")
        self.add("Identity", "basic", I, "identity/carry")
        self.add("ShiftRight1", "sequence", self.shift(1), "previous token/delay")
        self.add("ShiftLeft1", "sequence", self.shift(-1), "next token/lookahead")
        self.add("ShiftRight2", "sequence", self.shift(2), "two-step delay")
        self.add("ShiftLeft2", "sequence", self.shift(-2), "two-step lookahead")
        self.add("DiffBackward", "derivative", I - self.shift(1), "x[i]-x[i-1]")
        self.add("DiffForward", "derivative", self.shift(-1) - I, "x[i+1]-x[i]")
        self.add("Laplacian1D", "derivative", self.shift(-1) - 2 * I + self.shift(1), "second derivative")
        blur3 = self.conv([0.25, 0.50, 0.25], 1)
        blur5 = self.conv([1, 2, 3, 2, 1], 2)
        blur5 = blur5 / blur5.sum(dim=1, keepdim=True).clamp_min(1e-6)
        self.add("Blur3", "smoothing", blur3, "3-tap smoothing")
        self.add("Blur5", "smoothing", blur5, "5-tap smoothing")

        ones = torch.ones(D, 1, device=self.device, dtype=self.dtype)
        self.add("ProjectConst", "projection", ones @ ones.T / D, "mean/global projection")
        ramp = torch.linspace(-1, 1, D, device=self.device, dtype=self.dtype).view(D, 1)
        ramp = ramp / torch.norm(ramp).clamp_min(1e-12)
        self.add("ProjectRamp", "projection", ramp @ ramp.T, "trend projection")
        self.add("Reverse", "permutation", torch.flip(I, dims=[1]), "reverse order")

        C = dct_matrix(D, self.device, self.dtype)
        low = max(1, D // 4)
        mid0, mid1 = max(1, D // 4), max(2, D // 2)
        high = max(1, D // 4)
        self.add("DCTLowPass", "spectral", C[:low].T @ C[:low], "DCT lowpass")
        if mid1 > mid0:
            self.add("DCTMidPass", "spectral", C[mid0:mid1].T @ C[mid0:mid1], "DCT midband")
        self.add("DCTHighPass", "spectral", C[-high:].T @ C[-high:], "DCT highpass")

        pref = torch.tril(torch.ones(D, D, device=self.device, dtype=self.dtype))
        pref = pref / pref.sum(dim=1, keepdim=True).clamp_min(1.0)
        self.add("PrefixMean", "sequence", pref, "causal prefix mean")
        dist = torch.arange(D, device=self.device, dtype=self.dtype)
        dd = torch.abs(dist[:, None] - dist[None, :])
        self.add("DistanceDecay4", "sequence", torch.exp(-dd / 4.0), "local decay")
        self.add("DistanceDecay8", "sequence", torch.exp(-dd / 8.0), "longer decay")

        for b in (2, 4, 8):
            if b <= D:
                B = torch.zeros(D, D, device=self.device, dtype=self.dtype)
                for i in range(0, D, b):
                    j1 = min(i + b, D)
                    B[i:j1, i:j1] = 1.0 / max(1, j1 - i)
                self.add(f"BlockAvg{b}", "block", B, f"block average {b}")

        diag_ramp = torch.linspace(0.25, 1.25, D, device=self.device, dtype=self.dtype)
        self.add("DiagRampGate", "gate", torch.diag(diag_ramp), "diagonal ramp gate")
        diag_center = torch.exp(-((torch.linspace(-1, 1, D, device=self.device, dtype=self.dtype) / 0.55) ** 2))
        self.add("DiagCenterGate", "gate", torch.diag(diag_center), "center gate")
        self.add("DiagEdgeGate", "gate", torch.diag(1.0 - diag_center), "edge gate")
        self.add("LowRankA", "lowrank", self.lowrank_fixed(11, 2), "fixed low-rank A")
        self.add("LowRankB", "lowrank", self.lowrank_fixed(17, 3), "fixed low-rank B")
        self.add("LowRankC", "lowrank", self.lowrank_fixed(23, 4), "fixed low-rank C")

        A = torch.zeros(D, D, device=self.device, dtype=self.dtype)
        for i in range(D):
            A[i, (i - 1) % D] = 1.0
            A[i, (i + 1) % D] = 1.0
        RW = A / A.sum(dim=1, keepdim=True).clamp_min(1.0)
        L = torch.diag(A.sum(dim=1)) - A
        self.add("GraphRingRW", "graph", RW, "ring random walk")
        self.add("GraphDiffusion", "graph", I - 0.08 * L + 0.5 * 0.08 * 0.08 * (L @ L), "small diffusion")

    def add_mined_from_target(self, W: torch.Tensor, max_svd: int = 4, add_toeplitz: bool = True, add_blocks: bool = True) -> None:
        """Add target-derived atoms. These are useful for residual mining / auto dictionary.

        They are marked origin='mined_from_target', so downstream code can tell
        if reconstruction used general atoms or target-specific mined atoms.
        """
        W = sanitize_tensor(W).to(self.device, self.dtype)
        W = resample_matrix(W, self.D).to(self.device, self.dtype)
        if torch.norm(W) < 1e-12:
            return
        diag = torch.diag(torch.diag(W))
        if torch.norm(diag) > 1e-8:
            self.add("MinedDiagTarget", "mined_diag", safe_scale(diag, 1.0), "target diagonal residual atom", "mined_from_target")

        row = W.mean(dim=1, keepdim=True)
        col = W.mean(dim=0, keepdim=True)
        ones = torch.ones(self.D, 1, device=self.device, dtype=self.dtype)
        rowM = row @ ones.T / self.D
        colM = ones @ col / self.D
        if torch.norm(rowM) > 1e-8:
            self.add("MinedRowMeanTarget", "mined_lowrank", safe_scale(rowM, 1.0), "target row-mean atom", "mined_from_target")
        if torch.norm(colM) > 1e-8:
            self.add("MinedColMeanTarget", "mined_lowrank", safe_scale(colM, 1.0), "target col-mean atom", "mined_from_target")

        try:
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            for i in range(min(max_svd, S.numel())):
                M = S[i] * torch.outer(U[:, i], Vh[i, :])
                if torch.norm(M) > 1e-8:
                    self.add(f"MinedSVD{i+1}", "mined_svd", safe_scale(M, 1.0), f"target SVD rank-1 atom {i+1}", "mined_from_target")
        except Exception:
            pass

        if add_toeplitz:
            T = torch.zeros_like(W)
            for d in range(-(self.D - 1), self.D):
                vals = torch.diagonal(W, offset=d)
                if vals.numel() > 0:
                    val = vals.mean()
                    if d >= 0:
                        i = torch.arange(0, self.D - d, device=self.device)
                        T[i, i + d] = val
                    else:
                        i = torch.arange(0, self.D + d, device=self.device)
                        T[i - d, i] = val
            if torch.norm(T) > 1e-8:
                self.add("MinedToeplitzTarget", "mined_toeplitz", safe_scale(T, 1.0), "average-diagonal Toeplitz atom", "mined_from_target")

        if add_blocks:
            for b in (2, 4, 8):
                if b <= self.D:
                    B = torch.zeros_like(W)
                    for i in range(0, self.D, b):
                        for j in range(0, self.D, b):
                            ii, jj = min(i + b, self.D), min(j + b, self.D)
                            B[i:ii, j:jj] = W[i:ii, j:jj].mean()
                    if torch.norm(B) > 1e-8:
                        self.add(f"MinedBlockMean{b}", "mined_block", safe_scale(B, 1.0), f"target block-mean {b}", "mined_from_target")

    def coeffs_to_matrix(self, coeffs: torch.Tensor) -> torch.Tensor:
        return torch.einsum("p,pij->ij", coeffs.to(self.device, self.dtype), self.M)

    def format_formula(self, coeffs: torch.Tensor, threshold: float = 1e-5, max_terms: int = 20, name: str = "W") -> str:
        c = coeffs.detach().cpu().float()
        idx = [i for i, v in enumerate(c.tolist()) if abs(v) >= threshold]
        idx = sorted(idx, key=lambda i: abs(float(c[i])), reverse=True)[:max_terms]
        if not idx:
            return f"{name} = 0"
        terms = []
        for i in idx:
            terms.append(f"{float(c[i]):+.4g}*{self.names[i]}")
        rhs = " ".join(terms).replace("+ -", "- ")
        if rhs.startswith("+"):
            rhs = rhs[1:].strip()
        return f"{name} = {rhs}"


# =============================================================================
# Structured synthetic from real code skeleton
# =============================================================================

READ_TO_OPS = {
    "state": ["Identity", "DiagRampGate", "DCTLowPass"],
    "input": ["DCTLowPass", "Blur3", "ShiftRight1", "ProjectRamp"],
    "prev_step": ["Identity", "ShiftRight1", "DistanceDecay4"],
    "all_prev_steps": ["PrefixMean", "DistanceDecay8", "ProjectConst"],
    "layer_route": ["BlockAvg2", "BlockAvg4", "LowRankB"],
    "global": ["ProjectConst", "LowRankA", "DCTLowPass"],
    "memory": ["LowRankA", "LowRankB", "BlockAvg4", "DistanceDecay8"],
    "task": ["ProjectRamp", "DiagRampGate", "LowRankC"],
    "prev_same_block": ["Identity", "ShiftRight1", "BlockAvg2"],
    "prev_layer_mean": ["BlockAvg4", "ProjectConst", "LowRankB"],
    "global_mean": ["ProjectConst", "LowRankA"],
    "memory_mean": ["LowRankB", "DistanceDecay8"],
}
PRIMITIVE_TO_OPS = {
    "noop": ["Zero"],
    "identity": ["Identity"],
    "keep_state": ["Identity", "DiagRampGate"],
    "small_refine": ["Blur3", "DiffBackward", "DCTLowPass"],
    "mlp": ["LowRankA", "LowRankB", "DCTMidPass", "Blur5"],
    "matrix_mlp": ["BlockAvg2", "BlockAvg4", "LowRankC", "DCTLowPass", "DCTHighPass"],
    "compare": ["DiffBackward", "DiffForward", "Laplacian1D"],
    "memory_read": ["LowRankA", "LowRankB", "DistanceDecay8"],
    "global_read": ["ProjectConst", "LowRankA", "DCTLowPass"],
    "normalize": ["DiagCenterGate", "DCTLowPass", "ProjectRamp"],
    "suppress": ["DiagEdgeGate", "DiagRampGate", "DCTHighPass"],
    "attention": ["ProjectConst", "DistanceDecay8", "LowRankB"],
    "conv": ["Blur3", "Blur5", "DiffBackward"],
    "gate": ["DiagRampGate", "DiagCenterGate", "DiagEdgeGate"],
    "channel_butterfly": ["Identity", "ShiftRight1", "ShiftLeft1", "DCTLowPass", "DCTMidPass"],
    "block_butterfly": ["BlockAvg2", "BlockAvg4", "GraphRingRW", "GraphDiffusion"],
    "low_rank": ["LowRankA", "LowRankB", "LowRankC", "ProjectConst"],
    "ctx_matrix": ["ProjectConst", "ProjectRamp", "DCTLowPass", "LowRankA"],
    "product_gate": ["DiagRampGate", "DiagCenterGate", "DiagEdgeGate"],
    "phase_matrix": ["DCTLowPass", "DCTMidPass", "DCTHighPass", "DistanceDecay4", "DistanceDecay8"],
}
TRANSITION_TO_OPS = {
    "keep": ["Identity"],
    "replace": ["Identity", "LowRankA"],
    "residual": ["Identity", "Blur3", "DiffBackward"],
    "norm_residual": ["DCTLowPass", "DiagCenterGate"],
    "product_gate": ["DiagRampGate", "DiagCenterGate", "DiagEdgeGate"],
    "compare_mix": ["DiffBackward", "Laplacian1D", "LowRankC"],
}
WRITE_TO_OPS = {
    "state": ["Identity", "DiagRampGate"],
    "global": ["ProjectConst", "LowRankA"],
    "memory": ["LowRankB", "BlockAvg4"],
    "class": ["ProjectRamp", "LowRankC", "ProjectConst"],
}


def prefix_products(step_mats: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    if not step_mats:
        return []
    D = step_mats[0].shape[0]
    W = torch.eye(D, device=step_mats[0].device, dtype=step_mats[0].dtype)
    out: List[torch.Tensor] = []
    for S in step_mats:
        W = S @ W
        out.append(W)
    return out


def choose_ops(names: Sequence[str], lib: MatrixOpLibrary, rng: random.Random, strength: float = 1.0) -> Tuple[List[str], torch.Tensor, List[float]]:
    valid = [n for n in names if n in lib.name_to_idx]
    if not valid:
        valid = ["Identity"]
    k = rng.randint(1, min(3, len(valid)))
    picked = rng.sample(valid, k=k)
    raw = torch.tensor([rng.random() + 0.05 for _ in picked], device=lib.device, dtype=lib.dtype)
    weights = raw / raw.sum().clamp_min(1e-12)
    M = torch.zeros(lib.D, lib.D, device=lib.device, dtype=lib.dtype)
    for w, n in zip(weights, picked):
        M = M + w * lib.M[lib.name_to_idx[n]]
    return picked, strength * safe_scale(M, 1.0), [float(x) for x in weights.detach().cpu()]


class StructuredSynthesizer:
    def __init__(
        self,
        skeleton: CodeSkeleton,
        lib: MatrixOpLibrary,
        seed: int = 0,
        step_scale: float = 0.23,
        route_scale: float = 0.10,
        max_program_steps: Optional[int] = None,
    ):
        self.sk = skeleton
        self.lib = lib
        self.rng = random.Random(seed)
        self.step_scale = step_scale
        self.route_scale = route_scale
        self.max_program_steps = max_program_steps

    def ops_for_primitive(self, primitive: str) -> List[str]:
        if primitive in PRIMITIVE_TO_OPS:
            return PRIMITIVE_TO_OPS[primitive]
        p = primitive.lower()
        if "channel" in p and "butterfly" in p:
            return PRIMITIVE_TO_OPS["channel_butterfly"]
        if "block" in p and "butterfly" in p:
            return PRIMITIVE_TO_OPS["block_butterfly"]
        if "butterfly" in p:
            return ["ShiftRight1", "ShiftLeft1", "BlockAvg2", "DCTMidPass"]
        if "low_rank" in p or "lowrank" in p:
            return PRIMITIVE_TO_OPS["low_rank"]
        if "ctx" in p or "context" in p:
            return PRIMITIVE_TO_OPS["ctx_matrix"]
        if "product" in p:
            return PRIMITIVE_TO_OPS["product_gate"]
        if "phase" in p:
            return PRIMITIVE_TO_OPS["phase_matrix"]
        if "attn" in p or "attention" in p:
            return PRIMITIVE_TO_OPS["attention"]
        if "conv" in p:
            return PRIMITIVE_TO_OPS["conv"]
        if "gate" in p:
            return PRIMITIVE_TO_OPS["gate"]
        if "norm" in p:
            return PRIMITIVE_TO_OPS["normalize"]
        if "mlp" in p or "linear" in p or "proj" in p:
            return PRIMITIVE_TO_OPS["mlp"]
        return ["Identity", "LowRankA"]

    def sample_one(self, sample_id: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        D = self.lib.D
        I = torch.eye(D, device=self.lib.device, dtype=self.lib.dtype)
        step_mats: List[torch.Tensor] = []
        interactions: List[Dict[str, Any]] = []
        steps_json: List[Dict[str, Any]] = []
        used_ops: List[str] = []

        L, N, S, K = max(1, self.sk.L), max(1, self.sk.N), max(1, self.sk.S), max(1, self.sk.K)
        b = self.rng.randrange(N)
        logical_count = 0

        for l in range(L):
            if l > 0:
                route = self.rng.choice(self.sk.route_names or DEFAULT_LAYER_ROUTE_NAMES)
                rops, R, rw = choose_ops(READ_TO_OPS.get(route, ["BlockAvg2", "LowRankB"]), self.lib, self.rng, self.route_scale)
                step_mats.append(I + R)
                used_ops.extend(rops)
                interactions.append({
                    "kind": "layer_route",
                    "address": f"L{l}.B{b}.route",
                    "route": route,
                    "ops": rops,
                    "weights": rw,
                    "matrix_step": len(step_mats) - 1,
                })

            for s in range(S):
                if self.max_program_steps is not None and logical_count >= self.max_program_steps:
                    break
                addr = f"L{l}.B{b}.S{s}"
                read = self.rng.choice(self.sk.read_names or DEFAULT_STEP_READ_NAMES)
                read_ops, R_read, read_w = choose_ops(READ_TO_OPS.get(read, ["Identity"]), self.lib, self.rng, 1.0)

                P = R_read
                prim_names, prim_ops_all, trans_names, trans_ops_all = [], [], [], []
                for k in range(K):
                    prim = self.rng.choice(self.sk.primitive_names or DEFAULT_PRIMITIVE_NAMES)
                    prim_names.append(prim)
                    pop_names, Pop, pop_w = choose_ops(self.ops_for_primitive(prim), self.lib, self.rng, 1.0)
                    prim_ops_all.extend(pop_names)

                    if prim in ("noop", "identity", "keep_state"):
                        U = P
                    else:
                        U = Pop @ P

                    interactions.append({
                        "kind": "primitive",
                        "address": f"{addr}.P{k}",
                        "primitive": prim,
                        "ops": pop_names,
                        "weights": pop_w,
                    })

                    if k + 1 < K:
                        tr = self.rng.choice(self.sk.transition_names or DEFAULT_PRIM_TRANSITION_NAMES)
                        trans_names.append(tr)
                        top_names, Top, top_w = choose_ops(TRANSITION_TO_OPS.get(tr, ["Identity"]), self.lib, self.rng, 1.0)
                        trans_ops_all.extend(top_names)
                        if tr == "keep":
                            P = U
                        elif tr == "replace":
                            P = Top @ U
                        elif tr == "residual":
                            P = U + 0.35 * (Top @ U)
                        elif tr == "norm_residual":
                            P = safe_scale(U + 0.25 * (Top @ U), 1.25)
                        elif tr == "product_gate":
                            P = Top @ U
                        else:
                            P = 0.65 * U + 0.35 * (Top @ U)
                        interactions.append({
                            "kind": "primitive_transition",
                            "address": f"{addr}.P{k}->P{k+1}",
                            "transition": tr,
                            "ops": top_names,
                            "weights": top_w,
                        })
                    else:
                        P = U

                write_target = self.rng.choices(["state", "memory", "global"], weights=[0.70, 0.18, 0.12], k=1)[0]
                write_gate = self.rng.uniform(0.10, 0.85)
                wops, Wwrite, ww = choose_ops(WRITE_TO_OPS.get(write_target, ["Identity"]), self.lib, self.rng, 1.0)
                core = safe_scale(Wwrite @ P, 1.0)
                Smat = I + (self.step_scale * write_gate) * core
                Smat = safe_scale(Smat, 2.0)
                step_idx = len(step_mats)
                step_mats.append(Smat)

                used_ops.extend(read_ops + prim_ops_all + trans_ops_all + wops)
                interactions.append({
                    "kind": "step_read",
                    "address": f"{addr}.read",
                    "read": read,
                    "ops": read_ops,
                    "weights": read_w,
                })
                interactions.append({
                    "kind": "write",
                    "address": f"{addr}.{write_target}_write",
                    "target": write_target,
                    "write_gate": write_gate,
                    "ops": wops,
                    "weights": ww,
                    "matrix_step": step_idx,
                })
                steps_json.append({
                    "address": addr,
                    "layer": l,
                    "block": b,
                    "step": s,
                    "read": read,
                    "read_ops": read_ops,
                    "primitive_names": prim_names,
                    "primitive_ops": prim_ops_all,
                    "transition_names": trans_names,
                    "transition_ops": trans_ops_all,
                    "write_target": write_target,
                    "write_gate": write_gate,
                    "matrix_step": step_idx,
                })
                logical_count += 1

            if self.rng.random() < 0.35 and N > 1:
                b = self.rng.randrange(N)
            if self.max_program_steps is not None and logical_count >= self.max_program_steps:
                break

        prefixes = prefix_products(step_mats)
        Wfinal = prefixes[-1] if prefixes else I
        Wrev = prefix_products(list(reversed(step_mats)))[-1] if step_mats else I
        used_unique = sorted(set([u for u in used_ops if u in self.lib.name_to_idx]))
        used_vec = torch.zeros(len(self.lib.names), device=self.lib.device)
        for u in used_unique:
            used_vec[self.lib.name_to_idx[u]] = 1.0

        read_hist = torch.zeros(len(self.sk.read_names), device=self.lib.device)
        prim_hist = torch.zeros(len(self.sk.primitive_names), device=self.lib.device)
        trans_hist = torch.zeros(len(self.sk.transition_names), device=self.lib.device)
        primitive_transition_pair_hist = torch.zeros(
            len(self.sk.primitive_names),
            len(self.sk.transition_names),
            len(self.sk.primitive_names),
            device=self.lib.device,
        )
        rmap = {n: i for i, n in enumerate(self.sk.read_names)}
        pmap = {n: i for i, n in enumerate(self.sk.primitive_names)}
        tmap = {n: i for i, n in enumerate(self.sk.transition_names)}
        for st in steps_json:
            if st["read"] in rmap:
                read_hist[rmap[st["read"]]] += 1
            for p in st["primitive_names"]:
                if p in pmap:
                    prim_hist[pmap[p]] += 1
            for t in st["transition_names"]:
                if t in tmap:
                    trans_hist[tmap[t]] += 1
            prims = st["primitive_names"]
            trans = st["transition_names"]
            for k, t in enumerate(trans):
                if k + 1 < len(prims) and prims[k] in pmap and t in tmap and prims[k + 1] in pmap:
                    primitive_transition_pair_hist[pmap[prims[k]], tmap[t], pmap[prims[k + 1]]] += 1
        if read_hist.sum() > 0: read_hist /= read_hist.sum()
        if prim_hist.sum() > 0: prim_hist /= prim_hist.sum()
        if trans_hist.sum() > 0: trans_hist /= trans_hist.sum()
        if primitive_transition_pair_hist.sum() > 0:
            primitive_transition_pair_hist /= primitive_transition_pair_hist.sum()

        first_op = steps_json[0]["read_ops"][0] if steps_json and steps_json[0]["read_ops"] else "Identity"
        last_ops = []
        if steps_json:
            last_ops = steps_json[-1]["transition_ops"] + steps_json[-1]["primitive_ops"] + steps_json[-1]["read_ops"]
        last_op = last_ops[-1] if last_ops else "Identity"

        formula_lines = ["W = " + " @ ".join([f"S{i}" for i in reversed(range(len(step_mats)))])]
        for st in steps_json[:20]:
            formula_lines.append(
                f"S{st['matrix_step']} {st['address']}: read={st['read']} "
                f"prims={st['primitive_names']} trans={st['transition_names']} "
                f"write={st['write_target']} gate={st['write_gate']:.3f}"
            )
        if len(steps_json) > 20:
            formula_lines.append(f"... {len(steps_json)-20} more logical steps")

        record = {
            "sample_id": f"synth_{sample_id:08d}",
            "truth_level": "real_structure_synthetic_matrix",
            "skeleton_kind": self.sk.skeleton_kind,
            "used_operator_names": used_unique,
            "final_rel_to_identity": rel_err(Wfinal, I),
            "order_reverse_rel": rel_err(Wfinal, Wrev),
            "steps": steps_json,
            "interactions": interactions,
            "formula": "\n".join(formula_lines),
        }
        data = {
            "W": Wfinal.detach().float().cpu(),
            "used_ops": used_vec.detach().float().cpu(),
            "first_op": torch.tensor(self.lib.name_to_idx.get(first_op, self.lib.name_to_idx["Identity"]), dtype=torch.long),
            "last_op": torch.tensor(self.lib.name_to_idx.get(last_op, self.lib.name_to_idx["Identity"]), dtype=torch.long),
            "read_hist": read_hist.detach().float().cpu(),
            "primitive_hist": prim_hist.detach().float().cpu(),
            "transition_hist": trans_hist.detach().float().cpu(),
            "primitive_transition_pair_hist": primitive_transition_pair_hist.detach().float().cpu(),
            "num_steps": torch.tensor(len(step_mats), dtype=torch.long),
            "step_mats_list": [x.detach().float().cpu() for x in step_mats],
            "prefix_mats_list": [x.detach().float().cpu() for x in prefixes],
        }
        return data, record


def pad_stack_matrix_lists(items: List[List[torch.Tensor]], D: int) -> Tuple[torch.Tensor, torch.Tensor]:
    maxlen = max((len(x) for x in items), default=1)
    B = len(items)
    mats = torch.zeros(B, maxlen, D, D)
    mask = torch.zeros(B, maxlen)
    I = torch.eye(D)
    for i, lst in enumerate(items):
        for j in range(maxlen):
            if j < len(lst):
                mats[i, j] = lst[j]
                mask[i, j] = 1.0
            else:
                mats[i, j] = I
    return mats, mask


def run_build_synth(args: argparse.Namespace, sk: Optional[CodeSkeleton] = None) -> Dict[str, Any]:
    out = ensure_dir(Path(args.out) / "synthetic")
    if sk is None:
        sk, _ = parse_code_files(resolve_parse_files(args.parse_files, args.parse_dirs, args.max_parse_files))
    if args.layers is not None: sk.L = args.layers
    if args.blocks is not None: sk.N = args.blocks
    if args.steps is not None: sk.S = args.steps
    if args.primitive_slots is not None: sk.K = args.primitive_slots

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; using CPU", flush=True)
        device = "cpu"

    lib = MatrixOpLibrary(args.D, device=device)
    synth = StructuredSynthesizer(sk, lib, seed=args.seed, step_scale=args.step_scale, route_scale=args.route_scale, max_program_steps=args.max_program_steps)

    rows, records = [], []
    t0 = time.time()
    for i in range(args.n):
        row, rec = synth.sample_one(i)
        rows.append(row)
        records.append(rec)
        if args.log_every and (i + 1) % args.log_every == 0:
            print(f"[build-synth] {i+1}/{args.n} t={time.time()-t0:.1f}s", flush=True)

    write_jsonl(out / "programs.jsonl", records)
    W = torch.stack([r["W"] for r in rows])
    used_ops = torch.stack([r["used_ops"] for r in rows])
    first_op = torch.stack([r["first_op"] for r in rows])
    last_op = torch.stack([r["last_op"] for r in rows])
    read_hist = torch.stack([r["read_hist"] for r in rows])
    primitive_hist = torch.stack([r["primitive_hist"] for r in rows])
    transition_hist = torch.stack([r["transition_hist"] for r in rows])
    primitive_transition_pair_hist = torch.stack([r["primitive_transition_pair_hist"] for r in rows])
    num_steps = torch.stack([r["num_steps"] for r in rows])
    step_mats, step_mask = pad_stack_matrix_lists([r["step_mats_list"] for r in rows], args.D)
    prefix_mats, prefix_mask = pad_stack_matrix_lists([r["prefix_mats_list"] for r in rows], args.D)

    dataset = {
        "W": W,
        "used_ops": used_ops,
        "first_op": first_op,
        "last_op": last_op,
        "read_hist": read_hist,
        "primitive_hist": primitive_hist,
        "transition_hist": transition_hist,
        "primitive_transition_pair_hist": primitive_transition_pair_hist,
        "num_steps": num_steps,
        "step_mats": step_mats,
        "step_mask": step_mask,
        "prefix_mats": prefix_mats,
        "prefix_mask": prefix_mask,
        "meta": {
            "truth_level": "real_structure_synthetic_matrix",
            "D": args.D,
            "n": args.n,
            "op_names": lib.names,
            "op_families": lib.families,
            "read_names": sk.read_names,
            "primitive_names": sk.primitive_names,
            "transition_names": sk.transition_names,
            "primitive_transition_pair_shape": list(primitive_transition_pair_hist.shape[1:]),
            "skeleton": asdict(sk),
        },
    }
    torch.save(dataset, out / "dataset.pt")
    write_json(out / "build_summary.json", {
        "W_shape": list(W.shape),
        "step_mats_shape": list(step_mats.shape),
        "n": args.n,
        "D": args.D,
        "elapsed_sec": time.time() - t0,
        "skeleton_kind": sk.skeleton_kind,
        "L": sk.L, "N": sk.N, "S": sk.S, "K": sk.K,
    })
    print(f"[build-synth] saved {out/'dataset.pt'} W={tuple(W.shape)}", flush=True)
    return dataset


# =============================================================================
# Real weight harvester + matrix program decoder
# =============================================================================

@dataclass
class MatrixItem:
    name: str
    tensor: torch.Tensor
    source: str
    kind: str
    original_shape: List[int]


def unwrap_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "module"):
            if key in obj and isinstance(obj[key], dict):
                inner = obj[key]
                if any(torch.is_tensor(v) for v in inner.values()):
                    return {str(k): v for k, v in inner.items() if torch.is_tensor(v)}
        if any(torch.is_tensor(v) for v in obj.values()):
            return {str(k): v for k, v in obj.items() if torch.is_tensor(v)}
    return {}


def load_torch_checkpoint(path: Path) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    return unwrap_state_dict(obj)


def load_safetensors_checkpoint(path: Path) -> Dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load_file
    except Exception as e:
        print(f"[WARN] safetensors not installed, cannot read {path}: {e}", flush=True)
        return {}
    return load_file(str(path), device="cpu")


def resolve_checkpoint_files(checkpoints: Sequence[str], checkpoint_dirs: Sequence[str], max_files: int) -> List[str]:
    files: List[str] = []
    exts = {".pt", ".pth", ".bin", ".ckpt", ".safetensors"}
    for c in checkpoints or []:
        files.append(str(c))
    for d in checkpoint_dirs or []:
        root = Path(d)
        if root.is_file() and root.suffix in exts:
            files.append(str(root))
            continue
        if not root.exists():
            print(f"[WARN] checkpoint dir does not exist: {root}", flush=True)
            continue
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix in exts:
                files.append(str(p))
    seen, out = set(), []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    if max_files and max_files > 0:
        out = out[:max_files]
    return out


def tensor_to_matrix_items(name: str, t: torch.Tensor, source: str, min_dim: int, max_elements: int) -> List[MatrixItem]:
    t = sanitize_tensor(t)
    if t.numel() == 0 or t.numel() > max_elements:
        return []
    items: List[MatrixItem] = []
    if t.ndim == 2 and min(t.shape) >= min_dim:
        items.append(MatrixItem(name=name, tensor=t, source=source, kind="linear_weight_2d", original_shape=list(t.shape)))
    elif t.ndim == 4:
        # Conv weight [out, in, kh, kw] -> matrix [out, in*kh*kw]
        W = t.reshape(t.shape[0], -1)
        if min(W.shape) >= min_dim:
            items.append(MatrixItem(name=name, tensor=W, source=source, kind="conv2d_im2col_flat", original_shape=list(t.shape)))
    elif t.ndim == 3:
        # Common formats: [3,out,in] qkv packed or [heads,out,in]
        if t.shape[0] <= 64 and min(t.shape[1:]) >= min_dim:
            for i in range(min(t.shape[0], 16)):
                items.append(MatrixItem(name=f"{name}.slice{i}", tensor=t[i], source=source, kind="rank3_slice_2d", original_shape=list(t.shape)))
        else:
            W = t.reshape(t.shape[0], -1)
            if min(W.shape) >= min_dim:
                items.append(MatrixItem(name=name, tensor=W, source=source, kind="rank3_flat", original_shape=list(t.shape)))
    return items


def harvest_real_matrices(args: argparse.Namespace) -> List[MatrixItem]:
    files = resolve_checkpoint_files(args.checkpoint, args.checkpoint_dir, args.max_checkpoint_files)
    items: List[MatrixItem] = []
    name_re = re.compile(args.name_regex) if args.name_regex else None
    for fp in files:
        path = Path(fp)
        if not path.exists():
            print(f"[WARN] checkpoint missing: {path}", flush=True)
            continue
        try:
            if path.suffix == ".safetensors":
                sd = load_safetensors_checkpoint(path)
            else:
                sd = load_torch_checkpoint(path)
        except Exception as e:
            print(f"[WARN] failed loading {path}: {e}", flush=True)
            continue
        print(f"[harvest] {path} tensors={len(sd)}", flush=True)
        for name, t in sd.items():
            if name_re and not name_re.search(name):
                continue
            if any(fnmatch.fnmatch(name, pat) for pat in args.exclude_name_glob):
                continue
            found = tensor_to_matrix_items(name, t, str(path), args.min_matrix_dim, args.max_tensor_elements)
            items.extend(found)
            if args.max_matrices and len(items) >= args.max_matrices:
                return items[:args.max_matrices]
    return items


class SparseProgramDecoder:
    def __init__(self, lib: MatrixOpLibrary, ridge: float = 1e-5):
        self.lib = lib
        A = lib.M.reshape(len(lib.names), -1).T.contiguous().float()
        self.raw_A = A
        self.col_norm = torch.norm(A, dim=0).clamp_min(1e-12)
        self.A = A / self.col_norm.view(1, -1)
        self.AtA = self.A.T @ self.A
        self.ridge = ridge

    def decode_omp(self, W: torch.Tensor, max_atoms: int = 12, stop_rel: float = 1e-4) -> Tuple[torch.Tensor, Dict[str, Any]]:
        b = W.reshape(-1).to(self.A.device).float()
        b_norm = torch.norm(b).clamp_min(1e-12)
        residual = b.clone()
        selected: List[int] = []
        coeff_norm = torch.zeros(self.A.shape[1], device=self.A.device)
        for _ in range(min(max_atoms, self.A.shape[1])):
            corr = self.A.T @ residual
            if selected:
                corr[torch.tensor(selected, device=self.A.device)] = 0.0
            j = int(torch.argmax(corr.abs()).item())
            selected.append(j)
            idx = torch.tensor(selected, device=self.A.device)
            A2 = self.A[:, idx]
            eye = torch.eye(len(selected), device=self.A.device)
            coef = torch.linalg.solve(A2.T @ A2 + self.ridge * eye, A2.T @ b)
            residual = b - A2 @ coef
            if float(torch.norm(residual) / b_norm) <= stop_rel:
                break
        if selected:
            coeff_norm[torch.tensor(selected, device=self.A.device)] = coef
        coeff_raw = coeff_norm / self.col_norm
        Wh = self.lib.coeffs_to_matrix(coeff_raw)
        selected_names = [self.lib.names[i] for i in selected]
        used_mined = [self.lib.ops[i].origin == "mined_from_target" for i in selected]
        metrics = {
            "rec_err": rel_err(W.to(Wh.device), Wh),
            "active": len(selected),
            "selected_names": selected_names,
            "selected_origins": [self.lib.ops[i].origin for i in selected],
            "selected_families": [self.lib.ops[i].family for i in selected],
            "uses_mined_atoms": bool(any(used_mined)),
        }
        return coeff_raw.detach(), metrics


def guess_matrix_role(name: str) -> str:
    n = name.lower()
    if any(x in n for x in ["q_proj", ".q.", "query"]): return "attention_q"
    if any(x in n for x in ["k_proj", ".k.", "key"]): return "attention_k"
    if any(x in n for x in ["v_proj", ".v.", "value"]): return "attention_v"
    if any(x in n for x in ["o_proj", "out_proj", ".o."]): return "attention_o"
    if any(x in n for x in ["gate_proj", "up_proj", "down_proj", "mlp", "ffn"]): return "mlp"
    if "conv" in n: return "conv"
    if "embed" in n: return "embedding"
    if "lm_head" in n or "classifier" in n: return "head"
    if "norm" in n: return "norm"
    return "linear"


def run_decode_real(args: argparse.Namespace) -> Dict[str, Any]:
    out = ensure_dir(Path(args.out) / "real_decode")
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; using CPU", flush=True)
        device = "cpu"
    items = harvest_real_matrices(args)
    print(f"[decode-real] harvested matrices={len(items)}", flush=True)

    records: List[Dict[str, Any]] = []
    tensors: Dict[str, Any] = {
        "W": [],
        "W_hat": [],
        "coeffs": [],
        "names": [],
        "truth_level": [],
        "original_shapes": [],
        "rec_err": [],
        "functional_err": [],
    }
    t0 = time.time()
    for i, item in enumerate(items):
        Wd = resample_matrix(item.tensor, args.D).to(device)
        if torch.norm(Wd) < 1e-12:
            continue
        # normalize target for pattern decode, but store norm.
        target_norm = float(torch.norm(Wd).item())
        Wn = Wd / torch.norm(Wd).clamp_min(1e-12)

        lib = MatrixOpLibrary(args.D, device=device)
        if args.auto_mined_atoms:
            lib.add_mined_from_target(Wn, max_svd=args.mined_svd_atoms, add_toeplitz=True, add_blocks=True)
        dec = SparseProgramDecoder(lib, ridge=args.decode_ridge)
        coeff, met = dec.decode_omp(Wn, max_atoms=args.decode_topk, stop_rel=args.decode_stop_rel)
        Wh = lib.coeffs_to_matrix(coeff)
        ferr = functional_error_square(Wn, Wh, batch=args.functional_batch, mode="gaussian")

        formula = lib.format_formula(coeff, threshold=args.formula_threshold, max_terms=args.formula_max_terms, name="W_real")
        selected = []
        for j, val in enumerate(coeff.detach().cpu().float().tolist()):
            if abs(val) >= args.formula_threshold:
                op = lib.ops[j]
                selected.append({
                    "name": op.name,
                    "coef": float(val),
                    "family": op.family,
                    "origin": op.origin,
                    "desc": op.desc,
                })
        selected = sorted(selected, key=lambda x: abs(x["coef"]), reverse=True)[:args.formula_max_terms]

        record = {
            "sample_id": f"real_{i:08d}",
            "truth_level": "real_weight_program_decode",
            "source": item.source,
            "name": item.name,
            "kind": item.kind,
            "role_guess": guess_matrix_role(item.name),
            "original_shape": item.original_shape,
            "decoded_shape": [args.D, args.D],
            "target_norm_before_normalize": target_norm,
            "dictionary_size": len(lib.names),
            "dictionary_families": lib.families,
            "auto_mined_atoms": bool(args.auto_mined_atoms),
            "metrics": {
                "rec_err": met["rec_err"],
                "functional_err_gaussian": ferr,
                "active": met["active"],
                "uses_mined_atoms": met["uses_mined_atoms"],
            },
            "selected_terms": selected,
            "formula": formula,
            "honesty_note": (
                "For rectangular/conv matrices W was resampled to D x D; this is a pattern/program decode, "
                "not an exact replacement of original tensor shape."
            ) if item.original_shape != [args.D, args.D] else "Square D x D matrix decode.",
        }
        records.append(record)
        tensors["W"].append(Wn.detach().cpu())
        tensors["W_hat"].append(Wh.detach().cpu())
        tensors["coeffs"].append(coeff.detach().cpu())
        tensors["names"].append(item.name)
        tensors["truth_level"].append("real_weight_program_decode")
        tensors["original_shapes"].append(item.original_shape)
        tensors["rec_err"].append(float(met["rec_err"]))
        tensors["functional_err"].append(float(ferr))

        if args.log_every and (i + 1) % args.log_every == 0:
            print(f"[decode-real] {i+1}/{len(items)} rec_err={met['rec_err']:.4f} f_err={ferr:.4f}", flush=True)

    write_jsonl(out / "real_matrix_decodes.jsonl", records)
    if tensors["W"]:
        # coeffs may have same length because mined atoms count stable? Base + same mined counts, yes if same args.
        try:
            W = torch.stack(tensors["W"])
            Wh = torch.stack(tensors["W_hat"])
            C = torch.stack(tensors["coeffs"])
        except Exception:
            W, Wh, C = [], [], []
        torch.save({
            "W": W,
            "W_hat": Wh,
            "coeffs": C,
            "names": tensors["names"],
            "truth_level": tensors["truth_level"],
            "original_shapes": tensors["original_shapes"],
            "rec_err": torch.tensor(tensors["rec_err"]),
            "functional_err": torch.tensor(tensors["functional_err"]),
            "meta": {
                "D": args.D,
                "n": len(records),
                "description": "real model/checkpoint matrices decoded into readable operator programs",
                "auto_mined_atoms": bool(args.auto_mined_atoms),
                "decode_topk": args.decode_topk,
            },
        }, out / "real_matrix_dataset.pt")
    write_json(out / "decode_summary.json", {
        "harvested": len(items),
        "decoded": len(records),
        "D": args.D,
        "elapsed_sec": time.time() - t0,
        "mean_rec_err": sum(tensors["rec_err"]) / max(1, len(tensors["rec_err"])),
        "mean_functional_err": sum(tensors["functional_err"]) / max(1, len(tensors["functional_err"])),
    })
    print(f"[decode-real] saved {len(records)} decodes -> {out}", flush=True)
    return {"records": records}


# =============================================================================
# Baseline trainer for synthetic dataset
# =============================================================================

class BaselineDecoder(nn.Module):
    def __init__(self, D: int, n_ops: int, n_read: int, n_prim: int, n_trans: int, n_pair: int = 0, hidden: int = 384):
        super().__init__()
        self.n_pair = int(n_pair)
        inp = D * D
        self.backbone = nn.Sequential(
            nn.LayerNorm(inp),
            nn.Linear(inp, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.used = nn.Linear(hidden, n_ops)
        self.first = nn.Linear(hidden, n_ops)
        self.last = nn.Linear(hidden, n_ops)
        self.read = nn.Linear(hidden, n_read)
        self.prim = nn.Linear(hidden, n_prim)
        self.trans = nn.Linear(hidden, n_trans)
        self.pair = nn.Linear(hidden, n_pair) if n_pair > 0 else None

    def forward(self, W: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.backbone(W.flatten(1))
        out = {
            "used": self.used(z),
            "first": self.first(z),
            "last": self.last(z),
            "read": self.read(z),
            "prim": self.prim(z),
            "trans": self.trans(z),
        }
        if self.pair is not None:
            out["pair"] = self.pair(z)
        else:
            out["pair"] = W.new_empty(W.shape[0], 0)
        return out


@torch.no_grad()
def multilabel_f1(logits: torch.Tensor, target: torch.Tensor, th: float = 0.5) -> Dict[str, float]:
    pred = torch.sigmoid(logits) >= th
    tgt = target >= 0.5
    tp = (pred & tgt).sum(dim=1).float()
    fp = (pred & (~tgt)).sum(dim=1).float()
    fn = ((~pred) & tgt).sum(dim=1).float()
    f1 = (2 * tp / (2 * tp + fp + fn + 1e-9)).mean().item()
    exact = (pred == tgt).all(dim=1).float().mean().item()
    return {"f1": float(f1), "exact": float(exact)}


@torch.no_grad()
def hist_kl(logits: torch.Tensor, target: torch.Tensor) -> float:
    if logits.numel() == 0 or target.numel() == 0 or logits.shape[1] == 0:
        return 0.0
    q = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    return float(F.kl_div(F.log_softmax(logits, dim=-1), q, reduction="batchmean").item())


def batch_ids(ids: torch.Tensor, bs: int, shuffle: bool = True):
    if shuffle:
        ids = ids[torch.randperm(len(ids))]
    for i in range(0, len(ids), bs):
        yield ids[i:i+bs]


def corrupt_matrix_batch(W: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    """Denoising objective: hide parts of W while keeping full program targets."""

    mask_frac = float(getattr(args, "matrix_mask_frac", 0.0) or 0.0)
    rowcol_frac = float(getattr(args, "matrix_rowcol_mask_frac", 0.0) or 0.0)
    noise_std = float(getattr(args, "matrix_noise_std", 0.0) or 0.0)
    if mask_frac <= 0 and rowcol_frac <= 0 and noise_std <= 0:
        return W

    X = W.clone()
    if mask_frac > 0:
        keep = (torch.rand_like(X) >= mask_frac).to(X.dtype)
        X = X * keep
    if rowcol_frac > 0:
        B, D, _ = X.shape
        row_keep = (torch.rand(B, D, 1, device=X.device, dtype=X.dtype) >= rowcol_frac).to(X.dtype)
        col_keep = (torch.rand(B, 1, D, device=X.device, dtype=X.dtype) >= rowcol_frac).to(X.dtype)
        X = X * row_keep * col_keep
    if noise_std > 0:
        scale = W.flatten(1).std(dim=1, keepdim=True).view(-1, 1, 1).clamp_min(1e-6)
        X = X + noise_std * scale * torch.randn_like(X)
    return X


def run_train_synth(args: argparse.Namespace) -> Dict[str, Any]:
    out = ensure_dir(Path(args.out) / "synthetic")
    ds_path = Path(args.dataset or (out / "dataset.pt"))
    data = torch.load(ds_path, map_location="cpu")
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    W = data["W"].float()
    pair_target = data.get("primitive_transition_pair_hist")
    pair_flat = pair_target.flatten(1).float() if pair_target is not None else None
    N = W.shape[0]
    idx = torch.randperm(N)
    n_train = max(1, int(N * args.train_frac))
    tr, va = idx[:n_train], idx[n_train:]
    if len(va) == 0:
        va = tr

    model = BaselineDecoder(
        D=int(data["meta"]["D"]),
        n_ops=len(data["meta"]["op_names"]),
        n_read=len(data["meta"]["read_names"]),
        n_prim=len(data["meta"]["primitive_names"]),
        n_trans=len(data["meta"]["transition_names"]),
        n_pair=0 if pair_flat is None else pair_flat.shape[1],
        hidden=args.hidden,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for bi in batch_ids(tr, args.batch_size, True):
            bW = W[bi].to(device)
            outp = model(corrupt_matrix_batch(bW, args))
            used = data["used_ops"][bi].float().to(device)
            first = data["first_op"][bi].long().to(device)
            last = data["last_op"][bi].long().to(device)
            rh = data["read_hist"][bi].float().to(device)
            ph = data["primitive_hist"][bi].float().to(device)
            th = data["transition_hist"][bi].float().to(device)
            pair_h = pair_flat[bi].float().to(device) if pair_flat is not None else None

            loss = F.binary_cross_entropy_with_logits(outp["used"], used)
            loss = loss + 0.35 * F.cross_entropy(outp["first"], first)
            loss = loss + 0.35 * F.cross_entropy(outp["last"], last)
            if rh.shape[1] > 0:
                loss = loss + 0.20 * F.kl_div(F.log_softmax(outp["read"], dim=-1), rh, reduction="batchmean")
            if ph.shape[1] > 0:
                loss = loss + 0.20 * F.kl_div(F.log_softmax(outp["prim"], dim=-1), ph, reduction="batchmean")
            if th.shape[1] > 0:
                loss = loss + 0.15 * F.kl_div(F.log_softmax(outp["trans"], dim=-1), th, reduction="batchmean")
            if pair_h is not None and pair_h.shape[1] > 0:
                loss = loss + 0.20 * F.kl_div(F.log_softmax(outp["pair"], dim=-1), pair_h, reduction="batchmean")

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach().cpu()) * len(bi)
            seen += len(bi)
        if args.log_every_epoch and (ep % args.log_every_epoch == 0 or ep == args.epochs):
            print(f"[train-synth] epoch {ep:03d}/{args.epochs} loss={total/max(1,seen):.4f}", flush=True)

    model.eval()
    all_logits = {"used": [], "first": [], "last": [], "read": [], "prim": [], "trans": [], "pair": []}
    masked_logits = {"used": [], "first": [], "last": [], "read": [], "prim": [], "trans": [], "pair": []}
    with torch.no_grad():
        for bi in batch_ids(va, args.batch_size, False):
            clean_w = W[bi].to(device)
            outp = model(clean_w)
            for k, v in outp.items():
                all_logits[k].append(v.cpu())
            moutp = model(corrupt_matrix_batch(clean_w, args))
            for k, v in moutp.items():
                masked_logits[k].append(v.cpu())
    logits = {k: torch.cat(v, dim=0) if v else torch.empty(0) for k, v in all_logits.items()}
    mlogits = {k: torch.cat(v, dim=0) if v else torch.empty(0) for k, v in masked_logits.items()}
    vi = va
    f1 = multilabel_f1(logits["used"], data["used_ops"][vi].float())
    metrics = {
        "val_n": int(len(vi)),
        "used_ops_f1": f1["f1"],
        "used_ops_exact": f1["exact"],
        "first_op_acc": float((logits["first"].argmax(dim=-1) == data["first_op"][vi]).float().mean().item()),
        "last_op_acc": float((logits["last"].argmax(dim=-1) == data["last_op"][vi]).float().mean().item()),
        "read_hist_kl": hist_kl(logits["read"], data["read_hist"][vi].float()),
        "primitive_hist_kl": hist_kl(logits["prim"], data["primitive_hist"][vi].float()),
        "transition_hist_kl": hist_kl(logits["trans"], data["transition_hist"][vi].float()),
        "primitive_transition_pair_kl": hist_kl(logits["pair"], pair_flat[vi].float()) if pair_flat is not None else 0.0,
        "masked_read_hist_kl": hist_kl(mlogits["read"], data["read_hist"][vi].float()),
        "masked_primitive_hist_kl": hist_kl(mlogits["prim"], data["primitive_hist"][vi].float()),
        "masked_transition_hist_kl": hist_kl(mlogits["trans"], data["transition_hist"][vi].float()),
        "masked_primitive_transition_pair_kl": hist_kl(mlogits["pair"], pair_flat[vi].float()) if pair_flat is not None else 0.0,
        "matrix_mask_frac": float(getattr(args, "matrix_mask_frac", 0.0) or 0.0),
        "matrix_rowcol_mask_frac": float(getattr(args, "matrix_rowcol_mask_frac", 0.0) or 0.0),
        "matrix_noise_std": float(getattr(args, "matrix_noise_std", 0.0) or 0.0),
        "elapsed_sec": time.time() - t0,
    }
    torch.save({"model": model.state_dict(), "meta": data["meta"], "metrics": metrics}, out / "baseline_decoder.pt")
    write_json(out / "baseline_metrics.json", metrics)
    print("[train-synth] metrics:", json.dumps(metrics, ensure_ascii=False), flush=True)
    return metrics


# =============================================================================
# CLI
# =============================================================================

def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--out", default="./runs/neural_matrix_program_dataset_v3")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=500)


def add_parse_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--parse-files", nargs="*", default=[])
    ap.add_argument("--parse-dir", "--parse-dirs", dest="parse_dirs", nargs="*", default=[])
    ap.add_argument("--max-parse-files", type=int, default=300)


def add_synth_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--D", type=int, default=32)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--blocks", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--primitive-slots", type=int, default=None)
    ap.add_argument("--max-program-steps", type=int, default=None)
    ap.add_argument("--step-scale", type=float, default=0.23)
    ap.add_argument("--route-scale", type=float, default=0.10)


def add_train_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--train-frac", type=float, default=0.85)
    ap.add_argument("--log-every-epoch", type=int, default=5)
    ap.add_argument("--matrix-mask-frac", type=float, default=0.0)
    ap.add_argument("--matrix-rowcol-mask-frac", type=float, default=0.0)
    ap.add_argument("--matrix-noise-std", type=float, default=0.0)


def add_decode_args(ap: argparse.ArgumentParser, include_D: bool = True) -> None:
    ap.add_argument("--checkpoint", nargs="*", default=[])
    ap.add_argument("--checkpoint-dir", nargs="*", default=[])
    ap.add_argument("--max-checkpoint-files", type=int, default=20)
    ap.add_argument("--max-matrices", type=int, default=300)
    ap.add_argument("--min-matrix-dim", type=int, default=4)
    ap.add_argument("--max-tensor-elements", type=int, default=20_000_000)
    ap.add_argument("--name-regex", default="")
    ap.add_argument("--exclude-name-glob", nargs="*", default=["*bias*", "*norm*"])
    if include_D:
        ap.add_argument("--D", type=int, default=64)
    ap.add_argument("--auto-mined-atoms", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--mined-svd-atoms", type=int, default=4)
    ap.add_argument("--decode-topk", type=int, default=12)
    ap.add_argument("--decode-ridge", type=float, default=1e-5)
    ap.add_argument("--decode-stop-rel", type=float, default=1e-4)
    ap.add_argument("--formula-threshold", type=float, default=1e-5)
    ap.add_argument("--formula-max-terms", type=int, default=20)
    ap.add_argument("--functional-batch", type=int, default=512)


def main() -> None:
    root = argparse.ArgumentParser(description="Neural Matrix Program Dataset v3")
    sub = root.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("parse")
    add_common_args(p); add_parse_args(p)

    p = sub.add_parser("build-synth")
    add_common_args(p); add_parse_args(p); add_synth_args(p)

    p = sub.add_parser("decode-real")
    add_common_args(p); add_decode_args(p)

    p = sub.add_parser("train-synth")
    add_common_args(p); add_train_args(p)

    p = sub.add_parser("all")
    add_common_args(p); add_parse_args(p); add_synth_args(p); add_train_args(p); add_decode_args(p, include_D=False)

    args = root.parse_args()
    seed_all(args.seed)
    if args.cmd == "parse":
        run_parse(args)
    elif args.cmd == "build-synth":
        sk, _ = run_parse(args)
        run_build_synth(args, sk)
    elif args.cmd == "decode-real":
        run_decode_real(args)
    elif args.cmd == "train-synth":
        run_train_synth(args)
    elif args.cmd == "all":
        sk, _ = run_parse(args)
        run_build_synth(args, sk)
        run_train_synth(args)
        if args.checkpoint or args.checkpoint_dir:
            run_decode_real(args)
        else:
            print("[all] no checkpoint/checkpoint-dir passed; skipped decode-real", flush=True)


if __name__ == "__main__":
    main()
