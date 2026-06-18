#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audio transfer using Factorized MatrixProgramAssemblerCore.

Clean transfer:

    MatrixEvidence(audio) -> MatrixProgramAssemblerCore -> AudioHead

Train modes:

    freeze_core  : train only input adapter + head
    editor_delta : train only soft flow editor/phase deltas + adapter + head
    delta        : train only *_delta inside assembler + adapter + head
    full         : train full assembler + adapter + head
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from matrix_program_core.assembler_core import (  # noqa: E402
    AssemblerConfig,
    MatrixProgramAssemblerCore,
    PHASES,
    assembler_skill_loss,
)
from matrix_program_core.task_context_v2 import (  # noqa: E402
    TaskContextV2Config,
    TaskIOContextEncoder,
    ROLE_TASK_HEAD_CORE,
    INPUT_AUDIO_FEATURES,
    OUTPUT_CLASS_LOGITS,
    LOSS_CROSS_ENTROPY,
    READOUT_CLASS_QUERY,
)
from matrix_program_core.train_assembler_mechanism_skill_pretrain import (  # noqa: E402
    FLOW_KEYS,
    TASK_FAMILIES,
    MechanismEvidenceBuilder,
)
from simple_butterfly_matrix.simple_butterfly_matrix import PRIMITIVES  # noqa: E402
from simple_butterfly_matrix.simple_butterfly_matrix import (  # noqa: E402
    MatrixEvidence,
    amp_dtype,
    ensure_dir,
    make_loaders,
    set_seed,
    write_json,
)


SEMANTIC_KIND_NAMES = ("READ", "TRANSFORM", "FILTER_GATE", "MEMORY", "GLOBAL", "WRITE")


def phase_slot_matrix(layers: int, steps: int, blocks: int, device: torch.device, normalize_rows: bool) -> torch.Tensor:
    """Map assembler output slots to program phases.

    MatrixProgramAssemblerCore returns one slot per layer/step/block update:
    [L0.S0.B*, L0.S1.B*, L1.S0.B*, ...]. There is no hard routing here; this
    map is only a soft structural prior for class reads.
    """

    phase_ids: List[int] = []
    phase_count = len(PHASES)
    for layer in range(int(layers)):
        phase = min(layer, phase_count - 1)
        for _step in range(int(steps)):
            phase_ids.extend([phase] * int(blocks))
    mat = torch.zeros(phase_count, len(phase_ids), device=device)
    for slot, phase in enumerate(phase_ids):
        mat[phase, slot] = 1.0
    if normalize_rows:
        mat = mat / mat.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return mat


def class_phase_prior(classes: int) -> torch.Tensor:
    """Weak class/phase start prior copied from the v3 behavior.

    It does not assign classes to paths. It only breaks the symmetry where every
    class starts by reading the same aggregate slots.
    """

    phase_count = len(PHASES)
    p = torch.zeros(classes, phase_count)
    for c in range(classes):
        mode = c % 4
        if mode == 0:
            vals = {"compare": 0.45, "suppress": 0.35, "aggregate": 0.25}
        elif mode == 1:
            vals = {"extract": 0.35, "compare": 0.35, "suppress": 0.30}
        elif mode == 2:
            vals = {"extract": 0.45, "aggregate": 0.30, "compare": 0.20}
        else:
            vals = {"compare": 0.30, "suppress": 0.35, "aggregate": 0.25}
        for name, value in vals.items():
            if name in PHASES:
                p[c, PHASES.index(name)] = value
    if "extract" in PHASES:
        p[:, PHASES.index("extract")] += 0.08
    return p


class AudioAssemblerHead(nn.Module):
    def __init__(
        self,
        cfg: AssemblerConfig,
        dim: int,
        classes: int,
        dropout: float = 0.05,
        pair_slots: int = 12,
        phase_prior_strength: float = 0.85,
    ):
        super().__init__()
        self.cfg = cfg
        self.dim = int(dim)
        self.classes = int(classes)
        self.pair_slots = int(pair_slots)
        self.phase_prior_strength = float(phase_prior_strength)
        self.phasefree_head = bool(getattr(cfg, "latent_roles", False)) and self.phase_prior_strength <= 0.0

        self.class_state = nn.Parameter(torch.randn(classes, dim) * 0.05)
        self.class_phase_logits = nn.Parameter(class_phase_prior(classes))
        self.pair_state = nn.Parameter(torch.randn(self.pair_slots, dim) * 0.04)
        self.class_pair_logits = nn.Parameter(torch.randn(classes, self.pair_slots) * 0.04)

        self.key_w = nn.Parameter(torch.randn(dim, dim) * 0.04)
        self.value_w = nn.Parameter(torch.randn(dim, dim) * 0.04)
        self.class_q = nn.Parameter(torch.randn(dim, dim) * 0.04)
        self.pair_q = nn.Parameter(torch.randn(dim, dim) * 0.04)
        self.pair_k = nn.Parameter(torch.randn(dim, dim) * 0.04)
        self.pair_v = nn.Parameter(torch.randn(dim, dim) * 0.04)

        self.class_update = nn.Sequential(
            nn.LayerNorm(dim * 5),
            nn.Linear(dim * 5, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.class_norm = nn.LayerNorm(dim)
        self.logit_w = nn.Parameter(torch.randn(classes, dim) * 0.04)
        self.logit_bias = nn.Parameter(torch.zeros(classes))
        self.write_logit = nn.Parameter(torch.tensor(-0.35))
        self.drop = nn.Dropout(dropout)

    @property
    def query(self) -> torch.Tensor:
        return self.class_state

    def forward(self, slots: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch, slot_count, dim = slots.shape
        phase_count = len(PHASES)
        if self.phasefree_head:
            # Latent-role mode must not reintroduce extract/compare/suppress/
            # aggregate through the class head. Keep this diagnostic tensor
            # neutral so old report fields remain readable without steering.
            phase_map_mass = torch.full(
                (phase_count, slot_count),
                1.0 / float(max(1, phase_count)),
                device=slots.device,
                dtype=slots.dtype,
            )
            slot_prior = torch.full(
                (self.classes, slot_count),
                1.0 / float(max(1, slot_count)),
                device=slots.device,
                dtype=slots.dtype,
            )
        else:
            phase_map_prior = phase_slot_matrix(
                self.cfg.layers,
                self.cfg.steps,
                self.cfg.blocks,
                slots.device,
                normalize_rows=True,
            ).to(slots.dtype)
            phase_map_mass = phase_slot_matrix(
                self.cfg.layers,
                self.cfg.steps,
                self.cfg.blocks,
                slots.device,
                normalize_rows=False,
            ).to(slots.dtype)
            if phase_map_prior.shape[1] != slot_count:
                phase_map_prior = F.interpolate(
                    phase_map_prior.unsqueeze(0),
                    size=slot_count,
                    mode="linear",
                    align_corners=False,
                ).squeeze(0)
                phase_map_prior = phase_map_prior / phase_map_prior.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                phase_map_mass = phase_map_prior * float(slot_count) / float(max(1, len(PHASES)))

            phase_w = torch.softmax(self.class_phase_logits.float(), dim=-1).to(slots.dtype)
            slot_prior = torch.matmul(phase_w, phase_map_prior).clamp_min(1e-8)

        keys = slots @ self.key_w.to(device=slots.device, dtype=slots.dtype)
        values = self.drop(slots @ self.value_w.to(device=slots.device, dtype=slots.dtype))
        q = self.class_state.to(device=slots.device, dtype=slots.dtype) @ self.class_q.to(device=slots.device, dtype=slots.dtype)
        score = torch.einsum("cd,nsd->ncs", q, keys) / math.sqrt(dim)
        if self.phase_prior_strength > 0.0 and not self.phasefree_head:
            score = score + self.phase_prior_strength * slot_prior.log().view(1, self.classes, slot_count)
        attn = torch.softmax(score.float(), dim=-1).to(slots.dtype)
        class_read = torch.einsum("ncs,nsd->ncd", attn, values)

        pair_w = torch.softmax(self.class_pair_logits.float(), dim=-1).to(slots.dtype)
        pair_base = torch.matmul(pair_w, self.pair_state.to(device=slots.device, dtype=slots.dtype))
        pair_q = class_read @ self.pair_q.to(device=slots.device, dtype=slots.dtype)
        pair_k = pair_base @ self.pair_k.to(device=slots.device, dtype=slots.dtype)
        pair_score = torch.einsum("ncd,ed->nce", pair_q, pair_k) / math.sqrt(dim)
        pair_attn = torch.softmax(pair_score.float(), dim=-1).to(slots.dtype)
        pair_v = pair_base @ self.pair_v.to(device=slots.device, dtype=slots.dtype)
        pair_ctx = torch.einsum("nce,ed->ncd", pair_attn, pair_v)

        cls = self.class_state.to(device=slots.device, dtype=slots.dtype).view(1, self.classes, dim).expand(batch, -1, -1)
        delta = self.class_update(torch.cat([cls, class_read, pair_ctx, cls * class_read, class_read - pair_ctx], dim=-1))
        write = torch.sigmoid(self.write_logit.to(device=slots.device, dtype=slots.dtype))
        class_next = self.class_norm(cls + write * self.drop(delta))
        logits = (class_next * class_read * self.logit_w.to(device=slots.device, dtype=slots.dtype).view(1, self.classes, dim)).sum(dim=-1)
        logits = logits + self.logit_bias.to(device=slots.device, dtype=slots.dtype)

        phase_mass = torch.einsum("ncs,ps->ncp", attn.float(), phase_map_mass.float())
        return logits, {
            "class_slot_attention": attn,
            "class_phase_mass": phase_mass,
            "class_read": class_read,
            "class_next": class_next,
            "pair_attention": pair_attn,
            "pair_update_norm": pair_ctx.float().norm(dim=-1).mean(),
            "class_write": write.float(),
        }


class AudioMechanismContextAdapter(nn.Module):
    """Differentiable bridge from audio/head state to mechanism-flow summaries."""

    def __init__(self, cfg: AssemblerConfig, dim: int):
        super().__init__()
        self.cfg = cfg
        self.T = int(cfg.layers * cfg.steps)
        self.A = int(cfg.address_cells)
        self.B = int(cfg.blocks)
        self.K = int(cfg.primitive_slots)
        self.P = len(PRIMITIVES)
        self.audio_proj = nn.Linear(dim, dim)
        self.head_proj = nn.Linear(dim, dim, bias=False)
        self.mix = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.out = nn.ModuleDict({
            "read_flow": nn.Linear(dim, self.T * self.A),
            "primitive_slot_flow": nn.Linear(dim, self.T * self.P),
            "slot_transition_flow": nn.Linear(dim, self.T * self.K * self.K),
            "primitive_transition_flow": nn.Linear(dim, self.T * self.P * self.P),
            "slot_composition_flow": nn.Linear(dim, self.T * self.K),
            "write_flow": nn.Linear(dim, self.T * self.A),
        })

    def _dist(self, logits: torch.Tensor, shape, dim: int = -1) -> torch.Tensor:
        return torch.softmax(logits.view(*shape).float(), dim=dim)

    def forward(self, evidence: torch.Tensor, head_query: torch.Tensor) -> Dict[str, torch.Tensor]:
        N = int(evidence.shape[0])
        a = self.audio_proj(evidence.mean(dim=1).float())
        h = self.head_proj(head_query.float().mean(dim=0, keepdim=True)).expand(N, -1)
        z = self.mix(a + h)
        return {
            "read_flow": self._dist(self.out["read_flow"](z), (N, self.T, self.A)),
            "primitive_slot_flow": self._dist(self.out["primitive_slot_flow"](z), (N, self.T, self.P)),
            "slot_transition_flow": self._dist(self.out["slot_transition_flow"](z), (N, self.T, self.K * self.K)),
            "primitive_transition_flow": self._dist(self.out["primitive_transition_flow"](z), (N, self.T, self.P * self.P)),
            "slot_composition_flow": self._dist(self.out["slot_composition_flow"](z), (N, self.T, self.K)),
            "write_flow": self._dist(self.out["write_flow"](z), (N, self.T, self.A)),
        }


class AudioAssemblerModel(nn.Module):
    def __init__(self, classes: int, args):
        super().__init__()
        self.input_adapter = MatrixEvidence(args.sample_rate, args.n_mels, args.hop_length, args.evidence_cells, args.dim)
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
            operator_v2=bool(getattr(args, "operator_v2", False)),
            step_alive_init=float(getattr(args, "step_alive_init", 1.65)),
            phase_prior_strength=float(getattr(args, "phase_prior_strength", 0.0)),
            latent_roles=bool(getattr(args, "latent_roles", False)),
            role_count=int(getattr(args, "role_count", 6)),
            role_temperature=float(getattr(args, "role_temperature", 1.25)),
            role_init_std=float(getattr(args, "role_init_std", 0.02)),
            role_effect_init_std=float(getattr(args, "role_effect_init_std", 0.04)),
            role_bias_scale=float(getattr(args, "role_bias_scale", 1.0)),
        )
        self.assembler_core = MatrixProgramAssemblerCore(cfg)
        self.head = AudioAssemblerHead(
            cfg,
            args.dim,
            classes,
            args.head_dropout,
            pair_slots=args.pair_slots,
            phase_prior_strength=args.phase_prior_strength,
        )
        self.use_mechanism_context = bool(getattr(args, "use_mechanism_context", False))
        self.mechanism_builder = MechanismEvidenceBuilder(cfg, args.dim, dropout=args.dropout) if self.use_mechanism_context else None
        self.mechanism_adapter = AudioMechanismContextAdapter(cfg, args.dim) if self.use_mechanism_context else None
        self.mechanism_task_id = int(getattr(args, "mechanism_task_id", max(0, len(TASK_FAMILIES) - 1)))

        # Generic task I/O contract, not a hardcoded classification branch.
        # For attention/layer replacement another script can pass different IDs,
        # but the core still only sees dense context tokens and soft matrices.
        self.role_id = int(getattr(args, "role_id", ROLE_TASK_HEAD_CORE))
        self.input_kind_id = int(getattr(args, "input_kind_id", INPUT_AUDIO_FEATURES))
        self.output_kind_id = int(getattr(args, "output_kind_id", OUTPUT_CLASS_LOGITS))
        self.loss_kind_id = int(getattr(args, "loss_kind_id", LOSS_CROSS_ENTROPY))
        self.readout_kind_id = int(getattr(args, "readout_kind_id", READOUT_CLASS_QUERY))
        self.use_head_context = bool(getattr(args, "use_head_context", True))
        self.task_context = TaskIOContextEncoder(TaskContextV2Config(
            dim=args.dim,
            free_tokens=int(getattr(args, "task_context_tokens", 4)),
            max_head_tokens=int(getattr(args, "head_context_tokens", classes)),
            dropout=args.dropout,
        ))

    def forward(self, wav: torch.Tensor):
        evidence = self.input_adapter(wav)
        head_query = self.head.query if self.use_head_context else None
        context = self.task_context(
            evidence.shape[0],
            evidence.device,
            evidence.dtype,
            role_id=self.role_id,
            input_kind_id=self.input_kind_id,
            output_kind_id=self.output_kind_id,
            loss_kind_id=self.loss_kind_id,
            readout_kind_id=self.readout_kind_id,
            num_outputs=float(self.head.classes),
            sequence_length=float(evidence.shape[1]),
            hidden_dim=float(evidence.shape[-1]),
            head_query=head_query,
        )
        parts = [context]
        if self.use_mechanism_context and self.mechanism_builder is not None and self.mechanism_adapter is not None:
            summaries = self.mechanism_adapter(evidence, self.head.query)
            visible = torch.ones(evidence.shape[0], len(FLOW_KEYS), device=evidence.device, dtype=torch.float32)
            task_ids = torch.full((evidence.shape[0],), self.mechanism_task_id, device=evidence.device, dtype=torch.long)
            mech = self.mechanism_builder(summaries, visible, task_ids).to(dtype=evidence.dtype)
            parts.append(mech)
        parts.append(evidence)
        evidence = torch.cat(parts, dim=1)
        _cells, aux = self.assembler_core(evidence)
        logits, haux = self.head(aux.slots)
        return logits, aux, haux


def trainable_summary(model: AudioAssemblerModel) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    core_total = sum(p.numel() for p in model.assembler_core.parameters())
    core_train = sum(p.numel() for p in model.assembler_core.parameters() if p.requires_grad)
    context_total = sum(p.numel() for p in model.task_context.parameters())
    context_train = sum(p.numel() for p in model.task_context.parameters() if p.requires_grad)
    mech_total = 0
    mech_train = 0
    if model.mechanism_builder is not None:
        mech_total += sum(p.numel() for p in model.mechanism_builder.parameters())
        mech_train += sum(p.numel() for p in model.mechanism_builder.parameters() if p.requires_grad)
    if model.mechanism_adapter is not None:
        mech_total += sum(p.numel() for p in model.mechanism_adapter.parameters())
        mech_train += sum(p.numel() for p in model.mechanism_adapter.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "core_total": core_total,
        "core_trainable": core_train,
        "task_context_total": context_total,
        "task_context_trainable": context_train,
        "mechanism_context_total": mech_total,
        "mechanism_context_trainable": mech_train,
    }


def configure_train_mode(model: AudioAssemblerModel, mode: str, train_input_adapter: bool, train_head: bool, train_task_context: bool) -> None:
    model.assembler_core.freeze_for_mode(mode)
    for p in model.input_adapter.parameters():
        p.requires_grad = bool(train_input_adapter)
    for p in model.head.parameters():
        p.requires_grad = bool(train_head)
    for p in model.task_context.parameters():
        p.requires_grad = bool(train_task_context)
    if model.mechanism_builder is not None:
        for p in model.mechanism_builder.parameters():
            p.requires_grad = False
    if model.mechanism_adapter is not None:
        for p in model.mechanism_adapter.parameters():
            p.requires_grad = bool(train_input_adapter)


def _is_tensor_state(obj) -> bool:
    return isinstance(obj, dict) and bool(obj) and all(torch.is_tensor(v) for v in obj.values())


def _strip_prefix_state(state: Dict[str, torch.Tensor], prefixes: Tuple[str, ...]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        new_key = key
        for prefix in prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
                break
        out[new_key] = value
    return out


def _extract_core_state(ckpt) -> Dict[str, torch.Tensor]:
    if not isinstance(ckpt, dict):
        raise TypeError("assembler checkpoint must be a dict")
    for key in ("assembler_core", "core", "assembler_skill_base"):
        state = ckpt.get(key)
        if _is_tensor_state(state):
            return _strip_prefix_state(state, ("assembler_core.", "core."))
    model_state = ckpt.get("model")
    if isinstance(model_state, dict):
        state = {
            key: value
            for key, value in model_state.items()
            if key.startswith("assembler_core.") or key.startswith("core.")
        }
        if state:
            return _strip_prefix_state(state, ("assembler_core.", "core."))
    if _is_tensor_state(ckpt):
        return _strip_prefix_state(ckpt, ("assembler_core.", "core."))
    raise KeyError("assembler checkpoint has no assembler_core/core/assembler_skill_base/model core state")


def _extract_task_context_state(ckpt) -> Dict[str, torch.Tensor]:
    if not isinstance(ckpt, dict):
        return {}
    for key in ("task_context", "task_context_base"):
        state = ckpt.get(key)
        if _is_tensor_state(state):
            return _strip_prefix_state(state, ("task_context.",))
    model_state = ckpt.get("model")
    if isinstance(model_state, dict):
        state = {
            key: value
            for key, value in model_state.items()
            if key.startswith("task_context.")
        }
        if state:
            return _strip_prefix_state(state, ("task_context.",))
    return {}


def _extract_mechanism_builder_state(ckpt) -> Dict[str, torch.Tensor]:
    if not isinstance(ckpt, dict):
        return {}
    state = ckpt.get("mechanism_evidence_builder")
    if _is_tensor_state(state):
        return state
    state = ckpt.get("mechanism_context_base")
    if _is_tensor_state(state):
        return _strip_prefix_state(state, ("mechanism_evidence_builder.", "mechanism_builder."))
    return {}


def _normalize_flow_target(x: torch.Tensor, cfg: AssemblerConfig) -> torch.Tensor:
    x = x.float()
    total_steps = int(cfg.layers * cfg.steps)
    if x.ndim >= 2 and x.shape[0] != total_steps and x.shape[1] == total_steps:
        x = x.mean(dim=0)
    return x / x.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def load_skill_target_pack(path: str | Path, cfg: AssemblerConfig, device: torch.device) -> Dict[str, torch.Tensor]:
    pack = torch.load(path, map_location="cpu")
    targets: Dict[str, torch.Tensor] = {}
    for key in ("read_flow", "primitive_slot_flow", "slot_transition_flow", "primitive_transition_flow", "slot_composition_flow", "write_flow"):
        value = pack.get(key) if isinstance(pack, dict) else None
        if torch.is_tensor(value):
            targets[key] = _normalize_flow_target(value, cfg).to(device)
    if not targets:
        raise ValueError(f"skill target pack has no flow tensors: {path}")
    return targets


def build_prior_flow_targets(cfg: AssemblerConfig, device: torch.device) -> Dict[str, torch.Tensor]:
    # Kept as an explicit debug baseline. Normal transfer should use the skill
    # weights themselves, optionally regularized by a real/code flow pack.
    from matrix_program_core.train_assembler_pretrain import build_flow_targets

    return build_flow_targets(cfg, device)


def make_runtime_flow_targets(args, cfg: AssemblerConfig, device: torch.device) -> Dict[str, torch.Tensor]:
    if args.skill_target_pack:
        targets = load_skill_target_pack(args.skill_target_pack, cfg, device)
        print(f"loaded skill target pack: {args.skill_target_pack} keys={sorted(targets)}", flush=True)
        return targets
    if args.use_prior_skill_targets:
        print("using generic prior skill targets; this is a debug baseline, not dataset-transfer supervision", flush=True)
        return build_prior_flow_targets(cfg, device)
    if args.lambda_skill > 0:
        print("lambda_skill > 0 but no --skill-target-pack/--use-prior-skill-targets; disabling live skill anchor", flush=True)
    return {}


def class_read_diversity_loss(attn: torch.Tensor) -> torch.Tensor:
    # attn [N,C,S]. Penalize classes that read the same slot distribution.
    a = attn.float()
    a = F.normalize(a, dim=-1)
    sim = torch.einsum("ncs,nds->ncd", a, a)
    eye = torch.eye(sim.shape[-1], device=sim.device, dtype=sim.dtype).view(1, sim.shape[-1], sim.shape[-1])
    offdiag = sim - eye
    return F.relu(offdiag - 0.25).mean()


def class_slot_prior(attn: torch.Tensor, sigma: float) -> torch.Tensor:
    # Fixed soft windows over slots break the uniform-attention symmetry while
    # keeping all reads dense and differentiable.
    _, C, S = attn.shape
    slots = torch.arange(S, device=attn.device, dtype=torch.float32).view(1, S)
    centers = (torch.arange(C, device=attn.device, dtype=torch.float32) + 0.5) * (float(S) / float(C))
    centers = centers.view(C, 1)
    sigma_t = torch.tensor(max(0.5, float(sigma)), device=attn.device, dtype=torch.float32)
    dist = (slots - centers).abs()
    dist = torch.minimum(dist, torch.tensor(float(S), device=attn.device) - dist)
    prior = torch.softmax(-0.5 * (dist / sigma_t).pow(2), dim=-1)
    return prior


def class_slot_prior_loss(attn: torch.Tensor, sigma: float) -> torch.Tensor:
    a = attn.float().clamp_min(1e-8)
    prior = class_slot_prior(attn, sigma)
    return -(prior.view(1, *prior.shape) * a.log()).sum(dim=-1).mean()


def class_attention_entropy(attn: torch.Tensor) -> torch.Tensor:
    a = attn.float().clamp_min(1e-8)
    a = a / a.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return -(a * a.log()).sum(dim=-1).mean()


def phase_balance_loss(phase_mass: torch.Tensor, min_early: float, max_aggregate: float) -> torch.Tensor:
    usage = phase_mass.float().mean(dim=(0, 1))
    if usage.numel() <= 1:
        return torch.zeros((), device=usage.device)
    early = usage[:-1]
    early_loss = F.relu(torch.tensor(float(min_early), device=usage.device) - early).pow(2).mean()
    agg_loss = F.relu(usage[-1] - float(max_aggregate)).pow(2)
    ent = -(usage.clamp_min(1e-8) * usage.clamp_min(1e-8).log()).sum()
    ent_floor = F.relu(torch.tensor(1.15, device=usage.device) - ent).pow(2)
    return early_loss + agg_loss + 0.25 * ent_floor


def slot_diversity_loss(slots: torch.Tensor) -> torch.Tensor:
    s = slots.float()
    s = F.normalize(s, dim=-1)
    sim = torch.einsum("nsd,ntd->nst", s, s)
    eye = torch.eye(sim.shape[-1], device=sim.device, dtype=sim.dtype).view(1, sim.shape[-1], sim.shape[-1])
    offdiag = sim - eye
    return F.relu(offdiag - 0.55).mean()


def _load_balance_loss(p: torch.Tensor, active_floor: float = 0.0) -> torch.Tensor:
    # p [...,C], normalized distribution. Penalize extreme collapse while keeping it soft.
    q = p.float().mean(dim=tuple(range(max(0, p.ndim - 1))))
    q = q / q.sum().clamp_min(1e-8)
    C = q.numel()
    if C <= 1:
        return torch.zeros((), device=p.device)
    uniform = torch.full_like(q, 1.0 / float(C))
    mse = (q - uniform).pow(2).mean()
    ent = -(q.clamp_min(1e-8) * q.clamp_min(1e-8).log()).sum() / math.log(float(C))
    floor = F.relu(torch.tensor(float(active_floor), device=p.device) - ent).pow(2)
    return mse + floor


def primitive_load_balance_loss(aux, floor: float) -> torch.Tensor:
    return _load_balance_loss(aux.primitive_slot_flow, active_floor=floor)


def read_write_cell_balance_loss(aux, floor: float) -> torch.Tensor:
    rw = torch.cat([
        aux.read_flow.float().mean(dim=-2).reshape(-1, aux.read_flow.shape[-1]),
        aux.write_flow.float().reshape(-1, aux.write_flow.shape[-1]),
    ], dim=0)
    return _load_balance_loss(rw, active_floor=floor)


def entropy_band_loss(aux, low: float, high: float) -> torch.Tensor:
    vals = []
    for key in ("read", "primitive", "slot_transition", "primitive_transition", "write"):
        if key in aux.entropies:
            vals.append(aux.entropies[key].float())
    if not vals:
        return torch.zeros((), device=aux.cells.device)
    ent = torch.stack(vals).mean()
    return F.relu(torch.tensor(float(low), device=ent.device) - ent).pow(2) + F.relu(ent - float(high)).pow(2)


def step_alive_budget_loss(aux, target: float) -> torch.Tensor:
    if "step_alive" not in aux.entropies:
        return torch.zeros((), device=aux.cells.device)
    return (aux.entropies["step_alive"].float() - float(target)).pow(2)


def layer_program_similarity_loss(slots: torch.Tensor, layers: int, steps: int, blocks: int, margin: float) -> torch.Tensor:
    # slots [N,T*B,D]. Build layer summaries [N,L,D] and penalize too-similar layers.
    s = slots.float()
    N, SB, D = s.shape
    L, S, B = int(layers), int(steps), int(blocks)
    need = L * S * B
    if L <= 1 or SB < need:
        return torch.zeros((), device=slots.device)
    x = s[:, :need].view(N, L, S, B, D).mean(dim=(2, 3))
    x = F.normalize(x, dim=-1)
    sim = torch.einsum("nld,nmd->nlm", x, x)
    eye = torch.eye(L, device=slots.device, dtype=sim.dtype).view(1, L, L)
    off = sim - eye
    return F.relu(off - float(margin)).mean()


def step_program_similarity_loss(slots: torch.Tensor, layers: int, steps: int, blocks: int, margin: float) -> torch.Tensor:
    # Penalize adjacent/near step summaries becoming clones.
    s = slots.float()
    N, SB, D = s.shape
    L, S, B = int(layers), int(steps), int(blocks)
    T = L * S
    need = T * B
    if T <= 1 or SB < need:
        return torch.zeros((), device=slots.device)
    x = s[:, :need].view(N, T, B, D).mean(dim=2)
    x = F.normalize(x, dim=-1)
    sim = torch.einsum("ntd,nud->ntu", x, x)
    eye = torch.eye(T, device=slots.device, dtype=sim.dtype).view(1, T, T)
    off = sim - eye
    # Adjacent steps are allowed to be somewhat related, but not identical.
    return F.relu(off - float(margin)).mean()


def role_usage_balance_loss(aux, floor: float) -> torch.Tensor:
    if getattr(aux, "role_usage", None) is None:
        return torch.zeros((), device=aux.cells.device)
    usage = aux.role_usage.float()
    usage = usage / usage.sum().clamp_min(1e-8)
    R = usage.numel()
    if R <= 1:
        return torch.zeros((), device=aux.cells.device)
    entropy = -(usage.clamp_min(1e-8) * usage.clamp_min(1e-8).log()).sum() / math.log(float(R))
    uniform = torch.full_like(usage, 1.0 / float(R))
    return (usage - uniform).pow(2).mean() + F.relu(torch.tensor(float(floor), device=usage.device) - entropy).pow(2)


def role_entropy_band_loss(aux, low: float, high: float) -> torch.Tensor:
    if getattr(aux, "role_entropy", None) is None:
        return torch.zeros((), device=aux.cells.device)
    ent = aux.role_entropy.float().mean()
    R = max(2, int(aux.role_usage.numel())) if getattr(aux, "role_usage", None) is not None else 2
    ent_norm = ent / math.log(float(R))
    return F.relu(torch.tensor(float(low), device=ent.device) - ent_norm).pow(2) + F.relu(ent_norm - float(high)).pow(2)


def role_similarity_loss(aux, margin: float) -> torch.Tensor:
    if getattr(aux, "role_mix", None) is None:
        return torch.zeros((), device=aux.cells.device)
    mix = aux.role_mix.float().reshape(-1, aux.role_mix.shape[-1])
    if mix.shape[0] <= 1:
        return torch.zeros((), device=aux.cells.device)
    mix = F.normalize(mix, dim=-1)
    sim = torch.matmul(mix, mix.t())
    eye = torch.eye(sim.shape[0], device=sim.device, dtype=sim.dtype)
    off = sim - eye
    return F.relu(off - float(margin)).mean()


def role_effect_similarity_loss(aux, margin: float) -> torch.Tensor:
    sim = getattr(aux, "role_effect_similarity", None)
    if sim is None:
        return torch.zeros((), device=aux.cells.device)
    return F.relu(sim.float() - float(margin)).pow(2)


def late_input_shortcut_loss(aux, layers: int, steps: int, start_layer: int, target: float) -> torch.Tensor:
    x = getattr(aux, "input_proxy_read", None)
    if x is None:
        return torch.zeros((), device=aux.cells.device)
    # x [T,N] estimates how much a step reads cells still close to init_cells.
    # This is a soft proxy for raw-input shortcut in the current address-cell
    # design, where evidence itself is not an explicit read group.
    T = int(x.shape[0])
    layer_ids = torch.arange(T, device=x.device) // max(1, int(steps))
    mask = layer_ids >= int(start_layer)
    if not bool(mask.any()):
        return torch.zeros((), device=x.device)
    depth = (layer_ids.float() / float(max(1, int(layers) - 1))).view(T, 1)
    over = F.relu(x.float() - float(target))
    return (over[mask].pow(2) * (1.0 + depth[mask])).mean()


def _norm_entropy_focus(p: torch.Tensor, dim: int = -1) -> torch.Tensor:
    p = p.float().clamp_min(1e-8)
    p = p / p.sum(dim=dim, keepdim=True).clamp_min(1e-8)
    ent = -(p * p.log()).sum(dim=dim)
    denom = math.log(float(max(2, p.shape[dim])))
    return (1.0 - ent / denom).clamp(0.0, 1.0)


def semantic_frame_tensors(aux, haux, args) -> Dict[str, torch.Tensor]:
    # All fields are soft summaries of existing matrix-program flows.
    # Shapes mostly use [T,N,B] where T=L*S. Nothing here routes execution.
    read = aux.read_flow.float()
    prim = aux.primitive_slot_flow.float()
    write = aux.write_flow.float()
    T, N, B = int(read.shape[0]), int(read.shape[1]), int(read.shape[2])
    A = int(read.shape[-1])
    mem0 = int(args.blocks)
    glob0 = mem0 + int(args.memory_cells)

    read_addr = read.mean(dim=3)  # [T,N,B,A]
    read_state = read_addr[..., :mem0].sum(dim=-1) if mem0 > 0 else torch.zeros(T, N, B, device=read.device)
    read_memory = read_addr[..., mem0:glob0].sum(dim=-1) if int(args.memory_cells) > 0 else torch.zeros(T, N, B, device=read.device)
    read_global = read_addr[..., glob0:].sum(dim=-1) if int(args.global_cells) > 0 else torch.zeros(T, N, B, device=read.device)

    write_state = write[..., :mem0].sum(dim=-1) if mem0 > 0 else torch.zeros(T, N, B, device=write.device)
    write_memory = write[..., mem0:glob0].sum(dim=-1) if int(args.memory_cells) > 0 else torch.zeros(T, N, B, device=write.device)
    write_global = write[..., glob0:].sum(dim=-1) if int(args.global_cells) > 0 else torch.zeros(T, N, B, device=write.device)

    pidx = {name: i for i, name in enumerate(PRIMITIVES)}
    prim_mass = prim.mean(dim=3)  # [T,N,B,P]
    def pmass(names: Tuple[str, ...]) -> torch.Tensor:
        idxs = [pidx[n] for n in names if n in pidx]
        if not idxs:
            return torch.zeros(T, N, B, device=prim.device)
        return prim_mass[..., idxs].sum(dim=-1)

    transform = pmass(("channel_butterfly", "block_butterfly", "low_rank", "ctx_matrix"))
    filter_gate = pmass(("product_gate", "phase_matrix"))

    read_focus = _norm_entropy_focus(read_addr, dim=-1)
    write_focus = _norm_entropy_focus(write, dim=-1)
    primitive_focus = _norm_entropy_focus(prim_mass, dim=-1)

    attn = haux["class_slot_attention"].float()
    slot_count = T * B
    if attn.shape[-1] == slot_count:
        consumer = attn.mean(dim=1).transpose(0, 1).reshape(T, B, N).permute(0, 2, 1)
    else:
        consumer = torch.full((T, N, B), 1.0 / float(max(1, slot_count)), device=read.device, dtype=read.dtype)

    kind_raw = torch.stack([
        read_focus + 0.35 * read_state,
        transform + 0.35 * primitive_focus,
        filter_gate,
        read_memory + write_memory,
        read_global + write_global,
        write_focus + 0.35 * write_state,
    ], dim=-1).clamp_min(1e-8)
    kind = kind_raw / kind_raw.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    return {
        "kind": kind,
        "read_focus": read_focus,
        "write_focus": write_focus,
        "primitive_focus": primitive_focus,
        "read_memory": read_memory,
        "read_global": read_global,
        "write_memory": write_memory,
        "write_global": write_global,
        "consumer": consumer,
    }


def semantic_transition_loss(kind: torch.Tensor) -> torch.Tensor:
    if kind.shape[0] <= 1:
        return torch.zeros((), device=kind.device)
    C = kind.shape[-1]
    allowed = torch.tensor([
        # READ, TRANSFORM, FILTER_GATE, MEMORY, GLOBAL, WRITE
        [0.30, 0.95, 0.80, 0.75, 0.75, 0.35],  # READ
        [0.55, 0.75, 0.80, 0.70, 0.70, 0.90],  # TRANSFORM
        [0.45, 0.90, 0.55, 0.55, 0.55, 0.85],  # FILTER_GATE
        [0.65, 0.85, 0.65, 0.45, 0.70, 0.80],  # MEMORY
        [0.65, 0.85, 0.60, 0.65, 0.45, 0.85],  # GLOBAL
        [0.80, 0.60, 0.45, 0.45, 0.45, 0.35],  # WRITE
    ], device=kind.device, dtype=kind.dtype)
    if C != allowed.shape[0]:
        return torch.zeros((), device=kind.device)
    a = kind[:-1].mean(dim=(1, 2))
    b = kind[1:].mean(dim=(1, 2))
    bad = 1.0 - allowed
    return torch.einsum("tc,cd,td->t", a, bad, b).mean()


def semantic_lint_losses(aux, haux, args) -> Dict[str, torch.Tensor]:
    sem = semantic_frame_tensors(aux, haux, args)
    kind = sem["kind"]
    T, N, B = kind.shape[:3]
    slot_count = max(1, T * B)
    consumer_floor = float(args.semantic_dead_slot_frac) / float(slot_count)

    read_focus = sem["read_focus"].mean()
    primitive_focus = sem["primitive_focus"].mean()
    memory_global = (sem["read_memory"] + sem["read_global"] + sem["write_memory"] + sem["write_global"]).mean()
    consumer = sem["consumer"]
    write_pressure = (sem["write_focus"] + sem["write_memory"] + sem["write_global"]).detach()

    losses: Dict[str, torch.Tensor] = {}
    losses["semantic_no_read"] = F.relu(torch.tensor(float(args.semantic_min_read_focus), device=kind.device) - read_focus).pow(2)
    losses["semantic_no_transform"] = F.relu(torch.tensor(float(args.semantic_min_transform_focus), device=kind.device) - primitive_focus).pow(2)
    losses["semantic_memory_global"] = F.relu(torch.tensor(float(args.semantic_min_memory_global), device=kind.device) - memory_global).pow(2)
    losses["semantic_dead_slot"] = F.relu(torch.tensor(consumer_floor, device=kind.device) - consumer).mean()
    losses["semantic_write_consumer"] = (write_pressure * F.relu(torch.tensor(consumer_floor, device=kind.device) - consumer)).mean()
    if T > 1:
        early = max(1, T // 3)
        early_write = (sem["write_memory"][:early] + sem["write_global"][:early]).mean()
        losses["semantic_early_write"] = early_write.pow(2)
    else:
        losses["semantic_early_write"] = torch.zeros((), device=kind.device)
    losses["semantic_transition"] = semantic_transition_loss(kind)
    total = torch.zeros((), device=kind.device)
    for value in losses.values():
        total = total + value
    losses["semantic_lint"] = total
    return losses


@torch.no_grad()
def semantic_report(aux, haux, args) -> Dict[str, object]:
    sem = semantic_frame_tensors(aux, haux, args)
    kind = sem["kind"].detach().float()
    out: Dict[str, object] = {
        "kind_names": list(SEMANTIC_KIND_NAMES),
        "kind_usage": kind.mean(dim=(0, 1, 2)).cpu().tolist(),
        "read_focus": float(sem["read_focus"].mean().detach().cpu()),
        "write_focus": float(sem["write_focus"].mean().detach().cpu()),
        "primitive_focus": float(sem["primitive_focus"].mean().detach().cpu()),
        "memory_global_usage": float((sem["read_memory"] + sem["read_global"] + sem["write_memory"] + sem["write_global"]).mean().detach().cpu()),
        "consumer_min": float(sem["consumer"].min().detach().cpu()),
        "consumer_mean": float(sem["consumer"].mean().detach().cpu()),
    }
    if getattr(aux, "role_mix", None) is not None:
        T = kind.shape[0]
        role = aux.role_mix.detach().float().reshape(T, -1).to(kind.device)
        kind_t = kind.mean(dim=(1, 2))
        denom = role.sum(dim=0).clamp_min(1e-8).view(-1, 1)
        role_kind = torch.einsum("tr,tc->rc", role, kind_t) / denom
        prim = aux.primitive_slot_flow.detach().float().mean(dim=(1, 2, 3))
        role_prim = torch.einsum("tr,tp->rp", role, prim.to(kind.device)) / denom
        out["role_kind"] = role_kind.cpu().tolist()
        out["role_kind_top"] = [SEMANTIC_KIND_NAMES[int(i)] for i in role_kind.argmax(dim=-1).cpu().tolist()]
        out["role_primitive_top"] = [PRIMITIVES[int(i)] for i in role_prim.argmax(dim=-1).cpu().tolist()]
    return out


def aux_losses(logits: torch.Tensor, aux, haux, args, flow_targets, skill_weights) -> Dict[str, torch.Tensor]:
    if flow_targets:
        skill, flow_losses = assembler_skill_loss(aux, flow_targets, skill_weights)
    else:
        skill = torch.zeros((), device=logits.device)
        flow_losses = {}
    gate = aux.write_gates.float()
    upd = aux.update_norms.float()
    out: Dict[str, torch.Tensor] = dict(flow_losses)
    out["skill"] = skill
    out["write_budget"] = (gate.mean() - args.write_target).pow(2) if gate.numel() else torch.zeros((), device=logits.device)
    out["update_alive"] = F.relu(torch.tensor(float(args.min_update_norm), device=logits.device) - upd.mean()).pow(2) if upd.numel() else torch.zeros((), device=logits.device)
    out["class_read_div"] = class_read_diversity_loss(haux["class_slot_attention"])
    out["class_slot_prior"] = class_slot_prior_loss(haux["class_slot_attention"], args.class_slot_prior_sigma)
    out["class_attn_entropy"] = class_attention_entropy(haux["class_slot_attention"])
    out["phase_balance"] = phase_balance_loss(haux["class_phase_mass"], args.min_early_phase_mass, args.max_aggregate_phase_mass)
    out["slot_div"] = slot_diversity_loss(aux.slots)
    out["layer_sim"] = layer_program_similarity_loss(aux.slots, args.layers, args.steps, args.blocks, args.layer_sim_margin)
    out["step_sim"] = step_program_similarity_loss(aux.slots, args.layers, args.steps, args.blocks, args.step_sim_margin)
    out["primitive_balance"] = primitive_load_balance_loss(aux, args.primitive_balance_entropy_floor)
    out["cell_balance"] = read_write_cell_balance_loss(aux, args.cell_balance_entropy_floor)
    out["entropy_band"] = entropy_band_loss(aux, args.entropy_band_low, args.entropy_band_high)
    out["step_alive_budget"] = step_alive_budget_loss(aux, args.step_alive_target)
    out["role_usage_balance"] = role_usage_balance_loss(aux, args.role_usage_entropy_floor)
    out["role_entropy_band"] = role_entropy_band_loss(aux, args.role_entropy_low, args.role_entropy_high)
    out["role_similarity"] = role_similarity_loss(aux, args.role_similarity_margin)
    out["role_effect_similarity"] = role_effect_similarity_loss(aux, args.role_effect_similarity_margin)
    out["role_usage_max"] = aux.role_usage.float().max() if getattr(aux, "role_usage", None) is not None else torch.zeros((), device=logits.device)
    out["role_entropy"] = aux.role_entropy.float().mean() if getattr(aux, "role_entropy", None) is not None else torch.zeros((), device=logits.device)
    out["role_effect_sim_value"] = aux.role_effect_similarity.float() if getattr(aux, "role_effect_similarity", None) is not None else torch.zeros((), device=logits.device)
    out["late_input_shortcut"] = late_input_shortcut_loss(
        aux,
        args.layers,
        args.steps,
        args.late_input_start_layer,
        args.late_input_target,
    )
    out["input_proxy_read"] = aux.input_proxy_read.float().mean() if getattr(aux, "input_proxy_read", None) is not None else torch.zeros((), device=logits.device)
    out.update(semantic_lint_losses(aux, haux, args))
    out["logit_norm"] = logits.float().pow(2).mean()
    out["pair_update_norm"] = haux["pair_update_norm"].float()
    out["class_write"] = haux["class_write"].float()
    return out


def train_epoch(model, loader, opt, scaler, device, dtype, args, epoch: int, flow_targets, skill_weights):
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
            logits, aux, haux = model(wav)
            ce = F.cross_entropy(logits.float(), y)
            losses = aux_losses(logits, aux, haux, args, flow_targets, skill_weights)
            loss = ce
            loss = loss + args.lambda_skill * losses["skill"]
            loss = loss + args.lambda_write_budget * losses["write_budget"]
            loss = loss + args.lambda_update_alive * losses["update_alive"]
            loss = loss + args.lambda_class_read_div * losses["class_read_div"]
            loss = loss + args.lambda_class_slot_prior * losses["class_slot_prior"]
            loss = loss + args.lambda_class_attn_entropy * losses["class_attn_entropy"]
            loss = loss + args.lambda_phase_balance * losses["phase_balance"]
            loss = loss + args.lambda_slot_div * losses["slot_div"]
            loss = loss + args.lambda_layer_sim * losses["layer_sim"]
            loss = loss + args.lambda_step_sim * losses["step_sim"]
            loss = loss + args.lambda_primitive_balance * losses["primitive_balance"]
            loss = loss + args.lambda_cell_balance * losses["cell_balance"]
            loss = loss + args.lambda_entropy_band * losses["entropy_band"]
            loss = loss + args.lambda_step_alive_budget * losses["step_alive_budget"]
            loss = loss + args.lambda_role_usage_balance * losses["role_usage_balance"]
            loss = loss + args.lambda_role_entropy_band * losses["role_entropy_band"]
            loss = loss + args.lambda_role_similarity * losses["role_similarity"]
            loss = loss + args.lambda_role_effect_similarity * losses["role_effect_similarity"]
            loss = loss + args.lambda_late_input_shortcut * losses["late_input_shortcut"]
            loss = loss + args.lambda_semantic_lint * losses["semantic_lint"]
            loss = loss + args.lambda_logit_norm * losses["logit_norm"]
        if not torch.isfinite(loss):
            print("NONFINITE_LOSS skip", flush=True)
            continue
        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(opt)
            trainable = [p for p in model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
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
            print(f"epoch {epoch:03d} step {step:05d} loss={totals['loss']/max(1, totals['n']):.4f} ce={totals['ce']/max(1, totals['n']):.4f} acc={100*totals['correct']/max(1, totals['n']):.2f}%", flush=True)
    out = {k: v / max(1, totals["n"]) for k, v in totals.items() if k != "correct"}
    out["acc"] = totals["correct"] / max(1, totals["n"])
    for k, v in aux_sum.items():
        out[k] = v / max(1, totals["n"])
    return out


@torch.no_grad()
def evaluate(model, loader, device, dtype, args, flow_targets, skill_weights):
    model.eval()
    use_amp = device.startswith("cuda") and dtype != torch.float32
    total_loss, correct, n = 0.0, 0, 0
    last_report = None
    for step, (wav, y) in enumerate(loader, 1):
        if args.max_val_batches and step > args.max_val_batches:
            break
        wav = wav.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=use_amp):
            logits, aux, haux = model(wav)
            ce = F.cross_entropy(logits.float(), y)
            losses = aux_losses(logits, aux, haux, args, flow_targets, skill_weights)
            loss = ce + args.lambda_skill * losses["skill"]
        pred = logits.argmax(-1)
        bs = y.numel()
        total_loss += float(loss.detach().cpu()) * bs
        correct += int((pred == y).sum().detach().cpu())
        n += bs
        last_report = {
            "skill": float(losses["skill"].detach().cpu()),
            "write_gate_mean": float(aux.write_gates.float().mean().detach().cpu()) if aux.write_gates.numel() else 0.0,
            "update_norm_mean": float(aux.update_norms.float().mean().detach().cpu()) if aux.update_norms.numel() else 0.0,
            "memory_usage": float(aux.memory_usage.detach().cpu()),
            "global_usage": float(aux.global_usage.detach().cpu()),
            "entropy": {k: float(v.detach().cpu()) for k, v in aux.entropies.items()},
            "role_usage": aux.role_usage.detach().cpu().tolist() if getattr(aux, "role_usage", None) is not None else None,
            "role_mix": aux.role_mix.detach().cpu().tolist() if getattr(aux, "role_mix", None) is not None else None,
            "role_entropy_mean": float(aux.role_entropy.float().mean().detach().cpu()) if getattr(aux, "role_entropy", None) is not None else 0.0,
            "role_similarity": float(aux.role_similarity.detach().cpu()) if getattr(aux, "role_similarity", None) is not None else 0.0,
            "role_effect_similarity": float(aux.role_effect_similarity.detach().cpu()) if getattr(aux, "role_effect_similarity", None) is not None else 0.0,
            "input_proxy_read": aux.input_proxy_read.float().mean(dim=1).detach().cpu().tolist() if getattr(aux, "input_proxy_read", None) is not None else None,
            "read_group_mass": {
                "names": ["state", "memory", "global"],
                "by_step": aux.read_group_mass.detach().cpu().tolist() if getattr(aux, "read_group_mass", None) is not None else None,
                "mean": aux.read_group_mass.float().mean(dim=0).detach().cpu().tolist() if getattr(aux, "read_group_mass", None) is not None else None,
            },
            "semantic_frame_summary": semantic_report(aux, haux, args),
            "phase_mass_mean": {
                PHASES[i]: float(haux["class_phase_mass"].float().mean(dim=(0, 1))[i].detach().cpu())
                for i in range(len(PHASES))
            },
            "pair_update_norm": float(haux["pair_update_norm"].detach().cpu()),
            "class_write": float(haux["class_write"].detach().cpu()),
            "slot_count": len(aux.slot_names),
            "cell_names": aux.cell_names,
        }
    return {"loss": total_loss / max(1, n), "acc": correct / max(1, n), "n": n, "report": last_report}


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

    model = AudioAssemblerModel(len(classes), args).to(device)
    if args.assembler_checkpoint:
        ckpt = torch.load(args.assembler_checkpoint, map_location=device)
        state = _extract_core_state(ckpt)
        missing, unexpected = model.assembler_core.load_state_dict(state, strict=False)
        print(f"loaded assembler checkpoint: {args.assembler_checkpoint} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        ctx_state = _extract_task_context_state(ckpt)
        if ctx_state and hasattr(model, "task_context"):
            miss_ctx, unexp_ctx = model.task_context.load_state_dict(ctx_state, strict=False)
            print(f"loaded task_context from assembler checkpoint missing={len(miss_ctx)} unexpected={len(unexp_ctx)}", flush=True)
        mech_state = _extract_mechanism_builder_state(ckpt)
        if mech_state and model.mechanism_builder is not None:
            miss_mech, unexp_mech = model.mechanism_builder.load_state_dict(mech_state, strict=False)
            print(f"loaded mechanism_builder from assembler checkpoint missing={len(miss_mech)} unexpected={len(unexp_mech)}", flush=True)
    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location=device)
        state = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"loaded full task checkpoint: {args.init_checkpoint} missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    configure_train_mode(model, args.train_mode, args.train_input_adapter, args.train_head, args.train_task_context)
    summary = trainable_summary(model)
    print(f"loaded datasets: train={len(train_loader.dataset)} val={len(val_loader.dataset)} classes={classes}", flush=True)
    print(f"AudioAssemblerModel params={summary} mode={args.train_mode} device={device} amp={args.amp}", flush=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable parameters; change train mode")
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda") and dtype == torch.float16)

    cfg = model.assembler_core.cfg
    flow_targets = make_runtime_flow_targets(args, cfg, torch.device(device))
    skill_weights = {
        "read_flow_kl": args.w_read,
        "primitive_slot_kl": args.w_primitive,
        "slot_transition_kl": args.w_slot_transition,
        "primitive_transition_kl": args.w_primitive_transition,
        "slot_composition_kl": args.w_composition,
        "write_flow_kl": args.w_write,
    }

    fields = [
        "epoch", "train_loss", "train_ce", "train_acc", "val_loss", "val_acc", "best_acc",
        "skill", "read_flow_kl", "primitive_slot_kl", "slot_transition_kl", "primitive_transition_kl", "slot_composition_kl", "write_flow_kl",
        "write_budget", "update_alive", "class_read_div", "class_slot_prior", "class_attn_entropy", "phase_balance", "slot_div", "layer_sim", "step_sim",
        "primitive_balance", "cell_balance", "entropy_band", "step_alive_budget",
        "role_usage_balance", "role_entropy_band", "role_similarity", "role_effect_similarity", "role_effect_sim_value", "role_usage_max", "role_entropy",
        "late_input_shortcut", "input_proxy_read",
        "semantic_lint", "semantic_no_read", "semantic_no_transform", "semantic_memory_global", "semantic_dead_slot",
        "semantic_write_consumer", "semantic_early_write", "semantic_transition",
        "logit_norm",
        "pair_update_norm", "class_write",
    ]
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    best, best_epoch = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        tr = train_epoch(model, train_loader, opt, scaler, device, dtype, args, epoch, flow_targets, skill_weights)
        va = evaluate(model, val_loader, device, dtype, args, flow_targets, skill_weights)
        if va["acc"] > best:
            best, best_epoch = va["acc"], epoch
            torch.save({
                "model": model.state_dict(),
                "assembler_core": model.assembler_core.state_dict(),
                "args": vars(args),
                "classes": classes,
                "epoch": epoch,
                "best_acc": best,
                "trainable_summary": summary,
                "assembler_config": cfg.__dict__,
            }, out_dir / "best.pt")
        torch.save({
            "model": model.state_dict(),
            "assembler_core": model.assembler_core.state_dict(),
            "args": vars(args),
            "classes": classes,
            "epoch": epoch,
            "best_acc": best,
            "trainable_summary": summary,
            "assembler_config": cfg.__dict__,
        }, out_dir / "last.pt")
        row = {k: 0.0 for k in fields}
        row.update({
            "epoch": epoch,
            "train_loss": tr["loss"],
            "train_ce": tr["ce"],
            "train_acc": tr["acc"],
            "val_loss": va["loss"],
            "val_acc": va["acc"],
            "best_acc": best,
        })
        for k in row.keys():
            if k in tr:
                row[k] = tr[k]
        with (out_dir / "metrics.csv").open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        write_json(out_dir / f"analysis_epoch_{epoch:03d}.json", {
            "epoch": epoch,
            "train": tr,
            "val": va,
            "best_acc": best,
            "best_epoch": best_epoch,
            "classes": classes,
            "train_counts": train_counts,
            "val_counts": val_counts,
            "trainable_summary": summary,
        })
        print(f"epoch {epoch:03d}/{args.epochs} train={tr['loss']:.4f}/{100*tr['acc']:.2f}% val={va['loss']:.4f}/{100*va['acc']:.2f}% best={100*best:.2f}%@{best_epoch} skill={tr.get('skill',0.0):.4f}", flush=True)

    write_json(out_dir / "final_report.json", {
        "best_acc": best,
        "best_epoch": best_epoch,
        "args": vars(args),
        "classes": classes,
        "trainable_summary": summary,
        "assembler_config": cfg.__dict__,
    })


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
    p.add_argument("--primitive-slots", type=int, default=4)
    p.add_argument("--memory-cells", type=int, default=4)
    p.add_argument("--global-cells", type=int, default=2)
    p.add_argument("--channel-stages", type=int, default=3)
    p.add_argument("--operator-v2", action="store_true")
    p.add_argument("--step-alive-init", type=float, default=1.65)
    p.add_argument("--latent-roles", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--role-count", type=int, default=6)
    p.add_argument("--role-temperature", type=float, default=1.25)
    p.add_argument("--role-init-std", type=float, default=0.02)
    p.add_argument("--role-effect-init-std", type=float, default=0.04)
    p.add_argument("--role-bias-scale", type=float, default=1.0)
    p.add_argument("--pair-slots", type=int, default=12)
    p.add_argument("--dropout", type=float, default=0.04)
    p.add_argument("--head-dropout", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.012)
    p.add_argument("--grad-clip", type=float, default=0.75)
    p.add_argument("--write-target", type=float, default=0.44)
    p.add_argument("--min-update-norm", type=float, default=0.20)
    p.add_argument("--phase-prior-strength", type=float, default=0.85)
    p.add_argument("--min-early-phase-mass", type=float, default=0.07)
    p.add_argument("--max-aggregate-phase-mass", type=float, default=0.42)
    p.add_argument("--lambda-skill", type=float, default=0.0)
    p.add_argument("--lambda-write-budget", type=float, default=0.025)
    p.add_argument("--lambda-update-alive", type=float, default=0.005)
    p.add_argument("--lambda-class-read-div", type=float, default=0.020)
    p.add_argument("--lambda-class-slot-prior", type=float, default=0.030)
    p.add_argument("--lambda-class-attn-entropy", type=float, default=0.003)
    p.add_argument("--lambda-phase-balance", type=float, default=0.045)
    p.add_argument("--class-slot-prior-sigma", type=float, default=1.45)
    p.add_argument("--lambda-slot-div", type=float, default=0.002)
    p.add_argument("--lambda-layer-sim", type=float, default=0.000)
    p.add_argument("--lambda-step-sim", type=float, default=0.000)
    p.add_argument("--layer-sim-margin", type=float, default=0.25)
    p.add_argument("--step-sim-margin", type=float, default=0.45)
    p.add_argument("--lambda-primitive-balance", type=float, default=0.000)
    p.add_argument("--lambda-cell-balance", type=float, default=0.000)
    p.add_argument("--lambda-entropy-band", type=float, default=0.000)
    p.add_argument("--lambda-step-alive-budget", type=float, default=0.000)
    p.add_argument("--lambda-role-usage-balance", type=float, default=0.000)
    p.add_argument("--lambda-role-entropy-band", type=float, default=0.000)
    p.add_argument("--lambda-role-similarity", type=float, default=0.000)
    p.add_argument("--lambda-role-effect-similarity", type=float, default=0.000)
    p.add_argument("--lambda-late-input-shortcut", type=float, default=0.000)
    p.add_argument("--lambda-semantic-lint", type=float, default=0.000)
    p.add_argument("--primitive-balance-entropy-floor", type=float, default=0.72)
    p.add_argument("--cell-balance-entropy-floor", type=float, default=0.62)
    p.add_argument("--entropy-band-low", type=float, default=1.05)
    p.add_argument("--entropy-band-high", type=float, default=2.35)
    p.add_argument("--step-alive-target", type=float, default=0.78)
    p.add_argument("--role-usage-entropy-floor", type=float, default=0.72)
    p.add_argument("--role-entropy-low", type=float, default=0.35)
    p.add_argument("--role-entropy-high", type=float, default=0.92)
    p.add_argument("--role-similarity-margin", type=float, default=0.78)
    p.add_argument("--role-effect-similarity-margin", type=float, default=0.15)
    p.add_argument("--late-input-start-layer", type=int, default=1)
    p.add_argument("--late-input-target", type=float, default=0.55)
    p.add_argument("--semantic-min-read-focus", type=float, default=0.04)
    p.add_argument("--semantic-min-transform-focus", type=float, default=0.06)
    p.add_argument("--semantic-min-memory-global", type=float, default=0.03)
    p.add_argument("--semantic-dead-slot-frac", type=float, default=0.35)
    p.add_argument("--lambda-logit-norm", type=float, default=0.0007)
    p.add_argument("--w-read", type=float, default=0.25)
    p.add_argument("--w-primitive", type=float, default=0.30)
    p.add_argument("--w-slot-transition", type=float, default=0.20)
    p.add_argument("--w-primitive-transition", type=float, default=0.25)
    p.add_argument("--w-composition", type=float, default=0.15)
    p.add_argument("--w-write", type=float, default=0.25)
    p.add_argument("--amp", choices=["fp16", "bf16", "fp32", "off"], default="fp32")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--max-val-batches", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--out-dir", default="./matrix_program_core/runs/audio_assembler")
    p.add_argument("--assembler-checkpoint", default="")
    p.add_argument("--init-checkpoint", default="")
    p.add_argument("--skill-target-pack", default="")
    p.add_argument("--use-prior-skill-targets", action="store_true")
    p.add_argument("--train-mode", choices=["freeze_core", "editor_delta", "delta", "full"], default="delta")
    p.add_argument("--task-context-tokens", type=int, default=4)
    p.add_argument("--head-context-tokens", type=int, default=10)
    p.add_argument("--use-head-context", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use-mechanism-context", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--mechanism-task-id", type=int, default=4)
    p.add_argument("--train-input-adapter", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--train-head", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--train-task-context", action=argparse.BooleanOptionalAction, default=True)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
