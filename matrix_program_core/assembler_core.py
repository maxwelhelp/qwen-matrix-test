#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Factorized Matrix Program Assembler Core.

This is the corrected core for transferable matrix-program assembly.

It is NOT a W->labels decoder. It is the same trainable matrix program core that
synthetic pretrain and task transfer should both use.

No hard router, no top-k, no argmax path selection. All choices are dense
soft matrices:

    read_flow              [L,S,B,K,A]
    primitive_slot_flow    [L,S,B,K,P]
    slot_transition_flow   [L,S,B,K,K]
    primitive_transition   [L,S,P,P]
    slot_composition_flow  [L,S,B,K]
    write_flow             [L,S,B,A]

where:
    L = layers
    S = steps per layer
    B = blocks
    K = primitive slots per block-step
    P = primitive types
    A = address cells = state cells + memory cells + global cells

The expensive full tensor [K,K,P,P,A,...] is factorized into small matrices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from simple_butterfly_matrix.simple_butterfly_matrix import BlockButterfly, ChannelButterfly, PRIMITIVES


PHASES = ("extract", "compare", "suppress", "aggregate")


@dataclass
class AssemblerConfig:
    dim: int = 96
    evidence_cells: int = 48
    layers: int = 4
    blocks: int = 4
    steps: int = 2
    primitive_slots: int = 4
    memory_cells: int = 4
    global_cells: int = 2
    channel_stages: int = 3
    dropout: float = 0.04
    use_deltas: bool = True
    phase_prior_strength: float = 0.0
    operator_v2: bool = False
    step_alive_init: float = 1.65
    latent_roles: bool = False
    role_count: int = 6
    role_temperature: float = 1.25
    role_init_std: float = 0.02
    role_effect_init_std: float = 0.04
    role_bias_scale: float = 1.0

    @property
    def address_cells(self) -> int:
        return int(self.blocks + self.memory_cells + self.global_cells)


@dataclass
class AssemblerAux:
    cells: torch.Tensor
    slots: torch.Tensor
    read_flow: torch.Tensor
    primitive_slot_flow: torch.Tensor
    slot_transition_flow: torch.Tensor
    primitive_transition_flow: torch.Tensor
    slot_composition_flow: torch.Tensor
    write_flow: torch.Tensor
    write_gates: torch.Tensor
    update_norms: torch.Tensor
    memory_usage: torch.Tensor
    global_usage: torch.Tensor
    entropies: Dict[str, torch.Tensor]
    cell_names: List[str]
    slot_names: List[str]
    role_mix: torch.Tensor | None = None
    role_usage: torch.Tensor | None = None
    role_entropy: torch.Tensor | None = None
    role_similarity: torch.Tensor | None = None
    role_effect_similarity: torch.Tensor | None = None
    step_alive: torch.Tensor | None = None
    input_proxy_read: torch.Tensor | None = None
    read_group_mass: torch.Tensor | None = None


def _entropy(p: torch.Tensor, dim: int = -1) -> torch.Tensor:
    p = p.float().clamp_min(1e-8)
    p = p / p.sum(dim=dim, keepdim=True).clamp_min(1e-8)
    return -(p * p.log()).sum(dim=dim)


class MatrixButterflyAttention(nn.Module):
    """Factorized matrix attention over address cells.

    This is intentionally inside the assembler core, not an external adapter.
    It lets state/memory/global cells exchange information before read,
    primitive, transition, composition, and write matrices are chosen.
    """

    def __init__(self, dim: int, address_cells: int, heads: int = 4, rank: int = 8, dropout: float = 0.04):
        super().__init__()
        self.D = int(dim)
        self.A = int(address_cells)
        self.H = max(1, int(heads))
        self.R = max(4, int(rank))
        self.inner = self.H * self.R
        self.q = nn.Linear(self.D, self.inner, bias=False)
        self.k = nn.Linear(self.D, self.inner, bias=False)
        self.v = nn.Linear(self.D, self.inner, bias=False)
        self.out = nn.Linear(self.inner, self.D, bias=False)
        self.out_delta = nn.Linear(self.inner, self.D, bias=False)
        self.addr_left = nn.Parameter(torch.randn(self.H, self.A, self.R) * 0.02)
        self.addr_right = nn.Parameter(torch.randn(self.H, self.R, self.A) * 0.02)
        self.score_delta = nn.Parameter(torch.zeros(self.H, self.A, self.A))
        self.channel = ChannelButterfly(self.D, 2)
        self.gate = nn.Linear(self.D, self.D)
        self.norm = nn.LayerNorm(self.D)
        self.drop = nn.Dropout(dropout)
        nn.init.zeros_(self.out_delta.weight)
        nn.init.constant_(self.gate.bias, -2.2)

    def forward(self, cells: torch.Tensor, use_deltas: bool = True) -> torch.Tensor:
        N, A, D = cells.shape
        q = self.q(cells).view(N, A, self.H, self.R).transpose(1, 2)
        k = self.k(cells).view(N, A, self.H, self.R).transpose(1, 2)
        v = self.v(cells).view(N, A, self.H, self.R).transpose(1, 2)
        score = torch.einsum("nhar,nhbr->nhab", q, k) / math.sqrt(self.R)
        addr = torch.matmul(self.addr_left, self.addr_right) / math.sqrt(self.R)
        score = score + addr.to(device=cells.device, dtype=score.dtype).unsqueeze(0)
        if use_deltas:
            score = score + self.score_delta.to(device=cells.device, dtype=score.dtype).unsqueeze(0)
        attn = torch.softmax(score.float(), dim=-1).to(cells.dtype)
        mixed = torch.einsum("nhab,nhbr->nhar", attn, v).transpose(1, 2).reshape(N, A, self.inner)
        upd = self.out(mixed)
        if use_deltas:
            upd = upd + self.out_delta(mixed)
        upd = self.channel(upd)
        gate = torch.sigmoid(self.gate(cells.float())).to(cells.dtype)
        return self.norm(cells + gate * self.drop(upd.to(cells.dtype)))


class ProjectedVariantEditor(nn.Module):
    """Parallel low-rank edit variants with differentiable quality blending."""

    def __init__(self, dim: int, out_dim: int, variants: int = 4, rank: int = 8, max_scale: float = 0.20):
        super().__init__()
        self.D = int(dim)
        self.O = int(out_dim)
        self.V = max(2, int(variants))
        self.R = max(4, int(rank))
        self.max_scale = float(max_scale)
        self.quality = nn.Linear(self.D, self.V, bias=True)
        self.down = nn.Linear(self.D, self.V * self.R, bias=False)
        self.up = nn.Parameter(torch.zeros(self.V, self.R, self.O))
        self.up_delta = nn.Parameter(torch.zeros(self.V, self.R, self.O))
        self.gate = nn.Linear(self.D, 1, bias=True)
        self.gate_delta = nn.Linear(self.D, 1, bias=False)
        nn.init.zeros_(self.quality.weight)
        nn.init.zeros_(self.quality.bias)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.gate.bias, -3.0)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate_delta.weight)

    def forward(self, token: torch.Tensor, use_deltas: bool = True) -> torch.Tensor:
        token_f = token.float()
        quality = torch.softmax(self.quality(token_f), dim=-1)
        z = self.down(token_f).view(token.shape[0], self.V, self.R)
        up = self.up
        if use_deltas:
            up = up + self.up_delta
        variants = torch.einsum("nvr,vro->nvo", z, up.float())
        edit = torch.einsum("nv,nvo->no", quality, variants)
        gate_logits = self.gate(token_f)
        if use_deltas:
            gate_logits = gate_logits + self.gate_delta(token_f)
        gate = self.max_scale * torch.sigmoid(gate_logits)
        return (gate * edit).to(token.dtype)


class FlowEditAttention(nn.Module):
    """Small differentiable editor for flow logits.

    It does not choose a path. It proposes bounded residual edits for every
    read/primitive/operator/write matrix before softmax. The base program stays
    intact; the editor can softly add missing mass or move mass away from weak
    choices when gradients indicate that is useful.
    """

    def __init__(self, dim: int, blocks: int, slots: int, address_cells: int, primitive_count: int, rank: int = 16, dropout: float = 0.04):
        super().__init__()
        self.D = int(dim)
        self.B = int(blocks)
        self.K = int(slots)
        self.A = int(address_cells)
        self.P = int(primitive_count)
        self.M = 6
        self.R = max(8, int(rank))
        self.mechanism_embed = nn.Parameter(torch.randn(self.M, self.D) * 0.02)
        self.ctx = nn.Linear(self.D * 3, self.D)
        self.q = nn.Linear(self.D, self.R, bias=False)
        self.k = nn.Linear(self.D, self.R, bias=False)
        self.v = nn.Linear(self.D, self.R, bias=False)
        self.o = nn.Linear(self.R, self.D, bias=False)
        self.relation_left = nn.Parameter(torch.randn(self.M, self.R) * 0.02)
        self.relation_right = nn.Parameter(torch.randn(self.R, self.M) * 0.02)
        self.scale_logit = nn.Parameter(torch.full((self.M,), -3.0))
        self.scale_delta = nn.Parameter(torch.zeros(self.M))
        self.out = nn.ModuleDict({
            "read": nn.Linear(self.D, self.B * self.K * self.A, bias=False),
            "primitive": nn.Linear(self.D, self.B * self.K * self.P, bias=False),
            "slot_transition": nn.Linear(self.D, self.B * self.K * self.K, bias=False),
            "primitive_transition": nn.Linear(self.D, self.P * self.P, bias=False),
            "composition": nn.Linear(self.D, self.B * self.K, bias=False),
            "write": nn.Linear(self.D, self.B * self.A, bias=False),
        })
        self.out_delta = nn.ModuleDict({
            "read": nn.Linear(self.D, self.B * self.K * self.A, bias=False),
            "primitive": nn.Linear(self.D, self.B * self.K * self.P, bias=False),
            "slot_transition": nn.Linear(self.D, self.B * self.K * self.K, bias=False),
            "primitive_transition": nn.Linear(self.D, self.P * self.P, bias=False),
            "composition": nn.Linear(self.D, self.B * self.K, bias=False),
            "write": nn.Linear(self.D, self.B * self.A, bias=False),
        })
        self.variant_editors = nn.ModuleDict({
            "primitive": ProjectedVariantEditor(self.D, self.B * self.K * self.P, variants=5, rank=max(4, self.R // 2), max_scale=0.18),
            "slot_transition": ProjectedVariantEditor(self.D, self.B * self.K * self.K, variants=4, rank=max(4, self.R // 2), max_scale=0.14),
            "primitive_transition": ProjectedVariantEditor(self.D, self.P * self.P, variants=5, rank=max(4, self.R // 2), max_scale=0.18),
        })
        self.norm = nn.LayerNorm(self.D)
        self.drop = nn.Dropout(dropout)
        for mod in list(self.out.values()) + list(self.out_delta.values()):
            nn.init.zeros_(mod.weight)

    def _context(self, cells: torch.Tensor) -> torch.Tensor:
        state = cells[:, : self.B].mean(dim=1)
        memory = cells[:, self.B:].mean(dim=1) if cells.shape[1] > self.B else torch.zeros_like(state)
        global_ctx = cells.mean(dim=1)
        return self.ctx(torch.cat([state, memory, global_ctx], dim=-1).float()).to(cells.dtype)

    def forward(self, cells: torch.Tensor, use_deltas: bool = True):
        N = int(cells.shape[0])
        base = self._context(cells).unsqueeze(1) + self.mechanism_embed.to(device=cells.device, dtype=cells.dtype).unsqueeze(0)
        q = self.q(base.float())
        k = self.k(base.float())
        v = self.v(base.float())
        score = torch.einsum("bir,bjr->bij", q, k) / math.sqrt(self.R)
        relation = torch.matmul(self.relation_left, self.relation_right) / math.sqrt(self.R)
        score = score + relation.to(device=cells.device, dtype=score.dtype).unsqueeze(0)
        attn = torch.softmax(score, dim=-1).to(cells.dtype)
        mix = torch.einsum("bij,bjr->bir", attn, v.to(cells.dtype))
        token = self.norm(base + self.drop(self.o(mix.float()).to(cells.dtype)))
        scale_logits = self.scale_logit
        if use_deltas:
            scale_logits = scale_logits + self.scale_delta
        scale = (0.20 * torch.sigmoid(scale_logits.float())).to(device=cells.device, dtype=cells.dtype).view(1, self.M, 1)
        token = token * scale

        names = ("read", "primitive", "slot_transition", "primitive_transition", "composition", "write")
        flat = []
        for i, name in enumerate(names):
            x = self.out[name](token[:, i].float())
            if use_deltas:
                x = x + self.out_delta[name](token[:, i].float())
            if name in self.variant_editors:
                x = x + self.variant_editors[name](token[:, i], use_deltas=use_deltas).float()
            flat.append(x.to(cells.dtype))
        r, p, st, pt, comp, wr = flat
        return (
            r.view(N, self.B, self.K, self.A),
            p.view(N, self.B, self.K, self.P),
            st.view(N, self.B, self.K, self.K),
            pt.view(N, self.P, self.P),
            comp.view(N, self.B, self.K),
            wr.view(N, self.B, self.A),
        )


class AssemblerStep(nn.Module):
    """One factorized matrix-program assembly step.

    The step sees all address cells through read_flow, places all primitives into
    K primitive slots, mixes slots and primitive types through factorized
    transition matrices, composes update, then writes to all address cells.
    """

    def __init__(self, cfg: AssemblerConfig, layer: int, step: int):
        super().__init__()
        self.cfg = cfg
        self.layer = int(layer)
        self.step = int(step)
        self.phase = PHASES[min(layer, len(PHASES) - 1)]
        self.D = int(cfg.dim)
        self.B = int(cfg.blocks)
        self.K = int(cfg.primitive_slots)
        self.A = int(cfg.address_cells)
        self.P = len(PRIMITIVES)

        self.channel = ChannelButterfly(self.D, cfg.channel_stages)
        self.block = BlockButterfly(self.B)
        rank = max(8, self.D // 4)
        self.low_a = nn.Parameter(torch.randn(self.D, rank) * 0.04)
        self.low_b = nn.Parameter(torch.randn(rank, self.D) * 0.04)
        self.ctx_w = nn.Parameter(torch.randn(self.D, self.D) * 0.04)
        self.gate_h = nn.Parameter(torch.randn(self.D, self.D) * 0.02)
        self.gate_c = nn.Parameter(torch.randn(self.D, self.D) * 0.02)
        self.gate_bias = nn.Parameter(torch.full((self.D,), -0.35))
        self.layer_step_key = nn.Parameter(torch.randn(self.D) * 0.02)
        # Phase prior is optional. With phase_prior_strength=0, layers do not
        # start as extract/compare/suppress/aggregate; they must specialize from
        # data + losses. With strength>0, this becomes only a scaled soft init.
        phase_strength = 0.0 if bool(getattr(cfg, "latent_roles", False)) else float(getattr(cfg, "phase_prior_strength", 0.0))
        if phase_strength <= 0.0:
            phase_logits = torch.zeros((len(PHASES),))
        else:
            phase_logits = torch.full((len(PHASES),), -0.65 * phase_strength)
            phase_logits[min(layer, len(PHASES) - 1)] = 1.25 * phase_strength
        self.phase_mix_logits = nn.Parameter(phase_logits)
        self.phase_mix_delta = nn.Parameter(torch.zeros(len(PHASES)))
        self.matrix_attention = MatrixButterflyAttention(
            self.D,
            self.A,
            heads=4,
            rank=max(8, self.D // 8),
            dropout=cfg.dropout,
        )

        # Base assembly matrices.
        self.read_logits = nn.Parameter(self._read_prior())                     # [B,K,A]
        self.primitive_slot_logits = nn.Parameter(self._primitive_slot_prior()) # [B,K,P]
        self.slot_transition_logits = nn.Parameter(self._slot_transition_prior()) # [B,K,K]
        self.primitive_transition_logits = nn.Parameter(self._primitive_transition_prior()) # [P,P]
        self.slot_composition_logits = nn.Parameter(torch.zeros(self.B, self.K)) # [B,K]
        self.write_logits = nn.Parameter(self._write_prior())                   # [B,A]
        self.write_gate_logit = nn.Parameter(torch.full((self.B,), -0.15))

        # Context-conditioned dense soft-flow. This lets the same assembler core
        # choose different valid matrix programs for different evidence/cell states
        # without discrete routing. The base projection is learned in pretrain;
        # ctx_flow_delta is LoRA-like and trainable in delta mode.
        self.flow_context_size = (
            self.B * self.K * self.A
            + self.B * self.K * self.P
            + self.B * self.K * self.K
            + self.P * self.P
            + self.B * self.K
            + self.B * self.A
        )
        self.ctx_flow = nn.Linear(self.D, self.flow_context_size, bias=False)
        self.ctx_flow_delta = nn.Linear(self.D, self.flow_context_size, bias=False)
        nn.init.zeros_(self.ctx_flow.weight)
        nn.init.zeros_(self.ctx_flow_delta.weight)
        self.flow_editor = FlowEditAttention(
            self.D,
            self.B,
            self.K,
            self.A,
            self.P,
            rank=max(8, self.D // 6),
            dropout=cfg.dropout,
        )
        self.role_flow = nn.Linear(self.D, self.flow_context_size, bias=False)
        self.role_step_alive = nn.Linear(self.D, 1, bias=False)
        self.role_operator = nn.Linear(self.D, 20, bias=False)
        role_std = float(getattr(cfg, "role_init_std", 0.02))
        nn.init.normal_(self.role_flow.weight, mean=0.0, std=role_std)
        nn.init.normal_(self.role_step_alive.weight, mean=0.0, std=role_std)
        nn.init.normal_(self.role_operator.weight, mean=0.0, std=role_std)

        # LoRA-like task deltas. In delta mode only these are trainable.
        self.read_delta = nn.Parameter(torch.zeros(self.B, self.K, self.A))
        self.primitive_slot_delta = nn.Parameter(torch.zeros(self.B, self.K, self.P))
        self.slot_transition_delta = nn.Parameter(torch.zeros(self.B, self.K, self.K))
        self.primitive_transition_delta = nn.Parameter(torch.zeros(self.P, self.P))
        self.slot_composition_delta = nn.Parameter(torch.zeros(self.B, self.K))
        self.write_delta = nn.Parameter(torch.zeros(self.B, self.A))

        self.drop = nn.Dropout(cfg.dropout)
        self.norm = nn.LayerNorm(self.D)

        # OperatorBankV2 keeps the external primitive count P unchanged for
        # checkpoint/dataset compatibility, but each primitive becomes a soft
        # family of fast matrix operators: low-rank sizes, butterfly depths,
        # local Toeplitz-like smooth/diff, Haar wavelet-like channel transform,
        # diagonal gates, and richer phase variants.
        self.operator_v2_enabled = bool(getattr(cfg, "operator_v2", False))
        self.step_alive_logit = nn.Parameter(torch.tensor(float(getattr(cfg, "step_alive_init", 1.65))))
        self.step_alive_delta = nn.Parameter(torch.zeros(()))
        self.opv2_channel_logits = nn.Parameter(torch.tensor([1.20, -0.15, -0.35]))
        self.opv2_block_logits = nn.Parameter(torch.tensor([1.10, -0.10, -0.35]))
        self.opv2_lowrank_logits = nn.Parameter(torch.tensor([0.20, 0.70, 0.35]))
        self.opv2_ctx_logits = nn.Parameter(torch.tensor([1.00, -0.10, -0.25, -0.35]))
        self.opv2_product_logits = nn.Parameter(torch.tensor([0.90, -0.05, -0.25]))
        phase_extra_init = torch.zeros(4) if bool(getattr(cfg, "latent_roles", False)) else torch.tensor([-0.10, -0.20, -0.25, -0.35])
        self.phase_extra_logits = nn.Parameter(phase_extra_init)
        self.opv2_channel_delta = nn.Parameter(torch.zeros(3))
        self.opv2_block_delta = nn.Parameter(torch.zeros(3))
        self.opv2_lowrank_delta = nn.Parameter(torch.zeros(3))
        self.opv2_ctx_delta = nn.Parameter(torch.zeros(4))
        self.opv2_product_delta = nn.Parameter(torch.zeros(3))
        self.phase_extra_delta = nn.Parameter(torch.zeros(4))
        self.diag_gate = nn.Parameter(torch.zeros(self.D))
        self.diag_bias = nn.Parameter(torch.zeros(self.D))

    def _state_idx(self, b: int) -> int:
        return b

    def _memory_start(self) -> int:
        return self.B

    def _global_start(self) -> int:
        return self.B + int(self.cfg.memory_cells)

    def _read_prior(self) -> torch.Tensor:
        x = torch.zeros(self.B, self.K, self.A)
        mem0 = self._memory_start()
        glob0 = self._global_start()
        for b in range(self.B):
            for k in range(self.K):
                x[b, k, self._state_idx(b)] = 1.2
                if self.cfg.memory_cells > 0:
                    x[b, k, mem0 + (b + k) % self.cfg.memory_cells] = 0.45
                if self.cfg.global_cells > 0:
                    x[b, k, glob0 + k % self.cfg.global_cells] = 0.35
                if b > 0:
                    x[b, k, self._state_idx(b - 1)] = 0.20
                if b + 1 < self.B:
                    x[b, k, self._state_idx(b + 1)] = 0.20
        return x + 0.01 * torch.randn_like(x)

    def _write_prior(self) -> torch.Tensor:
        x = torch.full((self.B, self.A), -0.25)
        mem0 = self._memory_start()
        glob0 = self._global_start()
        for b in range(self.B):
            x[b, self._state_idx(b)] = 1.20
            if self.cfg.memory_cells > 0:
                x[b, mem0 + b % self.cfg.memory_cells] = 0.40
            if self.cfg.global_cells > 0:
                x[b, glob0 + b % self.cfg.global_cells] = 0.30
        return x + 0.01 * torch.randn_like(x)

    def _primitive_slot_prior(self) -> torch.Tensor:
        x = torch.zeros(self.B, self.K, self.P)
        phase_strength = 0.0 if bool(getattr(self.cfg, "latent_roles", False)) else float(getattr(self.cfg, "phase_prior_strength", 0.0))
        if phase_strength <= 0.0:
            # No role prior: all primitive families start equal. Tiny noise only
            # breaks exact symmetry; layer specialization must emerge from data.
            return x + 0.01 * torch.randn_like(x)
        idx = {name: i for i, name in enumerate(PRIMITIVES)}
        phase_sets = {
            "extract": ["ctx_matrix", "channel_butterfly", "phase_matrix"],
            "compare": ["low_rank", "product_gate", "phase_matrix"],
            "suppress": ["block_butterfly", "product_gate", "phase_matrix"],
            "aggregate": ["block_butterfly", "channel_butterfly", "phase_matrix"],
        }
        names = phase_sets.get(self.phase, list(PRIMITIVES))
        for b in range(self.B):
            for k in range(self.K):
                for j, name in enumerate(names):
                    if name in idx:
                        x[b, k, idx[name]] += phase_strength * 0.65 / (1 + abs(k - j))
                x[b, k, :] += 0.03
        return x + 0.01 * torch.randn_like(x)

    def _slot_transition_prior(self) -> torch.Tensor:
        x = torch.eye(self.K).view(1, self.K, self.K).repeat(self.B, 1, 1) * 0.75
        for b in range(self.B):
            for k in range(self.K):
                x[b, k, (k - 1) % self.K] += 0.20
                x[b, k, (k + 1) % self.K] += 0.15
        return x + 0.01 * torch.randn_like(x)

    def _primitive_transition_prior(self) -> torch.Tensor:
        phase_strength = 0.0 if bool(getattr(self.cfg, "latent_roles", False)) else float(getattr(self.cfg, "phase_prior_strength", 0.0))
        if phase_strength <= 0.0:
            # No role prior: transitions start near uniform logits with tiny noise.
            return torch.zeros(self.P, self.P) + 0.01 * torch.randn(self.P, self.P)
        x = torch.eye(self.P) * (0.35 * phase_strength)
        idx = {name: i for i, name in enumerate(PRIMITIVES)}
        def link(a: str, b: str, v: float) -> None:
            if a in idx and b in idx:
                x[idx[a], idx[b]] = v * phase_strength
        if self.phase == "extract":
            link("ctx_matrix", "channel_butterfly", 0.75)
            link("channel_butterfly", "phase_matrix", 0.60)
        elif self.phase == "compare":
            link("low_rank", "product_gate", 0.75)
            link("ctx_matrix", "phase_matrix", 0.55)
        elif self.phase == "suppress":
            link("block_butterfly", "phase_matrix", 0.80)
            link("product_gate", "phase_matrix", 0.60)
        elif self.phase == "aggregate":
            link("phase_matrix", "block_butterfly", 0.75)
            link("block_butterfly", "channel_butterfly", 0.60)
        return x + 0.01 * torch.randn_like(x)

    def _eff(self, base: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return base + delta if self.cfg.use_deltas else base

    def _context_flow_bias(self, cells: torch.Tensor, role_context: torch.Tensor | None = None):
        # cells: [N,A,D] -> per-example additive logits for all flow matrices.
        state_ctx = cells[:, : self.B].mean(dim=1)
        mem0 = self.B
        glob0 = self.B + int(self.cfg.memory_cells)
        if int(self.cfg.memory_cells) > 0:
            memory_ctx = cells[:, mem0:glob0].mean(dim=1)
        else:
            memory_ctx = torch.zeros_like(state_ctx)
        if int(self.cfg.global_cells) > 0:
            global_ctx = cells[:, glob0:].mean(dim=1)
        else:
            global_ctx = torch.zeros_like(state_ctx)
        # Explicit layer/step key tells the assembler where it is in the program grid.
        ctx = (state_ctx + 0.7 * memory_ctx + 0.7 * global_ctx + self.layer_step_key.to(device=cells.device, dtype=cells.dtype).view(1, -1)).float()
        flat = self.ctx_flow(ctx)
        if self.cfg.use_deltas:
            flat = flat + self.ctx_flow_delta(ctx)
        if bool(getattr(self.cfg, "latent_roles", False)) and role_context is not None:
            role = role_context.to(device=cells.device, dtype=torch.float32).view(1, -1)
            flat = flat + self.role_flow(role).expand_as(flat)
        flat = flat.to(device=cells.device, dtype=cells.dtype)
        sizes = [
            self.B * self.K * self.A,
            self.B * self.K * self.P,
            self.B * self.K * self.K,
            self.P * self.P,
            self.B * self.K,
            self.B * self.A,
        ]
        r, ps, st, pt, comp, wr = torch.split(flat, sizes, dim=-1)
        return (
            r.view(cells.shape[0], self.B, self.K, self.A),
            ps.view(cells.shape[0], self.B, self.K, self.P),
            st.view(cells.shape[0], self.B, self.K, self.K),
            pt.view(cells.shape[0], self.P, self.P),
            comp.view(cells.shape[0], self.B, self.K),
            wr.view(cells.shape[0], self.B, self.A),
        )

    def _opv2_weights(self, logits: torch.Tensor, delta: torch.Tensor | None = None, bias: torch.Tensor | None = None, dtype=None, device=None) -> torch.Tensor:
        z = logits
        if self.cfg.use_deltas and delta is not None:
            z = z + delta
        if bias is not None:
            z = z + bias.to(device=z.device, dtype=z.dtype)
        z = z.float()
        w = torch.softmax(z, dim=-1)
        if device is not None:
            w = w.to(device=device)
        if dtype is not None:
            w = w.to(dtype=dtype)
        return w

    def _local_smooth_flat(self, x: torch.Tensor) -> torch.Tensor:
        return 0.50 * x + 0.25 * torch.roll(x, 1, dims=-1) + 0.25 * torch.roll(x, -1, dims=-1)

    def _local_diff_flat(self, x: torch.Tensor) -> torch.Tensor:
        return x - self._local_smooth_flat(x)

    def _haar_flat(self, x: torch.Tensor) -> torch.Tensor:
        # Fast orthogonal-ish Haar step over channel pairs. Shape is preserved.
        D = x.shape[-1]
        if D < 2:
            return x
        even = x[..., 0::2]
        odd = x[..., 1::2]
        m = min(even.shape[-1], odd.shape[-1])
        avg = (even[..., :m] + odd[..., :m]) * 0.70710678
        dif = (even[..., :m] - odd[..., :m]) * 0.70710678
        y = x.clone()
        y[..., 0:2*m:2] = avg
        y[..., 1:2*m:2] = dif
        return y

    def _mix_last(self, variants: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # variants [...,V,D], weights [V]
        return torch.einsum("v,...vd->...d", weights, variants)

    def _operator_role_biases(
        self,
        role_context: torch.Tensor | None,
        device: torch.device,
        dtype: torch.dtype,
        shared_operator_bias: torch.Tensor | None = None,
    ):
        if not bool(getattr(self.cfg, "latent_roles", False)):
            return (None, None, None, None, None, None)
        if role_context is None:
            if shared_operator_bias is None:
                return (None, None, None, None, None, None)
            raw = shared_operator_bias.to(device=device, dtype=torch.float32)
        else:
            raw = self.role_operator(role_context.to(device=device, dtype=torch.float32).view(1, -1)).squeeze(0)
            if shared_operator_bias is not None:
                raw = raw + shared_operator_bias.to(device=device, dtype=torch.float32)
        raw = raw.to(device=device, dtype=dtype)
        return torch.split(raw, [3, 3, 3, 4, 3, 4], dim=0)

    def _primitive_outputs(
        self,
        read_ctx: torch.Tensor,
        role_context: torch.Tensor | None = None,
        role_operator_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # read_ctx: [N,B,K,D] -> [N,B,K,P,D]
        N, B, K, D = read_ctx.shape
        flat = read_ctx.reshape(N * B * K, D)
        dev, dtype = read_ctx.device, read_ctx.dtype
        ch_bias, bl_bias, low_bias, ctx_bias, pr_bias, phase_bias = self._operator_role_biases(
            role_context,
            dev,
            dtype,
            shared_operator_bias=role_operator_bias,
        )

        ctx_lin_flat = flat @ self.ctx_w.to(device=dev, dtype=dtype)
        smooth_flat = self._local_smooth_flat(flat)
        diff_flat = self._local_diff_flat(flat)
        haar_flat = self._haar_flat(flat)
        diag_flat = flat * torch.sigmoid(self.diag_gate.to(device=dev, dtype=dtype).view(1, -1) + self.diag_bias.to(device=dev, dtype=dtype).view(1, -1))

        # Low-rank size selection from slices of the max-rank factorization.
        rank_total = int(self.low_a.shape[1])
        r1 = max(4, rank_total // 4)
        r2 = max(r1, rank_total // 2)
        lows = []
        for r in (r1, r2, rank_total):
            la = self.low_a[:, :r].to(device=dev, dtype=dtype)
            lb = self.low_b[:r, :].to(device=dev, dtype=dtype)
            lows.append((flat @ la) @ lb)
        low_w = self._opv2_weights(self.opv2_lowrank_logits, self.opv2_lowrank_delta, bias=low_bias, dtype=dtype, device=dev)
        low = self._mix_last(torch.stack(lows, dim=1), low_w).view(N, B, K, D)

        # Context primitive becomes a selectable family: dense ctx, local smooth,
        # local diff, diagonal gate. All are linear/diagonal/Toeplitz-like fast ops.
        ctx_w = self._opv2_weights(self.opv2_ctx_logits, self.opv2_ctx_delta, bias=ctx_bias, dtype=dtype, device=dev)
        ctx_m = self._mix_last(torch.stack([ctx_lin_flat, smooth_flat, diff_flat, diag_flat], dim=1), ctx_w).view(N, B, K, D)

        # Channel primitive variants: one butterfly pass, two passes, Haar wavelet-like pass.
        ch1_flat = self.channel((flat + ctx_lin_flat).view(N * B * K, 1, D)).view(N * B * K, D)
        ch2_flat = self.channel((ch1_flat + 0.35 * ctx_lin_flat).view(N * B * K, 1, D)).view(N * B * K, D)
        ch_w = self._opv2_weights(self.opv2_channel_logits, self.opv2_channel_delta, bias=ch_bias, dtype=dtype, device=dev)
        channel = self._mix_last(torch.stack([ch1_flat, ch2_flat, haar_flat], dim=1), ch_w).view(N, B, K, D)

        # Block primitive variants: one block butterfly, two block butterfly passes,
        # and global block mean injection.
        block_in = read_ctx.permute(0, 2, 1, 3).reshape(N * K, B, D)
        block1 = self.block(block_in).reshape(N, K, B, D).permute(0, 2, 1, 3)
        block2_in = block1.permute(0, 2, 1, 3).reshape(N * K, B, D)
        block2 = self.block(block2_in).reshape(N, K, B, D).permute(0, 2, 1, 3)
        block_mean = read_ctx.mean(dim=1, keepdim=True).expand_as(read_ctx)
        bl_w = self._opv2_weights(self.opv2_block_logits, self.opv2_block_delta, bias=bl_bias, dtype=dtype, device=dev)
        block = self._mix_last(torch.stack([block1, block2, block_mean], dim=3), bl_w)

        # Product/gate primitive variants.
        product0 = read_ctx * torch.tanh(ctx_m)
        product1 = read_ctx * torch.sigmoid(ctx_m)
        product2 = diff_flat.view(N, B, K, D) * torch.sigmoid(ctx_m)
        pr_w = self._opv2_weights(self.opv2_product_logits, self.opv2_product_delta, bias=pr_bias, dtype=dtype, device=dev)
        product = self._mix_last(torch.stack([product0, product1, product2], dim=3), pr_w)

        gate = torch.sigmoid(
            read_ctx @ self.gate_h.to(device=dev, dtype=dtype)
            + ctx_m @ self.gate_c.to(device=dev, dtype=dtype)
            + self.gate_bias.to(device=dev, dtype=dtype)
        )
        smooth = smooth_flat.view(N, B, K, D)
        diff = diff_flat.view(N, B, K, D)
        haar = haar_flat.view(N, B, K, D)
        phase_candidates = torch.stack([
            channel + 0.50 * ctx_m,
            channel - low,
            -gate * block.mean(dim=1, keepdim=True),
            block + read_ctx.mean(dim=1, keepdim=True),
            smooth + 0.35 * ctx_m,
            diff + 0.25 * product,
            haar + 0.25 * channel,
            low + product,
        ], dim=3)
        phase_logits = torch.cat([self.phase_mix_logits, self.phase_extra_logits], dim=0)
        if self.cfg.use_deltas:
            phase_logits = phase_logits + torch.cat([self.phase_mix_delta, self.phase_extra_delta], dim=0)
        if phase_bias is not None:
            phase_logits = phase_logits + torch.cat([torch.zeros_like(self.phase_mix_logits), phase_bias.to(device=dev, dtype=phase_logits.dtype)], dim=0)
        phase_w = torch.softmax(phase_logits.float(), dim=-1).to(dtype)
        phase = torch.einsum("f,nbkfd->nbkd", phase_w, phase_candidates)
        return torch.stack([channel, block, low, ctx_m, product, phase], dim=3)

    def forward(
        self,
        cells: torch.Tensor,
        role_context: torch.Tensor | None = None,
        role_biases: Tuple[torch.Tensor, ...] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        # cells: [N,A,D]
        cells = self.matrix_attention(cells, use_deltas=self.cfg.use_deltas)
        read_logits = self._eff(self.read_logits, self.read_delta)
        prim_logits = self._eff(self.primitive_slot_logits, self.primitive_slot_delta)
        slot_trans_logits = self._eff(self.slot_transition_logits, self.slot_transition_delta)
        prim_trans_logits = self._eff(self.primitive_transition_logits, self.primitive_transition_delta)
        comp_logits = self._eff(self.slot_composition_logits, self.slot_composition_delta)
        write_logits = self._eff(self.write_logits, self.write_delta)

        read_b, prim_b, slot_b, ptrans_b, comp_b, write_b = self._context_flow_bias(cells, role_context=role_context)
        read_e, prim_e, slot_e, ptrans_e, comp_e, write_e = self.flow_editor(cells, use_deltas=self.cfg.use_deltas)
        role_alive_bias = None
        role_operator_bias = None
        if role_biases is not None:
            rb, pb, sb, ptb, cb, wb, role_alive_bias, role_operator_bias = role_biases
            read_b = read_b + rb.to(device=cells.device, dtype=read_b.dtype).view(1, self.B, self.K, self.A)
            prim_b = prim_b + pb.to(device=cells.device, dtype=prim_b.dtype).view(1, self.B, self.K, self.P)
            slot_b = slot_b + sb.to(device=cells.device, dtype=slot_b.dtype).view(1, self.B, self.K, self.K)
            ptrans_b = ptrans_b + ptb.to(device=cells.device, dtype=ptrans_b.dtype).view(1, self.P, self.P)
            comp_b = comp_b + cb.to(device=cells.device, dtype=comp_b.dtype).view(1, self.B, self.K)
            write_b = write_b + wb.to(device=cells.device, dtype=write_b.dtype).view(1, self.B, self.A)
        read_w = torch.softmax(read_logits.float().unsqueeze(0) + read_b.float() + read_e.float(), dim=-1).to(cells.dtype)        # [N,B,K,A]
        prim_w = torch.softmax(prim_logits.float().unsqueeze(0) + prim_b.float() + prim_e.float(), dim=-1).to(cells.dtype)        # [N,B,K,P]
        slot_trans = torch.softmax(slot_trans_logits.float().unsqueeze(0) + slot_b.float() + slot_e.float(), dim=-1).to(cells.dtype) # [N,B,K,K]
        prim_trans = torch.softmax(prim_trans_logits.float().unsqueeze(0) + ptrans_b.float() + ptrans_e.float(), dim=-1).to(cells.dtype) # [N,P,P]
        comp_w = torch.softmax(comp_logits.float().unsqueeze(0) + comp_b.float() + comp_e.float(), dim=-1).to(cells.dtype)        # [N,B,K]
        write_w = torch.softmax(write_logits.float().unsqueeze(0) + write_b.float() + write_e.float(), dim=-1).to(cells.dtype)      # [N,B,A]

        read_ctx = torch.einsum("nbka,nad->nbkd", read_w, cells)                   # [N,B,K,D]
        prim_out = self._primitive_outputs(
            read_ctx,
            role_context=role_context,
            role_operator_bias=role_operator_bias,
        )     # [N,B,K,P,D]

        # Factorized transitions: primitive type transition and slot transition.
        prim_mixed = torch.einsum("npq,nbkqd->nbkpd", prim_trans, prim_out)        # [N,B,K,P,D]
        slot_mixed = torch.einsum("nbkj,nbjpd->nbkpd", slot_trans, prim_mixed)     # [N,B,K,P,D]

        slot_val = torch.einsum("nbkp,nbkpd->nbkd", prim_w, slot_mixed)             # [N,B,K,D]
        update = torch.einsum("nbk,nbkd->nbd", comp_w, slot_val)                   # [N,B,D]
        update = self.norm(self.drop(update))
        step_alive_logit = self.step_alive_logit
        if self.cfg.use_deltas:
            step_alive_logit = step_alive_logit + self.step_alive_delta
        if bool(getattr(self.cfg, "latent_roles", False)) and role_context is not None:
            role_alive = self.role_step_alive(role_context.to(device=cells.device, dtype=torch.float32).view(1, -1)).squeeze()
            step_alive_logit = step_alive_logit + role_alive.to(device=cells.device, dtype=step_alive_logit.dtype)
        if role_alive_bias is not None:
            step_alive_logit = step_alive_logit + role_alive_bias.to(device=cells.device, dtype=step_alive_logit.dtype)
        step_alive = torch.sigmoid(step_alive_logit.to(device=cells.device, dtype=cells.dtype))
        active_update = step_alive * update

        write_gate = torch.sigmoid(self.write_gate_logit.to(device=cells.device, dtype=cells.dtype)).view(1, self.B, 1)
        delta_cells = torch.einsum("nba,nbd->nad", write_w, write_gate * active_update)    # [N,A,D]
        next_cells = self.norm(cells + delta_cells)

        info = {
            "read_flow": read_w.float(),
            "primitive_slot_flow": prim_w.float(),
            "slot_transition_flow": slot_trans.float(),
            "primitive_transition_flow": prim_trans.float(),
            "slot_composition_flow": comp_w.float(),
            "write_flow": write_w.float(),
            "write_gates": write_gate.detach().float().squeeze(0).squeeze(-1),
            "update_norms": active_update.detach().float().norm(dim=-1),
            "step_alive": step_alive.detach().float(),
            "slot_values": slot_val.detach(),
            "entropy_read": _entropy(read_w, dim=-1).mean().detach(),
            "entropy_primitive": _entropy(prim_w, dim=-1).mean().detach(),
            "entropy_slot_transition": _entropy(slot_trans, dim=-1).mean().detach(),
            "entropy_primitive_transition": _entropy(prim_trans, dim=-1).mean().detach(),
            "entropy_write": _entropy(write_w, dim=-1).mean().detach(),
        }
        return next_cells, active_update, info


class MatrixProgramAssemblerCore(nn.Module):
    """Transferable matrix-program assembly logic.

    Input: evidence [N,E,D]
    Output: cells/slots and full aux flow diagnostics.
    """

    def __init__(self, cfg: AssemblerConfig):
        super().__init__()
        self.cfg = cfg
        self.D = int(cfg.dim)
        self.A = int(cfg.address_cells)
        self.B = int(cfg.blocks)
        self.K = int(cfg.primitive_slots)
        self.P = len(PRIMITIVES)
        self.total_steps = int(cfg.layers * cfg.steps)
        self.latent_roles_enabled = bool(getattr(cfg, "latent_roles", False))
        self.role_count = max(1, int(getattr(cfg, "role_count", 6)))

        self.cell_query = nn.Parameter(torch.randn(self.A, self.D) * 0.04)
        self.cell_bias = nn.Parameter(torch.randn(self.A, self.D) * 0.02)
        self.evidence_norm = nn.LayerNorm(self.D)
        self.cell_norm = nn.LayerNorm(self.D)
        if self.latent_roles_enabled:
            role_std = float(getattr(cfg, "role_init_std", 0.02))
            effect_std = float(getattr(cfg, "role_effect_init_std", 0.04))
            self.role_embed = nn.Parameter(torch.randn(self.role_count, self.D) * role_std)
            self.role_logits = nn.Parameter(torch.randn(int(cfg.layers), int(cfg.steps), self.role_count) * role_std)
            # Shared role effects make role0/role1/... mean the same kind of
            # matrix-program pressure in every layer/step. The per-step
            # adapters still exist, but these tensors prevent roles from being
            # merely local decorative context.
            self.role_read_bias = nn.Parameter(torch.randn(self.role_count, self.B, self.K, self.A) * effect_std)
            self.role_primitive_bias = nn.Parameter(torch.randn(self.role_count, self.B, self.K, self.P) * effect_std)
            self.role_slot_transition_bias = nn.Parameter(torch.randn(self.role_count, self.B, self.K, self.K) * effect_std)
            self.role_primitive_transition_bias = nn.Parameter(torch.randn(self.role_count, self.P, self.P) * effect_std)
            self.role_composition_bias = nn.Parameter(torch.randn(self.role_count, self.B, self.K) * effect_std)
            self.role_write_bias = nn.Parameter(torch.randn(self.role_count, self.B, self.A) * effect_std)
            self.role_alive_bias = nn.Parameter(torch.randn(self.role_count) * effect_std)
            self.role_operator_bias = nn.Parameter(torch.randn(self.role_count, 20) * effect_std)

        self.steps = nn.ModuleList([
            AssemblerStep(cfg, layer=l, step=s)
            for l in range(cfg.layers)
            for s in range(cfg.steps)
        ])

    def _role_mix(self) -> torch.Tensor | None:
        if not self.latent_roles_enabled:
            return None
        tau = max(0.05, float(getattr(self.cfg, "role_temperature", 1.25)))
        return torch.softmax(self.role_logits.float() / tau, dim=-1)

    def _role_contexts(self, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
        mix = self._role_mix()
        if mix is None:
            return None, None
        embed = self.role_embed.to(device=device, dtype=torch.float32)
        ctx = torch.einsum("lsr,rd->lsd", mix.to(device=device), embed).to(dtype=dtype)
        return mix.to(device=device), ctx

    def _role_similarity(self, role_mix: torch.Tensor | None) -> torch.Tensor | None:
        if role_mix is None:
            return None
        flat = role_mix.float().reshape(-1, role_mix.shape[-1])
        flat = F.normalize(flat, dim=-1)
        sim = torch.matmul(flat, flat.t())
        eye = torch.eye(sim.shape[0], device=sim.device, dtype=sim.dtype)
        if sim.shape[0] <= 1:
            return torch.zeros((), device=sim.device, dtype=sim.dtype)
        return (sim - eye).sum() / float(sim.numel() - sim.shape[0])

    def _role_effect_biases(
        self,
        mix: torch.Tensor | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, ...] | None:
        if not self.latent_roles_enabled or mix is None:
            return None
        scale = float(getattr(self.cfg, "role_bias_scale", 1.0))
        m = mix.to(device=device, dtype=torch.float32)
        tensors = (
            self.role_read_bias,
            self.role_primitive_bias,
            self.role_slot_transition_bias,
            self.role_primitive_transition_bias,
            self.role_composition_bias,
            self.role_write_bias,
            self.role_alive_bias,
            self.role_operator_bias,
        )
        return tuple((scale * torch.einsum("r,r...->...", m, t.float())).to(device=device, dtype=dtype) for t in tensors)

    def _role_effect_similarity(self) -> torch.Tensor | None:
        if not self.latent_roles_enabled:
            return None
        parts = [
            self.role_read_bias.reshape(self.role_count, -1),
            self.role_primitive_bias.reshape(self.role_count, -1),
            self.role_slot_transition_bias.reshape(self.role_count, -1),
            self.role_primitive_transition_bias.reshape(self.role_count, -1),
            self.role_composition_bias.reshape(self.role_count, -1),
            self.role_write_bias.reshape(self.role_count, -1),
            self.role_alive_bias.reshape(self.role_count, -1),
            self.role_operator_bias.reshape(self.role_count, -1),
        ]
        flat = torch.cat([p.float() for p in parts], dim=-1)
        flat = F.normalize(flat, dim=-1)
        sim = torch.matmul(flat, flat.t())
        eye = torch.eye(sim.shape[0], device=sim.device, dtype=sim.dtype)
        if sim.shape[0] <= 1:
            return torch.zeros((), device=sim.device, dtype=sim.dtype)
        return (sim - eye).sum() / float(sim.numel() - sim.shape[0])

    def config_dict(self) -> Dict[str, object]:
        return {
            "dim": self.cfg.dim,
            "evidence_cells": self.cfg.evidence_cells,
            "layers": self.cfg.layers,
            "blocks": self.cfg.blocks,
            "steps": self.cfg.steps,
            "primitive_slots": self.cfg.primitive_slots,
            "memory_cells": self.cfg.memory_cells,
            "global_cells": self.cfg.global_cells,
            "address_cells": self.cfg.address_cells,
            "channel_stages": self.cfg.channel_stages,
            "dropout": self.cfg.dropout,
            "use_deltas": self.cfg.use_deltas,
            "operator_v2": bool(getattr(self.cfg, "operator_v2", False)),
            "step_alive_init": float(getattr(self.cfg, "step_alive_init", 1.65)),
            "phase_prior_strength": float(getattr(self.cfg, "phase_prior_strength", 0.0)),
            "latent_roles": bool(getattr(self.cfg, "latent_roles", False)),
            "role_count": int(getattr(self.cfg, "role_count", 6)),
            "role_temperature": float(getattr(self.cfg, "role_temperature", 1.25)),
            "role_init_std": float(getattr(self.cfg, "role_init_std", 0.02)),
            "role_effect_init_std": float(getattr(self.cfg, "role_effect_init_std", 0.04)),
            "role_bias_scale": float(getattr(self.cfg, "role_bias_scale", 1.0)),
            "primitives": list(PRIMITIVES),
        }

    def cell_names(self) -> List[str]:
        names = [f"state.B{b}" for b in range(self.cfg.blocks)]
        names.extend([f"memory.M{i}" for i in range(self.cfg.memory_cells)])
        names.extend([f"global.G{i}" for i in range(self.cfg.global_cells)])
        return names

    def set_delta_mode(self, enabled: bool = True) -> None:
        self.cfg.use_deltas = bool(enabled)
        for st in self.steps:
            st.cfg.use_deltas = bool(enabled)

    def freeze_for_mode(self, mode: str) -> None:
        mode = str(mode)
        if mode == "full":
            self.set_delta_mode(True)
            for p in self.parameters():
                p.requires_grad = True
            return
        if mode == "freeze_core":
            for p in self.parameters():
                p.requires_grad = False
            return
        if mode == "editor_delta":
            self.set_delta_mode(True)
            allowed = (
                "flow_editor.scale_delta",
                "flow_editor.variant_editors",
                "phase_mix_delta",
                "phase_extra_delta",
                "opv2_",
                "step_alive",
                "role_",
            )
            for name, p in self.named_parameters():
                p.requires_grad = (
                    (name.endswith("_delta") and any(key in name for key in allowed))
                    or ("step_alive_logit" in name)
                    or ("role_logits" in name)
                    or ("role_embed" in name)
                    or ("role_flow" in name)
                    or ("role_step_alive" in name)
                    or ("role_operator" in name)
                    or name.startswith("role_")
                )
            return
        if mode == "delta":
            self.set_delta_mode(True)
            for name, p in self.named_parameters():
                p.requires_grad = name.endswith("_delta") or "_delta" in name
            return
        raise ValueError(f"unknown train mode: {mode}")

    def init_cells(self, evidence: torch.Tensor) -> torch.Tensor:
        evidence = self.evidence_norm(evidence)
        q = self.cell_query.to(device=evidence.device, dtype=evidence.dtype)
        score = torch.einsum("ad,ned->nae", q, evidence) / math.sqrt(self.D)
        attn = torch.softmax(score.float(), dim=-1).to(evidence.dtype)
        cells = torch.einsum("nae,ned->nad", attn, evidence)
        cells = cells + self.cell_bias.to(device=evidence.device, dtype=evidence.dtype).view(1, self.A, self.D)
        return self.cell_norm(cells)

    def forward(self, evidence: torch.Tensor) -> Tuple[torch.Tensor, AssemblerAux]:
        if evidence.ndim != 3:
            raise ValueError(f"evidence must be [N,E,D], got {tuple(evidence.shape)}")
        if evidence.shape[-1] != self.D:
            raise ValueError(f"evidence dim mismatch: expected {self.D}, got {evidence.shape[-1]}")

        cells = self.init_cells(evidence)
        initial_cells = cells
        role_mix, role_contexts = self._role_contexts(evidence.device, evidence.dtype)
        update_slots: List[torch.Tensor] = []
        read_flows: List[torch.Tensor] = []
        prim_flows: List[torch.Tensor] = []
        slot_trans: List[torch.Tensor] = []
        prim_trans: List[torch.Tensor] = []
        comp_flows: List[torch.Tensor] = []
        write_flows: List[torch.Tensor] = []
        gates: List[torch.Tensor] = []
        updates: List[torch.Tensor] = []
        input_reads: List[torch.Tensor] = []
        read_groups: List[torch.Tensor] = []
        ent_acc: Dict[str, List[torch.Tensor]] = {
            "read": [], "primitive": [], "slot_transition": [], "primitive_transition": [], "write": [], "step_alive": []
        }

        slot_names: List[str] = []
        for ti, step in enumerate(self.steps):
            layer = ti // self.cfg.steps
            substep = ti % self.cfg.steps
            role_context = role_contexts[layer, substep] if role_contexts is not None else None
            role_biases = self._role_effect_biases(role_mix[layer, substep], evidence.device, evidence.dtype) if role_mix is not None else None
            pre_cells = cells
            cells, update, info = step(cells, role_context=role_context, role_biases=role_biases)
            update_slots.append(update)  # [N,B,D]
            read_flows.append(info["read_flow"])
            prim_flows.append(info["primitive_slot_flow"])
            slot_trans.append(info["slot_transition_flow"])
            prim_trans.append(info["primitive_transition_flow"])
            comp_flows.append(info["slot_composition_flow"])
            write_flows.append(info["write_flow"])
            gates.append(info["write_gates"])
            updates.append(info["update_norms"])
            raw_like = F.cosine_similarity(pre_cells.float(), initial_cells.float(), dim=-1).clamp(0.0, 1.0)  # [N,A]
            input_reads.append(torch.einsum("nbka,na->n", info["read_flow"].float(), raw_like) / float(max(1, self.B * self.K)))
            read_addr = info["read_flow"].float().mean(dim=(0, 1, 2))  # [A]
            mem0 = self.B
            glob0 = self.B + self.cfg.memory_cells
            state_mass = read_addr[:mem0].sum()
            memory_mass = read_addr[mem0:glob0].sum() if self.cfg.memory_cells > 0 else torch.zeros((), device=evidence.device)
            global_mass = read_addr[glob0:].sum() if self.cfg.global_cells > 0 else torch.zeros((), device=evidence.device)
            read_groups.append(torch.stack([state_mass, memory_mass, global_mass]))
            ent_acc["read"].append(info["entropy_read"])
            ent_acc["primitive"].append(info["entropy_primitive"])
            ent_acc["slot_transition"].append(info["entropy_slot_transition"])
            ent_acc["primitive_transition"].append(info["entropy_primitive_transition"])
            ent_acc["write"].append(info["entropy_write"])
            ent_acc["step_alive"].append(info["step_alive"])
            if role_mix is not None:
                rid = int(role_mix[layer, substep].argmax().detach().cpu())
                tag = f"role{rid}"
            else:
                phase = PHASES[min(layer, len(PHASES) - 1)]
                tag = phase
            slot_names.extend([f"L{layer}.{tag}.S{substep}.B{b}" for b in range(self.B)])

        slots = torch.cat(update_slots, dim=1) if update_slots else cells[:, : self.B]
        mem0 = self.B
        glob0 = self.B + self.cfg.memory_cells
        memory_usage = torch.stack(write_flows).float()[..., mem0:glob0].sum(dim=-1).mean() if self.cfg.memory_cells > 0 else torch.tensor(0.0, device=evidence.device)
        global_usage = torch.stack(write_flows).float()[..., glob0:].sum(dim=-1).mean() if self.cfg.global_cells > 0 else torch.tensor(0.0, device=evidence.device)
        role_usage = role_mix.float().mean(dim=(0, 1)) if role_mix is not None else None
        role_entropy = _entropy(role_mix, dim=-1) if role_mix is not None else None
        role_similarity = self._role_similarity(role_mix) if role_mix is not None else None
        role_effect_similarity = self._role_effect_similarity() if role_mix is not None else None
        step_alive = torch.stack(ent_acc["step_alive"]).detach() if ent_acc.get("step_alive") else None
        input_proxy_read = torch.stack(input_reads, dim=0) if input_reads else None
        read_group_mass = torch.stack(read_groups, dim=0) if read_groups else None

        aux = AssemblerAux(
            cells=cells,
            slots=slots,
            read_flow=torch.stack(read_flows, dim=0),
            primitive_slot_flow=torch.stack(prim_flows, dim=0),
            slot_transition_flow=torch.stack(slot_trans, dim=0),
            primitive_transition_flow=torch.stack(prim_trans, dim=0),
            slot_composition_flow=torch.stack(comp_flows, dim=0),
            write_flow=torch.stack(write_flows, dim=0),
            write_gates=torch.stack(gates, dim=0),
            update_norms=torch.stack(updates, dim=0),
            memory_usage=memory_usage.detach(),
            global_usage=global_usage.detach(),
            entropies={k: torch.stack(v).mean() if v else torch.tensor(0.0, device=evidence.device) for k, v in ent_acc.items()},
            cell_names=self.cell_names(),
            slot_names=slot_names,
            role_mix=role_mix if role_mix is not None else None,
            role_usage=role_usage,
            role_entropy=role_entropy,
            role_similarity=role_similarity,
            role_effect_similarity=role_effect_similarity,
            step_alive=step_alive,
            input_proxy_read=input_proxy_read,
            read_group_mass=read_group_mass,
        )
        return cells, aux


def flow_kl(pred: torch.Tensor, target: torch.Tensor, dim: int = -1) -> torch.Tensor:
    pred = pred.float()
    target = target.to(device=pred.device, dtype=pred.dtype)
    # Targets are usually [T,...], while context-conditioned predictions are [T,N,...].
    # Add broadcast dimensions after time until ranks match.
    while target.ndim < pred.ndim:
        target = target.unsqueeze(1)
    target = target / target.sum(dim=dim, keepdim=True).clamp_min(1e-8)
    pred = pred / pred.sum(dim=dim, keepdim=True).clamp_min(1e-8)
    target = target.expand_as(pred)
    return F.kl_div(
        pred.clamp_min(1e-8).log(),
        target,
        reduction="none",
    ).sum(dim=dim).mean()


def assembler_skill_loss(aux: AssemblerAux, targets: Dict[str, torch.Tensor], weights: Dict[str, float]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    losses: Dict[str, torch.Tensor] = {}
    if "read_flow" in targets:
        losses["read_flow_kl"] = flow_kl(aux.read_flow, targets["read_flow"], dim=-1)
    if "primitive_slot_flow" in targets:
        losses["primitive_slot_kl"] = flow_kl(aux.primitive_slot_flow, targets["primitive_slot_flow"], dim=-1)
    if "slot_transition_flow" in targets:
        losses["slot_transition_kl"] = flow_kl(aux.slot_transition_flow, targets["slot_transition_flow"], dim=-1)
    if "primitive_transition_flow" in targets:
        losses["primitive_transition_kl"] = flow_kl(aux.primitive_transition_flow, targets["primitive_transition_flow"], dim=-1)
    if "slot_composition_flow" in targets:
        losses["slot_composition_kl"] = flow_kl(aux.slot_composition_flow, targets["slot_composition_flow"], dim=-1)
    if "write_flow" in targets:
        losses["write_flow_kl"] = flow_kl(aux.write_flow, targets["write_flow"], dim=-1)

    total = torch.zeros((), device=aux.cells.device)
    for k, v in losses.items():
        total = total + float(weights.get(k, 1.0)) * v
    return total, losses
