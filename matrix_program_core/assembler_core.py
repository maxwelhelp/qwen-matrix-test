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
        phase_logits = torch.full((len(PHASES),), -0.65)
        phase_logits[min(layer, len(PHASES) - 1)] = 1.25
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

        # LoRA-like task deltas. In delta mode only these are trainable.
        self.read_delta = nn.Parameter(torch.zeros(self.B, self.K, self.A))
        self.primitive_slot_delta = nn.Parameter(torch.zeros(self.B, self.K, self.P))
        self.slot_transition_delta = nn.Parameter(torch.zeros(self.B, self.K, self.K))
        self.primitive_transition_delta = nn.Parameter(torch.zeros(self.P, self.P))
        self.slot_composition_delta = nn.Parameter(torch.zeros(self.B, self.K))
        self.write_delta = nn.Parameter(torch.zeros(self.B, self.A))

        self.drop = nn.Dropout(cfg.dropout)
        self.norm = nn.LayerNorm(self.D)

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
                        x[b, k, idx[name]] += 0.65 / (1 + abs(k - j))
                # Keep every primitive alive, but softly phase-biased.
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
        x = torch.eye(self.P) * 0.35
        idx = {name: i for i, name in enumerate(PRIMITIVES)}
        def link(a: str, b: str, v: float) -> None:
            if a in idx and b in idx:
                x[idx[a], idx[b]] = v
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

    def _context_flow_bias(self, cells: torch.Tensor):
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

    def _primitive_outputs(self, read_ctx: torch.Tensor) -> torch.Tensor:
        # read_ctx: [N,B,K,D] -> [N,B,K,P,D]
        N, B, K, D = read_ctx.shape
        flat = read_ctx.reshape(N * B * K, D)
        ctx_m = flat @ self.ctx_w.to(device=read_ctx.device, dtype=read_ctx.dtype)
        channel = self.channel((flat + ctx_m).view(N * B * K, 1, D)).view(N, B, K, D)
        low = ((flat @ self.low_a.to(device=read_ctx.device, dtype=read_ctx.dtype)) @ self.low_b.to(device=read_ctx.device, dtype=read_ctx.dtype)).view(N, B, K, D)
        ctx_m = ctx_m.view(N, B, K, D)
        product = read_ctx * torch.tanh(ctx_m)

        # Block primitive mixes across blocks for every slot k.
        block_in = read_ctx.permute(0, 2, 1, 3).reshape(N * K, B, D)
        block = self.block(block_in).reshape(N, K, B, D).permute(0, 2, 1, 3)

        gate = torch.sigmoid(
            read_ctx @ self.gate_h.to(device=read_ctx.device, dtype=read_ctx.dtype)
            + ctx_m @ self.gate_c.to(device=read_ctx.device, dtype=read_ctx.dtype)
            + self.gate_bias.to(device=read_ctx.device, dtype=read_ctx.dtype)
        )
        phase_candidates = torch.stack([
            channel + 0.50 * ctx_m,
            channel - low,
            -gate * block.mean(dim=1, keepdim=True),
            block + read_ctx.mean(dim=1, keepdim=True),
        ], dim=3)
        phase_logits = self.phase_mix_logits
        if self.cfg.use_deltas:
            phase_logits = phase_logits + self.phase_mix_delta
        phase_w = torch.softmax(phase_logits.float(), dim=-1).to(read_ctx.dtype)
        phase = torch.einsum("f,nbkfd->nbkd", phase_w, phase_candidates)
        return torch.stack([channel, block, low, ctx_m, product, phase], dim=3)

    def forward(self, cells: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        # cells: [N,A,D]
        cells = self.matrix_attention(cells, use_deltas=self.cfg.use_deltas)
        read_logits = self._eff(self.read_logits, self.read_delta)
        prim_logits = self._eff(self.primitive_slot_logits, self.primitive_slot_delta)
        slot_trans_logits = self._eff(self.slot_transition_logits, self.slot_transition_delta)
        prim_trans_logits = self._eff(self.primitive_transition_logits, self.primitive_transition_delta)
        comp_logits = self._eff(self.slot_composition_logits, self.slot_composition_delta)
        write_logits = self._eff(self.write_logits, self.write_delta)

        read_b, prim_b, slot_b, ptrans_b, comp_b, write_b = self._context_flow_bias(cells)
        read_e, prim_e, slot_e, ptrans_e, comp_e, write_e = self.flow_editor(cells, use_deltas=self.cfg.use_deltas)
        read_w = torch.softmax(read_logits.float().unsqueeze(0) + read_b.float() + read_e.float(), dim=-1).to(cells.dtype)        # [N,B,K,A]
        prim_w = torch.softmax(prim_logits.float().unsqueeze(0) + prim_b.float() + prim_e.float(), dim=-1).to(cells.dtype)        # [N,B,K,P]
        slot_trans = torch.softmax(slot_trans_logits.float().unsqueeze(0) + slot_b.float() + slot_e.float(), dim=-1).to(cells.dtype) # [N,B,K,K]
        prim_trans = torch.softmax(prim_trans_logits.float().unsqueeze(0) + ptrans_b.float() + ptrans_e.float(), dim=-1).to(cells.dtype) # [N,P,P]
        comp_w = torch.softmax(comp_logits.float().unsqueeze(0) + comp_b.float() + comp_e.float(), dim=-1).to(cells.dtype)        # [N,B,K]
        write_w = torch.softmax(write_logits.float().unsqueeze(0) + write_b.float() + write_e.float(), dim=-1).to(cells.dtype)      # [N,B,A]

        read_ctx = torch.einsum("nbka,nad->nbkd", read_w, cells)                   # [N,B,K,D]
        prim_out = self._primitive_outputs(read_ctx)                               # [N,B,K,P,D]

        # Factorized transitions: primitive type transition and slot transition.
        prim_mixed = torch.einsum("npq,nbkqd->nbkpd", prim_trans, prim_out)        # [N,B,K,P,D]
        slot_mixed = torch.einsum("nbkj,nbjpd->nbkpd", slot_trans, prim_mixed)     # [N,B,K,P,D]

        slot_val = torch.einsum("nbkp,nbkpd->nbkd", prim_w, slot_mixed)             # [N,B,K,D]
        update = torch.einsum("nbk,nbkd->nbd", comp_w, slot_val)                   # [N,B,D]
        update = self.norm(self.drop(update))

        write_gate = torch.sigmoid(self.write_gate_logit.to(device=cells.device, dtype=cells.dtype)).view(1, self.B, 1)
        delta_cells = torch.einsum("nba,nbd->nad", write_w, write_gate * update)    # [N,A,D]
        next_cells = self.norm(cells + delta_cells)

        info = {
            "read_flow": read_w.float(),
            "primitive_slot_flow": prim_w.float(),
            "slot_transition_flow": slot_trans.float(),
            "primitive_transition_flow": prim_trans.float(),
            "slot_composition_flow": comp_w.float(),
            "write_flow": write_w.float(),
            "write_gates": write_gate.detach().float().squeeze(0).squeeze(-1),
            "update_norms": update.detach().float().norm(dim=-1),
            "slot_values": slot_val.detach(),
            "entropy_read": _entropy(read_w, dim=-1).mean().detach(),
            "entropy_primitive": _entropy(prim_w, dim=-1).mean().detach(),
            "entropy_slot_transition": _entropy(slot_trans, dim=-1).mean().detach(),
            "entropy_primitive_transition": _entropy(prim_trans, dim=-1).mean().detach(),
            "entropy_write": _entropy(write_w, dim=-1).mean().detach(),
        }
        return next_cells, update, info


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

        self.cell_query = nn.Parameter(torch.randn(self.A, self.D) * 0.04)
        self.cell_bias = nn.Parameter(torch.randn(self.A, self.D) * 0.02)
        self.evidence_norm = nn.LayerNorm(self.D)
        self.cell_norm = nn.LayerNorm(self.D)

        self.steps = nn.ModuleList([
            AssemblerStep(cfg, layer=l, step=s)
            for l in range(cfg.layers)
            for s in range(cfg.steps)
        ])

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
            )
            for name, p in self.named_parameters():
                p.requires_grad = name.endswith("_delta") and any(key in name for key in allowed)
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
        update_slots: List[torch.Tensor] = []
        read_flows: List[torch.Tensor] = []
        prim_flows: List[torch.Tensor] = []
        slot_trans: List[torch.Tensor] = []
        prim_trans: List[torch.Tensor] = []
        comp_flows: List[torch.Tensor] = []
        write_flows: List[torch.Tensor] = []
        gates: List[torch.Tensor] = []
        updates: List[torch.Tensor] = []
        ent_acc: Dict[str, List[torch.Tensor]] = {
            "read": [], "primitive": [], "slot_transition": [], "primitive_transition": [], "write": []
        }

        slot_names: List[str] = []
        for ti, step in enumerate(self.steps):
            cells, update, info = step(cells)
            update_slots.append(update)  # [N,B,D]
            read_flows.append(info["read_flow"])
            prim_flows.append(info["primitive_slot_flow"])
            slot_trans.append(info["slot_transition_flow"])
            prim_trans.append(info["primitive_transition_flow"])
            comp_flows.append(info["slot_composition_flow"])
            write_flows.append(info["write_flow"])
            gates.append(info["write_gates"])
            updates.append(info["update_norms"])
            ent_acc["read"].append(info["entropy_read"])
            ent_acc["primitive"].append(info["entropy_primitive"])
            ent_acc["slot_transition"].append(info["entropy_slot_transition"])
            ent_acc["primitive_transition"].append(info["entropy_primitive_transition"])
            ent_acc["write"].append(info["entropy_write"])
            layer = ti // self.cfg.steps
            substep = ti % self.cfg.steps
            phase = PHASES[min(layer, len(PHASES) - 1)]
            slot_names.extend([f"L{layer}.{phase}.S{substep}.B{b}" for b in range(self.B)])

        slots = torch.cat(update_slots, dim=1) if update_slots else cells[:, : self.B]
        mem0 = self.B
        glob0 = self.B + self.cfg.memory_cells
        memory_usage = torch.stack(write_flows).float()[..., mem0:glob0].sum(dim=-1).mean() if self.cfg.memory_cells > 0 else torch.tensor(0.0, device=evidence.device)
        global_usage = torch.stack(write_flows).float()[..., glob0:].sum(dim=-1).mean() if self.cfg.global_cells > 0 else torch.tensor(0.0, device=evidence.device)

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
