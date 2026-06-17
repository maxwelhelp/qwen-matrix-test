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

        # Base assembly matrices.
        self.read_logits = nn.Parameter(self._read_prior())                     # [B,K,A]
        self.primitive_slot_logits = nn.Parameter(self._primitive_slot_prior()) # [B,K,P]
        self.slot_transition_logits = nn.Parameter(self._slot_transition_prior()) # [B,K,K]
        self.primitive_transition_logits = nn.Parameter(self._primitive_transition_prior()) # [P,P]
        self.slot_composition_logits = nn.Parameter(torch.zeros(self.B, self.K)) # [B,K]
        self.write_logits = nn.Parameter(self._write_prior())                   # [B,A]
        self.write_gate_logit = nn.Parameter(torch.full((self.B,), -0.15))

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
        if self.phase == "extract":
            phase = channel + 0.50 * ctx_m
        elif self.phase == "compare":
            phase = channel - low
        elif self.phase == "suppress":
            phase = -gate * block.mean(dim=1, keepdim=True)
        elif self.phase == "aggregate":
            phase = block + read_ctx.mean(dim=1, keepdim=True)
        else:
            phase = channel
        return torch.stack([channel, block, low, ctx_m, product, phase], dim=3)

    def forward(self, cells: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        # cells: [N,A,D]
        read_logits = self._eff(self.read_logits, self.read_delta)
        prim_logits = self._eff(self.primitive_slot_logits, self.primitive_slot_delta)
        slot_trans_logits = self._eff(self.slot_transition_logits, self.slot_transition_delta)
        prim_trans_logits = self._eff(self.primitive_transition_logits, self.primitive_transition_delta)
        comp_logits = self._eff(self.slot_composition_logits, self.slot_composition_delta)
        write_logits = self._eff(self.write_logits, self.write_delta)

        read_w = torch.softmax(read_logits.float(), dim=-1).to(cells.dtype)        # [B,K,A]
        prim_w = torch.softmax(prim_logits.float(), dim=-1).to(cells.dtype)        # [B,K,P]
        slot_trans = torch.softmax(slot_trans_logits.float(), dim=-1).to(cells.dtype) # [B,K,K]
        prim_trans = torch.softmax(prim_trans_logits.float(), dim=-1).to(cells.dtype) # [P,P]
        comp_w = torch.softmax(comp_logits.float(), dim=-1).to(cells.dtype)        # [B,K]
        write_w = torch.softmax(write_logits.float(), dim=-1).to(cells.dtype)      # [B,A]

        read_ctx = torch.einsum("bka,nad->nbkd", read_w, cells)                   # [N,B,K,D]
        prim_out = self._primitive_outputs(read_ctx)                               # [N,B,K,P,D]

        # Factorized transitions: primitive type transition and slot transition.
        prim_mixed = torch.einsum("pq,nbkqd->nbkpd", prim_trans, prim_out)        # [N,B,K,P,D]
        slot_mixed = torch.einsum("bkj,nbjpd->nbkpd", slot_trans, prim_mixed)     # [N,B,K,P,D]

        slot_val = torch.einsum("bkp,nbkpd->nbkd", prim_w, slot_mixed)             # [N,B,K,D]
        update = torch.einsum("bk,nbkd->nbd", comp_w, slot_val)                   # [N,B,D]
        update = self.norm(self.drop(update))

        write_gate = torch.sigmoid(self.write_gate_logit.to(device=cells.device, dtype=cells.dtype)).view(1, self.B, 1)
        delta_cells = torch.einsum("ba,nbd->nad", write_w, write_gate * update)    # [N,A,D]
        next_cells = self.norm(cells + delta_cells)

        info = {
            "read_flow": read_w.detach().float(),
            "primitive_slot_flow": prim_w.detach().float(),
            "slot_transition_flow": slot_trans.detach().float(),
            "primitive_transition_flow": prim_trans.detach().float(),
            "slot_composition_flow": comp_w.detach().float(),
            "write_flow": write_w.detach().float(),
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
    target = target / target.sum(dim=dim, keepdim=True).clamp_min(1e-8)
    pred = pred / pred.sum(dim=dim, keepdim=True).clamp_min(1e-8)
    return F.kl_div(pred.clamp_min(1e-8).log(), target, reduction="batchmean")


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
