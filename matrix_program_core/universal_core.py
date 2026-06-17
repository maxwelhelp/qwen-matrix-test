#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Universal Matrix Program Core v4.

The important correction vs the old `BaselineDecoder`:

* this module is not a separate W->labels decoder;
* this is the actual transferable architecture core;
* synthetic pretrain and real tasks use the same parameters.

Task-specific code should provide only:

    input -> evidence [B, E, D]
    core(evidence) -> slots/program-flow diagnostics
    head(slots) -> task output

Finetune modes:

    freeze_core: train only input adapter and task head
    delta:       LoRA-like train only *_delta parameters inside the core + adapters/head
    full:        train the whole core
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
class ProgramCoreConfig:
    dim: int = 96
    evidence_cells: int = 48
    layers: int = 4
    blocks: int = 4
    steps: int = 2
    variants: int = 3
    channel_stages: int = 3
    dropout: float = 0.04
    use_deltas: bool = True


@dataclass
class CoreAux:
    slots: torch.Tensor
    slot_names: List[str]
    write_gates: torch.Tensor
    update_norms: torch.Tensor
    primitive_probs: torch.Tensor
    transition_probs: torch.Tensor
    variant_transport: torch.Tensor
    pair_flow: torch.Tensor
    transport_entropy: torch.Tensor


class UniversalMatrixProgramStep(nn.Module):
    """One matrix-program step shared by synthetic and real tasks.

    There is no discrete router. All primitive paths stay differentiable. The
    trainable objects are exactly the things we want to transfer: primitive flow,
    primitive transitions, variant transport, pair-flow, and low-rank/context
    operators.
    """

    def __init__(self, dim: int, blocks: int, variants: int, phase: str, channel_stages: int, dropout: float, use_deltas: bool = True):
        super().__init__()
        self.dim = int(dim)
        self.blocks = int(blocks)
        self.variants = int(variants)
        self.phase = str(phase)
        self.use_deltas = bool(use_deltas)
        self.p_count = len(PRIMITIVES)

        self.channel = ChannelButterfly(dim, channel_stages)
        self.block = BlockButterfly(blocks)
        rank = max(8, dim // 4)
        self.low_a = nn.Parameter(torch.randn(dim, rank) * 0.04)
        self.low_b = nn.Parameter(torch.randn(rank, dim) * 0.04)
        self.ctx_w = nn.Parameter(torch.randn(dim, dim) * 0.04)
        self.gate_h = nn.Parameter(torch.randn(dim, dim) * 0.02)
        self.gate_c = nn.Parameter(torch.randn(dim, dim) * 0.02)
        self.gate_bias = nn.Parameter(torch.full((dim,), -0.35))

        self.primitive_flow = nn.Parameter(self._primitive_prior(phase))
        self.primitive_flow_delta = nn.Parameter(torch.zeros(self.p_count, self.p_count))
        self.variant_transport = nn.Parameter(self._variant_prior(variants))
        self.variant_transport_delta = nn.Parameter(torch.zeros(variants, variants))
        self.variant_primitive_bias = nn.Parameter(torch.zeros(variants, self.p_count))
        self.variant_primitive_delta = nn.Parameter(torch.zeros(variants, self.p_count))

        # Pair-flow is the explicit transferable memory of which primitive tends
        # to follow which primitive inside this phase/step.
        self.pair_flow = nn.Parameter(self._pair_prior(phase))
        self.pair_flow_delta = nn.Parameter(torch.zeros(self.p_count, self.p_count))

        self.write_logit = nn.Parameter(torch.tensor(-0.15))
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    @staticmethod
    def _primitive_prior(phase: str) -> torch.Tensor:
        p = len(PRIMITIVES)
        m = torch.eye(p) * 0.55
        idx = {name: i for i, name in enumerate(PRIMITIVES)}
        if phase == "extract":
            m[idx["channel_butterfly"], idx["ctx_matrix"]] = 0.35
            m[idx["phase_matrix"], idx["channel_butterfly"]] = 0.35
        elif phase == "compare":
            m[idx["phase_matrix"], idx["low_rank"]] = 0.35
            m[idx["product_gate"], idx["ctx_matrix"]] = 0.25
        elif phase == "suppress":
            m[idx["phase_matrix"], idx["block_butterfly"]] = 0.45
            m[idx["product_gate"], idx["phase_matrix"]] = 0.25
        elif phase == "aggregate":
            m[idx["block_butterfly"], idx["channel_butterfly"]] = 0.35
            m[idx["phase_matrix"], idx["block_butterfly"]] = 0.40
        return m + 0.01 * torch.randn(p, p)

    @staticmethod
    def _variant_prior(variants: int) -> torch.Tensor:
        m = torch.eye(variants) * 1.30
        for i in range(variants):
            m[i, (i + 1) % variants] = 0.20
            m[i, (i - 1) % variants] = 0.20
        return m

    @staticmethod
    def _pair_prior(phase: str) -> torch.Tensor:
        p = len(PRIMITIVES)
        m = torch.eye(p) * 0.25
        idx = {name: i for i, name in enumerate(PRIMITIVES)}
        def link(a: str, b: str, v: float) -> None:
            if a in idx and b in idx:
                m[idx[a], idx[b]] = v
        if phase == "extract":
            link("ctx_matrix", "channel_butterfly", 0.45)
            link("channel_butterfly", "phase_matrix", 0.40)
        elif phase == "compare":
            link("low_rank", "product_gate", 0.45)
            link("ctx_matrix", "phase_matrix", 0.38)
        elif phase == "suppress":
            link("block_butterfly", "phase_matrix", 0.50)
            link("product_gate", "phase_matrix", 0.35)
        elif phase == "aggregate":
            link("block_butterfly", "channel_butterfly", 0.45)
            link("phase_matrix", "block_butterfly", 0.42)
        return m + 0.01 * torch.randn(p, p)

    def _effective(self, base: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return base + delta if self.use_deltas else base

    def _primitive_outputs(self, h: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        # h/ctx: [B,V,N,D], output [B,V,N,P,D]
        B, V, N, D = h.shape
        flat_h = h.reshape(B * V, N, D)
        flat_ctx = ctx.reshape(B * V, N, D)
        ctx_m = flat_ctx @ self.ctx_w.to(device=h.device, dtype=h.dtype)
        channel = self.channel(flat_h + ctx_m)
        block = self.block(flat_h)
        low = (flat_h @ self.low_a.to(device=h.device, dtype=h.dtype)) @ self.low_b.to(device=h.device, dtype=h.dtype)
        product = flat_h * torch.tanh(ctx_m)
        gate = torch.sigmoid(
            flat_h @ self.gate_h.to(device=h.device, dtype=h.dtype)
            + ctx_m @ self.gate_c.to(device=h.device, dtype=h.dtype)
            + self.gate_bias.to(device=h.device, dtype=h.dtype)
        )
        if self.phase == "extract":
            phase = channel + 0.50 * ctx_m
        elif self.phase == "compare":
            phase = self.channel(flat_h - ctx_m)
        elif self.phase == "suppress":
            phase = -gate * block.mean(dim=1, keepdim=True)
        elif self.phase == "aggregate":
            phase = self.block(flat_h + flat_h.mean(dim=1, keepdim=True))
        else:
            phase = channel
        cands = torch.stack([channel, block, low, ctx_m, product, phase], dim=2)
        return cands.reshape(B, V, N, self.p_count, D)

    def forward(self, h: torch.Tensor, ctx: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        cands = self._primitive_outputs(h, ctx)

        flow_logits = self._effective(self.primitive_flow, self.primitive_flow_delta)
        flow = flow_logits.to(device=h.device, dtype=h.dtype)
        mixed = torch.einsum("pq,bvnqd->bvnpd", flow, cands)

        prim_logits = self._effective(self.variant_primitive_bias, self.variant_primitive_delta)
        primitive_gate = torch.sigmoid(prim_logits.to(device=h.device, dtype=h.dtype)).view(1, self.variants, 1, self.p_count, 1)
        update = (primitive_gate * mixed).sum(dim=3) / primitive_gate.sum(dim=3).clamp_min(1e-4)

        pair_logits = self._effective(self.pair_flow, self.pair_flow_delta)
        pair_prob = torch.softmax(pair_logits.float().reshape(-1), dim=0).reshape(self.p_count, self.p_count).to(h.dtype)
        # pair_gain lightly modulates primitive mixture without making a discrete route.
        pair_gain = pair_prob.sum(dim=0).view(1, 1, 1, self.p_count, 1)
        update = update + 0.10 * ((pair_gain * mixed).sum(dim=3) / pair_gain.sum(dim=3).clamp_min(1e-4))

        write = torch.sigmoid(self.write_logit.to(device=h.device, dtype=h.dtype))
        local_next = self.norm(h + write * self.drop(update))

        transport_logits = self._effective(self.variant_transport, self.variant_transport_delta)
        transport = torch.softmax(transport_logits.float(), dim=-1).to(h.dtype)
        transported = torch.einsum("uv,bvnd->bund", transport, local_next)
        h_next = self.norm(0.65 * local_next + 0.35 * transported)

        primitive_prob = torch.sigmoid(prim_logits.float()).detach()
        transition_prob = torch.softmax(flow_logits.float(), dim=-1).detach()
        transport_prob = transport.detach().float()
        update_norm = update.detach().float().norm(dim=-1).mean(dim=1)
        variant_entropy = -(transport_prob.clamp_min(1e-8) * transport_prob.clamp_min(1e-8).log()).sum(dim=-1).mean()
        info = {
            "write": write.detach().expand(h.shape[0], self.blocks),
            "update_norm": update_norm,
            "primitive_prob": primitive_prob,
            "transition_prob": transition_prob,
            "pair_flow": pair_prob.detach().float(),
            "variant_transport": transport_prob,
            "transport_entropy": variant_entropy.detach(),
        }
        return h_next, info


class UniversalMatrixProgramCore(nn.Module):
    """Task-independent matrix program core.

    Input is an evidence tensor [B, evidence_cells, dim]. Evidence can come from
    audio, text, images, or synthetic tasks. The core state and program-flow
    weights are the transferable skill.
    """

    def __init__(self, cfg: ProgramCoreConfig):
        super().__init__()
        self.cfg = cfg
        self.dim = int(cfg.dim)
        self.evidence_cells = int(cfg.evidence_cells)
        self.layers = int(cfg.layers)
        self.blocks = int(cfg.blocks)
        self.steps = int(cfg.steps)
        self.variants = int(cfg.variants)

        self.block_query = nn.Parameter(torch.randn(self.blocks, self.dim) * 0.04)
        self.variant_bias = nn.Parameter(torch.randn(self.variants, self.blocks, self.dim) * 0.025)
        self.phase_ctx = nn.Parameter(torch.randn(self.layers, self.steps, self.variants, self.dim, self.dim) * 0.022)
        self.phase_ctx_delta = nn.Parameter(torch.zeros(self.layers, self.steps, self.variants, self.dim, self.dim))
        self.layer_bias = nn.Parameter(torch.randn(self.layers, 1, 1, self.dim) * 0.02)
        self.step_bias = nn.Parameter(torch.randn(self.steps, 1, 1, self.dim) * 0.02)
        self.norm = nn.LayerNorm(self.dim)

        phase_names = [PHASES[min(i, len(PHASES) - 1)] for i in range(self.layers)]
        self.units = nn.ModuleList([
            UniversalMatrixProgramStep(
                self.dim,
                self.blocks,
                self.variants,
                phase_names[l],
                cfg.channel_stages,
                cfg.dropout,
                cfg.use_deltas,
            )
            for l in range(self.layers)
            for _s in range(self.steps)
        ])

    def config_dict(self) -> Dict[str, int | float | bool]:
        return {
            "dim": self.dim,
            "evidence_cells": self.evidence_cells,
            "layers": self.layers,
            "blocks": self.blocks,
            "steps": self.steps,
            "variants": self.variants,
            "channel_stages": self.cfg.channel_stages,
            "dropout": self.cfg.dropout,
            "use_deltas": self.cfg.use_deltas,
            "primitives": list(PRIMITIVES),
        }

    def set_delta_mode(self, enabled: bool = True) -> None:
        self.cfg.use_deltas = bool(enabled)
        for u in self.units:
            u.use_deltas = bool(enabled)

    def freeze_for_mode(self, mode: str) -> None:
        """Configure trainable parameters inside the core.

        mode values:
          full        - train all core params
          freeze_core - train no core params
          delta       - train only LoRA-like delta params
        """
        mode = str(mode)
        if mode == "full":
            for p in self.parameters():
                p.requires_grad = True
            self.set_delta_mode(True)
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
        raise ValueError(f"unknown core train mode: {mode}")

    def _block_init(self, evidence: torch.Tensor) -> torch.Tensor:
        q = self.block_query.to(device=evidence.device, dtype=evidence.dtype)
        score = torch.einsum("nd,bed->bne", q, evidence) / math.sqrt(evidence.shape[-1])
        attn = torch.softmax(score.float(), dim=-1).to(evidence.dtype)
        base = torch.einsum("bne,bed->bnd", attn, evidence)
        h = base[:, None, :, :] + self.variant_bias.to(device=evidence.device, dtype=evidence.dtype).view(1, self.variants, self.blocks, self.dim)
        return self.norm(h)

    def _context(self, h: torch.Tensor, evidence: torch.Tensor, l: int, s: int) -> torch.Tensor:
        w = self.phase_ctx[l, s]
        if self.cfg.use_deltas:
            w = w + self.phase_ctx_delta[l, s]
        w = w.to(device=h.device, dtype=h.dtype)
        q = torch.einsum("bvnd,vdh->bvnh", h, w)
        score = torch.einsum("bvnd,bed->bvne", q, evidence) / math.sqrt(h.shape[-1])
        attn = torch.softmax(score.float(), dim=-1).to(h.dtype)
        ctx = torch.einsum("bvne,bed->bvnd", attn, evidence)
        return ctx / float(1 + l + s)

    def forward(self, evidence: torch.Tensor) -> Tuple[torch.Tensor, CoreAux]:
        if evidence.ndim != 3:
            raise ValueError(f"evidence must be [B,E,D], got {tuple(evidence.shape)}")
        if evidence.shape[-1] != self.dim:
            raise ValueError(f"evidence dim mismatch: expected {self.dim}, got {evidence.shape[-1]}")
        evidence = self.norm(evidence)
        h = self._block_init(evidence)
        slot_banks = [h]
        names = [f"input.V{v}.B{b}" for v in range(self.variants) for b in range(self.blocks)]
        gates: List[torch.Tensor] = []
        updates: List[torch.Tensor] = []
        prims: List[torch.Tensor] = []
        trans: List[torch.Tensor] = []
        pairs: List[torch.Tensor] = []
        transports: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []

        for l in range(self.layers):
            h = self.norm(h + self.layer_bias[l].to(device=h.device, dtype=h.dtype))
            for s in range(self.steps):
                h = h + self.step_bias[s].to(device=h.device, dtype=h.dtype)
                ctx = self._context(h, evidence, l, s)
                unit = self.units[l * self.steps + s]
                h, info = unit(h, ctx)
                slot_banks.append(h)
                gates.append(info["write"])
                updates.append(info["update_norm"])
                prims.append(info["primitive_prob"])
                trans.append(info["transition_prob"])
                pairs.append(info["pair_flow"])
                transports.append(info["variant_transport"])
                entropies.append(info["transport_entropy"])
                phase = PHASES[min(l, len(PHASES) - 1)]
                names.extend([f"L{l}.{phase}.V{v}.B{b}.S{s}" for v in range(self.variants) for b in range(self.blocks)])

        slots = torch.stack(slot_banks, dim=1).reshape(evidence.shape[0], -1, self.dim)
        aux = CoreAux(
            slots=slots,
            slot_names=names,
            write_gates=torch.stack(gates, dim=1) if gates else torch.empty(evidence.shape[0], 0, self.blocks, device=evidence.device),
            update_norms=torch.stack(updates, dim=1) if updates else torch.empty(evidence.shape[0], 0, self.blocks, device=evidence.device),
            primitive_probs=torch.stack(prims, dim=0) if prims else torch.empty(0, device=evidence.device),
            transition_probs=torch.stack(trans, dim=0) if trans else torch.empty(0, device=evidence.device),
            variant_transport=torch.stack(transports, dim=0) if transports else torch.empty(0, device=evidence.device),
            pair_flow=torch.stack(pairs, dim=0) if pairs else torch.empty(0, device=evidence.device),
            transport_entropy=torch.stack(entropies).mean() if entropies else torch.tensor(0.0, device=evidence.device),
        )
        return slots, aux


def core_skill_loss(aux: CoreAux, phase_targets: torch.Tensor, pair_targets: torch.Tensor, weight_pair: float = 0.35) -> torch.Tensor:
    """Program-flow supervision for synthetic pretrain.

    phase_targets: [T, P] desired primitive use per logical step.
    pair_targets:  [T, P, P] desired primitive->primitive pair-flow.
    """
    loss = torch.zeros((), device=aux.slots.device)
    if aux.primitive_probs.numel() and phase_targets.numel():
        T = min(aux.primitive_probs.shape[0], phase_targets.shape[0])
        pred = aux.primitive_probs[:T].float().mean(dim=1)  # [T,P]
        tgt = phase_targets[:T].to(device=pred.device, dtype=pred.dtype)
        tgt = tgt / tgt.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        pred = pred / pred.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        loss = loss + F.kl_div(pred.clamp_min(1e-8).log(), tgt, reduction="batchmean")
    if aux.pair_flow.numel() and pair_targets.numel():
        T = min(aux.pair_flow.shape[0], pair_targets.shape[0])
        predp = aux.pair_flow[:T].float().flatten(1)
        tgtp = pair_targets[:T].to(device=predp.device, dtype=predp.dtype).flatten(1)
        tgtp = tgtp / tgtp.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        loss = loss + weight_pair * F.kl_div(predp.clamp_min(1e-8).log(), tgtp, reduction="batchmean")
    return loss
