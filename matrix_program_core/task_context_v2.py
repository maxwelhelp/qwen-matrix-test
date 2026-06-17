#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task / head / consumer context encoder v2.

This module avoids hardcoding `task_type=classification` inside the assembler.
It converts a generic I/O contract plus actual head/consumer query vectors into
context tokens that can be prepended to evidence.

The assembler still only sees tensors and soft matrices. There is no router.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class TaskContextV2Config:
    dim: int = 96
    max_roles: int = 32
    max_input_kinds: int = 32
    max_output_kinds: int = 32
    max_loss_kinds: int = 32
    max_readout_kinds: int = 32
    free_tokens: int = 4
    numeric_tokens: int = 2
    max_head_tokens: int = 16
    dropout: float = 0.02


class TaskIOContextEncoder(nn.Module):
    """Build generic task/consumer context tokens.

    Inputs are IDs and numeric descriptors, not a hardcoded task branch.

    Example audio classification is only one possible contract:
      role_id=task_head_core, input_kind=audio_features,
      output_kind=class_logits, loss_kind=cross_entropy,
      readout_kind=class_query_readout, num_outputs=10.

    A layer replacement can use a different contract:
      role_id=sequence_mixer, input_kind=token_states,
      output_kind=token_states, readout_kind=residual_writeback.
    """

    def __init__(self, cfg: TaskContextV2Config):
        super().__init__()
        self.cfg = cfg
        D = int(cfg.dim)
        self.role = nn.Embedding(cfg.max_roles, D)
        self.input_kind = nn.Embedding(cfg.max_input_kinds, D)
        self.output_kind = nn.Embedding(cfg.max_output_kinds, D)
        self.loss_kind = nn.Embedding(cfg.max_loss_kinds, D)
        self.readout_kind = nn.Embedding(cfg.max_readout_kinds, D)
        self.free_context = nn.Parameter(torch.randn(max(1, cfg.free_tokens), D) * 0.02)
        self.numeric_proj = nn.Sequential(
            nn.Linear(4, D),
            nn.GELU(),
            nn.Linear(D, D),
        )
        self.head_proj = nn.Linear(D, D, bias=False)
        self.norm = nn.LayerNorm(D)
        self.drop = nn.Dropout(cfg.dropout)

    def _id_tensor(self, value, batch: int, device: torch.device) -> torch.Tensor:
        if torch.is_tensor(value):
            v = value.to(device=device, dtype=torch.long).view(-1)
            if v.numel() == 1:
                return v.expand(batch)
            if v.numel() != batch:
                raise ValueError(f"context id tensor has {v.numel()} items, expected {batch}")
            return v
        return torch.full((batch,), int(value), device=device, dtype=torch.long)

    def _num_tensor(self, value, batch: int, device: torch.device, default: float = 0.0) -> torch.Tensor:
        if torch.is_tensor(value):
            v = value.to(device=device, dtype=torch.float32).view(-1)
            if v.numel() == 1:
                return v.expand(batch)
            if v.numel() != batch:
                raise ValueError(f"context numeric tensor has {v.numel()} items, expected {batch}")
            return v
        return torch.full((batch,), float(value if value is not None else default), device=device, dtype=torch.float32)

    def forward(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        *,
        role_id: int = 0,
        input_kind_id: int = 0,
        output_kind_id: int = 0,
        loss_kind_id: int = 0,
        readout_kind_id: int = 0,
        num_outputs: float = 1.0,
        sequence_length: float = 1.0,
        hidden_dim: float = 1.0,
        extra_scalar: float = 0.0,
        head_query: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = int(batch_size)
        role = self.role(self._id_tensor(role_id, B, device)).unsqueeze(1)
        inp = self.input_kind(self._id_tensor(input_kind_id, B, device)).unsqueeze(1)
        out = self.output_kind(self._id_tensor(output_kind_id, B, device)).unsqueeze(1)
        loss = self.loss_kind(self._id_tensor(loss_kind_id, B, device)).unsqueeze(1)
        readout = self.readout_kind(self._id_tensor(readout_kind_id, B, device)).unsqueeze(1)
        nums = torch.stack([
            self._num_tensor(num_outputs, B, device, 1.0),
            self._num_tensor(sequence_length, B, device, 1.0),
            self._num_tensor(hidden_dim, B, device, 1.0),
            self._num_tensor(extra_scalar, B, device, 0.0),
        ], dim=-1)
        # log scale keeps dimensions/counts from dominating.
        nums = torch.log1p(nums.clamp_min(0.0))
        num_tok = self.numeric_proj(nums).unsqueeze(1)
        free = self.free_context.to(device=device, dtype=torch.float32).unsqueeze(0).expand(B, -1, -1)
        tokens = [role, inp, out, loss, readout, num_tok, free]
        if head_query is not None and self.cfg.max_head_tokens > 0:
            hq = head_query[: self.cfg.max_head_tokens].to(device=device, dtype=torch.float32)
            hq = self.head_proj(hq).unsqueeze(0).expand(B, -1, -1)
            tokens.append(hq)
        ctx = torch.cat(tokens, dim=1)
        return self.drop(self.norm(ctx)).to(dtype=dtype)


# Conventional IDs. These are not branches; they are just default embedding IDs
# for scripts. Experiments can use other integer IDs without changing core code.
ROLE_TASK_HEAD_CORE = 0
ROLE_SEQUENCE_MIXER = 1
ROLE_ATTENTION_REPLACEMENT = 2
ROLE_MATRIX_DECOMPILER = 3

INPUT_AUDIO_FEATURES = 0
INPUT_TOKEN_STATES = 1
INPUT_MATRIX_FEATURES = 2
INPUT_GENERIC_EVIDENCE = 3

OUTPUT_CLASS_LOGITS = 0
OUTPUT_TOKEN_STATES = 1
OUTPUT_MATRIX_RECON = 2
OUTPUT_PROGRAM_SLOTS = 3

LOSS_CROSS_ENTROPY = 0
LOSS_RECONSTRUCTION = 1
LOSS_DISTILLATION = 2
LOSS_DOWNSTREAM = 3

READOUT_CLASS_QUERY = 0
READOUT_RESIDUAL_WRITEBACK = 1
READOUT_PROGRAM_SLOT = 2
READOUT_GENERIC_HEAD = 3
