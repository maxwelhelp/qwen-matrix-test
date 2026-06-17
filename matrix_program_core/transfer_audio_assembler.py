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
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from matrix_program_core.assembler_core import (  # noqa: E402
    AssemblerConfig,
    MatrixProgramAssemblerCore,
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


class AudioAssemblerHead(nn.Module):
    def __init__(self, dim: int, classes: int, dropout: float = 0.05):
        super().__init__()
        self.classes = int(classes)
        self.query = nn.Parameter(torch.randn(classes, dim) * 0.04)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.update = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm = nn.LayerNorm(dim)
        self.logit_w = nn.Parameter(torch.randn(classes, dim) * 0.04)
        self.logit_bias = nn.Parameter(torch.zeros(classes))

    def forward(self, slots: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        q = self.query.to(device=slots.device, dtype=slots.dtype)
        k = self.key(slots)
        v = self.value(slots)
        score = torch.einsum("cd,nsd->ncs", q, k) / math.sqrt(slots.shape[-1])
        attn = torch.softmax(score.float(), dim=-1).to(slots.dtype)
        read = torch.einsum("ncs,nsd->ncd", attn, v)
        global_read = slots.mean(dim=1, keepdim=True).expand_as(read)
        cls_state = q.view(1, self.classes, -1).expand(slots.shape[0], -1, -1)
        h = self.norm(cls_state + self.update(torch.cat([cls_state, read, global_read], dim=-1)))
        logits = (h * self.logit_w.to(device=slots.device, dtype=slots.dtype).view(1, self.classes, -1)).sum(dim=-1)
        logits = logits + self.logit_bias.to(device=slots.device, dtype=slots.dtype)
        return logits, {"class_slot_attention": attn.detach(), "class_read": read.detach()}


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
        )
        self.assembler_core = MatrixProgramAssemblerCore(cfg)
        self.head = AudioAssemblerHead(args.dim, classes, args.head_dropout)
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
    a = attn.float().mean(dim=0)
    a = F.normalize(a, dim=-1)
    sim = a @ a.t()
    offdiag = sim - torch.eye(sim.shape[0], device=sim.device)
    return F.relu(offdiag - 0.25).mean()


def slot_diversity_loss(slots: torch.Tensor) -> torch.Tensor:
    s = slots.float().mean(dim=0)
    s = F.normalize(s, dim=-1)
    sim = s @ s.t()
    offdiag = sim - torch.eye(sim.shape[0], device=sim.device)
    return F.relu(offdiag - 0.55).mean()


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
    out["slot_div"] = slot_diversity_loss(aux.slots)
    out["logit_norm"] = logits.float().pow(2).mean()
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
            loss = loss + args.lambda_slot_div * losses["slot_div"]
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
        "write_budget", "update_alive", "class_read_div", "slot_div", "logit_norm",
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
    p.add_argument("--lambda-skill", type=float, default=0.0)
    p.add_argument("--lambda-write-budget", type=float, default=0.025)
    p.add_argument("--lambda-update-alive", type=float, default=0.005)
    p.add_argument("--lambda-class-read-div", type=float, default=0.020)
    p.add_argument("--lambda-slot-div", type=float, default=0.002)
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
