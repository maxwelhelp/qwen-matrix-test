#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mechanism-level MatrixProgramAssembler skill pretrain.

This is not a W->label decoder and not a hard operator classifier.

It trains the same MatrixProgramAssemblerCore used by live transfer, but the
input evidence contains only partial matrix-program flow summaries. The core
must reconstruct hidden mechanisms:

  - primitive choice from read/write/context;
  - operator transitions from primitive/read/write context;
  - read/write choices from primitive/transition context;
  - full denoising of all mechanism flows.

All targets are soft matrices. There is no router, top-k, or argmax path choice.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from matrix_program_core.assembler_core import (  # noqa: E402
    AssemblerConfig,
    MatrixProgramAssemblerCore,
    flow_kl,
)
from matrix_program_core.task_context_v2 import TaskContextV2Config, TaskIOContextEncoder  # noqa: E402
from simple_butterfly_matrix.simple_butterfly_matrix import PRIMITIVES  # noqa: E402


FLOW_KEYS = (
    "read_flow",
    "primitive_slot_flow",
    "slot_transition_flow",
    "primitive_transition_flow",
    "slot_composition_flow",
    "write_flow",
)

TASK_FAMILIES = (
    "primitive_from_io",
    "operators_from_primitives",
    "read_write_from_ops",
    "write_from_read_primitive",
    "full_denoise",
)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MechanismFlowDataset(Dataset):
    def __init__(self, pack: Dict[str, torch.Tensor], indices: Sequence[int]):
        self.pack = pack
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        j = self.indices[idx]
        out = {"idx": torch.tensor(j, dtype=torch.long)}
        for key in ("role_id", "input_kind_id", "output_kind_id", "loss_kind_id", "readout_kind_id"):
            out[key] = self.pack[key][j].long()
        for key in ("num_outputs", "sequence_length", "hidden_dim", "extra_scalar"):
            out[key] = self.pack[key][j].float()
        for key in FLOW_KEYS:
            out[key] = self.pack[key][j].float()
        return out


def infer_cfg_from_pack(pack: Dict[str, torch.Tensor], args) -> AssemblerConfig:
    read = pack["read_flow"]
    prim = pack["primitive_slot_flow"]
    if read.ndim != 5 or prim.ndim != 5:
        raise ValueError("expected read_flow [N,T,B,K,A] and primitive_slot_flow [N,T,B,K,P]")
    T, B, K, A = map(int, read.shape[1:])
    steps = int(args.steps_core)
    if T % steps != 0:
        raise ValueError(f"T={T} is not divisible by steps={steps}")
    memory_cells = int(args.memory_cells)
    global_cells = int(args.global_cells)
    if B + memory_cells + global_cells != A:
        memory_cells = max(0, A - B - global_cells)
    return AssemblerConfig(
        dim=args.dim,
        evidence_cells=args.evidence_cells,
        layers=T // steps,
        blocks=B,
        steps=steps,
        primitive_slots=K,
        memory_cells=memory_cells,
        global_cells=global_cells,
        channel_stages=args.channel_stages,
        dropout=args.dropout,
        use_deltas=True,
    )


def make_context_tokens(task_context: TaskIOContextEncoder, batch: Dict[str, torch.Tensor], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    B = int(batch["role_id"].shape[0])
    return task_context(
        B,
        device,
        dtype,
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


def flow_summaries(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # All summaries are [N,T,F]. They describe visible mechanisms without
    # exposing full high-rank tensors as one giant token.
    read = batch["read_flow"].mean(dim=(2, 3))
    prim = batch["primitive_slot_flow"].mean(dim=(2, 3))
    slot = batch["slot_transition_flow"].mean(dim=2).flatten(start_dim=2)
    ptrans = batch["primitive_transition_flow"].flatten(start_dim=2)
    comp = batch["slot_composition_flow"].mean(dim=2)
    write = batch["write_flow"].mean(dim=2)
    return {
        "read_flow": read,
        "primitive_slot_flow": prim,
        "slot_transition_flow": slot,
        "primitive_transition_flow": ptrans,
        "slot_composition_flow": comp,
        "write_flow": write,
    }


def task_masks(task_ids: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    # visible/target: [N,FLOW_KEYS]. Hidden mechanisms are the main targets;
    # visible mechanisms get a small consistency loss.
    N = int(task_ids.numel())
    visible = torch.zeros(N, len(FLOW_KEYS), device=device)
    target = torch.zeros(N, len(FLOW_KEYS), device=device)
    idx = {k: i for i, k in enumerate(FLOW_KEYS)}
    for n, tid in enumerate(task_ids.tolist()):
        fam = TASK_FAMILIES[int(tid) % len(TASK_FAMILIES)]
        if fam == "primitive_from_io":
            show = ("read_flow", "write_flow", "slot_composition_flow")
            hide = ("primitive_slot_flow", "primitive_transition_flow", "slot_transition_flow")
        elif fam == "operators_from_primitives":
            show = ("read_flow", "primitive_slot_flow", "write_flow")
            hide = ("primitive_transition_flow", "slot_transition_flow", "slot_composition_flow")
        elif fam == "read_write_from_ops":
            show = ("primitive_slot_flow", "primitive_transition_flow", "slot_transition_flow")
            hide = ("read_flow", "write_flow", "slot_composition_flow")
        elif fam == "write_from_read_primitive":
            show = ("read_flow", "primitive_slot_flow", "primitive_transition_flow", "slot_transition_flow")
            hide = ("write_flow", "slot_composition_flow")
        else:
            show = FLOW_KEYS
            hide = FLOW_KEYS
        for key in show:
            visible[n, idx[key]] = 1.0
        for key in hide:
            target[n, idx[key]] = 1.0
    return visible, target


class MechanismEvidenceBuilder(nn.Module):
    def __init__(self, cfg: AssemblerConfig, dim: int, dropout: float = 0.04):
        super().__init__()
        D, T, A, P, K = dim, cfg.layers * cfg.steps, cfg.address_cells, len(PRIMITIVES), cfg.primitive_slots
        self.flow_keys = FLOW_KEYS
        self.proj = nn.ModuleDict({
            "read_flow": nn.Linear(A, D),
            "primitive_slot_flow": nn.Linear(P, D),
            "slot_transition_flow": nn.Linear(K * K, D),
            "primitive_transition_flow": nn.Linear(P * P, D),
            "slot_composition_flow": nn.Linear(K, D),
            "write_flow": nn.Linear(A, D),
        })
        self.type_embed = nn.Parameter(torch.randn(len(FLOW_KEYS), D) * 0.02)
        self.step_embed = nn.Parameter(torch.randn(T, D) * 0.02)
        self.visible_embed = nn.Linear(1, D, bias=False)
        self.feedback_proj = nn.Sequential(
            nn.Linear(4, D),
            nn.GELU(),
            nn.Linear(D, D),
        )
        self.task_embed = nn.Embedding(len(TASK_FAMILIES), D)
        self.free = nn.Parameter(torch.randn(4, D) * 0.02)
        self.norm = nn.LayerNorm(D)
        self.drop = nn.Dropout(dropout)

    def forward(self, summaries: Dict[str, torch.Tensor], visible: torch.Tensor, task_ids: torch.Tensor, feedback: torch.Tensor | None = None) -> torch.Tensor:
        first = next(iter(summaries.values()))
        N, T, Dvc = first.shape
        tokens: List[torch.Tensor] = []
        for i, key in enumerate(self.flow_keys):
            x = summaries[key].to(device=visible.device, dtype=torch.float32)
            v = visible[:, i].view(N, 1, 1)
            # Hidden mechanisms are zeroed, but the model still sees which
            # mechanism type and step are being queried.
            tok = self.proj[key](x * v)
            tok = tok + self.type_embed[i].view(1, 1, -1)
            tok = tok + self.step_embed[:T].view(1, T, -1)
            tok = tok + self.visible_embed(v.expand(N, T, 1))
            tokens.append(tok)
        task = self.task_embed(task_ids.to(visible.device).long()).view(N, 1, -1)
        if feedback is None:
            feedback = torch.ones(N, 4, device=visible.device, dtype=torch.float32)
        feedback_tok = self.feedback_proj(feedback.to(device=visible.device, dtype=torch.float32)).view(N, 1, -1)
        free = self.free.view(1, -1, self.free.shape[-1]).expand(N, -1, -1)
        return self.drop(self.norm(torch.cat([task, feedback_tok, free] + tokens, dim=1)))


def targets_from_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: batch[key].to(device, non_blocking=True).transpose(0, 1).contiguous() for key in FLOW_KEYS}


def flow_kl_per_sample(pred: torch.Tensor, target: torch.Tensor, dim: int = -1) -> torch.Tensor:
    pred = pred.float()
    target = target.to(device=pred.device, dtype=pred.dtype)
    while target.ndim < pred.ndim:
        target = target.unsqueeze(1)
    target = target.expand_as(pred)
    target = target / target.sum(dim=dim, keepdim=True).clamp_min(1e-8)
    pred = pred / pred.sum(dim=dim, keepdim=True).clamp_min(1e-8)
    kl = F.kl_div(pred.clamp_min(1e-8).log(), target, reduction="none").sum(dim=dim)
    if kl.ndim < 2:
        return kl.reshape(1)
    reduce_dims = [d for d in range(kl.ndim) if d != 1]
    return kl.mean(dim=reduce_dims)


def masked_mechanism_loss(aux, targets: Dict[str, torch.Tensor], target_mask: torch.Tensor, visible_mask: torch.Tensor, sample_weight: torch.Tensor, args) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # target_mask/visible_mask are [N,6]. KL tensors reduce over distribution
    # dimensions but keep [T,N,...] instance axes where needed.
    losses: Dict[str, torch.Tensor] = {}
    total = torch.zeros((), device=aux.cells.device)
    pred_target_pairs = {
        "read_flow": (aux.read_flow, targets["read_flow"], -1),
        "primitive_slot_flow": (aux.primitive_slot_flow, targets["primitive_slot_flow"], -1),
        "slot_transition_flow": (aux.slot_transition_flow, targets["slot_transition_flow"], -1),
        "primitive_transition_flow": (aux.primitive_transition_flow, targets["primitive_transition_flow"], -1),
        "slot_composition_flow": (aux.slot_composition_flow, targets["slot_composition_flow"], -1),
        "write_flow": (aux.write_flow, targets["write_flow"], -1),
    }
    weights = {
        "read_flow": args.w_read,
        "primitive_slot_flow": args.w_primitive,
        "slot_transition_flow": args.w_slot_transition,
        "primitive_transition_flow": args.w_primitive_transition,
        "slot_composition_flow": args.w_composition,
        "write_flow": args.w_write,
    }
    for i, key in enumerate(FLOW_KEYS):
        pred, target, dim = pred_target_pairs[key]
        kl = flow_kl(pred, target, dim=dim)
        per_sample = flow_kl_per_sample(pred, target, dim=dim)
        # Hidden mechanisms are primary; visible mechanisms are weak consistency.
        coef = target_mask[:, i] + float(args.visible_consistency) * visible_mask[:, i]
        weighted = coef * sample_weight.to(device=coef.device, dtype=coef.dtype)
        value = (per_sample * weighted).sum() / weighted.sum().clamp_min(1.0)
        losses[f"{key}_kl"] = kl.detach()
        losses[f"{key}_weighted"] = value.detach()
        total = total + float(weights[key]) * value
    entropy_keep = torch.zeros((), device=aux.cells.device)
    if args.lambda_entropy_keep > 0:
        for ent in aux.entropies.values():
            entropy_keep = entropy_keep + F.relu(torch.tensor(float(args.min_entropy), device=aux.cells.device) - ent.float()).pow(2)
    total = total + float(args.lambda_entropy_keep) * entropy_keep
    losses["entropy_keep"] = entropy_keep.detach()
    return total, losses


def corrupt_summaries(summaries: Dict[str, torch.Tensor], batch_idx: torch.Tensor, epoch: int, train: bool, args, device: torch.device) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    first = next(iter(summaries.values()))
    N = int(first.shape[0])
    g = torch.Generator(device="cpu")
    base_seed = int(args.seed + epoch * 9176 + (13 if train else 777_777))
    if torch.is_tensor(batch_idx):
        base_seed += int(batch_idx.detach().cpu().sum().item()) % 100_000
    g.manual_seed(base_seed)
    if args.feedback_noise_max <= 0:
        quality = torch.ones(N, device=device)
    else:
        quality = 1.0 - torch.rand(N, generator=g, device="cpu").to(device) * float(args.feedback_noise_max)
        quality = quality.clamp(float(args.feedback_quality_min), 1.0)
    loss_proxy = 1.0 - quality
    pressure = loss_proxy.pow(0.5)
    epoch_frac = torch.full_like(quality, float(epoch) / max(1.0, float(args.epochs)))
    feedback = torch.stack([quality, loss_proxy, pressure, epoch_frac], dim=-1)
    out: Dict[str, torch.Tensor] = {}
    for key, x in summaries.items():
        q = quality.view(N, 1, 1)
        if x.shape[-1] > 1:
            uniform = torch.full_like(x, 1.0 / float(x.shape[-1]))
        else:
            uniform = torch.zeros_like(x)
        perm = torch.randperm(N, generator=g, device="cpu").to(device)
        mixed_other = x.index_select(0, perm)
        noise = torch.randn(x.shape, generator=g, device="cpu").to(device=device, dtype=x.dtype) * float(args.feedback_noise_std)
        bad = 0.55 * uniform + 0.35 * mixed_other + 0.10 * (x + noise)
        bad = bad.clamp_min(1e-6)
        bad = bad / bad.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        y = q * x + (1.0 - q) * bad
        out[key] = y / y.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    sample_weight = 1.0 + float(args.feedback_importance) * loss_proxy
    return out, feedback, sample_weight


def run_epoch(core, task_context, evidence_builder, loader, device, dtype, args, train: bool, opt=None, scaler=None, epoch: int = 0):
    core.train(train)
    task_context.train(train)
    evidence_builder.train(train)
    use_amp = str(device).startswith("cuda") and dtype != torch.float32
    total, n = 0.0, 0
    acc: Dict[str, float] = {}
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(args.seed + epoch * 1009 + (0 if train else 500000)))
    for step, batch in enumerate(loader, 1):
        bs = int(batch["role_id"].shape[0])
        task_ids = torch.randint(0, len(TASK_FAMILIES), (bs,), generator=rng)
        visible, target_mask = task_masks(task_ids, device)
        clean_summaries = {k: v.to(device, non_blocking=True) for k, v in flow_summaries(batch).items()}
        summaries, feedback, sample_weight = corrupt_summaries(clean_summaries, batch["idx"], epoch, train, args, device)
        if train:
            opt.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=str(device).split(":")[0], dtype=dtype, enabled=use_amp):
                ctx = make_context_tokens(task_context, batch, device, dtype)
                mech = evidence_builder(summaries, visible, task_ids.to(device), feedback=feedback).to(dtype=dtype)
                evidence = torch.cat([ctx, mech], dim=1)
                _cells, aux = core(evidence)
                loss, parts = masked_mechanism_loss(aux, targets_from_batch(batch, device), target_mask, visible, sample_weight, args)
        if train:
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                params = list(core.parameters()) + list(task_context.parameters()) + list(evidence_builder.parameters())
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            scaler.step(opt)
            scaler.update()
        total += float(loss.detach().cpu()) * bs
        n += bs
        for k, v in parts.items():
            acc[k] = acc.get(k, 0.0) + float(v.detach().cpu()) * bs
        if train and args.log_every and step % args.log_every == 0:
            print(f"epoch {epoch:03d} step {step:05d} loss={total/max(1,n):.4f} prim={acc.get('primitive_slot_flow_kl',0)/max(1,n):.3f} op={acc.get('primitive_transition_flow_kl',0)/max(1,n):.3f} write={acc.get('write_flow_kl',0)/max(1,n):.3f}", flush=True)
    out = {"loss": total / max(1, n)}
    for k, v in acc.items():
        out[k] = v / max(1, n)
    return out


def load_init(core: MatrixProgramAssemblerCore, task_context: TaskIOContextEncoder, path: str, device: torch.device) -> None:
    if not path:
        return
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("assembler_core", ckpt.get("assembler_skill_base", ckpt.get("core", ckpt)))
    missing, unexpected = core.load_state_dict(state, strict=False)
    print(f"loaded init assembler {path} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    ctx = ckpt.get("task_context", ckpt.get("task_context_base"))
    if isinstance(ctx, dict):
        ctx = {k[len("task_context."):] if k.startswith("task_context.") else k: v for k, v in ctx.items()}
        miss, unexp = task_context.load_state_dict(ctx, strict=False)
        print(f"loaded init task_context missing={len(miss)} unexpected={len(unexp)}", flush=True)


def run(args) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    if str(device).startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(args.amp, torch.float32)
    out = ensure_dir(args.out_dir)
    pack = torch.load(args.dataset, map_location="cpu")
    cfg = infer_cfg_from_pack(pack, args)
    N = int(pack["read_flow"].shape[0])
    idx = torch.randperm(N)
    n_train = max(1, int(N * args.train_frac))
    train = DataLoader(MechanismFlowDataset(pack, idx[:n_train].tolist()), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=args.pin_memory)
    val = DataLoader(MechanismFlowDataset(pack, idx[n_train:].tolist() or idx[:n_train].tolist()), batch_size=args.eval_batch_size, shuffle=False, num_workers=args.workers, pin_memory=args.pin_memory)

    core = MatrixProgramAssemblerCore(cfg).to(device)
    task_context = TaskIOContextEncoder(TaskContextV2Config(dim=args.dim, free_tokens=args.task_context_tokens, max_head_tokens=args.head_context_tokens, dropout=args.dropout)).to(device)
    evidence_builder = MechanismEvidenceBuilder(cfg, args.dim, dropout=args.dropout).to(device)
    load_init(core, task_context, args.init_assembler, device)

    params = list(core.parameters()) + list(task_context.parameters()) + list(evidence_builder.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=str(device).startswith("cuda") and dtype == torch.float16)
    fields = [
        "epoch", "train_loss", "val_loss", "best_val",
        "read_flow_kl", "primitive_slot_flow_kl", "slot_transition_flow_kl",
        "primitive_transition_flow_kl", "slot_composition_flow_kl", "write_flow_kl",
    ]
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()
    best, best_epoch = 1e18, 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(core, task_context, evidence_builder, train, device, dtype, args, True, opt, scaler, epoch)
        with torch.no_grad():
            va = run_epoch(core, task_context, evidence_builder, val, device, dtype, args, False, epoch=epoch)
        if va["loss"] < best:
            best, best_epoch = va["loss"], epoch
            torch.save({
                "assembler_core": core.state_dict(),
                "task_context": task_context.state_dict(),
                "mechanism_evidence_builder": evidence_builder.state_dict(),
                "config": cfg.__dict__,
                "args": vars(args),
                "best_val_loss": best,
                "epoch": epoch,
                "meta": {"source_dataset": args.dataset, "skill": "mechanism_flow_masked", "task_families": list(TASK_FAMILIES), "pack_meta": pack.get("meta", {})},
            }, out / "assembler_mechanism_skill_best.pt")
        torch.save({
            "assembler_core": core.state_dict(),
            "task_context": task_context.state_dict(),
            "mechanism_evidence_builder": evidence_builder.state_dict(),
            "config": cfg.__dict__,
            "args": vars(args),
            "best_val_loss": best,
            "epoch": epoch,
            "meta": {"source_dataset": args.dataset, "skill": "mechanism_flow_masked", "task_families": list(TASK_FAMILIES), "pack_meta": pack.get("meta", {})},
        }, out / "assembler_mechanism_skill_last.pt")
        row = {"epoch": epoch, "train_loss": tr["loss"], "val_loss": va["loss"], "best_val": best}
        for k in fields:
            if k in va:
                row[k] = va[k]
        with (out / "metrics.csv").open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writerow({k: row.get(k, 0.0) for k in fields})
        print(f"epoch {epoch:03d}/{args.epochs} train={tr['loss']:.4f} val={va['loss']:.4f} best={best:.4f}@{best_epoch}", flush=True)
    write_json(out / "final_report.json", {
        "best_val_loss": best,
        "best_epoch": best_epoch,
        "elapsed_sec": time.time() - t0,
        "checkpoint": str(out / "assembler_mechanism_skill_best.pt"),
        "dataset": args.dataset,
        "config": cfg.__dict__,
        "task_families": list(TASK_FAMILIES),
    })


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--out-dir", default="matrix_program_core/runs/mechanism_skill")
    p.add_argument("--init-assembler", default="")
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", choices=["fp16", "bf16", "fp32", "off"], default="fp32")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dim", type=int, default=96)
    p.add_argument("--evidence-cells", type=int, default=48)
    p.add_argument("--steps-core", type=int, default=2)
    p.add_argument("--memory-cells", type=int, default=4)
    p.add_argument("--global-cells", type=int, default=2)
    p.add_argument("--channel-stages", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.04)
    p.add_argument("--task-context-tokens", type=int, default=4)
    p.add_argument("--head-context-tokens", type=int, default=10)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--train-frac", type=float, default=0.85)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=0.75)
    p.add_argument("--visible-consistency", type=float, default=0.15)
    p.add_argument("--feedback-noise-max", type=float, default=0.70)
    p.add_argument("--feedback-quality-min", type=float, default=0.20)
    p.add_argument("--feedback-noise-std", type=float, default=0.03)
    p.add_argument("--feedback-importance", type=float, default=1.50)
    p.add_argument("--lambda-entropy-keep", type=float, default=0.01)
    p.add_argument("--min-entropy", type=float, default=0.55)
    p.add_argument("--w-read", type=float, default=0.25)
    p.add_argument("--w-primitive", type=float, default=0.35)
    p.add_argument("--w-slot-transition", type=float, default=0.20)
    p.add_argument("--w-primitive-transition", type=float, default=0.30)
    p.add_argument("--w-composition", type=float, default=0.12)
    p.add_argument("--w-write", type=float, default=0.25)
    p.add_argument("--log-every", type=int, default=50)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
