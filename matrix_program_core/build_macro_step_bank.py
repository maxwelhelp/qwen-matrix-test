#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build an offline MacroStepBank from existing matrix-program flow recipes.

MVP rules:
- no primitive birth;
- no online promotion;
- no hard routing;
- macros are soft prototypes of existing read/primitive/transition/write flows.

Input pack must contain standard flow tensors:
  read_flow                     [N,T,B,K,A]
  primitive_slot_flow           [N,T,B,K,P]
  slot_transition_flow          [N,T,B,K,K]
  primitive_transition_flow     [N,T,P,P]
  slot_composition_flow         [N,T,B,K]
  write_flow                    [N,T,B,A]

For every (program, step, block), this script creates a fingerprint and clusters
it into M macro-step prototypes. The saved bank contains average full flows per
macro. It is intentionally frozen/static for the first controlled experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simple_butterfly_matrix.simple_butterfly_matrix import PRIMITIVES  # noqa: E402

FLOW_KEYS = (
    "read_flow",
    "primitive_slot_flow",
    "slot_transition_flow",
    "primitive_transition_flow",
    "slot_composition_flow",
    "write_flow",
)


def _norm_last(x: torch.Tensor) -> torch.Tensor:
    x = x.float().clamp_min(1e-8)
    return x / x.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def _load_pack(path: Path) -> Dict[str, torch.Tensor]:
    pack = torch.load(path, map_location="cpu")
    missing = [k for k in FLOW_KEYS if k not in pack]
    if missing:
        raise KeyError(f"input pack is missing flow keys: {missing}")
    return pack


def _maybe_reduce_to_programs(pack: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Repair episode packs repeat target programs. Keep all rows for MVP.

    We deliberately do not deduplicate by episode_program_index here because the
    repeated rows are harmless and can act as frequency weighting. The report
    still exposes episode metadata if present.
    """

    return pack


def _flatten_step_block_recipes(pack: Dict[str, torch.Tensor], max_items: int, seed: int):
    read = _norm_last(pack["read_flow"])                       # [N,T,B,K,A]
    prim = _norm_last(pack["primitive_slot_flow"])             # [N,T,B,K,P]
    slot = _norm_last(pack["slot_transition_flow"])            # [N,T,B,K,K]
    ptrans = _norm_last(pack["primitive_transition_flow"])     # [N,T,P,P]
    comp = _norm_last(pack["slot_composition_flow"])           # [N,T,B,K]
    write = _norm_last(pack["write_flow"])                     # [N,T,B,A]

    N, T, B, K, A = read.shape
    P = prim.shape[-1]
    total = N * T * B
    idx = torch.arange(total)
    if max_items > 0 and total > max_items:
        g = torch.Generator(device="cpu").manual_seed(seed)
        idx = idx[torch.randperm(total, generator=g)[:max_items]]

    n_idx = idx // (T * B)
    rem = idx % (T * B)
    t_idx = rem // B
    b_idx = rem % B

    r = read[n_idx, t_idx, b_idx]           # [M,K,A]
    pr = prim[n_idx, t_idx, b_idx]          # [M,K,P]
    st = slot[n_idx, t_idx, b_idx]          # [M,K,K]
    pt = ptrans[n_idx, t_idx]               # [M,P,P]
    cp = comp[n_idx, t_idx, b_idx]          # [M,K]
    wr = write[n_idx, t_idx, b_idx]         # [M,A]

    # Fingerprints are normalized histograms, not giant raw tensors.
    # Keep enough structure to separate read/primitive/transition/write families.
    fp_parts = [
        r.reshape(r.shape[0], -1),
        pr.reshape(pr.shape[0], -1),
        st.reshape(st.shape[0], -1),
        pt.reshape(pt.shape[0], -1),
        cp.reshape(cp.shape[0], -1),
        wr.reshape(wr.shape[0], -1),
        F.one_hot(t_idx % max(1, T), num_classes=T).float() * 0.15,
        F.one_hot(b_idx % max(1, B), num_classes=B).float() * 0.10,
    ]
    fp = torch.cat(fp_parts, dim=-1).float()
    fp = F.normalize(fp, dim=-1)
    recipes = {"read": r, "primitive": pr, "slot_transition": st, "primitive_transition": pt, "composition": cp, "write": wr}
    meta = {"n_idx": n_idx, "t_idx": t_idx, "b_idx": b_idx, "N": N, "T": T, "B": B, "K": K, "A": A, "P": P, "total_step_blocks": total}
    return fp, recipes, meta


def _farthest_init(x: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = x.shape[0]
    first = int(torch.randint(0, n, (1,), generator=g))
    centers = [first]
    best = torch.cdist(x[first:first + 1], x).squeeze(0).pow(2)
    for _ in range(1, k):
        nxt = int(torch.argmax(best))
        centers.append(nxt)
        d = torch.cdist(x[nxt:nxt + 1], x).squeeze(0).pow(2)
        best = torch.minimum(best, d)
    return x[torch.tensor(centers, dtype=torch.long)].clone()


def _kmeans(x: torch.Tensor, k: int, iters: int, seed: int, log_every: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = x.shape[0]
    k = min(int(k), int(n))
    centers = _farthest_init(x, k, seed)
    labels = torch.zeros(n, dtype=torch.long)
    for it in range(max(1, int(iters))):
        # cosine-ish because x and centers are normalized; use euclidean for stability.
        dist = torch.cdist(x, centers).pow(2)
        labels = dist.argmin(dim=-1)
        new = torch.zeros_like(centers)
        counts = torch.bincount(labels, minlength=k).float().clamp_min(0)
        new.index_add_(0, labels, x)
        nonempty = counts > 0
        new[nonempty] = new[nonempty] / counts[nonempty].view(-1, 1)
        # Re-seed empty centers from worst points.
        if (~nonempty).any():
            min_dist = dist.min(dim=1).values
            worst = torch.argsort(min_dist, descending=True)
            empty_ids = torch.where(~nonempty)[0]
            for j, cid in enumerate(empty_ids):
                new[cid] = x[worst[j % len(worst)]]
                counts[cid] = 1.0
        centers = F.normalize(new, dim=-1)
        if log_every and (it + 1) % log_every == 0:
            print(f"[macro-kmeans] iter={it+1}/{iters} active={(counts>0).sum().item()} max_count={counts.max().item():.0f}", flush=True)
    dist = torch.cdist(x, centers).pow(2)
    labels = dist.argmin(dim=-1)
    min_dist = dist.gather(1, labels.view(-1, 1)).squeeze(1)
    return centers, labels, min_dist


def _avg_by_label(values: torch.Tensor, labels: torch.Tensor, k: int) -> torch.Tensor:
    flat = values.reshape(values.shape[0], -1).float()
    out = torch.zeros(k, flat.shape[1], dtype=torch.float32)
    out.index_add_(0, labels, flat)
    counts = torch.bincount(labels, minlength=k).float().clamp_min(1.0)
    out = out / counts.view(-1, 1)
    return out.view(k, *values.shape[1:])


def _kl_rows(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = _norm_last(pred)
    target = _norm_last(target)
    return F.kl_div(pred.clamp_min(1e-8).log(), target, reduction="none").sum(dim=-1).mean()


def _build_bank(recipes: Dict[str, torch.Tensor], labels: torch.Tensor, k: int) -> Dict[str, torch.Tensor]:
    bank = {
        "macro_read": _norm_last(_avg_by_label(recipes["read"], labels, k)),
        "macro_primitive": _norm_last(_avg_by_label(recipes["primitive"], labels, k)),
        "macro_slot_transition": _norm_last(_avg_by_label(recipes["slot_transition"], labels, k)),
        "macro_primitive_transition": _norm_last(_avg_by_label(recipes["primitive_transition"], labels, k)),
        "macro_composition": _norm_last(_avg_by_label(recipes["composition"], labels, k)),
        "macro_write": _norm_last(_avg_by_label(recipes["write"], labels, k)),
    }
    return bank


def _evaluate_bank(bank: Dict[str, torch.Tensor], recipes: Dict[str, torch.Tensor], labels: torch.Tensor, min_dist: torch.Tensor) -> Dict[str, float]:
    pred = {
        "read": bank["macro_read"][labels],
        "primitive": bank["macro_primitive"][labels],
        "slot_transition": bank["macro_slot_transition"][labels],
        "primitive_transition": bank["macro_primitive_transition"][labels],
        "composition": bank["macro_composition"][labels],
        "write": bank["macro_write"][labels],
    }
    mean_pred = {k: _norm_last(v.mean(dim=0, keepdim=True)).expand_as(v) for k, v in recipes.items()}
    metrics: Dict[str, float] = {}
    weights = {"read": 1.0, "primitive": 1.0, "slot_transition": 0.5, "primitive_transition": 1.0, "composition": 0.5, "write": 1.0}
    bank_total = 0.0
    mean_total = 0.0
    for key in recipes:
        b = float(_kl_rows(pred[key], recipes[key]))
        m = float(_kl_rows(mean_pred[key], recipes[key]))
        metrics[f"bank_{key}_kl"] = b
        metrics[f"mean_{key}_kl"] = m
        bank_total += weights[key] * b
        mean_total += weights[key] * m
    metrics["bank_weighted_kl"] = bank_total / sum(weights.values())
    metrics["mean_weighted_kl"] = mean_total / sum(weights.values())
    metrics["kl_improvement"] = metrics["mean_weighted_kl"] - metrics["bank_weighted_kl"]
    metrics["mean_cluster_distance"] = float(min_dist.mean())
    metrics["p95_cluster_distance"] = float(min_dist.quantile(0.95))
    return metrics


def _write_preview(path: Path, bank: Dict[str, torch.Tensor], counts: torch.Tensor, top: int = 64) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prim_names = list(PRIMITIVES)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["macro", "count", "top_read", "top_write", "top_primitives", "top_primitive_transitions"])
        for m in range(min(int(bank["macro_read"].shape[0]), int(top))):
            read_hist = bank["macro_read"][m].mean(dim=0)
            write_hist = bank["macro_write"][m]
            prim_hist = bank["macro_primitive"][m].mean(dim=0)
            ptrans = bank["macro_primitive_transition"][m]
            read_top = ";".join(f"A{int(i)}:{float(v):.3f}" for v, i in zip(*torch.topk(read_hist, min(5, read_hist.numel()))))
            write_top = ";".join(f"A{int(i)}:{float(v):.3f}" for v, i in zip(*torch.topk(write_hist, min(5, write_hist.numel()))))
            prim_top = ";".join(f"{prim_names[int(i)]}:{float(v):.3f}" for v, i in zip(*torch.topk(prim_hist, min(5, prim_hist.numel()))))
            flat = ptrans.flatten()
            vals, ids = torch.topk(flat, min(6, flat.numel()))
            pairs = []
            P = ptrans.shape[-1]
            for v, idx in zip(vals, ids):
                a = int(idx) // P; b = int(idx) % P
                pairs.append(f"{prim_names[a]}->{prim_names[b]}:{float(v):.3f}")
            w.writerow([m, int(counts[m]), read_top, write_top, prim_top, ";".join(pairs)])


def build(args) -> Dict[str, object]:
    t0 = time.time()
    pack = _maybe_reduce_to_programs(_load_pack(Path(args.input)))
    fp, recipes, meta = _flatten_step_block_recipes(pack, args.max_items, args.seed)
    print(json.dumps({"items": int(fp.shape[0]), "total_step_blocks": int(meta["total_step_blocks"]), "K": int(meta["K"]), "A": int(meta["A"]), "P": int(meta["P"])}, indent=2), flush=True)
    centers, labels, min_dist = _kmeans(fp, args.macros, args.kmeans_iters, args.seed, log_every=max(0, args.log_every))
    k = int(centers.shape[0])
    bank = _build_bank(recipes, labels, k)
    counts = torch.bincount(labels, minlength=k).long()
    metrics = _evaluate_bank(bank, recipes, labels, min_dist)
    count_p = counts.float() / counts.sum().clamp_min(1)
    macro_entropy = float((-(count_p.clamp_min(1e-8) * count_p.clamp_min(1e-8).log()).sum() / math.log(max(2, k))))
    metrics.update({
        "macro_count": k,
        "items": int(fp.shape[0]),
        "total_step_blocks": int(meta["total_step_blocks"]),
        "active_macros": int((counts > 0).sum()),
        "macro_entropy_norm": macro_entropy,
        "macro_top1_usage": float(counts.max().float() / counts.sum().clamp_min(1)),
        "elapsed_sec": time.time() - t0,
    })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_obj = {
        **bank,
        "macro_feature": centers.float(),
        "macro_counts": counts,
        "macro_metrics": metrics,
        "config": {
            "macros": k,
            "K": int(meta["K"]),
            "A": int(meta["A"]),
            "P": int(meta["P"]),
            "source": str(args.input),
            "max_items": int(args.max_items),
            "kmeans_iters": int(args.kmeans_iters),
            "seed": int(args.seed),
        },
        "meta": {
            "truth_level": "offline_macro_step_prototypes_from_flow_recipes",
            "source_meta": pack.get("meta", {}),
        },
    }
    torch.save(save_obj, out)
    report_path = Path(args.report) if args.report else out.with_suffix(".report.json")
    preview_path = Path(args.preview) if args.preview else out.with_suffix(".preview.csv")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_preview(preview_path, bank, counts, top=args.preview_top)
    print(json.dumps({"out": str(out), "report": str(report_path), "preview": str(preview_path), "metrics": metrics}, ensure_ascii=False, indent=2), flush=True)
    return save_obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="flow pack or repair episode dataset .pt")
    ap.add_argument("--out", required=True, help="output macro_step_bank.pt")
    ap.add_argument("--report", default="")
    ap.add_argument("--preview", default="")
    ap.add_argument("--macros", type=int, default=32)
    ap.add_argument("--max-items", type=int, default=160000, help="sampled step-block recipes; 0 means all")
    ap.add_argument("--kmeans-iters", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--preview-top", type=int, default=64)
    ap.add_argument("--log-every", type=int, default=5)
    args = ap.parse_args()
    build(args)


if __name__ == "__main__":
    main()
