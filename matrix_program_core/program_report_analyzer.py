#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Program Central Mind Phase 0: report-only analyzer.

This script does not load weights and does not change training. It reads an
agent report directory and produces a structured diagnosis:

- baseline vs macro variants;
- best/last accuracy;
- macro entropy/top1/gain;
- class/slot collapse signals;
- skill drift;
- recommended next actions.

Use it before implementing the active central edit controller.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _f(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def _i(x: Any, default: int = -1) -> int:
    try:
        if x is None or x == "":
            return default
        return int(float(x))
    except Exception:
        return default


def _fmt(x: float, digits: int = 4) -> str:
    if x != x:
        return "nan"
    return f"{x:.{digits}f}"


def _variant_name_from_report(path: Path) -> str:
    name = path.name
    if name.startswith("audio_"):
        name = name[len("audio_"):]
    if name.endswith("_final_report.json"):
        name = name[: -len("_final_report.json")]
    return name


def _load_audio_variants(report_dir: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for f in sorted(report_dir.glob("audio_*_final_report.json")):
        name = _variant_name_from_report(f)
        final = _read_json(f) or {}
        metrics = _read_csv(report_dir / f"audio_{name}_metrics.csv")
        out[name] = {"final": final, "metrics": metrics, "path": str(f)}
    return out


def _load_macro_reports(report_dir: Path) -> Dict[str, Dict[str, Any]]:
    out = {}
    for f in sorted(report_dir.glob("macro_step_bank_M*_report.json")):
        key = f.stem.replace("macro_step_bank_", "")
        out[key] = _read_json(f) or {}
    return out


def _best_from_rows(rows: List[Dict[str, str]]) -> Tuple[float, int]:
    best = -1e9
    epoch = -1
    for r in rows:
        b = _f(r.get("best_acc"), default=float("nan"))
        v = _f(r.get("val_acc"), default=float("nan"))
        cand = b if b == b else v
        if cand == cand and cand > best:
            best = cand
            epoch = _i(r.get("epoch"), epoch)
    return best, epoch


def _last_float(rows: List[Dict[str, str]], key: str) -> float:
    if not rows:
        return float("nan")
    return _f(rows[-1].get(key))


def _analyze_variant(name: str, obj: Dict[str, Any]) -> Dict[str, Any]:
    final = obj.get("final", {}) or {}
    rows = obj.get("metrics", []) or []
    best = _f(final.get("best_acc"), default=float("nan"))
    best_epoch = _i(final.get("best_epoch"), default=-1)
    if best != best:
        best, best_epoch = _best_from_rows(rows)
    last = rows[-1] if rows else {}
    return {
        "name": name,
        "best_acc": best,
        "best_epoch": best_epoch,
        "last_epoch": _i(last.get("epoch"), -1),
        "last_val_acc": _f(last.get("val_acc")),
        "last_train_acc": _f(last.get("train_acc")),
        "last_train_ce": _f(last.get("train_ce")),
        "last_skill": _f(last.get("skill")),
        "class_read_div": _f(last.get("class_read_div")),
        "slot_div": _f(last.get("slot_div")),
        "class_attn_entropy": _f(last.get("class_attn_entropy")),
        "class_slot_prior": _f(last.get("class_slot_prior")),
        "phase_balance": _f(last.get("phase_balance")),
        "macro_entropy": _f(last.get("macro_entropy")),
        "macro_top1_usage": _f(last.get("macro_top1_usage")),
        "macro_gain": _f(last.get("macro_gain")),
        "rows": len(rows),
        "trainable_summary": final.get("trainable_summary"),
    }


def _diagnose_macro(macro_reports: Dict[str, Dict[str, Any]]) -> List[str]:
    recs: List[str] = []
    for name, r in sorted(macro_reports.items()):
        imp = _f(r.get("kl_improvement"))
        ent = _f(r.get("macro_entropy_norm"))
        top = _f(r.get("macro_top1_usage"))
        bank = _f(r.get("bank_weighted_kl"))
        mean = _f(r.get("mean_weighted_kl"))
        if imp == imp and imp > 0.03:
            recs.append(f"{name}: macro bank has useful structure: weighted KL {bank:.4f} vs mean {mean:.4f}, improvement {imp:.4f}.")
        else:
            recs.append(f"{name}: macro bank structure weak; consider more macros, better fingerprints, or source pack filtering.")
        if ent == ent and ent < 0.55:
            recs.append(f"{name}: macro prototype usage is imbalanced in offline clustering; entropy={ent:.3f}.")
        if top == top and top > 0.55:
            recs.append(f"{name}: top macro dominates offline bank; top1={top:.3f}. Try more clusters or diversity filtering.")
    return recs


def _diagnose_variants(variants: Dict[str, Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
    rows = [_analyze_variant(n, v) for n, v in variants.items()]
    rows = sorted(rows, key=lambda x: x["best_acc"], reverse=True)
    recs: List[str] = []
    if not rows:
        return ["No audio variants found."], rows
    best = rows[0]
    recs.append(f"Best audio variant: {best['name']} best_acc={_fmt(best['best_acc'])} at epoch {best['best_epoch']}.")

    baseline = None
    for r in rows:
        if r["name"].startswith("baseline"):
            baseline = r
            break
    if baseline is not None:
        for r in rows:
            if r is baseline:
                continue
            delta = r["best_acc"] - baseline["best_acc"]
            if delta == delta:
                sign = "+" if delta >= 0 else ""
                recs.append(f"{r['name']} vs {baseline['name']}: delta_best={sign}{delta:.4f}.")

    for r in rows:
        if r["macro_entropy"] == r["macro_entropy"]:
            if r["macro_entropy"] < 0.35:
                recs.append(f"{r['name']}: macro selector collapse risk: macro_entropy={r['macro_entropy']:.3f}. Lower MACRO_GAIN or add macro diversity.")
            if r["macro_top1_usage"] == r["macro_top1_usage"] and r["macro_top1_usage"] > 0.75:
                recs.append(f"{r['name']}: one macro dominates: top1={r['macro_top1_usage']:.3f}. Try MACRO_GAIN=0.08 or macro reuse penalty.")
            if r["macro_gain"] == r["macro_gain"] and r["macro_gain"] < 0.01:
                recs.append(f"{r['name']}: macro path effectively unused: gain={r['macro_gain']:.4f}. Try MACRO_GAIN=0.25 or unfreeze selector/gain.")

        if r["class_read_div"] == r["class_read_div"] and r["class_read_div"] > 0.55:
            recs.append(f"{r['name']}: class reads still too similar: class_read_div={r['class_read_div']:.3f}. Need central analyzer on class-slot pressure or stronger class-slot prior.")
        if r["slot_div"] == r["slot_div"] and r["slot_div"] > 0.25:
            recs.append(f"{r['name']}: slot collapse likely: slot_div={r['slot_div']:.3f}. Need slot pressure trace and slot dropout/usage balance.")
        if r["last_train_acc"] == r["last_train_acc"] and r["last_val_acc"] == r["last_val_acc"]:
            if abs(r["last_train_acc"] - r["last_val_acc"]) < 0.03 and r["best_acc"] < 0.65:
                recs.append(f"{r['name']}: not overfit; train and val are both capped. Bottleneck likely evidence/capacity/program usage, not regularization only.")
    return recs, rows


def analyze(report_dir: Path) -> Dict[str, Any]:
    macro_reports = _load_macro_reports(report_dir)
    variants = _load_audio_variants(report_dir)
    macro_recs = _diagnose_macro(macro_reports)
    variant_recs, variant_rows = _diagnose_variants(variants)
    summary = {
        "report_dir": str(report_dir),
        "macro_reports": macro_reports,
        "audio_variants": variant_rows,
        "recommendations": macro_recs + variant_recs,
    }
    return summary


def write_markdown(obj: Dict[str, Any], path: Path) -> None:
    lines: List[str] = []
    lines.append("# Program report analyzer")
    lines.append("")
    lines.append(f"report_dir: `{obj['report_dir']}`")
    lines.append("")
    lines.append("## Audio variants")
    lines.append("")
    rows = obj.get("audio_variants", [])
    if rows:
        lines.append("| variant | best | best_epoch | last_val | train | skill | class_read_div | slot_div | macro_entropy | macro_top1 | macro_gain |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in rows:
            lines.append(
                f"| {r['name']} | {_fmt(r['best_acc'])} | {r['best_epoch']} | {_fmt(r['last_val_acc'])} | {_fmt(r['last_train_acc'])} | {_fmt(r['last_skill'])} | {_fmt(r['class_read_div'])} | {_fmt(r['slot_div'])} | {_fmt(r['macro_entropy'])} | {_fmt(r['macro_top1_usage'])} | {_fmt(r['macro_gain'])} |"
            )
    else:
        lines.append("No audio variants found.")
    lines.append("")
    lines.append("## Macro bank reports")
    lines.append("")
    mr = obj.get("macro_reports", {})
    if mr:
        lines.append("| bank | M | bank_KL | mean_KL | improvement | entropy | top1 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for name, r in sorted(mr.items()):
            lines.append(
                f"| {name} | {_i(r.get('macro_count'), -1)} | {_fmt(_f(r.get('bank_weighted_kl')))} | {_fmt(_f(r.get('mean_weighted_kl')))} | {_fmt(_f(r.get('kl_improvement')))} | {_fmt(_f(r.get('macro_entropy_norm')))} | {_fmt(_f(r.get('macro_top1_usage')))} |"
            )
    else:
        lines.append("No macro bank reports found.")
    lines.append("")
    lines.append("## Recommendations")
    lines.append("")
    for x in obj.get("recommendations", []):
        lines.append(f"- {x}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-dir", required=True)
    ap.add_argument("--out-json", default="")
    ap.add_argument("--out-md", default="")
    args = ap.parse_args()
    rd = Path(args.report_dir)
    obj = analyze(rd)
    out_json = Path(args.out_json) if args.out_json else rd / "program_report_analysis.json"
    out_md = Path(args.out_md) if args.out_md else rd / "program_report_analysis.md"
    out_json.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(obj, out_md)
    print(out_md)
    print(out_json)
    print("\n" + out_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
