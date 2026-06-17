#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python - <<'PY'
from pathlib import Path

p = Path("matrix_program_core/train_assembler_multitask_pretrain.py")
txt = p.read_text(encoding="utf-8")
orig = txt

needle = '''    train_ds = MultiTaskAssemblerDataset(args.train_n, args.classes, args.evidence_cells, args.dim, task_families, solution_families, args.seed, args.noise)
    val_ds = MultiTaskAssemblerDataset(args.val_n, args.classes, args.evidence_cells, args.dim, task_families, solution_families, args.seed + 1000, args.noise)
    train = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=args.pin_memory)
'''
repl = '''    train_ds = MultiTaskAssemblerDataset(args.train_n, args.classes, args.evidence_cells, args.dim, task_families, solution_families, args.seed, args.noise)
    val_ds = MultiTaskAssemblerDataset(args.val_n, args.classes, args.evidence_cells, args.dim, task_families, solution_families, args.seed + 1000, args.noise)
    # Critical: validation must test new examples/noise, not a different label universe.
    # Keep the same class/task/head/solution prototypes across train/val so class ids
    # and task ids mean the same thing. Otherwise validation is effectively impossible.
    val_ds.class_proto = train_ds.class_proto.clone()
    val_ds.task_proto = train_ds.task_proto.clone()
    val_ds.solution_proto = train_ds.solution_proto.clone()
    val_ds.head_proto = train_ds.head_proto.clone()
    train = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=args.pin_memory)
'''
if needle in txt:
    txt = txt.replace(needle, repl)
elif "val_ds.class_proto = train_ds.class_proto.clone()" not in txt:
    raise SystemExit("PATCH_FAIL: train/val dataset block not found")

if txt != orig:
    p.write_text(txt, encoding="utf-8")
    print("[patch] multitask train/val now share class/task/head/solution prototypes")
else:
    print("[patch] already applied")
PY

grep -n "same class/task/head\|val_ds.class_proto\|val_ds.task_proto\|val_ds.solution_proto\|val_ds.head_proto" matrix_program_core/train_assembler_multitask_pretrain.py
