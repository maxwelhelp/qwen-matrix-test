#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT="${1:-matrix_program_core/runs/core_pretrain_debug}"

python matrix_program_core/train_core_pretrain.py \
  --out-dir "$OUT" \
  --device cuda \
  --amp bf16 \
  --train-n 6000 \
  --val-n 1000 \
  --classes 10 \
  --dim 96 \
  --evidence-cells 48 \
  --layers 4 \
  --blocks 4 \
  --steps 2 \
  --variants 3 \
  --channel-stages 3 \
  --batch-size 128 \
  --eval-batch-size 256 \
  --workers 2 \
  --pin-memory \
  --epochs 3 \
  --lr 5e-4 \
  --lambda-skill 0.35 \
  --lambda-pair 0.45 \
  --lambda-recon 0.20 \
  --log-every 25
