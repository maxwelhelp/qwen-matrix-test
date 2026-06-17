#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CKPT="${1:-simple_butterfly_matrix_v3/runs/speechcommands_v3/best.pt}"
OUT="${2:-simple_butterfly_matrix_v3/runs/speechcommands_v3_p40_continue}"

python simple_butterfly_matrix_v3/class_matrix_transport.py \
  --data-root ../architecture_builder/data/speechcommands \
  --init-checkpoint "$CKPT" \
  --epochs 20 \
  --batch-size 128 \
  --eval-batch-size 256 \
  --workers 4 \
  --pin-memory \
  --amp fp32 \
  --device cuda \
  --lr 2.5e-4 \
  --weight-decay 0.012 \
  --grad-clip 0.65 \
  --lambda-phase-balance 0.065 \
  --max-aggregate-phase-mass 0.38 \
  --min-early-phase-mass 0.085 \
  --lambda-class-read-div 0.045 \
  --lambda-logit-norm 0.001 \
  --log-every 25 \
  --out-dir "$OUT"

