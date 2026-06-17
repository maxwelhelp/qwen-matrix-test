#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

python simple_butterfly_matrix_v3/class_matrix_transport.py \
  --data-root ../architecture_builder/data/speechcommands \
  --epochs 30 \
  --batch-size 128 \
  --eval-batch-size 256 \
  --workers 4 \
  --pin-memory \
  --amp fp32 \
  --device cuda \
  --lr 4e-4 \
  --weight-decay 0.012 \
  --grad-clip 0.75 \
  --log-every 25 \
  --out-dir simple_butterfly_matrix_v3/runs/speechcommands_v3_p40_fp32

