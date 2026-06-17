#!/usr/bin/env bash
set -euo pipefail

python simple_butterfly_matrix_v3/class_matrix_transport.py \
  --synthetic \
  --classes c0,c1,c2,c3,c4,c5,c6,c7,c8,c9 \
  --train-limit 512 \
  --val-limit 256 \
  --synthetic-length 512 \
  --dim 64 \
  --evidence-cells 36 \
  --layers 4 \
  --blocks 4 \
  --steps 2 \
  --variants 3 \
  --pair-slots 12 \
  --channel-stages 3 \
  --epochs 3 \
  --batch-size 64 \
  --eval-batch-size 128 \
  --workers 0 \
  --amp fp32 \
  --device cpu \
  --log-every 4 \
  --out-dir simple_butterfly_matrix_v3/runs/smoke
