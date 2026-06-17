#!/usr/bin/env bash
set -euo pipefail

python simple_butterfly_matrix_v3/class_matrix_transport.py \
  --data-root ../architecture_builder/data/speechcommands \
  --epochs 15 \
  --batch-size 128 \
  --eval-batch-size 256 \
  --amp bf16 \
  --device cuda \
  --dim 96 \
  --evidence-cells 48 \
  --variants 3 \
  --pair-slots 12 \
  --workers 2 \
  --log-every 50 \
  --out-dir simple_butterfly_matrix_v3/runs/speechcommands_v3
