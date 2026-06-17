#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

python neural_matrix_program_dataset_v3/neural_matrix_program_dataset_v3.py all \
  --parse-dir ./simple_butterfly_matrix_v3 ./simple_butterfly_matrix_v2 ./simple_butterfly_matrix \
  --out neural_matrix_program_dataset_v3/runs/syntax_pretrain_cpu_medium \
  --n 2048 \
  --D 32 \
  --layers 3 \
  --blocks 4 \
  --steps 2 \
  --primitive-slots 3 \
  --max-program-steps 12 \
  --device cpu \
  --epochs 8 \
  --batch-size 256 \
  --hidden 384 \
  --max-parse-files 300 \
  --log-every 256 \
  --log-every-epoch 1

