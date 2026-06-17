#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

python neural_matrix_program_dataset_v3/neural_matrix_program_dataset_v3.py all \
  --parse-dir ./simple_butterfly_matrix_v3 ./simple_butterfly_matrix_v2 ./simple_butterfly_matrix \
  --out neural_matrix_program_dataset_v3/runs/local_smoke \
  --n 128 \
  --D 16 \
  --layers 2 \
  --blocks 2 \
  --steps 2 \
  --primitive-slots 2 \
  --max-program-steps 4 \
  --device cpu \
  --epochs 2 \
  --batch-size 64 \
  --hidden 128 \
  --max-parse-files 30 \
  --log-every 64 \
  --log-every-epoch 1

