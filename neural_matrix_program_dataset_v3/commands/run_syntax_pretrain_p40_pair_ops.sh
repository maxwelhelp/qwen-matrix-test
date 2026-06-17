#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

python neural_matrix_program_dataset_v3/neural_matrix_program_dataset_v3.py all \
  --parse-dir ./simple_butterfly_matrix_v3 ./simple_butterfly_matrix_v2 ./simple_butterfly_matrix \
  --out neural_matrix_program_dataset_v3/runs/syntax_pretrain_p40_pair_ops \
  --n 20000 \
  --D 32 \
  --layers 4 \
  --blocks 4 \
  --steps 2 \
  --primitive-slots 3 \
  --max-program-steps 16 \
  --device cuda \
  --epochs 35 \
  --batch-size 1024 \
  --hidden 768 \
  --max-parse-files 300 \
  --log-every 1000 \
  --log-every-epoch 1

