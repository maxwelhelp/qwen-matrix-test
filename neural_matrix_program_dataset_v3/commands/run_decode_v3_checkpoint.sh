#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CKPT="${1:-simple_butterfly_matrix_v3/runs/speechcommands_v3/best.pt}"
OUT="${2:-neural_matrix_program_dataset_v3/runs/real_decode_v3_speechcommands}"

python neural_matrix_program_dataset_v3/neural_matrix_program_dataset_v3.py decode-real \
  --checkpoint "$CKPT" \
  --out "$OUT" \
  --D 64 \
  --device cuda \
  --max-matrices 500 \
  --decode-topk 12 \
  --mined-svd-atoms 4 \
  --functional-batch 512 \
  --log-every 25

