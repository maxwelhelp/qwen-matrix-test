#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CORE_CKPT="${1:?usage: bash matrix_program_core/commands/run_audio_transfer_v4_debug.sh CORE_PRETRAIN_PT [OUT_DIR] [MODE]}"
OUT="${2:-matrix_program_core/runs/audio_v4_delta_debug}"
MODE="${3:-delta}"

python matrix_program_core/transfer_audio_v4.py \
  --data-root ../architecture_builder/data/speechcommands \
  --core-checkpoint "$CORE_CKPT" \
  --out-dir "$OUT" \
  --train-mode "$MODE" \
  --device cuda \
  --amp fp32 \
  --classes yes,no,up,down,left,right,on,off,stop,go \
  --train-limit 12000 \
  --val-limit 2000 \
  --batch-size 128 \
  --eval-batch-size 256 \
  --workers 4 \
  --pin-memory \
  --dim 96 \
  --evidence-cells 48 \
  --layers 4 \
  --blocks 4 \
  --steps 2 \
  --variants 3 \
  --channel-stages 3 \
  --epochs 3 \
  --lr 3e-4 \
  --lambda-skill 0.05 \
  --lambda-pair 0.45 \
  --lambda-write-budget 0.025 \
  --lambda-update-alive 0.005 \
  --lambda-logit-norm 0.0007 \
  --grad-clip 0.75 \
  --log-every 25
