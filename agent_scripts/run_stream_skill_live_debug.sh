#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"
REPORT_DIR="agent_reports/stream_skill_live_${STAMP}"
LATEST_LINK="agent_reports/latest_stream_skill_live"
mkdir -p "$REPORT_DIR"
rm -f "$LATEST_LINK"
ln -s "stream_skill_live_${STAMP}" "$LATEST_LINK"

LOG="$REPORT_DIR/full_run.log"
exec > >(tee -a "$LOG") 2>&1

DATASET="${DATASET:-matrix_program_core/runs/code_real_decode_20260617_072934/code_only_build/code_context_dataset.pt}"
INIT_ASSEMBLER="${INIT_ASSEMBLER:-}"
STREAM_CKPT_OVERRIDE="${STREAM_CKPT_OVERRIDE:-}"
EPOCHS_STREAM="${EPOCHS_STREAM:-8}"
EPOCHS_AUDIO="${EPOCHS_AUDIO:-8}"
AUDIO_MODES="${AUDIO_MODES:-editor_delta freeze_core full}"
AMP_STREAM="${AMP_STREAM:-fp32}"
AMP_AUDIO="${AMP_AUDIO:-fp32}"
TRAIN_TASK_CONTEXT="${TRAIN_TASK_CONTEXT:-0}"
USE_MECHANISM_CONTEXT="${USE_MECHANISM_CONTEXT:-0}"
AUDIO_LAMBDA_SKILL="${AUDIO_LAMBDA_SKILL:-0}"
AUDIO_LR="${AUDIO_LR:-3e-4}"
AUDIO_CLASS_READ_DIV="${AUDIO_CLASS_READ_DIV:-0.08}"
AUDIO_CLASS_SLOT_PRIOR="${AUDIO_CLASS_SLOT_PRIOR:-0.03}"
AUDIO_CLASS_ATTN_ENTROPY="${AUDIO_CLASS_ATTN_ENTROPY:-0.003}"
AUDIO_CLASS_SLOT_SIGMA="${AUDIO_CLASS_SLOT_SIGMA:-1.45}"
AUDIO_SLOT_DIV="${AUDIO_SLOT_DIV:-0.01}"
STREAM_TRAIN_MODE="${STREAM_TRAIN_MODE:-editor_delta}"
STREAM_LR="${STREAM_LR:-2e-4}"
NOISE_MIN="${NOISE_MIN:-0.15}"
NOISE_MAX="${NOISE_MAX:-0.85}"
REPAIR_MIN="${REPAIR_MIN:-0.20}"
REPAIR_MAX="${REPAIR_MAX:-0.55}"
LOG_EVERY="${LOG_EVERY:-50}"

WORK="matrix_program_core/runs/stream_skill_${STAMP}"
AUDIO_ROOT="matrix_program_core/runs/stream_audio_${STAMP}"
EXPORT_ROOT="$REPORT_DIR/exported_checkpoints"
mkdir -p "$WORK" "$AUDIO_ROOT" "$EXPORT_ROOT"

TASK_CONTEXT_FLAG="--train-task-context"
if [[ "$TRAIN_TASK_CONTEXT" == "0" ]]; then
  TASK_CONTEXT_FLAG="--no-train-task-context"
fi
MECHANISM_CONTEXT_FLAGS=()
if [[ "$USE_MECHANISM_CONTEXT" == "1" ]]; then
  MECHANISM_CONTEXT_FLAGS=(--use-mechanism-context --mechanism-task-id 4)
fi

printf '\n[stream] repo=%s\n' "$ROOT"
printf '[stream] report_dir=%s\n' "$REPORT_DIR"
printf '[stream] dataset=%s\n' "$DATASET"
printf '[stream] init_assembler=%s\n' "$INIT_ASSEMBLER"
printf '[stream] epochs_stream=%s epochs_audio=%s modes=%s train_mode=%s\n\n' "$EPOCHS_STREAM" "$EPOCHS_AUDIO" "$AUDIO_MODES" "$STREAM_TRAIN_MODE"
git rev-parse --short HEAD 2>/dev/null | sed 's/^/[stream] git_head=/' || true

if [[ -n "$STREAM_CKPT_OVERRIDE" ]]; then
  STREAM_CKPT="$STREAM_CKPT_OVERRIDE"
  printf '\n[stream] using STREAM_CKPT_OVERRIDE=%s\n' "$STREAM_CKPT"
else
  if [[ -z "$INIT_ASSEMBLER" ]]; then
    echo "[stream] ERROR: set INIT_ASSEMBLER to a mechanism/code skill checkpoint, or set STREAM_CKPT_OVERRIDE" >&2
    exit 2
  fi
  printf '\n[stream] train stream repair skill\n'
  python matrix_program_core/train_assembler_stream_skill_pretrain.py \
    --dataset "$DATASET" \
    --out-dir "$WORK" \
    --init-assembler "$INIT_ASSEMBLER" \
    --device cuda --amp "$AMP_STREAM" \
    --epochs "$EPOCHS_STREAM" \
    --batch-size 128 --eval-batch-size 256 --workers 2 --pin-memory \
    --train-mode "$STREAM_TRAIN_MODE" \
    --lr "$STREAM_LR" --weight-decay 0.01 --grad-clip 0.75 \
    --noise-min "$NOISE_MIN" --noise-max "$NOISE_MAX" \
    --repair-min "$REPAIR_MIN" --repair-max "$REPAIR_MAX" \
    --feedback-importance 1.50 --lambda-entropy-keep 0.01 \
    --log-every "$LOG_EVERY"
  STREAM_CKPT="$WORK/assembler_stream_skill_best.pt"
  cp "$WORK/final_report.json" "$REPORT_DIR/stream_final_report.json" || true
  cp "$WORK/metrics.csv" "$REPORT_DIR/stream_metrics.csv" || true
fi

python matrix_program_core/checkpoint_split.py --input "$STREAM_CKPT" --out-dir "$EXPORT_ROOT" --tag stream_skill

for MODE in $AUDIO_MODES; do
  OUT="$AUDIO_ROOT/$MODE"
  printf '\n[stream] live audio transfer mode=%s\n' "$MODE"
  python matrix_program_core/transfer_audio_assembler.py \
    --data-root ../architecture_builder/data/speechcommands \
    --assembler-checkpoint "$STREAM_CKPT" \
    --skill-target-pack "$DATASET" \
    --out-dir "$OUT" \
    --train-mode "$MODE" \
    --task-context-tokens 4 --head-context-tokens 10 --use-head-context \
    "${MECHANISM_CONTEXT_FLAGS[@]}" \
    "$TASK_CONTEXT_FLAG" \
    --device cuda --amp "$AMP_AUDIO" \
    --classes yes,no,up,down,left,right,on,off,stop,go \
    --train-limit 12000 --val-limit 2000 \
    --batch-size 128 --eval-batch-size 256 --workers 4 --pin-memory \
    --dim 96 --evidence-cells 48 --layers 4 --blocks 4 --steps 2 --primitive-slots 4 --memory-cells 4 --global-cells 2 --channel-stages 3 \
    --epochs "$EPOCHS_AUDIO" --lr "$AUDIO_LR" \
    --lambda-skill "$AUDIO_LAMBDA_SKILL" --lambda-write-budget 0.025 --lambda-update-alive 0.005 \
    --lambda-class-read-div "$AUDIO_CLASS_READ_DIV" \
    --lambda-class-slot-prior "$AUDIO_CLASS_SLOT_PRIOR" --lambda-class-attn-entropy "$AUDIO_CLASS_ATTN_ENTROPY" \
    --class-slot-prior-sigma "$AUDIO_CLASS_SLOT_SIGMA" \
    --lambda-slot-div "$AUDIO_SLOT_DIV" --lambda-logit-norm 0.0007 \
    --grad-clip 0.75 --log-every 25
  python matrix_program_core/checkpoint_split.py --input "$OUT/best.pt" --out-dir "$EXPORT_ROOT" --tag "audio_${MODE}"
  cp "$OUT/final_report.json" "$REPORT_DIR/audio_${MODE}_final_report.json" || true
  cp "$OUT/metrics.csv" "$REPORT_DIR/audio_${MODE}_metrics.csv" || true
done

python - "$REPORT_DIR" "$EXPORT_ROOT" <<'PY' | tee "$REPORT_DIR/summary.txt"
import csv, json, sys
from pathlib import Path
rd = Path(sys.argv[1]); ex = Path(sys.argv[2])
print("# stream skill live summary")
sf = rd / "stream_final_report.json"
if sf.exists():
    o = json.loads(sf.read_text())
    print("\n## stream_skill")
    print("best_val_loss:", o.get("best_val_loss"))
    print("best_epoch:", o.get("best_epoch"))
    print("checkpoint:", o.get("checkpoint"))
sm = rd / "stream_metrics.csv"
if sm.exists():
    rows = list(csv.DictReader(sm.open()))
    print("stream_last_row:", rows[-1] if rows else None)
for f in sorted(rd.glob("audio_*_final_report.json")):
    mode = f.name.replace("audio_", "").replace("_final_report.json", "")
    o = json.loads(f.read_text())
    print(f"\n## audio_{mode}")
    print("best_acc:", o.get("best_acc"))
    print("best_epoch:", o.get("best_epoch"))
    print("trainable_summary:", o.get("trainable_summary"))
print("\n## split reports")
for f in sorted(ex.glob("*_split_report.json")):
    o = json.loads(f.read_text())
    print(f.name, {k:o.get(k) for k in ["base_params","context_base_params","delta_params","task_params","base_keys","context_base_keys","delta_keys","task_keys"]})
PY

REPORT="$REPORT_DIR/REPORT_TO_CHATGPT.txt"
{
  echo "# REPORT_TO_CHATGPT stream skill live"
  echo
  echo "## git"; git rev-parse HEAD 2>/dev/null || true; git status --short 2>/dev/null || true
  echo
  echo "## config"; echo "DATASET=$DATASET"; echo "INIT_ASSEMBLER=$INIT_ASSEMBLER"; echo "STREAM_CKPT=$STREAM_CKPT"; echo "EPOCHS_STREAM=$EPOCHS_STREAM"; echo "EPOCHS_AUDIO=$EPOCHS_AUDIO"; echo "AUDIO_MODES=$AUDIO_MODES"; echo "STREAM_TRAIN_MODE=$STREAM_TRAIN_MODE"; echo "TRAIN_TASK_CONTEXT=$TRAIN_TASK_CONTEXT"; echo "USE_MECHANISM_CONTEXT=$USE_MECHANISM_CONTEXT"; echo "AUDIO_LR=$AUDIO_LR"; echo "AUDIO_CLASS_READ_DIV=$AUDIO_CLASS_READ_DIV"; echo "AUDIO_CLASS_SLOT_PRIOR=$AUDIO_CLASS_SLOT_PRIOR"; echo "AUDIO_CLASS_ATTN_ENTROPY=$AUDIO_CLASS_ATTN_ENTROPY"; echo "AUDIO_CLASS_SLOT_SIGMA=$AUDIO_CLASS_SLOT_SIGMA"; echo "AUDIO_SLOT_DIV=$AUDIO_SLOT_DIV"; echo "NOISE_MIN=$NOISE_MIN"; echo "NOISE_MAX=$NOISE_MAX"; echo "REPAIR_MIN=$REPAIR_MIN"; echo "REPAIR_MAX=$REPAIR_MAX"
  echo
  echo "## summary"; cat "$REPORT_DIR/summary.txt"
  echo
  echo "## stream metrics tail"; tail -n 30 "$REPORT_DIR/stream_metrics.csv" 2>/dev/null || true
  echo
  for f in "$REPORT_DIR"/audio_*_metrics.csv; do [[ -f "$f" ]] || continue; echo "## $(basename "$f") tail"; tail -n 20 "$f"; echo; done
  echo "## tail full_run.log"; tail -n 240 "$LOG"
} > "$REPORT"

printf '\n[stream] done. Paste this into ChatGPT:\n'
printf 'cat %s\n' "$REPORT"
