#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"
REPORT_DIR="agent_reports/mechanism_skill_live_${STAMP}"
LATEST_LINK="agent_reports/latest_mechanism_skill_live"
mkdir -p "$REPORT_DIR"
rm -f "$LATEST_LINK"
ln -s "mechanism_skill_live_${STAMP}" "$LATEST_LINK"

LOG="$REPORT_DIR/full_run.log"
exec > >(tee -a "$LOG") 2>&1

DATASET="${DATASET:-matrix_program_core/runs/code_real_decode_20260617_072934/code_only_build/code_context_dataset.pt}"
INIT_ASSEMBLER="${INIT_ASSEMBLER:-}"
MECH_CKPT_OVERRIDE="${MECH_CKPT_OVERRIDE:-}"
EPOCHS_MECH="${EPOCHS_MECH:-5}"
EPOCHS_AUDIO="${EPOCHS_AUDIO:-3}"
AUDIO_MODES="${AUDIO_MODES:-editor_delta delta freeze_core full}"
AUDIO_LAMBDA_SKILL="${AUDIO_LAMBDA_SKILL:-0}"
TRAIN_TASK_CONTEXT="${TRAIN_TASK_CONTEXT:-0}"
USE_MECHANISM_CONTEXT="${USE_MECHANISM_CONTEXT:-0}"
AMP_SKILL="${AMP_SKILL:-fp32}"
AMP_AUDIO="${AMP_AUDIO:-fp32}"
LOG_EVERY="${LOG_EVERY:-50}"
FEEDBACK_NOISE_MAX="${FEEDBACK_NOISE_MAX:-0.70}"
FEEDBACK_IMPORTANCE="${FEEDBACK_IMPORTANCE:-1.50}"

WORK="matrix_program_core/runs/mechanism_skill_${STAMP}"
AUDIO_ROOT="matrix_program_core/runs/mechanism_audio_${STAMP}"
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

printf '\n[mechanism] repo=%s\n' "$ROOT"
printf '[mechanism] report_dir=%s\n' "$REPORT_DIR"
printf '[mechanism] dataset=%s\n' "$DATASET"
printf '[mechanism] init_assembler=%s\n' "$INIT_ASSEMBLER"
printf '[mechanism] epochs_mech=%s epochs_audio=%s modes=%s\n\n' "$EPOCHS_MECH" "$EPOCHS_AUDIO" "$AUDIO_MODES"
git rev-parse --short HEAD 2>/dev/null | sed 's/^/[mechanism] git_head=/' || true

if [[ -n "$MECH_CKPT_OVERRIDE" ]]; then
  MECH_CKPT="$MECH_CKPT_OVERRIDE"
  printf '\n[mechanism] using MECH_CKPT_OVERRIDE=%s\n' "$MECH_CKPT"
else
  printf '\n[mechanism] train mechanism-level assembly skill\n'
  python matrix_program_core/train_assembler_mechanism_skill_pretrain.py \
    --dataset "$DATASET" \
    --out-dir "$WORK" \
    ${INIT_ASSEMBLER:+--init-assembler "$INIT_ASSEMBLER"} \
    --device cuda --amp "$AMP_SKILL" \
    --epochs "$EPOCHS_MECH" \
    --batch-size 128 --eval-batch-size 256 --workers 2 --pin-memory \
    --lr 3e-4 --visible-consistency 0.15 --lambda-entropy-keep 0.01 \
    --feedback-noise-max "$FEEDBACK_NOISE_MAX" --feedback-importance "$FEEDBACK_IMPORTANCE" \
    --log-every "$LOG_EVERY"
  MECH_CKPT="$WORK/assembler_mechanism_skill_best.pt"
  cp "$WORK/final_report.json" "$REPORT_DIR/mechanism_final_report.json" || true
  cp "$WORK/metrics.csv" "$REPORT_DIR/mechanism_metrics.csv" || true
fi
python matrix_program_core/checkpoint_split.py --input "$MECH_CKPT" --out-dir "$EXPORT_ROOT" --tag mechanism_skill

for MODE in $AUDIO_MODES; do
  OUT="$AUDIO_ROOT/$MODE"
  printf '\n[mechanism] live audio transfer mode=%s\n' "$MODE"
  python matrix_program_core/transfer_audio_assembler.py \
    --data-root ../architecture_builder/data/speechcommands \
    --assembler-checkpoint "$MECH_CKPT" \
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
    --epochs "$EPOCHS_AUDIO" --lr 3e-4 \
    --lambda-skill "$AUDIO_LAMBDA_SKILL" --lambda-write-budget 0.025 --lambda-update-alive 0.005 --lambda-logit-norm 0.0007 \
    --grad-clip 0.75 --log-every 25
  python matrix_program_core/checkpoint_split.py --input "$OUT/best.pt" --out-dir "$EXPORT_ROOT" --tag "audio_${MODE}"
  cp "$OUT/final_report.json" "$REPORT_DIR/audio_${MODE}_final_report.json" || true
  cp "$OUT/metrics.csv" "$REPORT_DIR/audio_${MODE}_metrics.csv" || true
done

python - "$REPORT_DIR" "$EXPORT_ROOT" <<'PY' | tee "$REPORT_DIR/summary.txt"
import csv, json, sys
from pathlib import Path
rd = Path(sys.argv[1]); ex = Path(sys.argv[2])
print("# mechanism skill live summary")
f = rd / "mechanism_final_report.json"
if f.exists():
    o=json.loads(f.read_text())
    print("\n## mechanism_skill")
    print("best_val_loss:", o.get("best_val_loss"))
    print("best_epoch:", o.get("best_epoch"))
    print("checkpoint:", o.get("checkpoint"))
mp = rd / "mechanism_metrics.csv"
if mp.exists():
    rows=list(csv.DictReader(mp.open()))
    print("mechanism_last_row:", rows[-1] if rows else None)
for f in sorted(rd.glob("audio_*_final_report.json")):
    mode=f.name.replace("audio_","").replace("_final_report.json","")
    o=json.loads(f.read_text())
    print(f"\n## audio_{mode}")
    print("best_acc:", o.get("best_acc"))
    print("best_epoch:", o.get("best_epoch"))
    print("trainable_summary:", o.get("trainable_summary"))
print("\n## split reports")
for f in sorted(ex.glob("*_split_report.json")):
    o=json.loads(f.read_text())
    print(f.name, {k:o.get(k) for k in ["base_params","context_base_params","delta_params","task_params","base_keys","context_base_keys","delta_keys","task_keys"]})
PY

REPORT="$REPORT_DIR/REPORT_TO_CHATGPT.txt"
{
  echo "# REPORT_TO_CHATGPT mechanism skill live"
  echo
  echo "## git"; git rev-parse HEAD 2>/dev/null || true; git status --short 2>/dev/null || true
  echo
  echo "## config"; echo "DATASET=$DATASET"; echo "INIT_ASSEMBLER=$INIT_ASSEMBLER"; echo "MECH_CKPT=$MECH_CKPT"; echo "EPOCHS_MECH=$EPOCHS_MECH"; echo "EPOCHS_AUDIO=$EPOCHS_AUDIO"; echo "AUDIO_MODES=$AUDIO_MODES"; echo "TRAIN_TASK_CONTEXT=$TRAIN_TASK_CONTEXT"; echo "USE_MECHANISM_CONTEXT=$USE_MECHANISM_CONTEXT"; echo "AUDIO_LAMBDA_SKILL=$AUDIO_LAMBDA_SKILL"; echo "FEEDBACK_NOISE_MAX=$FEEDBACK_NOISE_MAX"; echo "FEEDBACK_IMPORTANCE=$FEEDBACK_IMPORTANCE"
  echo
  echo "## summary"; cat "$REPORT_DIR/summary.txt"
  echo
  echo "## mechanism metrics tail"; tail -n 30 "$REPORT_DIR/mechanism_metrics.csv" 2>/dev/null || true
  echo
  for f in "$REPORT_DIR"/audio_*_metrics.csv; do [[ -f "$f" ]] || continue; echo "## $(basename "$f") tail"; tail -n 20 "$f"; echo; done
  echo "## tail full_run.log"; tail -n 240 "$LOG"
} > "$REPORT"

printf '\n[mechanism] done. Paste this into ChatGPT:\n'
printf 'cat %s\n' "$REPORT"
