#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# This touches the NEW transfer_audio_assembler.py only.
# It does not patch the old v3 decoder/parser file.
bash agent_scripts/patch_differentiable_head_diversity.sh

STAMP="$(date +%Y%m%d_%H%M%S)"
REPORT_DIR="agent_reports/mechanism_specialize40_${STAMP}"
LATEST_LINK="agent_reports/latest_mechanism_specialize40"
mkdir -p "$REPORT_DIR"
rm -f "$LATEST_LINK"
ln -s "mechanism_specialize40_${STAMP}" "$LATEST_LINK"

LOG="$REPORT_DIR/full_run.log"
exec > >(tee -a "$LOG") 2>&1

DATASET="${DATASET:-matrix_program_core/runs/code_real_decode_20260617_072934/code_only_build/code_context_dataset.pt}"
MECH_CKPT_OVERRIDE="${MECH_CKPT_OVERRIDE:-matrix_program_core/runs/mechanism_skill_20260617_085250/assembler_mechanism_skill_best.pt}"
EPOCHS_AUDIO="${EPOCHS_AUDIO:-40}"
AUDIO_MODES="${AUDIO_MODES:-editor_delta freeze_core}"
AMP_AUDIO="${AMP_AUDIO:-fp32}"
TRAIN_TASK_CONTEXT="${TRAIN_TASK_CONTEXT:-0}"
USE_MECHANISM_CONTEXT="${USE_MECHANISM_CONTEXT:-0}"
AUDIO_LAMBDA_SKILL="${AUDIO_LAMBDA_SKILL:-0}"
LAMBDA_CLASS_READ_DIV="${LAMBDA_CLASS_READ_DIV:-0.060}"
LAMBDA_SLOT_DIV="${LAMBDA_SLOT_DIV:-0.006}"
LAMBDA_WRITE_BUDGET="${LAMBDA_WRITE_BUDGET:-0.025}"
LAMBDA_UPDATE_ALIVE="${LAMBDA_UPDATE_ALIVE:-0.005}"
LAMBDA_LOGIT_NORM="${LAMBDA_LOGIT_NORM:-0.0007}"
LR="${LR:-3e-4}"
SEED="${SEED:-42}"
LOG_EVERY="${LOG_EVERY:-25}"

AUDIO_ROOT="matrix_program_core/runs/mechanism_specialize40_audio_${STAMP}"
EXPORT_ROOT="$REPORT_DIR/exported_checkpoints"
mkdir -p "$AUDIO_ROOT" "$EXPORT_ROOT"

TASK_CONTEXT_FLAG="--train-task-context"
if [[ "$TRAIN_TASK_CONTEXT" == "0" ]]; then
  TASK_CONTEXT_FLAG="--no-train-task-context"
fi
MECHANISM_CONTEXT_FLAGS=()
if [[ "$USE_MECHANISM_CONTEXT" == "1" ]]; then
  MECHANISM_CONTEXT_FLAGS=(--use-mechanism-context --mechanism-task-id 4)
fi

printf '\n[specialize40] repo=%s\n' "$ROOT"
printf '[specialize40] report_dir=%s\n' "$REPORT_DIR"
printf '[specialize40] skill_ckpt=%s\n' "$MECH_CKPT_OVERRIDE"
printf '[specialize40] dataset=%s\n' "$DATASET"
printf '[specialize40] modes=%s epochs=%s amp=%s seed=%s\n' "$AUDIO_MODES" "$EPOCHS_AUDIO" "$AMP_AUDIO" "$SEED"
printf '[specialize40] lambda_class_read_div=%s lambda_slot_div=%s\n' "$LAMBDA_CLASS_READ_DIV" "$LAMBDA_SLOT_DIV"
printf '[specialize40] train_task_context=%s use_mechanism_context=%s lambda_skill=%s\n\n' "$TRAIN_TASK_CONTEXT" "$USE_MECHANISM_CONTEXT" "$AUDIO_LAMBDA_SKILL"

git rev-parse --short HEAD 2>/dev/null | sed 's/^/[specialize40] git_head=/' || true

grep -n "class_slot_attention" matrix_program_core/transfer_audio_assembler.py || true
python matrix_program_core/checkpoint_split.py --input "$MECH_CKPT_OVERRIDE" --out-dir "$EXPORT_ROOT" --tag mechanism_skill

for MODE in $AUDIO_MODES; do
  OUT="$AUDIO_ROOT/$MODE"
  printf '\n[specialize40] live audio transfer mode=%s\n' "$MODE"
  python matrix_program_core/transfer_audio_assembler.py \
    --data-root ../architecture_builder/data/speechcommands \
    --assembler-checkpoint "$MECH_CKPT_OVERRIDE" \
    --skill-target-pack "$DATASET" \
    --out-dir "$OUT" \
    --train-mode "$MODE" \
    --task-context-tokens 4 --head-context-tokens 10 --use-head-context \
    "${MECHANISM_CONTEXT_FLAGS[@]}" \
    "$TASK_CONTEXT_FLAG" \
    --device cuda --amp "$AMP_AUDIO" \
    --seed "$SEED" \
    --classes yes,no,up,down,left,right,on,off,stop,go \
    --train-limit 12000 --val-limit 2000 \
    --batch-size 128 --eval-batch-size 256 --workers 4 --pin-memory \
    --dim 96 --evidence-cells 48 --layers 4 --blocks 4 --steps 2 --primitive-slots 4 --memory-cells 4 --global-cells 2 --channel-stages 3 \
    --epochs "$EPOCHS_AUDIO" --lr "$LR" \
    --lambda-skill "$AUDIO_LAMBDA_SKILL" \
    --lambda-write-budget "$LAMBDA_WRITE_BUDGET" \
    --lambda-update-alive "$LAMBDA_UPDATE_ALIVE" \
    --lambda-class-read-div "$LAMBDA_CLASS_READ_DIV" \
    --lambda-slot-div "$LAMBDA_SLOT_DIV" \
    --lambda-logit-norm "$LAMBDA_LOGIT_NORM" \
    --grad-clip 0.75 --log-every "$LOG_EVERY"
  python matrix_program_core/checkpoint_split.py --input "$OUT/best.pt" --out-dir "$EXPORT_ROOT" --tag "audio_${MODE}"
  cp "$OUT/final_report.json" "$REPORT_DIR/audio_${MODE}_final_report.json" || true
  cp "$OUT/metrics.csv" "$REPORT_DIR/audio_${MODE}_metrics.csv" || true
done

python - "$REPORT_DIR" "$EXPORT_ROOT" <<'PY' | tee "$REPORT_DIR/summary.txt"
import csv, json, sys
from pathlib import Path
rd = Path(sys.argv[1]); ex = Path(sys.argv[2])
print("# mechanism specialize40 summary")
for f in sorted(rd.glob("audio_*_final_report.json")):
    mode = f.name.replace("audio_", "").replace("_final_report.json", "")
    obj = json.loads(f.read_text(encoding="utf-8"))
    print(f"\n## audio_{mode}")
    print("best_acc:", obj.get("best_acc"))
    print("best_epoch:", obj.get("best_epoch"))
    print("trainable_summary:", obj.get("trainable_summary"))
    mpath = rd / f"audio_{mode}_metrics.csv"
    if mpath.exists():
        rows = list(csv.DictReader(mpath.open()))
        if rows:
            print("last_row:", rows[-1])
            keys = ["epoch", "train_ce", "train_acc", "val_acc", "best_acc", "class_read_div", "slot_div", "skill"]
            print("tail_focus:")
            for r in rows[-8:]:
                print({k: r.get(k) for k in keys})
print("\n## split reports")
for f in sorted(ex.glob("*_split_report.json")):
    obj = json.loads(f.read_text(encoding="utf-8"))
    print(f.name, {k: obj.get(k) for k in ["base_params", "context_base_params", "mechanism_base_params", "delta_params", "task_params", "base_keys", "context_base_keys", "mechanism_base_keys", "delta_keys", "task_keys"]})
PY

REPORT="$REPORT_DIR/REPORT_TO_CHATGPT.txt"
{
  echo "# REPORT_TO_CHATGPT mechanism specialize40"
  echo
  echo "## git"; git rev-parse HEAD 2>/dev/null || true; git status --short 2>/dev/null || true
  echo
  echo "## config"
  echo "DATASET=$DATASET"
  echo "MECH_CKPT_OVERRIDE=$MECH_CKPT_OVERRIDE"
  echo "EPOCHS_AUDIO=$EPOCHS_AUDIO"
  echo "AUDIO_MODES=$AUDIO_MODES"
  echo "TRAIN_TASK_CONTEXT=$TRAIN_TASK_CONTEXT"
  echo "USE_MECHANISM_CONTEXT=$USE_MECHANISM_CONTEXT"
  echo "AUDIO_LAMBDA_SKILL=$AUDIO_LAMBDA_SKILL"
  echo "LAMBDA_CLASS_READ_DIV=$LAMBDA_CLASS_READ_DIV"
  echo "LAMBDA_SLOT_DIV=$LAMBDA_SLOT_DIV"
  echo "LR=$LR"
  echo
  echo "## summary"; cat "$REPORT_DIR/summary.txt"
  echo
  for f in "$REPORT_DIR"/audio_*_metrics.csv; do
    [[ -f "$f" ]] || continue
    echo "## $(basename "$f") tail"
    tail -n 25 "$f"
    echo
  done
  echo "## tail full_run.log"; tail -n 240 "$LOG"
} > "$REPORT"

printf '\n[specialize40] done. Paste this into ChatGPT:\n'
printf 'cat %s\n' "$REPORT"
