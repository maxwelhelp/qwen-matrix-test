#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

APPLY_PATCHES="${APPLY_PATCHES:-0}"
if [[ "$APPLY_PATCHES" == "1" ]]; then
  bash agent_scripts/patch_assembler_context_softflow.sh
  bash agent_scripts/patch_task_context_v2_force.sh
  bash agent_scripts/patch_task_context_batch_and_load.sh
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
REPORT_DIR="agent_reports/context_skill_live_${STAMP}"
LATEST_LINK="agent_reports/latest_context_skill_live"
mkdir -p "$REPORT_DIR"
rm -f "$LATEST_LINK"
ln -s "context_skill_live_${STAMP}" "$LATEST_LINK"

LOG="$REPORT_DIR/full_run.log"
exec > >(tee -a "$LOG") 2>&1

CONTEXT_OUT="matrix_program_core/runs/context_skill_${STAMP}"
AUDIO_ROOT="matrix_program_core/runs/context_skill_audio_${STAMP}"
EXPORT_ROOT="$REPORT_DIR/exported_checkpoints"
mkdir -p "$EXPORT_ROOT"

EPOCHS_CONTEXT="${EPOCHS_CONTEXT:-5}"
EPOCHS_AUDIO="${EPOCHS_AUDIO:-3}"
AUDIO_MODES="${AUDIO_MODES:-delta freeze_core full}"
TRAIN_N="${TRAIN_N:-12000}"
VAL_N="${VAL_N:-2000}"
LOG_EVERY="${LOG_EVERY:-10}"
AUDIO_LAMBDA_SKILL="${AUDIO_LAMBDA_SKILL:-0.001}"
TASK_CONTEXT_TOKENS="${TASK_CONTEXT_TOKENS:-4}"
HEAD_CONTEXT_TOKENS="${HEAD_CONTEXT_TOKENS:-10}"
TRAIN_TASK_CONTEXT="${TRAIN_TASK_CONTEXT:-1}"
AMP_SKILL="${AMP_SKILL:-fp32}"
AMP_AUDIO="${AMP_AUDIO:-fp32}"

printf '\n[context-skill] repo=%s\n' "$ROOT"
printf '[context-skill] report_dir=%s\n' "$REPORT_DIR"
printf '[context-skill] context_epochs=%s audio_epochs=%s modes=%s train_n=%s val_n=%s\n' "$EPOCHS_CONTEXT" "$EPOCHS_AUDIO" "$AUDIO_MODES" "$TRAIN_N" "$VAL_N"
printf '[context-skill] export_dir=%s\n\n' "$EXPORT_ROOT"
TASK_CONTEXT_FLAG="--train-task-context"
if [[ "$TRAIN_TASK_CONTEXT" == "0" ]]; then
  TASK_CONTEXT_FLAG="--no-train-task-context"
fi

git rev-parse --short HEAD 2>/dev/null | sed 's/^/[context-skill] git_head=/' || true

printf '\n[context-skill] pretrain assembly skill from context only: no classifier head, no raw audio\n'
python matrix_program_core/train_assembler_context_skill_pretrain.py \
  --out-dir "$CONTEXT_OUT" \
  --device cuda \
  --amp "$AMP_SKILL" \
  --train-n "$TRAIN_N" \
  --val-n "$VAL_N" \
  --dim 96 \
  --evidence-cells 48 \
  --layers 4 \
  --blocks 4 \
  --steps 2 \
  --primitive-slots 4 \
  --memory-cells 4 \
  --global-cells 2 \
  --channel-stages 3 \
  --batch-size 128 \
  --eval-batch-size 256 \
  --workers 2 \
  --pin-memory \
  --epochs "$EPOCHS_CONTEXT" \
  --lr 4e-4 \
  --lambda-entropy-keep 0.01 \
  --lambda-delta-l2 0.01 \
  --holdout-mod 5 \
  --holdout-rem 0 \
  --log-every "$LOG_EVERY"

CONTEXT_CKPT="$CONTEXT_OUT/assembler_context_skill_best.pt"
python matrix_program_core/checkpoint_split.py --input "$CONTEXT_CKPT" --out-dir "$EXPORT_ROOT" --tag context_skill
cp "$CONTEXT_OUT/final_report.json" "$REPORT_DIR/context_skill_final_report.json" || true
cp "$CONTEXT_OUT/metrics.csv" "$REPORT_DIR/context_skill_metrics.csv" || true

for MODE in $AUDIO_MODES; do
  OUT="$AUDIO_ROOT/$MODE"
  printf '\n[context-skill] live audio transfer mode=%s\n' "$MODE"
  python matrix_program_core/transfer_audio_assembler.py \
    --data-root ../architecture_builder/data/speechcommands \
    --assembler-checkpoint "$CONTEXT_CKPT" \
    --out-dir "$OUT" \
    --train-mode "$MODE" \
    --task-context-tokens "$TASK_CONTEXT_TOKENS" \
    --head-context-tokens "$HEAD_CONTEXT_TOKENS" \
    --use-head-context \
    "$TASK_CONTEXT_FLAG" \
    --device cuda \
    --amp "$AMP_AUDIO" \
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
    --primitive-slots 4 \
    --memory-cells 4 \
    --global-cells 2 \
    --channel-stages 3 \
    --epochs "$EPOCHS_AUDIO" \
    --lr 3e-4 \
    --lambda-skill "$AUDIO_LAMBDA_SKILL" \
    --lambda-write-budget 0.025 \
    --lambda-update-alive 0.005 \
    --lambda-logit-norm 0.0007 \
    --grad-clip 0.75 \
    --log-every 25
  python matrix_program_core/checkpoint_split.py --input "$OUT/best.pt" --out-dir "$EXPORT_ROOT" --tag "audio_${MODE}"
  cp "$OUT/final_report.json" "$REPORT_DIR/audio_${MODE}_final_report.json" || true
  cp "$OUT/metrics.csv" "$REPORT_DIR/audio_${MODE}_metrics.csv" || true
  cp "$OUT/analysis_epoch_$(printf '%03d' "$EPOCHS_AUDIO").json" "$REPORT_DIR/audio_${MODE}_last_analysis.json" || true
done

python - "$REPORT_DIR" "$EXPORT_ROOT" <<'PY' | tee "$REPORT_DIR/summary.txt"
import csv, json, sys
from pathlib import Path
rd = Path(sys.argv[1]); ex = Path(sys.argv[2])
print("# context-skill then live summary")
pre = rd / "context_skill_final_report.json"
if pre.exists():
    obj = json.loads(pre.read_text(encoding="utf-8"))
    print("\n## context_skill_pretrain")
    print("best_val_loss:", obj.get("best_val_loss"))
    print("best_epoch:", obj.get("best_epoch"))
    print("checkpoint:", obj.get("checkpoint"))
mp = rd / "context_skill_metrics.csv"
if mp.exists():
    rows = list(csv.DictReader(mp.open()))
    if rows: print("context_skill_last_row:", rows[-1])
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
        if rows: print("last_row:", rows[-1])
print("\n## split reports")
for f in sorted(ex.glob("*_split_report.json")):
    obj = json.loads(f.read_text(encoding="utf-8"))
    print(f.name, {k: obj.get(k) for k in ["base_params", "delta_params", "task_params", "base_keys", "delta_keys", "task_keys"]})
PY

REPORT="$REPORT_DIR/REPORT_TO_CHATGPT.txt"
{
  echo "# REPORT_TO_CHATGPT context skill live"
  echo
  echo "## git"
  git rev-parse HEAD 2>/dev/null || true
  git status --short 2>/dev/null || true
  echo
  echo "## config"
  echo "CONTEXT_OUT=$CONTEXT_OUT"
  echo "AUDIO_ROOT=$AUDIO_ROOT"
  echo "EXPORT_ROOT=$EXPORT_ROOT"
  echo "EPOCHS_CONTEXT=$EPOCHS_CONTEXT"
  echo "EPOCHS_AUDIO=$EPOCHS_AUDIO"
  echo "AUDIO_MODES=$AUDIO_MODES"
  echo "AUDIO_LAMBDA_SKILL=$AUDIO_LAMBDA_SKILL"
  echo
  echo "## summary"
  cat "$REPORT_DIR/summary.txt"
  echo
  echo "## context skill metrics tail"
  tail -n 30 "$REPORT_DIR/context_skill_metrics.csv" 2>/dev/null || true
  echo
  for f in "$REPORT_DIR"/audio_*_metrics.csv; do
    [[ -f "$f" ]] || continue
    echo "## $(basename "$f") tail"
    tail -n 20 "$f"
    echo
  done
  echo "## split reports"
  for f in "$EXPORT_ROOT"/*_split_report.json; do
    [[ -f "$f" ]] || continue
    echo "### $(basename "$f")"
    cat "$f"
    echo
  done
  echo "## tail full_run.log"
  tail -n 240 "$LOG"
} > "$REPORT"

printf '\n[context-skill] done. Paste this into ChatGPT:\n'
printf 'cat %s\n' "$REPORT"
printf '\n[context-skill] local .pt split packages are in: %s\n' "$EXPORT_ROOT"
