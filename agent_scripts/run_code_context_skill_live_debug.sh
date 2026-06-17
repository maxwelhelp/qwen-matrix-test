#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

bash agent_scripts/patch_assembler_context_softflow.sh
bash agent_scripts/patch_task_context_v2_force.sh
bash agent_scripts/patch_task_context_batch_and_load.sh

STAMP="$(date +%Y%m%d_%H%M%S)"
REPORT_DIR="agent_reports/code_context_skill_live_${STAMP}"
LATEST_LINK="agent_reports/latest_code_context_skill_live"
mkdir -p "$REPORT_DIR"
rm -f "$LATEST_LINK"
ln -s "code_context_skill_live_${STAMP}" "$LATEST_LINK"

LOG="$REPORT_DIR/full_run.log"
exec > >(tee -a "$LOG") 2>&1

CODE_OUT="matrix_program_core/runs/code_context_skill_${STAMP}"
AUDIO_ROOT="matrix_program_core/runs/code_context_audio_${STAMP}"
EXPORT_ROOT="$REPORT_DIR/exported_checkpoints"
mkdir -p "$EXPORT_ROOT"

EPOCHS_CODE="${EPOCHS_CODE:-5}"
EPOCHS_AUDIO="${EPOCHS_AUDIO:-3}"
AUDIO_MODES="${AUDIO_MODES:-delta freeze_core full}"
N_CODE="${N_CODE:-6000}"
LOG_EVERY="${LOG_EVERY:-50}"
AUDIO_LAMBDA_SKILL="${AUDIO_LAMBDA_SKILL:-0.001}"
PARSE_DIRS="${PARSE_DIRS:-simple_butterfly_matrix_v3 simple_butterfly_matrix_v4 matrix_program_core}"
MAX_PARSE_FILES="${MAX_PARSE_FILES:-128}"

printf '\n[code-context] repo=%s\n' "$ROOT"
printf '[code-context] report_dir=%s\n' "$REPORT_DIR"
printf '[code-context] parse_dirs=%s\n' "$PARSE_DIRS"
printf '[code-context] code_epochs=%s audio_epochs=%s modes=%s n_code=%s\n\n' "$EPOCHS_CODE" "$EPOCHS_AUDIO" "$AUDIO_MODES" "$N_CODE"

git rev-parse --short HEAD 2>/dev/null | sed 's/^/[code-context] git_head=/' || true

printf '\n[code-context] train assembly skill from parsed code context\n'
# shellcheck disable=SC2086
python matrix_program_core/train_assembler_code_context_pretrain.py \
  --old-v3-path neural_matrix_program_dataset_v3/neural_matrix_program_dataset_v3.py \
  --parse-dirs $PARSE_DIRS \
  --max-parse-files "$MAX_PARSE_FILES" \
  --out-dir "$CODE_OUT" \
  --device cuda \
  --amp bf16 \
  --n "$N_CODE" \
  --matrix-D 32 \
  --dim 96 \
  --layers-core 4 \
  --blocks-core 4 \
  --steps-core 2 \
  --primitive-slots-core 4 \
  --memory-cells 4 \
  --global-cells 2 \
  --channel-stages 3 \
  --batch-size 128 \
  --eval-batch-size 256 \
  --workers 2 \
  --pin-memory \
  --epochs "$EPOCHS_CODE" \
  --lr 4e-4 \
  --lambda-entropy-keep 0.01 \
  --lambda-delta-l2 0.01 \
  --log-every "$LOG_EVERY"

CODE_CKPT="$CODE_OUT/assembler_code_context_best.pt"
python matrix_program_core/checkpoint_split.py --input "$CODE_CKPT" --out-dir "$EXPORT_ROOT" --tag code_context
cp "$CODE_OUT/final_report.json" "$REPORT_DIR/code_context_final_report.json" || true
cp "$CODE_OUT/metrics.csv" "$REPORT_DIR/code_context_metrics.csv" || true
cp "$CODE_OUT/ast_summary.json" "$REPORT_DIR/code_context_ast_summary.json" || true

for MODE in $AUDIO_MODES; do
  OUT="$AUDIO_ROOT/$MODE"
  printf '\n[code-context] live audio transfer mode=%s\n' "$MODE"
  python matrix_program_core/transfer_audio_assembler.py \
    --data-root ../architecture_builder/data/speechcommands \
    --assembler-checkpoint "$CODE_CKPT" \
    --out-dir "$OUT" \
    --train-mode "$MODE" \
    --task-context-tokens 4 \
    --head-context-tokens 10 \
    --use-head-context \
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
print("# code-context skill then live summary")
pre = rd / "code_context_final_report.json"
if pre.exists():
    obj = json.loads(pre.read_text(encoding="utf-8"))
    print("\n## code_context_skill")
    print("best_val_loss:", obj.get("best_val_loss"))
    print("best_epoch:", obj.get("best_epoch"))
    print("checkpoint:", obj.get("checkpoint"))
    print("dataset:", obj.get("dataset"))
mp = rd / "code_context_metrics.csv"
if mp.exists():
    rows = list(csv.DictReader(mp.open()))
    if rows: print("code_context_last_row:", rows[-1])
ast = rd / "code_context_ast_summary.json"
if ast.exists():
    print("ast_summary:", json.loads(ast.read_text(encoding="utf-8")))
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
  echo "# REPORT_TO_CHATGPT code context skill live"
  echo
  echo "## git"
  git rev-parse HEAD 2>/dev/null || true
  git status --short 2>/dev/null || true
  echo
  echo "## config"
  echo "CODE_OUT=$CODE_OUT"
  echo "AUDIO_ROOT=$AUDIO_ROOT"
  echo "EXPORT_ROOT=$EXPORT_ROOT"
  echo "PARSE_DIRS=$PARSE_DIRS"
  echo "EPOCHS_CODE=$EPOCHS_CODE"
  echo "EPOCHS_AUDIO=$EPOCHS_AUDIO"
  echo "AUDIO_MODES=$AUDIO_MODES"
  echo "AUDIO_LAMBDA_SKILL=$AUDIO_LAMBDA_SKILL"
  echo
  echo "## summary"
  cat "$REPORT_DIR/summary.txt"
  echo
  echo "## code context metrics tail"
  tail -n 30 "$REPORT_DIR/code_context_metrics.csv" 2>/dev/null || true
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

printf '\n[code-context] done. Paste this into ChatGPT:\n'
printf 'cat %s\n' "$REPORT"
printf '\n[code-context] local .pt split packages are in: %s\n' "$EXPORT_ROOT"
