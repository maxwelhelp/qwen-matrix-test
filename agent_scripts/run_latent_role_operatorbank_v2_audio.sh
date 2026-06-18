#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"
REPORT_DIR="agent_reports/latent_role_operatorbank_v2_${STAMP}"
LATEST_LINK="agent_reports/latest_latent_role_operatorbank_v2"
mkdir -p "$REPORT_DIR"
rm -f "$LATEST_LINK"
ln -s "latent_role_operatorbank_v2_${STAMP}" "$LATEST_LINK"

LOG="$REPORT_DIR/full_run.log"
exec > >(tee -a "$LOG") 2>&1

DATA_ROOT="${DATA_ROOT:-../architecture_builder/data/speechcommands}"
AUDIO_MODES="${AUDIO_MODES:-full}"
EPOCHS_AUDIO="${EPOCHS_AUDIO:-20}"
AUDIO_LR="${AUDIO_LR:-2e-4}"
AMP_AUDIO="${AMP_AUDIO:-fp32}"
TRAIN_N="${TRAIN_N:-12000}"
VAL_N="${VAL_N:-2000}"
LOG_EVERY="${LOG_EVERY:-25}"
WORKERS="${WORKERS:-4}"
PIN_MEMORY="${PIN_MEMORY:-1}"
LAYERS="${LAYERS:-4}"
STEPS="${STEPS:-3}"
BLOCKS="${BLOCKS:-4}"
PRIMITIVE_SLOTS="${PRIMITIVE_SLOTS:-4}"
MEMORY_CELLS="${MEMORY_CELLS:-4}"
GLOBAL_CELLS="${GLOBAL_CELLS:-2}"
DIM="${DIM:-96}"
EVIDENCE_CELLS="${EVIDENCE_CELLS:-48}"
PAIR_SLOTS="${PAIR_SLOTS:-12}"
ROLE_COUNT="${ROLE_COUNT:-6}"
ROLE_TEMPERATURE="${ROLE_TEMPERATURE:-1.25}"
ROLE_INIT_STD="${ROLE_INIT_STD:-0.02}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
ASSEMBLER_CKPT="${ASSEMBLER_CKPT:-}"
PURE_LATENT="${PURE_LATENT:-1}"

LAMBDA_LAYER_SIM="${LAMBDA_LAYER_SIM:-0.020}"
LAMBDA_STEP_SIM="${LAMBDA_STEP_SIM:-0.006}"
LAYER_SIM_MARGIN="${LAYER_SIM_MARGIN:-0.22}"
STEP_SIM_MARGIN="${STEP_SIM_MARGIN:-0.42}"
LAMBDA_CLASS_READ_DIV="${LAMBDA_CLASS_READ_DIV:-0.075}"
if [[ "$PURE_LATENT" == "1" || "$PURE_LATENT" == "true" || "$PURE_LATENT" == "yes" ]]; then
  LAMBDA_CLASS_SLOT_PRIOR="${LAMBDA_CLASS_SLOT_PRIOR:-0.000}"
else
  LAMBDA_CLASS_SLOT_PRIOR="${LAMBDA_CLASS_SLOT_PRIOR:-0.020}"
fi
LAMBDA_CLASS_ATTN_ENTROPY="${LAMBDA_CLASS_ATTN_ENTROPY:-0.0035}"
LAMBDA_ROLE_USAGE_BALANCE="${LAMBDA_ROLE_USAGE_BALANCE:-0.010}"
LAMBDA_ROLE_ENTROPY_BAND="${LAMBDA_ROLE_ENTROPY_BAND:-0.006}"
LAMBDA_ROLE_SIMILARITY="${LAMBDA_ROLE_SIMILARITY:-0.012}"
LAMBDA_SEMANTIC_LINT="${LAMBDA_SEMANTIC_LINT:-0.002}"
ROLE_USAGE_ENTROPY_FLOOR="${ROLE_USAGE_ENTROPY_FLOOR:-0.72}"
ROLE_ENTROPY_LOW="${ROLE_ENTROPY_LOW:-0.35}"
ROLE_ENTROPY_HIGH="${ROLE_ENTROPY_HIGH:-0.92}"
ROLE_SIMILARITY_MARGIN="${ROLE_SIMILARITY_MARGIN:-0.78}"

printf '\n[latent-role] repo=%s\n' "$ROOT"
printf '[latent-role] report_dir=%s\n' "$REPORT_DIR"
printf '[latent-role] modes=%s epochs=%s lr=%s amp=%s\n' "$AUDIO_MODES" "$EPOCHS_AUDIO" "$AUDIO_LR" "$AMP_AUDIO"
printf '[latent-role] train_n=%s val_n=%s workers=%s pin_memory=%s\n' "$TRAIN_N" "$VAL_N" "$WORKERS" "$PIN_MEMORY"
printf '[latent-role] architecture: layers=%s steps=%s blocks=%s K=%s memory=%s global=%s dim=%s roles=%s tau=%s\n' \
  "$LAYERS" "$STEPS" "$BLOCKS" "$PRIMITIVE_SLOTS" "$MEMORY_CELLS" "$GLOBAL_CELLS" "$DIM" "$ROLE_COUNT" "$ROLE_TEMPERATURE"
printf '[latent-role] pure_latent=%s\n' "$PURE_LATENT"
printf '[latent-role] role losses: usage=%s entropy=%s sim=%s\n\n' \
  "$LAMBDA_ROLE_USAGE_BALANCE" "$LAMBDA_ROLE_ENTROPY_BAND" "$LAMBDA_ROLE_SIMILARITY"
printf '[latent-role] class losses: read_div=%s slot_prior=%s attn_entropy=%s\n' \
  "$LAMBDA_CLASS_READ_DIV" "$LAMBDA_CLASS_SLOT_PRIOR" "$LAMBDA_CLASS_ATTN_ENTROPY"
printf '[latent-role] semantic_lint=%s\n\n' "$LAMBDA_SEMANTIC_LINT"

git rev-parse --short HEAD 2>/dev/null | sed 's/^/[latent-role] git_head=/' || true

python -m py_compile matrix_program_core/assembler_core.py matrix_program_core/transfer_audio_assembler.py

for MODE in $AUDIO_MODES; do
  OUT="matrix_program_core/runs/latent_role_operatorbank_v2_${STAMP}/${MODE}"
  mkdir -p "$OUT"
  CKPT_ARGS=()
  if [[ -n "$ASSEMBLER_CKPT" ]]; then CKPT_ARGS+=(--assembler-checkpoint "$ASSEMBLER_CKPT"); fi
  if [[ -n "$INIT_CHECKPOINT" ]]; then CKPT_ARGS+=(--init-checkpoint "$INIT_CHECKPOINT"); fi
  PIN_ARGS=()
  if [[ "$PIN_MEMORY" == "1" || "$PIN_MEMORY" == "true" || "$PIN_MEMORY" == "yes" ]]; then PIN_ARGS+=(--pin-memory); fi

  printf '\n[latent-role] run mode=%s out=%s\n' "$MODE" "$OUT"
  python matrix_program_core/transfer_audio_assembler.py \
    --data-root "$DATA_ROOT" \
    "${CKPT_ARGS[@]}" \
    --out-dir "$OUT" \
    --train-mode "$MODE" \
    --task-context-tokens 4 --head-context-tokens 10 --use-head-context --train-task-context \
    --device cuda --amp "$AMP_AUDIO" \
    --classes yes,no,up,down,left,right,on,off,stop,go \
    --train-limit "$TRAIN_N" --val-limit "$VAL_N" \
    --batch-size 128 --eval-batch-size 256 --workers "$WORKERS" "${PIN_ARGS[@]}" \
    --dim "$DIM" --evidence-cells "$EVIDENCE_CELLS" \
    --layers "$LAYERS" --steps "$STEPS" --blocks "$BLOCKS" \
    --primitive-slots "$PRIMITIVE_SLOTS" --memory-cells "$MEMORY_CELLS" --global-cells "$GLOBAL_CELLS" \
    --channel-stages 3 --operator-v2 --step-alive-init 1.55 \
    --latent-roles --role-count "$ROLE_COUNT" --role-temperature "$ROLE_TEMPERATURE" --role-init-std "$ROLE_INIT_STD" \
    --pair-slots "$PAIR_SLOTS" \
    --phase-prior-strength 0.0 \
    --epochs "$EPOCHS_AUDIO" --lr "$AUDIO_LR" \
    --lambda-skill 0 \
    --lambda-write-budget 0.020 --lambda-update-alive 0.004 \
    --lambda-phase-balance 0.000 --min-early-phase-mass 0.000 --max-aggregate-phase-mass 1.00 \
    --lambda-class-read-div "$LAMBDA_CLASS_READ_DIV" --lambda-class-slot-prior "$LAMBDA_CLASS_SLOT_PRIOR" --lambda-class-attn-entropy "$LAMBDA_CLASS_ATTN_ENTROPY" \
    --class-slot-prior-sigma 1.55 \
    --lambda-slot-div 0.010 \
    --lambda-layer-sim "$LAMBDA_LAYER_SIM" --lambda-step-sim "$LAMBDA_STEP_SIM" \
    --layer-sim-margin "$LAYER_SIM_MARGIN" --step-sim-margin "$STEP_SIM_MARGIN" \
    --lambda-primitive-balance 0.004 \
    --lambda-cell-balance 0.003 \
    --lambda-entropy-band 0.002 \
    --lambda-step-alive-budget 0.004 \
    --lambda-role-usage-balance "$LAMBDA_ROLE_USAGE_BALANCE" \
    --lambda-role-entropy-band "$LAMBDA_ROLE_ENTROPY_BAND" \
    --lambda-role-similarity "$LAMBDA_ROLE_SIMILARITY" \
    --lambda-semantic-lint "$LAMBDA_SEMANTIC_LINT" \
    --role-usage-entropy-floor "$ROLE_USAGE_ENTROPY_FLOOR" \
    --role-entropy-low "$ROLE_ENTROPY_LOW" --role-entropy-high "$ROLE_ENTROPY_HIGH" \
    --role-similarity-margin "$ROLE_SIMILARITY_MARGIN" \
    --primitive-balance-entropy-floor 0.72 \
    --cell-balance-entropy-floor 0.62 \
    --entropy-band-low 1.05 --entropy-band-high 2.45 \
    --step-alive-target 0.76 \
    --lambda-logit-norm 0.0007 \
    --grad-clip 0.75 --log-every "$LOG_EVERY"

  cp "$OUT/final_report.json" "$REPORT_DIR/audio_${MODE}_final_report.json" || true
  cp "$OUT/metrics.csv" "$REPORT_DIR/audio_${MODE}_metrics.csv" || true
  for f in "$OUT"/analysis_epoch_*.json; do
    [[ -f "$f" ]] || continue
    cp "$f" "$REPORT_DIR/$(basename "$OUT")_$(basename "$f")" || true
  done
done

python - "$REPORT_DIR" <<'PY' | tee "$REPORT_DIR/summary.txt"
import csv, json, sys
from pathlib import Path
rd = Path(sys.argv[1])
print("# Latent role OperatorBankV2 summary")
for f in sorted(rd.glob("audio_*_final_report.json")):
    name = f.name.replace("audio_", "").replace("_final_report.json", "")
    o = json.loads(f.read_text(encoding="utf-8"))
    print(f"\n## {name}")
    print("best_acc:", o.get("best_acc"))
    print("best_epoch:", o.get("best_epoch"))
    cfg = o.get("assembler_config", {})
    print("latent_roles:", cfg.get("latent_roles"), "role_count:", cfg.get("role_count"), "operator_v2:", cfg.get("operator_v2"))
    m = rd / f"audio_{name}_metrics.csv"
    if m.exists():
        rows = list(csv.DictReader(m.open()))
        if rows:
            keys = [
                "epoch", "train_ce", "train_acc", "val_acc", "best_acc",
                "class_read_div", "slot_div", "layer_sim", "step_sim",
                "role_usage_balance", "role_entropy_band", "role_similarity", "role_usage_max", "role_entropy",
                "semantic_lint", "semantic_no_read", "semantic_no_transform", "semantic_memory_global",
                "semantic_dead_slot", "semantic_write_consumer", "semantic_early_write", "semantic_transition",
                "primitive_balance", "cell_balance", "entropy_band", "step_alive_budget",
            ]
            print("last_focus:", {k: rows[-1].get(k) for k in keys})
PY

REPORT="$REPORT_DIR/REPORT_TO_CHATGPT.txt"
{
  echo "# REPORT_TO_CHATGPT latent role OperatorBankV2"
  echo
  echo "## git"; git rev-parse HEAD 2>/dev/null || true; git status --short 2>/dev/null || true
  echo
  echo "## config"
  echo "AUDIO_MODES=$AUDIO_MODES"; echo "EPOCHS_AUDIO=$EPOCHS_AUDIO"; echo "AUDIO_LR=$AUDIO_LR"
  echo "PURE_LATENT=$PURE_LATENT"
  echo "LAYERS=$LAYERS"; echo "STEPS=$STEPS"; echo "ROLE_COUNT=$ROLE_COUNT"; echo "ROLE_TEMPERATURE=$ROLE_TEMPERATURE"
  echo "LAMBDA_CLASS_READ_DIV=$LAMBDA_CLASS_READ_DIV"; echo "LAMBDA_CLASS_SLOT_PRIOR=$LAMBDA_CLASS_SLOT_PRIOR"
  echo "LAMBDA_SEMANTIC_LINT=$LAMBDA_SEMANTIC_LINT"
  echo
  echo "## summary"; cat "$REPORT_DIR/summary.txt"
  echo
  for f in "$REPORT_DIR"/audio_*_metrics.csv; do [[ -f "$f" ]] || continue; echo "## $(basename "$f") tail"; tail -n 35 "$f"; echo; done
  echo "## tail full_run.log"; tail -n 260 "$LOG"
} > "$REPORT"

printf '\n[latent-role] done. Paste this into ChatGPT:\n'
printf 'cat %s\n' "$REPORT"
