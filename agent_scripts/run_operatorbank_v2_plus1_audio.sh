#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Main patch is allowed to fail if it was already applied locally; the hotfix
# below repairs partial/idempotent states and compiles the files.
bash agent_scripts/patch_operatorbank_v2_plus1.sh || echo "[opv2] main patch already applied or partially applied; running hotfix"
bash agent_scripts/patch_operatorbank_v2_hotfix.sh

STAMP="$(date +%Y%m%d_%H%M%S)"
REPORT_DIR="agent_reports/operatorbank_v2_plus1_${STAMP}"
LATEST_LINK="agent_reports/latest_operatorbank_v2_plus1"
mkdir -p "$REPORT_DIR"
rm -f "$LATEST_LINK"
ln -s "operatorbank_v2_plus1_${STAMP}" "$LATEST_LINK"

LOG="$REPORT_DIR/full_run.log"
exec > >(tee -a "$LOG") 2>&1

DATA_ROOT="${DATA_ROOT:-../architecture_builder/data/speechcommands}"
ASSEMBLER_CKPT="${ASSEMBLER_CKPT:-}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
AUDIO_MODES="${AUDIO_MODES:-full}"
EPOCHS_AUDIO="${EPOCHS_AUDIO:-25}"
AUDIO_LR="${AUDIO_LR:-2e-4}"
AMP_AUDIO="${AMP_AUDIO:-fp32}"
TRAIN_N="${TRAIN_N:-12000}"
VAL_N="${VAL_N:-2000}"
LOG_EVERY="${LOG_EVERY:-25}"
LAYERS="${LAYERS:-5}"
STEPS="${STEPS:-3}"
BLOCKS="${BLOCKS:-4}"
PRIMITIVE_SLOTS="${PRIMITIVE_SLOTS:-4}"
MEMORY_CELLS="${MEMORY_CELLS:-4}"
GLOBAL_CELLS="${GLOBAL_CELLS:-2}"
DIM="${DIM:-96}"
EVIDENCE_CELLS="${EVIDENCE_CELLS:-48}"
PAIR_SLOTS="${PAIR_SLOTS:-12}"

printf '\n[opv2] repo=%s\n' "$ROOT"
printf '[opv2] report_dir=%s\n' "$REPORT_DIR"
printf '[opv2] modes=%s epochs=%s lr=%s amp=%s\n' "$AUDIO_MODES" "$EPOCHS_AUDIO" "$AUDIO_LR" "$AMP_AUDIO"
printf '[opv2] architecture: layers=%s steps=%s blocks=%s K=%s memory=%s global=%s dim=%s\n' "$LAYERS" "$STEPS" "$BLOCKS" "$PRIMITIVE_SLOTS" "$MEMORY_CELLS" "$GLOBAL_CELLS" "$DIM"
printf '[opv2] checkpoint=%s init=%s\n\n' "$ASSEMBLER_CKPT" "$INIT_CHECKPOINT"

git rev-parse --short HEAD 2>/dev/null | sed 's/^/[opv2] git_head=/' || true

for MODE in $AUDIO_MODES; do
  OUT="matrix_program_core/runs/operatorbank_v2_plus1_${STAMP}/${MODE}"
  mkdir -p "$OUT"
  CKPT_ARGS=()
  if [[ -n "$ASSEMBLER_CKPT" ]]; then
    CKPT_ARGS+=(--assembler-checkpoint "$ASSEMBLER_CKPT")
  fi
  if [[ -n "$INIT_CHECKPOINT" ]]; then
    CKPT_ARGS+=(--init-checkpoint "$INIT_CHECKPOINT")
  fi
  printf '\n[opv2] run mode=%s out=%s\n' "$MODE" "$OUT"
  python matrix_program_core/transfer_audio_assembler.py \
    --data-root "$DATA_ROOT" \
    "${CKPT_ARGS[@]}" \
    --out-dir "$OUT" \
    --train-mode "$MODE" \
    --task-context-tokens 4 --head-context-tokens 10 --use-head-context --train-task-context \
    --device cuda --amp "$AMP_AUDIO" \
    --classes yes,no,up,down,left,right,on,off,stop,go \
    --train-limit "$TRAIN_N" --val-limit "$VAL_N" \
    --batch-size 128 --eval-batch-size 256 --workers 4 --pin-memory \
    --dim "$DIM" --evidence-cells "$EVIDENCE_CELLS" \
    --layers "$LAYERS" --steps "$STEPS" --blocks "$BLOCKS" \
    --primitive-slots "$PRIMITIVE_SLOTS" --memory-cells "$MEMORY_CELLS" --global-cells "$GLOBAL_CELLS" \
    --channel-stages 3 --operator-v2 --step-alive-init 1.65 \
    --pair-slots "$PAIR_SLOTS" --phase-prior-strength 0.75 \
    --epochs "$EPOCHS_AUDIO" --lr "$AUDIO_LR" \
    --lambda-skill 0 \
    --lambda-write-budget 0.020 --lambda-update-alive 0.004 \
    --lambda-phase-balance 0.030 --min-early-phase-mass 0.055 --max-aggregate-phase-mass 0.45 \
    --lambda-class-read-div 0.075 --lambda-class-slot-prior 0.020 --lambda-class-attn-entropy 0.0035 \
    --class-slot-prior-sigma 1.55 \
    --lambda-slot-div 0.010 \
    --lambda-primitive-balance 0.004 \
    --lambda-cell-balance 0.003 \
    --lambda-entropy-band 0.002 \
    --lambda-step-alive-budget 0.004 \
    --primitive-balance-entropy-floor 0.72 \
    --cell-balance-entropy-floor 0.62 \
    --entropy-band-low 1.05 --entropy-band-high 2.45 \
    --step-alive-target 0.78 \
    --lambda-logit-norm 0.0006 \
    --grad-clip 0.75 --log-every "$LOG_EVERY"

  cp "$OUT/final_report.json" "$REPORT_DIR/audio_${MODE}_final_report.json" || true
  cp "$OUT/metrics.csv" "$REPORT_DIR/audio_${MODE}_metrics.csv" || true
  for f in "$OUT"/analysis_epoch_*.json; do
    [[ -f "$f" ]] || continue
    cp "$f" "$REPORT_DIR/$(basename "$OUT")_$(basename "$f")" || true
  done
  python matrix_program_core/checkpoint_split.py --input "$OUT/best.pt" --out-dir "$REPORT_DIR/exported_checkpoints" --tag "operatorbank_v2_plus1_${MODE}" || true
done

python - "$REPORT_DIR" <<'PY' | tee "$REPORT_DIR/summary.txt"
import csv, json, sys
from pathlib import Path
rd = Path(sys.argv[1])
print("# OperatorBankV2 + one layer summary")
for f in sorted(rd.glob("audio_*_final_report.json")):
    name = f.name.replace("audio_", "").replace("_final_report.json", "")
    o = json.loads(f.read_text(encoding='utf-8'))
    print(f"\n## {name}")
    print("best_acc:", o.get("best_acc"))
    print("best_epoch:", o.get("best_epoch"))
    print("assembler_config:", o.get("assembler_config"))
    m = rd / f"audio_{name}_metrics.csv"
    if m.exists():
        rows = list(csv.DictReader(m.open()))
        if rows:
            keys = ["epoch","train_ce","train_acc","val_acc","best_acc","class_read_div","slot_div","primitive_balance","cell_balance","entropy_band","step_alive_budget","phase_balance"]
            print("last_focus:", {k: rows[-1].get(k) for k in keys})
PY

REPORT="$REPORT_DIR/REPORT_TO_CHATGPT.txt"
{
  echo "# REPORT_TO_CHATGPT OperatorBankV2 + plus one layer"
  echo
  echo "## git"; git rev-parse HEAD 2>/dev/null || true; git status --short 2>/dev/null || true
  echo
  echo "## config"
  echo "DATA_ROOT=$DATA_ROOT"; echo "ASSEMBLER_CKPT=$ASSEMBLER_CKPT"; echo "INIT_CHECKPOINT=$INIT_CHECKPOINT"
  echo "AUDIO_MODES=$AUDIO_MODES"; echo "EPOCHS_AUDIO=$EPOCHS_AUDIO"; echo "AUDIO_LR=$AUDIO_LR"
  echo "LAYERS=$LAYERS"; echo "STEPS=$STEPS"; echo "BLOCKS=$BLOCKS"; echo "PRIMITIVE_SLOTS=$PRIMITIVE_SLOTS"; echo "MEMORY_CELLS=$MEMORY_CELLS"; echo "GLOBAL_CELLS=$GLOBAL_CELLS"; echo "DIM=$DIM"
  echo
  echo "## summary"; cat "$REPORT_DIR/summary.txt"
  echo
  for f in "$REPORT_DIR"/audio_*_metrics.csv; do [[ -f "$f" ]] || continue; echo "## $(basename "$f") tail"; tail -n 35 "$f"; echo; done
  echo "## tail full_run.log"; tail -n 260 "$LOG"
} > "$REPORT"

printf '\n[opv2] done. Paste this into ChatGPT:\n'
printf 'cat %s\n' "$REPORT"
