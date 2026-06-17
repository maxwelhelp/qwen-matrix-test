#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

bash agent_scripts/patch_macro_step_injection.sh

STAMP="$(date +%Y%m%d_%H%M%S)"
REPORT_DIR="agent_reports/macro_step_bank_compare_${STAMP}"
LATEST_LINK="agent_reports/latest_macro_step_bank_compare"
mkdir -p "$REPORT_DIR"
rm -f "$LATEST_LINK"
ln -s "macro_step_bank_compare_${STAMP}" "$LATEST_LINK"

LOG="$REPORT_DIR/full_run.log"
exec > >(tee -a "$LOG") 2>&1

DATASET="${DATASET:-matrix_program_core/runs/code_real_decode_20260617_073803/merged_flow_pack.pt}"
EPISODE_DATASET="${EPISODE_DATASET:-matrix_program_core/runs/code_real_decode_20260617_073803/repair_episode_dataset_v2.pt}"
MACRO_SOURCE="${MACRO_SOURCE:-$EPISODE_DATASET}"
MECH_CKPT_OVERRIDE="${MECH_CKPT_OVERRIDE:-}"
EPOCHS_MECH="${EPOCHS_MECH:-10}"
EPOCHS_AUDIO="${EPOCHS_AUDIO:-30}"
AUDIO_MODES="${AUDIO_MODES:-editor_delta}"
AUDIO_LR="${AUDIO_LR:-2e-4}"
AUDIO_LAMBDA_SKILL="${AUDIO_LAMBDA_SKILL:-0}"
TRAIN_TASK_CONTEXT="${TRAIN_TASK_CONTEXT:-1}"
USE_MECHANISM_CONTEXT="${USE_MECHANISM_CONTEXT:-0}"
FEEDBACK_STREAM="${FEEDBACK_STREAM:-1}"
REPAIR_CURRICULUM="${REPAIR_CURRICULUM:-1}"
LAMBDA_LIVE_SYNTH="${LAMBDA_LIVE_SYNTH:-0.12}"
AMP_SKILL="${AMP_SKILL:-fp32}"
AMP_AUDIO="${AMP_AUDIO:-fp32}"
LOG_EVERY="${LOG_EVERY:-50}"
MACRO_MAX_ITEMS="${MACRO_MAX_ITEMS:-160000}"
MACRO_KMEANS_ITERS="${MACRO_KMEANS_ITERS:-25}"
MACRO_GAIN="${MACRO_GAIN:-0.18}"
RUN_VARIANTS="${RUN_VARIANTS:-baseline macro16 macro32}"

WORK="matrix_program_core/runs/macro_step_compare_${STAMP}"
MACRO_DIR="$WORK/macros"
MECH_DIR="$WORK/mechanism_skill"
AUDIO_ROOT="$WORK/audio"
EXPORT_ROOT="$REPORT_DIR/exported_checkpoints"
mkdir -p "$WORK" "$MACRO_DIR" "$MECH_DIR" "$AUDIO_ROOT" "$EXPORT_ROOT"

printf '\n[macro-compare] repo=%s\n' "$ROOT"
printf '[macro-compare] report_dir=%s\n' "$REPORT_DIR"
printf '[macro-compare] dataset=%s\n' "$DATASET"
printf '[macro-compare] episode_dataset=%s\n' "$EPISODE_DATASET"
printf '[macro-compare] macro_source=%s\n' "$MACRO_SOURCE"
printf '[macro-compare] variants=%s modes=%s epochs_audio=%s macro_gain=%s\n' "$RUN_VARIANTS" "$AUDIO_MODES" "$EPOCHS_AUDIO" "$MACRO_GAIN"
printf '[macro-compare] mech_override=%s epochs_mech=%s\n\n' "$MECH_CKPT_OVERRIDE" "$EPOCHS_MECH"

git rev-parse --short HEAD 2>/dev/null | sed 's/^/[macro-compare] git_head=/' || true

printf '\n[macro-compare] build macro bank M=16\n'
python matrix_program_core/build_macro_step_bank.py \
  --input "$MACRO_SOURCE" \
  --out "$MACRO_DIR/macro_step_bank_M16.pt" \
  --report "$REPORT_DIR/macro_step_bank_M16_report.json" \
  --preview "$REPORT_DIR/macro_step_bank_M16_preview.csv" \
  --macros 16 --max-items "$MACRO_MAX_ITEMS" --kmeans-iters "$MACRO_KMEANS_ITERS" --log-every 5

printf '\n[macro-compare] build macro bank M=32\n'
python matrix_program_core/build_macro_step_bank.py \
  --input "$MACRO_SOURCE" \
  --out "$MACRO_DIR/macro_step_bank_M32.pt" \
  --report "$REPORT_DIR/macro_step_bank_M32_report.json" \
  --preview "$REPORT_DIR/macro_step_bank_M32_preview.csv" \
  --macros 32 --max-items "$MACRO_MAX_ITEMS" --kmeans-iters "$MACRO_KMEANS_ITERS" --log-every 5

if [[ -n "$MECH_CKPT_OVERRIDE" ]]; then
  MECH_CKPT="$MECH_CKPT_OVERRIDE"
  printf '\n[macro-compare] using existing mechanism ckpt=%s\n' "$MECH_CKPT"
else
  printf '\n[macro-compare] train mechanism skill once\n'
  python matrix_program_core/train_assembler_mechanism_skill_pretrain.py \
    --dataset "$EPISODE_DATASET" \
    --out-dir "$MECH_DIR" \
    --device cuda --amp "$AMP_SKILL" \
    --epochs "$EPOCHS_MECH" \
    --batch-size 128 --eval-batch-size 256 --workers 2 --pin-memory \
    --lr 3e-4 --visible-consistency 0.15 --lambda-entropy-keep 0.01 \
    "$([[ "$FEEDBACK_STREAM" == "1" ]] && echo --feedback-stream || echo --no-feedback-stream)" \
    "$([[ "$REPAIR_CURRICULUM" == "1" ]] && echo --repair-curriculum || echo --no-repair-curriculum)" \
    --lambda-live-synth "$LAMBDA_LIVE_SYNTH" --live-classes 8 \
    --log-every "$LOG_EVERY"
  MECH_CKPT="$MECH_DIR/assembler_mechanism_skill_best.pt"
  cp "$MECH_DIR/final_report.json" "$REPORT_DIR/mechanism_final_report.json" || true
  cp "$MECH_DIR/metrics.csv" "$REPORT_DIR/mechanism_metrics.csv" || true
fi

TASK_CONTEXT_FLAG="--train-task-context"
if [[ "$TRAIN_TASK_CONTEXT" == "0" ]]; then TASK_CONTEXT_FLAG="--no-train-task-context"; fi
MECHANISM_CONTEXT_FLAGS=()
if [[ "$USE_MECHANISM_CONTEXT" == "1" ]]; then MECHANISM_CONTEXT_FLAGS=(--use-mechanism-context --mechanism-task-id 4); fi

for VARIANT in $RUN_VARIANTS; do
  MACRO_ARGS=()
  if [[ "$VARIANT" == "macro16" ]]; then
    MACRO_ARGS=(--macro-step-bank "$MACRO_DIR/macro_step_bank_M16.pt" --macro-gain "$MACRO_GAIN")
  elif [[ "$VARIANT" == "macro32" ]]; then
    MACRO_ARGS=(--macro-step-bank "$MACRO_DIR/macro_step_bank_M32.pt" --macro-gain "$MACRO_GAIN")
  elif [[ "$VARIANT" == "baseline" ]]; then
    MACRO_ARGS=()
  else
    echo "[macro-compare][WARN] unknown variant=$VARIANT, skipping" >&2
    continue
  fi
  for MODE in $AUDIO_MODES; do
    OUT="$AUDIO_ROOT/${VARIANT}_${MODE}"
    printf '\n[macro-compare] audio variant=%s mode=%s\n' "$VARIANT" "$MODE"
    python matrix_program_core/transfer_audio_assembler.py \
      --data-root ../architecture_builder/data/speechcommands \
      --assembler-checkpoint "$MECH_CKPT" \
      --skill-target-pack "$DATASET" \
      --out-dir "$OUT" \
      --train-mode "$MODE" \
      --task-context-tokens 4 --head-context-tokens 10 --use-head-context \
      "${MECHANISM_CONTEXT_FLAGS[@]}" \
      "$TASK_CONTEXT_FLAG" \
      "${MACRO_ARGS[@]}" \
      --device cuda --amp "$AMP_AUDIO" \
      --classes yes,no,up,down,left,right,on,off,stop,go \
      --train-limit 12000 --val-limit 2000 \
      --batch-size 128 --eval-batch-size 256 --workers 4 --pin-memory \
      --dim 96 --evidence-cells 48 --layers 4 --blocks 4 --steps 2 --primitive-slots 4 --memory-cells 4 --global-cells 2 --channel-stages 3 \
      --epochs "$EPOCHS_AUDIO" --lr "$AUDIO_LR" \
      --lambda-skill "$AUDIO_LAMBDA_SKILL" --lambda-write-budget 0.025 --lambda-update-alive 0.005 \
      --pair-slots 12 --phase-prior-strength 0.85 \
      --lambda-phase-balance 0.045 --min-early-phase-mass 0.07 --max-aggregate-phase-mass 0.42 \
      --lambda-class-read-div 0.08 --lambda-class-slot-prior 0.0 --lambda-class-attn-entropy 0.003 \
      --lambda-slot-div 0.01 --lambda-logit-norm 0.0007 \
      --grad-clip 0.75 --log-every 25
    cp "$OUT/final_report.json" "$REPORT_DIR/audio_${VARIANT}_${MODE}_final_report.json" || true
    cp "$OUT/metrics.csv" "$REPORT_DIR/audio_${VARIANT}_${MODE}_metrics.csv" || true
    python matrix_program_core/checkpoint_split.py --input "$OUT/best.pt" --out-dir "$EXPORT_ROOT" --tag "audio_${VARIANT}_${MODE}" || true
  done
done

python - "$REPORT_DIR" <<'PY' | tee "$REPORT_DIR/summary.txt"
import csv, json, sys
from pathlib import Path
rd = Path(sys.argv[1])
print("# macro step bank comparison summary")
for f in sorted(rd.glob("macro_step_bank_M*_report.json")):
    o=json.loads(f.read_text(encoding='utf-8'))
    print(f"\n## {f.name}")
    for k in ["macro_count","items","bank_weighted_kl","mean_weighted_kl","kl_improvement","macro_entropy_norm","macro_top1_usage","mean_cluster_distance","p95_cluster_distance"]:
        print(f"{k}: {o.get(k)}")
print("\n## audio")
for f in sorted(rd.glob("audio_*_final_report.json")):
    name=f.name.replace("audio_","").replace("_final_report.json","")
    o=json.loads(f.read_text(encoding='utf-8'))
    print(f"\n### {name}")
    print("best_acc:", o.get("best_acc"))
    print("best_epoch:", o.get("best_epoch"))
    print("trainable_summary:", o.get("trainable_summary"))
    m=rd / f"audio_{name}_metrics.csv"
    if m.exists():
        rows=list(csv.DictReader(m.open()))
        if rows:
            keys=["epoch","train_ce","train_acc","val_acc","best_acc","class_read_div","slot_div","macro_entropy","macro_top1_usage","macro_gain","skill"]
            print("last_focus:", {k: rows[-1].get(k) for k in keys})
PY

REPORT="$REPORT_DIR/REPORT_TO_CHATGPT.txt"
{
  echo "# REPORT_TO_CHATGPT macro step bank comparison"
  echo
  echo "## git"; git rev-parse HEAD 2>/dev/null || true; git status --short 2>/dev/null || true
  echo
  echo "## config"; echo "DATASET=$DATASET"; echo "EPISODE_DATASET=$EPISODE_DATASET"; echo "MACRO_SOURCE=$MACRO_SOURCE"; echo "MECH_CKPT=$MECH_CKPT"; echo "RUN_VARIANTS=$RUN_VARIANTS"; echo "AUDIO_MODES=$AUDIO_MODES"; echo "EPOCHS_AUDIO=$EPOCHS_AUDIO"; echo "MACRO_GAIN=$MACRO_GAIN"; echo "MACRO_MAX_ITEMS=$MACRO_MAX_ITEMS"; echo "MACRO_KMEANS_ITERS=$MACRO_KMEANS_ITERS"
  echo
  echo "## summary"; cat "$REPORT_DIR/summary.txt"
  echo
  for f in "$REPORT_DIR"/audio_*_metrics.csv; do [[ -f "$f" ]] || continue; echo "## $(basename "$f") tail"; tail -n 20 "$f"; echo; done
  echo "## macro reports"; for f in "$REPORT_DIR"/macro_step_bank_M*_report.json; do [[ -f "$f" ]] || continue; echo "### $(basename "$f")"; cat "$f"; echo; done
  echo "## tail full_run.log"; tail -n 260 "$LOG"
} > "$REPORT"

printf '\n[macro-compare] done. Paste this into ChatGPT:\n'
printf 'cat %s\n' "$REPORT"
