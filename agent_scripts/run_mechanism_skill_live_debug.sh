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
EPISODE_DATASET="${EPISODE_DATASET:-}"
BUILD_EPISODE_CACHE="${BUILD_EPISODE_CACHE:-0}"
EPISODE_MAX_PROGRAMS="${EPISODE_MAX_PROGRAMS:-0}"
EPISODE_MODE="${EPISODE_MODE:-all}"
INIT_ASSEMBLER="${INIT_ASSEMBLER:-}"
MECH_CKPT_OVERRIDE="${MECH_CKPT_OVERRIDE:-}"
EPOCHS_MECH="${EPOCHS_MECH:-5}"
EPOCHS_AUDIO="${EPOCHS_AUDIO:-3}"
AUDIO_MODES="${AUDIO_MODES:-editor_delta delta freeze_core full}"
AUDIO_LAMBDA_SKILL="${AUDIO_LAMBDA_SKILL:-0}"
AUDIO_LR="${AUDIO_LR:-3e-4}"
AUDIO_CLASS_READ_DIV="${AUDIO_CLASS_READ_DIV:-0.08}"
AUDIO_CLASS_SLOT_PRIOR="${AUDIO_CLASS_SLOT_PRIOR:-0.0}"
AUDIO_CLASS_ATTN_ENTROPY="${AUDIO_CLASS_ATTN_ENTROPY:-0.003}"
AUDIO_CLASS_SLOT_SIGMA="${AUDIO_CLASS_SLOT_SIGMA:-1.45}"
AUDIO_PHASE_PRIOR_STRENGTH="${AUDIO_PHASE_PRIOR_STRENGTH:-0.85}"
AUDIO_LAMBDA_PHASE_BALANCE="${AUDIO_LAMBDA_PHASE_BALANCE:-0.045}"
AUDIO_MIN_EARLY_PHASE_MASS="${AUDIO_MIN_EARLY_PHASE_MASS:-0.07}"
AUDIO_MAX_AGGREGATE_PHASE_MASS="${AUDIO_MAX_AGGREGATE_PHASE_MASS:-0.42}"
AUDIO_PAIR_SLOTS="${AUDIO_PAIR_SLOTS:-12}"
AUDIO_SLOT_DIV="${AUDIO_SLOT_DIV:-0.01}"
TRAIN_TASK_CONTEXT="${TRAIN_TASK_CONTEXT:-0}"
USE_MECHANISM_CONTEXT="${USE_MECHANISM_CONTEXT:-0}"
AMP_SKILL="${AMP_SKILL:-fp32}"
AMP_AUDIO="${AMP_AUDIO:-fp32}"
LOG_EVERY="${LOG_EVERY:-50}"
FEEDBACK_NOISE_MAX="${FEEDBACK_NOISE_MAX:-0.70}"
FEEDBACK_IMPORTANCE="${FEEDBACK_IMPORTANCE:-1.50}"
FEEDBACK_STREAM="${FEEDBACK_STREAM:-1}"
REPAIR_CURRICULUM="${REPAIR_CURRICULUM:-1}"
REPAIR_CYCLE_STEPS="${REPAIR_CYCLE_STEPS:-6}"
REPAIR_STRENGTH="${REPAIR_STRENGTH:-1.0}"
LAMBDA_LIVE_SYNTH="${LAMBDA_LIVE_SYNTH:-0.12}"
LIVE_CLASSES="${LIVE_CLASSES:-8}"

WORK="matrix_program_core/runs/mechanism_skill_${STAMP}"
AUDIO_ROOT="matrix_program_core/runs/mechanism_audio_${STAMP}"
EXPORT_ROOT="$REPORT_DIR/exported_checkpoints"
mkdir -p "$WORK" "$AUDIO_ROOT" "$EXPORT_ROOT"
TRAIN_DATASET="$DATASET"

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
printf '[mechanism] episode_dataset=%s build_episode_cache=%s\n' "$EPISODE_DATASET" "$BUILD_EPISODE_CACHE"
printf '[mechanism] episode_mode=%s episode_max_programs=%s\n' "$EPISODE_MODE" "$EPISODE_MAX_PROGRAMS"
printf '[mechanism] init_assembler=%s\n' "$INIT_ASSEMBLER"
printf '[mechanism] epochs_mech=%s epochs_audio=%s modes=%s\n\n' "$EPOCHS_MECH" "$EPOCHS_AUDIO" "$AUDIO_MODES"
git rev-parse --short HEAD 2>/dev/null | sed 's/^/[mechanism] git_head=/' || true

if [[ -n "$MECH_CKPT_OVERRIDE" ]]; then
  MECH_CKPT="$MECH_CKPT_OVERRIDE"
  printf '\n[mechanism] using MECH_CKPT_OVERRIDE=%s\n' "$MECH_CKPT"
else
  if [[ -n "$EPISODE_DATASET" ]]; then
    TRAIN_DATASET="$EPISODE_DATASET"
  elif [[ "$BUILD_EPISODE_CACHE" == "1" ]]; then
    TRAIN_DATASET="$WORK/repair_episode_dataset.pt"
    printf '\n[mechanism] build cached repair/from-scratch episode dataset\n'
    python matrix_program_core/build_repair_episode_dataset.py \
      --input "$DATASET" \
      --out "$TRAIN_DATASET" \
      --steps-core 2 \
      --episode-mode "$EPISODE_MODE" \
      --max-programs "$EPISODE_MAX_PROGRAMS" \
      --log-every "$LOG_EVERY"
    python - "$TRAIN_DATASET" "$REPORT_DIR/repair_episode_dataset_meta.json" <<'PY' || true
import json, sys, torch
pack = torch.load(sys.argv[1], map_location="cpu")
meta = dict(pack.get("meta", {}))
meta["path"] = sys.argv[1]
open(sys.argv[2], "w", encoding="utf-8").write(json.dumps(meta, ensure_ascii=False, indent=2))
PY
  fi
  printf '\n[mechanism] train mechanism-level assembly skill\n'
  python matrix_program_core/train_assembler_mechanism_skill_pretrain.py \
    --dataset "$TRAIN_DATASET" \
    --out-dir "$WORK" \
    ${INIT_ASSEMBLER:+--init-assembler "$INIT_ASSEMBLER"} \
    --device cuda --amp "$AMP_SKILL" \
    --epochs "$EPOCHS_MECH" \
    --batch-size 128 --eval-batch-size 256 --workers 2 --pin-memory \
    --lr 3e-4 --visible-consistency 0.15 --lambda-entropy-keep 0.01 \
    --feedback-noise-max "$FEEDBACK_NOISE_MAX" --feedback-importance "$FEEDBACK_IMPORTANCE" \
    "$([[ "$FEEDBACK_STREAM" == "1" ]] && echo --feedback-stream || echo --no-feedback-stream)" \
    "$([[ "$REPAIR_CURRICULUM" == "1" ]] && echo --repair-curriculum || echo --no-repair-curriculum)" \
    --repair-cycle-steps "$REPAIR_CYCLE_STEPS" --repair-strength "$REPAIR_STRENGTH" \
    --lambda-live-synth "$LAMBDA_LIVE_SYNTH" --live-classes "$LIVE_CLASSES" \
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
    --epochs "$EPOCHS_AUDIO" --lr "$AUDIO_LR" \
    --lambda-skill "$AUDIO_LAMBDA_SKILL" --lambda-write-budget 0.025 --lambda-update-alive 0.005 \
    --pair-slots "$AUDIO_PAIR_SLOTS" --phase-prior-strength "$AUDIO_PHASE_PRIOR_STRENGTH" \
    --lambda-phase-balance "$AUDIO_LAMBDA_PHASE_BALANCE" \
    --min-early-phase-mass "$AUDIO_MIN_EARLY_PHASE_MASS" --max-aggregate-phase-mass "$AUDIO_MAX_AGGREGATE_PHASE_MASS" \
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
  echo "## config"; echo "DATASET=$DATASET"; echo "TRAIN_DATASET=$TRAIN_DATASET"; echo "EPISODE_DATASET=$EPISODE_DATASET"; echo "BUILD_EPISODE_CACHE=$BUILD_EPISODE_CACHE"; echo "EPISODE_MODE=$EPISODE_MODE"; echo "EPISODE_MAX_PROGRAMS=$EPISODE_MAX_PROGRAMS"; echo "INIT_ASSEMBLER=$INIT_ASSEMBLER"; echo "MECH_CKPT=$MECH_CKPT"; echo "EPOCHS_MECH=$EPOCHS_MECH"; echo "EPOCHS_AUDIO=$EPOCHS_AUDIO"; echo "AUDIO_MODES=$AUDIO_MODES"; echo "TRAIN_TASK_CONTEXT=$TRAIN_TASK_CONTEXT"; echo "USE_MECHANISM_CONTEXT=$USE_MECHANISM_CONTEXT"; echo "AUDIO_LAMBDA_SKILL=$AUDIO_LAMBDA_SKILL"; echo "AUDIO_LR=$AUDIO_LR"; echo "AUDIO_CLASS_READ_DIV=$AUDIO_CLASS_READ_DIV"; echo "AUDIO_CLASS_SLOT_PRIOR=$AUDIO_CLASS_SLOT_PRIOR"; echo "AUDIO_CLASS_ATTN_ENTROPY=$AUDIO_CLASS_ATTN_ENTROPY"; echo "AUDIO_CLASS_SLOT_SIGMA=$AUDIO_CLASS_SLOT_SIGMA"; echo "AUDIO_PHASE_PRIOR_STRENGTH=$AUDIO_PHASE_PRIOR_STRENGTH"; echo "AUDIO_LAMBDA_PHASE_BALANCE=$AUDIO_LAMBDA_PHASE_BALANCE"; echo "AUDIO_MIN_EARLY_PHASE_MASS=$AUDIO_MIN_EARLY_PHASE_MASS"; echo "AUDIO_MAX_AGGREGATE_PHASE_MASS=$AUDIO_MAX_AGGREGATE_PHASE_MASS"; echo "AUDIO_PAIR_SLOTS=$AUDIO_PAIR_SLOTS"; echo "AUDIO_SLOT_DIV=$AUDIO_SLOT_DIV"; echo "FEEDBACK_NOISE_MAX=$FEEDBACK_NOISE_MAX"; echo "FEEDBACK_IMPORTANCE=$FEEDBACK_IMPORTANCE"; echo "FEEDBACK_STREAM=$FEEDBACK_STREAM"; echo "REPAIR_CURRICULUM=$REPAIR_CURRICULUM"; echo "REPAIR_CYCLE_STEPS=$REPAIR_CYCLE_STEPS"; echo "REPAIR_STRENGTH=$REPAIR_STRENGTH"; echo "LAMBDA_LIVE_SYNTH=$LAMBDA_LIVE_SYNTH"; echo "LIVE_CLASSES=$LIVE_CLASSES"
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
