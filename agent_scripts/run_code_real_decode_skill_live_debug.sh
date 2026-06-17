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
REPORT_DIR="agent_reports/code_real_decode_skill_live_${STAMP}"
LATEST_LINK="agent_reports/latest_code_real_decode_skill_live"
mkdir -p "$REPORT_DIR"
rm -f "$LATEST_LINK"
ln -s "code_real_decode_skill_live_${STAMP}" "$LATEST_LINK"

LOG="$REPORT_DIR/full_run.log"
exec > >(tee -a "$LOG") 2>&1

WORK="matrix_program_core/runs/code_real_decode_${STAMP}"
EXPORT_ROOT="$REPORT_DIR/exported_checkpoints"
AUDIO_ROOT="matrix_program_core/runs/code_real_decode_audio_${STAMP}"
mkdir -p "$WORK" "$EXPORT_ROOT"

EPOCHS_SKILL="${EPOCHS_SKILL:-5}"
EPOCHS_AUDIO="${EPOCHS_AUDIO:-3}"
AUDIO_MODES="${AUDIO_MODES:-delta freeze_core full}"
N_CODE="${N_CODE:-4000}"
LOG_EVERY="${LOG_EVERY:-50}"
AUDIO_LAMBDA_SKILL="${AUDIO_LAMBDA_SKILL:-0.001}"
AUDIO_LR="${AUDIO_LR:-3e-4}"
AUDIO_CLASS_READ_DIV="${AUDIO_CLASS_READ_DIV:-0.04}"
AUDIO_CLASS_SLOT_PRIOR="${AUDIO_CLASS_SLOT_PRIOR:-0.0}"
AUDIO_CLASS_ATTN_ENTROPY="${AUDIO_CLASS_ATTN_ENTROPY:-0.003}"
AUDIO_PHASE_PRIOR_STRENGTH="${AUDIO_PHASE_PRIOR_STRENGTH:-0.85}"
AUDIO_LAMBDA_PHASE_BALANCE="${AUDIO_LAMBDA_PHASE_BALANCE:-0.03}"
AUDIO_PAIR_SLOTS="${AUDIO_PAIR_SLOTS:-12}"
AUDIO_SLOT_DIV="${AUDIO_SLOT_DIV:-0.003}"
TRAIN_TASK_CONTEXT="${TRAIN_TASK_CONTEXT:-1}"
AMP_SKILL="${AMP_SKILL:-fp32}"
AMP_AUDIO="${AMP_AUDIO:-fp32}"
PARSE_DIRS="${PARSE_DIRS:-matrix_program_core simple_butterfly_matrix_v3 simple_butterfly_matrix}"
MAX_PARSE_FILES="${MAX_PARSE_FILES:-96}"
CHECKPOINTS="${CHECKPOINTS:-}"
CHECKPOINT_DIRS="${CHECKPOINT_DIRS:-}"
MAX_MATRICES="${MAX_MATRICES:-128}"
CACHE_ROOT="${CACHE_ROOT:-matrix_program_core/cache/code_real_decode}"
FORCE_REBUILD_CODE_CACHE="${FORCE_REBUILD_CODE_CACHE:-0}"
CODE_PACK_OVERRIDE="${CODE_PACK_OVERRIDE:-}"
SKILL_CKPT_OVERRIDE="${SKILL_CKPT_OVERRIDE:-}"

printf '\n[code-real] repo=%s\n' "$ROOT"
printf '[code-real] report_dir=%s\n' "$REPORT_DIR"
printf '[code-real] parse_dirs=%s checkpoints=%s checkpoint_dirs=%s\n' "$PARSE_DIRS" "$CHECKPOINTS" "$CHECKPOINT_DIRS"
printf '[code-real] skill_epochs=%s audio_epochs=%s modes=%s n_code=%s\n\n' "$EPOCHS_SKILL" "$EPOCHS_AUDIO" "$AUDIO_MODES" "$N_CODE"
TASK_CONTEXT_FLAG="--train-task-context"
if [[ "$TRAIN_TASK_CONTEXT" == "0" ]]; then
  TASK_CONTEXT_FLAG="--no-train-task-context"
fi

git rev-parse --short HEAD 2>/dev/null | sed 's/^/[code-real] git_head=/' || true

mkdir -p "$CACHE_ROOT"
if [[ -n "$CODE_PACK_OVERRIDE" ]]; then
  CODE_PACK="$CODE_PACK_OVERRIDE"
  CODE_CACHE_DIR="$(dirname "$CODE_PACK")"
  printf '\n[code-real] using CODE_PACK_OVERRIDE=%s\n' "$CODE_PACK"
elif [[ -z "$CODE_PACK_OVERRIDE" ]]; then
  CODE_CACHE_KEY="$(
  python - "$PARSE_DIRS" "$MAX_PARSE_FILES" "$N_CODE" "$AMP_SKILL" <<'PY'
import hashlib, sys
print(hashlib.sha1("\n".join(sys.argv[1:]).encode("utf-8")).hexdigest()[:16])
PY
)"
  CODE_CACHE_DIR="$CACHE_ROOT/code_${CODE_CACHE_KEY}"
  CODE_PACK="$CODE_CACHE_DIR/code_context_dataset.pt"
  printf '\n[code-real] code cache key=%s path=%s\n' "$CODE_CACHE_KEY" "$CODE_PACK"
  if [[ "$FORCE_REBUILD_CODE_CACHE" == "1" || ! -f "$CODE_PACK" ]]; then
    printf '\n[code-real] build code-context flow pack\n'
    mkdir -p "$CODE_CACHE_DIR"
    # shellcheck disable=SC2086
    python matrix_program_core/train_assembler_code_context_pretrain.py \
      --old-v3-path neural_matrix_program_dataset_v3/neural_matrix_program_dataset_v3.py \
      --parse-dirs $PARSE_DIRS \
      --max-parse-files "$MAX_PARSE_FILES" \
      --out-dir "$CODE_CACHE_DIR" \
      --device cuda \
      --amp "$AMP_SKILL" \
      --n "$N_CODE" \
      --epochs 0 \
      --log-every "$LOG_EVERY" || true
    if [[ ! -f "$CODE_PACK" ]]; then
      echo "[code-real][WARN] code pack was not created by epochs=0 path; rebuilding with 1 dry epoch small" >&2
      # shellcheck disable=SC2086
      python matrix_program_core/train_assembler_code_context_pretrain.py \
        --old-v3-path neural_matrix_program_dataset_v3/neural_matrix_program_dataset_v3.py \
        --parse-dirs $PARSE_DIRS \
        --max-parse-files "$MAX_PARSE_FILES" \
        --out-dir "$CODE_CACHE_DIR" \
        --device cuda --amp "$AMP_SKILL" --n "$N_CODE" --epochs 1 --log-every "$LOG_EVERY"
    fi
  else
    printf '[code-real] reuse cached code-context flow pack\n'
  fi
fi
cp "$CODE_CACHE_DIR/ast_summary.json" "$REPORT_DIR/code_context_ast_summary.json" 2>/dev/null || true

REAL_PACK=""
if [[ -n "$CHECKPOINTS$CHECKPOINT_DIRS" ]]; then
  printf '\n[code-real] run old real matrix decoder\n'
  # shellcheck disable=SC2086
  python neural_matrix_program_dataset_v3/neural_matrix_program_dataset_v3.py decode-real \
    --out "$WORK/old_real_decode" \
    --device cuda \
    --D 32 \
    --max-matrices "$MAX_MATRICES" \
    --decode-topk 12 \
    --formula-max-terms 12 \
    --functional-batch 256 \
    ${CHECKPOINTS:+--checkpoint $CHECKPOINTS} \
    ${CHECKPOINT_DIRS:+--checkpoint-dir $CHECKPOINT_DIRS}
  REAL_JSONL="$WORK/old_real_decode/real_decode/real_matrix_decodes.jsonl"
  REAL_PACK="$WORK/real_decode_flow_pack.pt"
  python matrix_program_core/real_decode_to_flow_pack.py \
    --input-jsonl "$REAL_JSONL" \
    --out "$REAL_PACK" \
    --dim 96 --layers 4 --blocks 4 --steps 2 --primitive-slots 4 --memory-cells 4 --global-cells 2
else
  printf '\n[code-real] no CHECKPOINTS/CHECKPOINT_DIRS passed; using code-context pack only\n'
fi

MERGED_PACK="$WORK/merged_flow_pack.pt"
python - "$MERGED_PACK" "$CODE_PACK" ${REAL_PACK:+"$REAL_PACK"} <<'PY'
import sys, json, torch
from pathlib import Path
out = Path(sys.argv[1]); paths = [Path(p) for p in sys.argv[2:] if p]
keys = [
    "read_flow","primitive_slot_flow","slot_transition_flow","primitive_transition_flow","slot_composition_flow","write_flow",
    "role_id","input_kind_id","output_kind_id","loss_kind_id","readout_kind_id","num_outputs","sequence_length","hidden_dim","extra_scalar",
    "primitive_hist","primitive_transition_hist","read_hist","write_hist","slot_transition_hist","composition_hist",
]
packs = [torch.load(p, map_location="cpu") for p in paths]
merged = {}
for k in keys:
    vals = [p[k] for p in packs if k in p]
    if vals:
        merged[k] = torch.cat(vals, dim=0)
merged["meta"] = {
    "truth_level":"code_context_plus_real_decode_flow_pack",
    "inputs":[str(p) for p in paths],
    "n": int(merged["read_flow"].shape[0]),
    "program_sketch_keys":[k for k in keys if k.endswith("_hist")],
}
out.parent.mkdir(parents=True, exist_ok=True)
torch.save(merged, out)
print(json.dumps(merged["meta"], ensure_ascii=False, indent=2))
PY
cp "$MERGED_PACK" "$REPORT_DIR/merged_flow_pack.meta.pt" 2>/dev/null || true

if [[ -n "$SKILL_CKPT_OVERRIDE" ]]; then
  SKILL_CKPT="$SKILL_CKPT_OVERRIDE"
  printf '\n[code-real] using SKILL_CKPT_OVERRIDE=%s\n' "$SKILL_CKPT"
else
  printf '\n[code-real] train assembler skill on merged flow pack\n'
  python matrix_program_core/train_assembler_code_context_pretrain.py \
    --dataset "$MERGED_PACK" \
    --out-dir "$WORK/skill_train" \
    --device cuda --amp "$AMP_SKILL" \
    --epochs "$EPOCHS_SKILL" \
    --batch-size 128 --eval-batch-size 256 --workers 2 --pin-memory \
    --lr 4e-4 --lambda-entropy-keep 0.01 --lambda-delta-l2 0.01 \
    --log-every "$LOG_EVERY"
  SKILL_CKPT="$WORK/skill_train/assembler_code_context_best.pt"
  cp "$WORK/skill_train/final_report.json" "$REPORT_DIR/skill_final_report.json" || true
  cp "$WORK/skill_train/metrics.csv" "$REPORT_DIR/skill_metrics.csv" || true
fi
python matrix_program_core/checkpoint_split.py --input "$SKILL_CKPT" --out-dir "$EXPORT_ROOT" --tag code_real_skill

for MODE in $AUDIO_MODES; do
  OUT="$AUDIO_ROOT/$MODE"
  printf '\n[code-real] live audio transfer mode=%s\n' "$MODE"
  python matrix_program_core/transfer_audio_assembler.py \
    --data-root ../architecture_builder/data/speechcommands \
    --assembler-checkpoint "$SKILL_CKPT" \
    --skill-target-pack "$MERGED_PACK" \
    --out-dir "$OUT" \
    --train-mode "$MODE" \
    --task-context-tokens 4 --head-context-tokens 10 --use-head-context \
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
    --lambda-class-read-div "$AUDIO_CLASS_READ_DIV" \
    --lambda-class-slot-prior "$AUDIO_CLASS_SLOT_PRIOR" --lambda-class-attn-entropy "$AUDIO_CLASS_ATTN_ENTROPY" \
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
print("# code + real decode skill live summary")
f = rd / "skill_final_report.json"
if f.exists():
    o=json.loads(f.read_text()); print("\n## skill_train"); print("best_val_loss:",o.get("best_val_loss")); print("checkpoint:",o.get("checkpoint")); print("meta:",o.get("meta"))
mp=rd/"skill_metrics.csv"
if mp.exists():
    rows=list(csv.DictReader(mp.open())); print("skill_last_row:", rows[-1] if rows else None)
for f in sorted(rd.glob("audio_*_final_report.json")):
    mode=f.name.replace("audio_","").replace("_final_report.json","")
    o=json.loads(f.read_text()); print(f"\n## audio_{mode}"); print("best_acc:",o.get("best_acc")); print("best_epoch:",o.get("best_epoch")); print("trainable_summary:",o.get("trainable_summary"))
print("\n## split reports")
for f in sorted(ex.glob("*_split_report.json")):
    o=json.loads(f.read_text()); print(f.name,{k:o.get(k) for k in ["base_params","context_base_params","delta_params","task_params","base_keys","context_base_keys","delta_keys","task_keys"]})
PY

REPORT="$REPORT_DIR/REPORT_TO_CHATGPT.txt"
{
  echo "# REPORT_TO_CHATGPT code real decode skill live"
  echo
  echo "## git"; git rev-parse HEAD 2>/dev/null || true; git status --short 2>/dev/null || true
  echo
  echo "## config"; echo "WORK=$WORK"; echo "CHECKPOINTS=$CHECKPOINTS"; echo "CHECKPOINT_DIRS=$CHECKPOINT_DIRS"; echo "PARSE_DIRS=$PARSE_DIRS"; echo "AUDIO_MODES=$AUDIO_MODES"; echo "TRAIN_TASK_CONTEXT=$TRAIN_TASK_CONTEXT"; echo "AMP_SKILL=$AMP_SKILL"; echo "AMP_AUDIO=$AMP_AUDIO"; echo "SKILL_CKPT=$SKILL_CKPT"
  echo
  echo "## summary"; cat "$REPORT_DIR/summary.txt"
  echo
  echo "## skill metrics tail"; tail -n 30 "$REPORT_DIR/skill_metrics.csv" 2>/dev/null || true
  echo
  for f in "$REPORT_DIR"/audio_*_metrics.csv; do [[ -f "$f" ]] || continue; echo "## $(basename "$f") tail"; tail -n 20 "$f"; echo; done
  echo "## split reports"; for f in "$EXPORT_ROOT"/*_split_report.json; do [[ -f "$f" ]] || continue; echo "### $(basename "$f")"; cat "$f"; echo; done
  echo "## tail full_run.log"; tail -n 240 "$LOG"
} > "$REPORT"

printf '\n[code-real] done. Paste this into ChatGPT:\n'
printf 'cat %s\n' "$REPORT"
