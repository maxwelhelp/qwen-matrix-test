#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"
REPORT_DIR="agent_reports/assembler_core_${STAMP}"
LATEST_LINK="agent_reports/latest_assembler_core"
mkdir -p "$REPORT_DIR"
rm -f "$LATEST_LINK"
ln -s "assembler_core_${STAMP}" "$LATEST_LINK"

LOG="$REPORT_DIR/full_run.log"
exec > >(tee -a "$LOG") 2>&1

PRETRAIN_OUT="matrix_program_core/runs/assembler_pretrain_${STAMP}"
AUDIO_ROOT="matrix_program_core/runs/audio_assembler_${STAMP}"
EPOCHS_PRETRAIN="${EPOCHS_PRETRAIN:-3}"
EPOCHS_AUDIO="${EPOCHS_AUDIO:-3}"
AUDIO_MODES="${AUDIO_MODES:-delta freeze_core full}"
TRAIN_N="${TRAIN_N:-6000}"
VAL_N="${VAL_N:-1000}"

printf '\n[assembler] repo=%s\n' "$ROOT"
printf '[assembler] report_dir=%s\n' "$REPORT_DIR"
printf '[assembler] pretrain_epochs=%s audio_epochs=%s modes=%s train_n=%s val_n=%s\n\n' "$EPOCHS_PRETRAIN" "$EPOCHS_AUDIO" "$AUDIO_MODES" "$TRAIN_N" "$VAL_N"

git rev-parse --short HEAD 2>/dev/null | sed 's/^/[assembler] git_head=/' || true

printf '\n[assembler] run pretrain: factorized MatrixProgramAssemblerCore\n'
python matrix_program_core/train_assembler_pretrain.py \
  --out-dir "$PRETRAIN_OUT" \
  --device cuda \
  --amp bf16 \
  --train-n "$TRAIN_N" \
  --val-n "$VAL_N" \
  --classes 10 \
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
  --epochs "$EPOCHS_PRETRAIN" \
  --lr 5e-4 \
  --lambda-skill 0.35 \
  --lambda-recon 0.20 \
  --log-every 25

ASSEMBLER_CKPT="$PRETRAIN_OUT/assembler_core_best.pt"
if [[ ! -f "$ASSEMBLER_CKPT" ]]; then
  echo "[assembler][ERROR] missing assembler checkpoint: $ASSEMBLER_CKPT" >&2
  exit 2
fi
cp "$PRETRAIN_OUT/final_report.json" "$REPORT_DIR/assembler_pretrain_final_report.json" || true
cp "$PRETRAIN_OUT/metrics.csv" "$REPORT_DIR/assembler_pretrain_metrics.csv" || true

printf '\n[assembler] transfer assembler checkpoint: %s\n' "$ASSEMBLER_CKPT"
for MODE in $AUDIO_MODES; do
  OUT="$AUDIO_ROOT/$MODE"
  printf '\n[assembler] run audio transfer mode=%s\n' "$MODE"
  python matrix_program_core/transfer_audio_assembler.py \
    --data-root ../architecture_builder/data/speechcommands \
    --assembler-checkpoint "$ASSEMBLER_CKPT" \
    --out-dir "$OUT" \
    --train-mode "$MODE" \
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
    --lambda-skill 0.05 \
    --lambda-write-budget 0.025 \
    --lambda-update-alive 0.005 \
    --lambda-logit-norm 0.0007 \
    --grad-clip 0.75 \
    --log-every 25
  cp "$OUT/final_report.json" "$REPORT_DIR/audio_${MODE}_final_report.json" || true
  cp "$OUT/metrics.csv" "$REPORT_DIR/audio_${MODE}_metrics.csv" || true
  cp "$OUT/analysis_epoch_$(printf '%03d' "$EPOCHS_AUDIO").json" "$REPORT_DIR/audio_${MODE}_last_analysis.json" || true
done

printf '\n[assembler] build summary\n'
python - "$REPORT_DIR" <<'PY' | tee "$REPORT_DIR/summary.txt"
import csv, json, sys
from pathlib import Path
rd = Path(sys.argv[1])
print("# assembler core summary")
pre = rd / "assembler_pretrain_final_report.json"
if pre.exists():
    obj = json.loads(pre.read_text(encoding="utf-8"))
    print("\n## assembler_pretrain")
    print("best_acc:", obj.get("best_acc"))
    print("best_epoch:", obj.get("best_epoch"))
    print("checkpoint:", obj.get("checkpoint"))
    print("config:", obj.get("config"))
mp = rd / "assembler_pretrain_metrics.csv"
if mp.exists():
    rows = list(csv.DictReader(mp.open()))
    if rows:
        print("pretrain_last_row:", rows[-1])
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
    apath = rd / f"audio_{mode}_last_analysis.json"
    if apath.exists():
        a = json.loads(apath.read_text(encoding="utf-8"))
        report = a.get("val", {}).get("report", {})
        print("flow_report:", report)
PY

REPORT="$REPORT_DIR/REPORT_TO_CHATGPT.txt"
{
  echo "# REPORT_TO_CHATGPT assembler core"
  echo
  echo "## git"
  git rev-parse HEAD 2>/dev/null || true
  git status --short 2>/dev/null || true
  echo
  echo "## config"
  echo "PRETRAIN_OUT=$PRETRAIN_OUT"
  echo "AUDIO_ROOT=$AUDIO_ROOT"
  echo "EPOCHS_PRETRAIN=$EPOCHS_PRETRAIN"
  echo "EPOCHS_AUDIO=$EPOCHS_AUDIO"
  echo "AUDIO_MODES=$AUDIO_MODES"
  echo "TRAIN_N=$TRAIN_N"
  echo "VAL_N=$VAL_N"
  echo
  echo "## summary"
  cat "$REPORT_DIR/summary.txt"
  echo
  echo "## pretrain metrics tail"
  tail -n 20 "$REPORT_DIR/assembler_pretrain_metrics.csv" 2>/dev/null || true
  echo
  for f in "$REPORT_DIR"/audio_*_metrics.csv; do
    [[ -f "$f" ]] || continue
    echo "## $(basename "$f") tail"
    tail -n 20 "$f"
    echo
  done
  echo "## tail full_run.log"
  tail -n 220 "$LOG"
} > "$REPORT"

printf '\n[assembler] done. Paste this into ChatGPT:\n'
printf 'cat %s\n' "$REPORT"
printf '\n[assembler] no .pt files copied to report_dir. PT checkpoints stayed under runs.\n'
find "$REPORT_DIR" -maxdepth 1 -type f -printf '[assembler] report file: %p\n'
