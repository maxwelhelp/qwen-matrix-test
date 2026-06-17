#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"
REPORT_DIR="agent_reports/skill_debug_${STAMP}"
LATEST_LINK="agent_reports/latest_skill_debug"
mkdir -p "$REPORT_DIR"
rm -f "$LATEST_LINK"
ln -s "skill_debug_${STAMP}" "$LATEST_LINK"

LOG="$REPORT_DIR/full_run.log"
exec > >(tee -a "$LOG") 2>&1

EPOCHS="${EPOCHS:-3}"
N_SYNTH="${N_SYNTH:-2000}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
HIDDEN="${HIDDEN:-512}"
DEVICE="${DEVICE:-cuda}"
# Optional: pass an existing dataset.pt to skip parse/build entirely.
# Example: DATASET_PATH=neural_matrix_program_dataset_v3/runs/.../synthetic/dataset.pt bash agent_scripts/run_skill_debug_3ep.sh
DATASET_PATH="${DATASET_PATH:-}"

TOOL="neural_matrix_program_dataset_v3/neural_matrix_program_dataset_v3.py"
RUN_ROOT="neural_matrix_program_dataset_v3/runs/agent_skill_debug_${STAMP}"
DATASET_RUN="$RUN_ROOT/dataset_base"

printf '\n[agent] repo=%s\n' "$ROOT"
printf '[agent] report_dir=%s\n' "$REPORT_DIR"
printf '[agent] epochs=%s n=%s batch=%s hidden=%s device=%s\n\n' "$EPOCHS" "$N_SYNTH" "$BATCH_SIZE" "$HIDDEN" "$DEVICE"

git rev-parse --short HEAD 2>/dev/null | sed 's/^/[agent] git_head=/' || true

printf '\n[agent] applying local resume patch to %s\n' "$TOOL"
python - <<'PY'
from pathlib import Path

p = Path("neural_matrix_program_dataset_v3/neural_matrix_program_dataset_v3.py")
txt = p.read_text(encoding="utf-8")
orig = txt

# 1) CLI args for loading/resuming the synthetic skill decoder.
if '--init-decoder' not in txt:
    needle = '    ap.add_argument("--matrix-noise-std", type=float, default=0.0)\n'
    repl = needle + (
        '    ap.add_argument("--init-decoder", default="", help="Load baseline_decoder.pt before train-synth")\n'
        '    ap.add_argument("--resume-decoder", action="store_true", help="Resume from <out>/synthetic/baseline_decoder.pt if it exists")\n'
    )
    if needle not in txt:
        raise SystemExit("PATCH_FAIL: train args insertion point not found")
    txt = txt.replace(needle, repl)

# 2) Load decoder checkpoint after model creation and before optimizer creation.
if '[train-synth] loaded init decoder:' not in txt:
    old = '''    model = BaselineDecoder(
        D=int(data["meta"]["D"]),
        n_ops=len(data["meta"]["op_names"]),
        n_read=len(data["meta"]["read_names"]),
        n_prim=len(data["meta"]["primitive_names"]),
        n_trans=len(data["meta"]["transition_names"]),
        n_pair=0 if pair_flat is None else pair_flat.shape[1],
        hidden=args.hidden,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
'''
    new = '''    model = BaselineDecoder(
        D=int(data["meta"]["D"]),
        n_ops=len(data["meta"]["op_names"]),
        n_read=len(data["meta"]["read_names"]),
        n_prim=len(data["meta"]["primitive_names"]),
        n_trans=len(data["meta"]["transition_names"]),
        n_pair=0 if pair_flat is None else pair_flat.shape[1],
        hidden=args.hidden,
    ).to(device)

    init_decoder = str(getattr(args, "init_decoder", "") or "")
    if bool(getattr(args, "resume_decoder", False)) and not init_decoder:
        candidate = out / "baseline_decoder.pt"
        if candidate.exists():
            init_decoder = str(candidate)
    if init_decoder:
        init_path = Path(init_decoder)
        if not init_path.exists():
            raise FileNotFoundError(f"--init-decoder not found: {init_path}")
        ckpt = torch.load(init_path, map_location=device)
        state = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(
            f"[train-synth] loaded init decoder: {init_path} "
            f"missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
'''
    if old not in txt:
        raise SystemExit("PATCH_FAIL: decoder model block not found")
    txt = txt.replace(old, new)

if txt != orig:
    p.write_text(txt, encoding="utf-8")
    print("[patch] changed neural_matrix_program_dataset_v3.py")
else:
    print("[patch] already applied")
PY

printf '\n[agent] patch confirmation:\n'
grep -n "init-decoder\|resume-decoder\|loaded init decoder" "$TOOL" || true

COMMON_BUILD=(
  --parse-dir ./simple_butterfly_matrix_v3 ./simple_butterfly_matrix_v2 ./simple_butterfly_matrix
  --n "$N_SYNTH"
  --D 32
  --layers 4
  --blocks 4
  --steps 2
  --primitive-slots 3
  --max-program-steps 16
  --device "$DEVICE"
  --max-parse-files 300
  --log-every 500
)

COMMON_TRAIN=(
  --device "$DEVICE"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --hidden "$HIDDEN"
  --log-every-epoch 1
)

copy_metrics() {
  local name="$1"
  local path="$2/synthetic/baseline_metrics.json"
  if [[ -f "$path" ]]; then
    cp "$path" "$REPORT_DIR/${name}_metrics.json"
  else
    echo "[WARN] metrics not found: $path"
  fi
}

if [[ -n "$DATASET_PATH" ]]; then
  if [[ ! -f "$DATASET_PATH" ]]; then
    echo "[agent][ERROR] DATASET_PATH does not exist: $DATASET_PATH" >&2
    exit 2
  fi
  DATASET="$DATASET_PATH"
  printf '\n[agent] using existing cached dataset: %s\n' "$DATASET"
else
  printf '\n[agent] build dataset once only; all 4 train runs reuse this dataset.pt\n'
  python "$TOOL" build-synth \
    "${COMMON_BUILD[@]}" \
    --out "$DATASET_RUN"
  DATASET="$DATASET_RUN/synthetic/dataset.pt"
fi

printf '\n[agent] dataset=%s\n' "$DATASET"
ls -lh "$DATASET" || true

printf '\n[agent] run 1/4: fresh baseline, no resume\n'
python "$TOOL" train-synth \
  --dataset "$DATASET" \
  --out "$RUN_ROOT/fresh_a" \
  "${COMMON_TRAIN[@]}"
copy_metrics fresh_a "$RUN_ROOT/fresh_a"

printf '\n[agent] run 2/4: resume from fresh_a baseline_decoder.pt on same dataset\n'
python "$TOOL" train-synth \
  --dataset "$DATASET" \
  --out "$RUN_ROOT/resume_from_fresh_a" \
  "${COMMON_TRAIN[@]}" \
  --init-decoder "$RUN_ROOT/fresh_a/synthetic/baseline_decoder.pt"
copy_metrics resume_from_fresh_a "$RUN_ROOT/resume_from_fresh_a"

printf '\n[agent] run 3/4: masked/denoising fresh on same dataset\n'
python "$TOOL" train-synth \
  --dataset "$DATASET" \
  --out "$RUN_ROOT/masked_fresh" \
  "${COMMON_TRAIN[@]}" \
  --matrix-mask-frac 0.15 \
  --matrix-rowcol-mask-frac 0.05 \
  --matrix-noise-std 0.02
copy_metrics masked_fresh "$RUN_ROOT/masked_fresh"

printf '\n[agent] run 4/4: masked/denoising resume from masked_fresh baseline_decoder.pt\n'
python "$TOOL" train-synth \
  --dataset "$DATASET" \
  --out "$RUN_ROOT/masked_resume" \
  "${COMMON_TRAIN[@]}" \
  --init-decoder "$RUN_ROOT/masked_fresh/synthetic/baseline_decoder.pt" \
  --matrix-mask-frac 0.15 \
  --matrix-rowcol-mask-frac 0.05 \
  --matrix-noise-std 0.02
copy_metrics masked_resume "$RUN_ROOT/masked_resume"

printf '\n[agent] building metric summary\n'
python - "$REPORT_DIR" <<'PY' | tee "$REPORT_DIR/summary.txt"
import json
import sys
from pathlib import Path

rd = Path(sys.argv[1])
keys = [
    "used_ops_f1",
    "used_ops_exact",
    "first_op_acc",
    "last_op_acc",
    "read_hist_kl",
    "primitive_hist_kl",
    "transition_hist_kl",
    "primitive_transition_pair_kl",
    "masked_read_hist_kl",
    "masked_primitive_hist_kl",
    "masked_transition_hist_kl",
    "masked_primitive_transition_pair_kl",
]
files = [
    "fresh_a_metrics.json",
    "resume_from_fresh_a_metrics.json",
    "masked_fresh_metrics.json",
    "masked_resume_metrics.json",
]
print("# skill debug summary")
for fn in files:
    path = rd / fn
    print(f"\n## {fn}")
    if not path.exists():
        print("MISSING")
        continue
    m = json.loads(path.read_text(encoding="utf-8"))
    for k in keys:
        if k in m:
            v = m[k]
            if isinstance(v, float):
                print(f"{k}: {v:.6f}")
            else:
                print(f"{k}: {v}")

def load(name):
    p = rd / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

fresh = load("fresh_a_metrics.json")
resume = load("resume_from_fresh_a_metrics.json")
masked = load("masked_fresh_metrics.json")
masked_resume = load("masked_resume_metrics.json")
print("\n# deltas")
for label, a, b in [("resume-fresh", fresh, resume), ("masked_resume-masked_fresh", masked, masked_resume)]:
    print(f"\n## {label}")
    for k in keys:
        if k in a and k in b:
            print(f"{k}: {b[k] - a[k]:+.6f}")
PY

REPORT="$REPORT_DIR/REPORT_TO_CHATGPT.txt"
{
  echo "# REPORT_TO_CHATGPT skill debug"
  echo
  echo "## git"
  git rev-parse HEAD 2>/dev/null || true
  git status --short 2>/dev/null || true
  echo
  echo "## config"
  echo "EPOCHS=$EPOCHS"
  echo "N_SYNTH=$N_SYNTH"
  echo "BATCH_SIZE=$BATCH_SIZE"
  echo "HIDDEN=$HIDDEN"
  echo "DEVICE=$DEVICE"
  echo "DATASET=$DATASET"
  echo
  echo "## patch grep"
  grep -n "init-decoder\|resume-decoder\|loaded init decoder" "$TOOL" || true
  echo
  echo "## summary"
  cat "$REPORT_DIR/summary.txt"
  echo
  echo "## raw metrics"
  for f in "$REPORT_DIR"/*_metrics.json; do
    echo "### $(basename "$f")"
    cat "$f"
    echo
  done
  echo
  echo "## tail full_run.log"
  tail -n 120 "$LOG"
} > "$REPORT"

printf '\n[agent] done. Paste this into ChatGPT:\n'
printf 'cat %s\n' "$REPORT"
printf '\n[agent] no .pt files were copied into report_dir. Run outputs are under: %s\n' "$RUN_ROOT"
find "$REPORT_DIR" -maxdepth 1 -type f -printf '[agent] report file: %p\n'
