#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT="neural_matrix_program_dataset_v3/runs/check_auto_primitives"

python neural_matrix_program_dataset_v3/neural_matrix_program_dataset_v3.py parse \
  --parse-dir ./simple_butterfly_matrix_v3 ./simple_butterfly_matrix_v2 ./simple_butterfly_matrix \
  --out "$OUT" \
  --max-parse-files 30

python - <<'PY'
import json
from pathlib import Path

sk = json.loads(Path("neural_matrix_program_dataset_v3/runs/check_auto_primitives/ast/skeleton.json").read_text())
print("primitive_count", len(sk["primitive_names"]))
print("primitive_names", sk["primitive_names"])
print("transition_count", len(sk["transition_names"]))
print("transition_names", sk["transition_names"])
print("notes", sk["notes"])
PY

