#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python - <<'PY'
from pathlib import Path

p = Path("matrix_program_core/assembler_core.py")
txt = p.read_text(encoding="utf-8")
orig = txt

repls = {
    '"read_flow": read_w.detach().float(),': '"read_flow": read_w.float(),',
    '"primitive_slot_flow": prim_w.detach().float(),': '"primitive_slot_flow": prim_w.float(),',
    '"slot_transition_flow": slot_trans.detach().float(),': '"slot_transition_flow": slot_trans.float(),',
    '"primitive_transition_flow": prim_trans.detach().float(),': '"primitive_transition_flow": prim_trans.float(),',
    '"slot_composition_flow": comp_w.detach().float(),': '"slot_composition_flow": comp_w.float(),',
    '"write_flow": write_w.detach().float(),': '"write_flow": write_w.float(),',
}
missing = []
for a, b in repls.items():
    if a in txt:
        txt = txt.replace(a, b)
    elif b not in txt:
        missing.append(a)
if missing:
    raise SystemExit("PATCH_FAIL missing patterns:\n" + "\n".join(missing))
if txt != orig:
    p.write_text(txt, encoding="utf-8")
    print("[patch] assembler flow tensors now keep gradient for assembler_skill_loss")
else:
    print("[patch] already applied")
PY

grep -n '"read_flow"\|"primitive_slot_flow"\|"slot_transition_flow"\|"primitive_transition_flow"\|"slot_composition_flow"\|"write_flow"' matrix_program_core/assembler_core.py | head -30
