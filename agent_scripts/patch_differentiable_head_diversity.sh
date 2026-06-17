#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python - <<'PY'
from pathlib import Path
p = Path("matrix_program_core/transfer_audio_assembler.py")
txt = p.read_text(encoding="utf-8")
old = 'return logits, {"class_slot_attention": attn.detach(), "class_read": read.detach()}'
new = 'return logits, {"class_slot_attention": attn, "class_read": read}'
if old in txt:
    txt = txt.replace(old, new)
    p.write_text(txt, encoding="utf-8")
    print("[patch] class_read_div is now differentiable: removed detach from head attention/read")
elif new in txt:
    print("[patch] already applied: head attention/read are differentiable")
else:
    raise SystemExit("PATCH_FAIL: expected AudioAssemblerHead return line not found")
PY

python -m py_compile matrix_program_core/transfer_audio_assembler.py

grep -n "class_slot_attention" matrix_program_core/transfer_audio_assembler.py
