#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python - <<'PY'
from pathlib import Path
p = Path("matrix_program_core/assembler_core.py")
txt = p.read_text(encoding="utf-8")
orig = txt

# Fix partial OperatorBankV2 patch state: step_alive append was inserted but
# ent_acc was still missing the key in some local trees.
old = '''        ent_acc: Dict[str, List[torch.Tensor]] = {
            "read": [], "primitive": [], "slot_transition": [], "primitive_transition": [], "write": []
        }
'''
new = '''        ent_acc: Dict[str, List[torch.Tensor]] = {
            "read": [], "primitive": [], "slot_transition": [], "primitive_transition": [], "write": [], "step_alive": []
        }
'''
if old in txt:
    txt = txt.replace(old, new)

# If the exact block differs, repair any ent_acc literal that has write but no step_alive.
needle = '"read": [], "primitive": [], "slot_transition": [], "primitive_transition": [], "write": []'
if needle in txt and '"step_alive": []' not in txt[txt.find('ent_acc:'):txt.find('slot_names:', txt.find('ent_acc:'))]:
    txt = txt.replace(needle, '"read": [], "primitive": [], "slot_transition": [], "primitive_transition": [], "write": [], "step_alive": []')

# Ensure info step_alive exists. If old patch did not add it, add a safe default.
if '"step_alive": step_alive.detach().float(),' not in txt:
    txt = txt.replace('''            "update_norms": update.detach().float().norm(dim=-1),
            "slot_values": slot_val.detach(),
''', '''            "update_norms": update.detach().float().norm(dim=-1),
            "step_alive": torch.ones((), device=cells.device).float(),
            "slot_values": slot_val.detach(),
''')
    txt = txt.replace('''            "update_norms": active_update.detach().float().norm(dim=-1),
            "slot_values": slot_val.detach(),
''', '''            "update_norms": active_update.detach().float().norm(dim=-1),
            "step_alive": step_alive.detach().float(),
            "slot_values": slot_val.detach(),
''')

# Ensure the append is safe even if older checkpoints/code produce no key.
txt = txt.replace('''            ent_acc["step_alive"].append(info["step_alive"])
''', '''            ent_acc.setdefault("step_alive", []).append(info.get("step_alive", torch.ones((), device=evidence.device)))
''')

# Ensure config dict survives if fields were not patched.
if '"operator_v2": bool(getattr(self.cfg, "operator_v2", False)),' not in txt:
    txt = txt.replace('''            "use_deltas": self.cfg.use_deltas,
            "primitives": list(PRIMITIVES),
''', '''            "use_deltas": self.cfg.use_deltas,
            "operator_v2": bool(getattr(self.cfg, "operator_v2", False)),
            "step_alive_init": float(getattr(self.cfg, "step_alive_init", 1.65)),
            "primitives": list(PRIMITIVES),
''')

if txt != orig:
    p.write_text(txt, encoding="utf-8")
    print("[hotfix] fixed OperatorBankV2 step_alive bookkeeping")
else:
    print("[hotfix] nothing to change")
PY

python -m py_compile matrix_program_core/assembler_core.py matrix_program_core/transfer_audio_assembler.py

grep -n "ent_acc\|step_alive" matrix_program_core/assembler_core.py | head -80
