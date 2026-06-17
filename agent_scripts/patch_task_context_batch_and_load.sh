#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python - <<'PY'
from pathlib import Path

# Patch TaskIOContextEncoder to accept scalar ids OR batch tensors.
p = Path("matrix_program_core/task_context_v2.py")
txt = p.read_text(encoding="utf-8")
orig = txt
old = '''    def _id_tensor(self, value: int, batch: int, device: torch.device) -> torch.Tensor:
        return torch.full((batch,), int(value), device=device, dtype=torch.long)
'''
new = '''    def _id_tensor(self, value, batch: int, device: torch.device) -> torch.Tensor:
        if torch.is_tensor(value):
            v = value.to(device=device, dtype=torch.long).view(-1)
            if v.numel() == 1:
                return v.expand(batch)
            if v.numel() != batch:
                raise ValueError(f"context id tensor has {v.numel()} items, expected {batch}")
            return v
        return torch.full((batch,), int(value), device=device, dtype=torch.long)

    def _num_tensor(self, value, batch: int, device: torch.device, default: float = 0.0) -> torch.Tensor:
        if torch.is_tensor(value):
            v = value.to(device=device, dtype=torch.float32).view(-1)
            if v.numel() == 1:
                return v.expand(batch)
            if v.numel() != batch:
                raise ValueError(f"context numeric tensor has {v.numel()} items, expected {batch}")
            return v
        return torch.full((batch,), float(value if value is not None else default), device=device, dtype=torch.float32)
'''
if old in txt:
    txt = txt.replace(old, new)

old = '''        nums = torch.tensor(
            [float(num_outputs), float(sequence_length), float(hidden_dim), float(extra_scalar)],
            device=device,
            dtype=torch.float32,
        ).view(1, 4).expand(B, 4)
'''
new = '''        nums = torch.stack([
            self._num_tensor(num_outputs, B, device, 1.0),
            self._num_tensor(sequence_length, B, device, 1.0),
            self._num_tensor(hidden_dim, B, device, 1.0),
            self._num_tensor(extra_scalar, B, device, 0.0),
        ], dim=-1)
'''
if old in txt:
    txt = txt.replace(old, new)
if txt != orig:
    p.write_text(txt, encoding="utf-8")
    print("[patch] TaskIOContextEncoder accepts batch ids/numerics")
else:
    print("[patch] TaskIOContextEncoder already ok")

# Patch transfer_audio_assembler to load task_context from checkpoint when present.
p = Path("matrix_program_core/transfer_audio_assembler.py")
txt = p.read_text(encoding="utf-8")
orig = txt
old = '''        missing, unexpected = model.assembler_core.load_state_dict(state, strict=False)
        print(f"loaded assembler checkpoint: {args.assembler_checkpoint} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
'''
new = '''        missing, unexpected = model.assembler_core.load_state_dict(state, strict=False)
        print(f"loaded assembler checkpoint: {args.assembler_checkpoint} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        if "task_context" in ckpt and hasattr(model, "task_context"):
            miss_ctx, unexp_ctx = model.task_context.load_state_dict(ckpt["task_context"], strict=False)
            print(f"loaded task_context from assembler checkpoint missing={len(miss_ctx)} unexpected={len(unexp_ctx)}", flush=True)
'''
if old in txt:
    txt = txt.replace(old, new)
if txt != orig:
    p.write_text(txt, encoding="utf-8")
    print("[patch] transfer loads task_context from assembler checkpoint")
else:
    print("[patch] transfer task_context load already ok or marker missing")
PY

grep -n "def _num_tensor\|loaded task_context\|context numeric tensor" matrix_program_core/task_context_v2.py matrix_program_core/transfer_audio_assembler.py
