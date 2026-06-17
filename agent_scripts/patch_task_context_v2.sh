#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python - <<'PY'
from pathlib import Path

# 1) Patch assembler_core.py: add explicit layer_step_key into context flow.
p = Path("matrix_program_core/assembler_core.py")
txt = p.read_text(encoding="utf-8")
orig = txt
if "self.layer_step_key = nn.Parameter" not in txt:
    needle = '''        self.gate_bias = nn.Parameter(torch.full((self.D,), -0.35))

        # Base assembly matrices.
'''
    repl = '''        self.gate_bias = nn.Parameter(torch.full((self.D,), -0.35))
        self.layer_step_key = nn.Parameter(torch.randn(self.D) * 0.02)

        # Base assembly matrices.
'''
    if needle not in txt:
        raise SystemExit("PATCH_FAIL assembler_core: layer_step_key insertion point not found")
    txt = txt.replace(needle, repl)

# If context-softflow patch has been applied, strengthen ctx composition.
old = '''        ctx = cells.mean(dim=1).float()
        flat = self.ctx_flow(ctx)
'''
new = '''        state_ctx = cells[:, : self.B].mean(dim=1)
        mem0 = self.B
        glob0 = self.B + int(self.cfg.memory_cells)
        if int(self.cfg.memory_cells) > 0:
            memory_ctx = cells[:, mem0:glob0].mean(dim=1)
        else:
            memory_ctx = torch.zeros_like(state_ctx)
        if int(self.cfg.global_cells) > 0:
            global_ctx = cells[:, glob0:].mean(dim=1)
        else:
            global_ctx = torch.zeros_like(state_ctx)
        # Explicit layer/step key tells the assembler where it is in the program grid.
        ctx = (state_ctx + 0.7 * memory_ctx + 0.7 * global_ctx + self.layer_step_key.to(device=cells.device, dtype=cells.dtype).view(1, -1)).float()
        flat = self.ctx_flow(ctx)
'''
if old in txt:
    txt = txt.replace(old, new)
elif "state_ctx = cells[:, : self.B].mean(dim=1)" not in txt:
    print("[patch] assembler_core: context-softflow block not found; layer_step_key added but context split not applied")

if txt != orig:
    p.write_text(txt, encoding="utf-8")
    print("[patch] assembler_core layer_step_key/context split applied")
else:
    print("[patch] assembler_core already patched")

# 2) Patch transfer_audio_assembler.py: use TaskIOContextEncoder if old head-context patch is present.
p = Path("matrix_program_core/transfer_audio_assembler.py")
txt = p.read_text(encoding="utf-8")
orig = txt
if "from matrix_program_core.task_context_v2 import" not in txt:
    needle = '''from matrix_program_core.train_assembler_pretrain import build_flow_targets  # noqa: E402
from simple_butterfly_matrix.simple_butterfly_matrix import (  # noqa: E402
'''
    repl = '''from matrix_program_core.train_assembler_pretrain import build_flow_targets  # noqa: E402
from matrix_program_core.task_context_v2 import (  # noqa: E402
    TaskContextV2Config,
    TaskIOContextEncoder,
    ROLE_TASK_HEAD_CORE,
    INPUT_AUDIO_FEATURES,
    OUTPUT_CLASS_LOGITS,
    LOSS_CROSS_ENTROPY,
    READOUT_CLASS_QUERY,
)
from simple_butterfly_matrix.simple_butterfly_matrix import (  # noqa: E402
'''
    if needle not in txt:
        raise SystemExit("PATCH_FAIL transfer: import insertion point not found")
    txt = txt.replace(needle, repl)

old = '''        self.head = AudioAssemblerHead(args.dim, classes, args.head_dropout)
        self.task_context_tokens = int(getattr(args, "task_context_tokens", 4))
        self.head_context_tokens = int(getattr(args, "head_context_tokens", classes))
        self.use_head_context = bool(getattr(args, "use_head_context", True))
        self.task_context = nn.Parameter(torch.randn(max(1, self.task_context_tokens), args.dim) * 0.02)
        self.context_norm = nn.LayerNorm(args.dim)
'''
new = '''        self.head = AudioAssemblerHead(args.dim, classes, args.head_dropout)
        self.use_head_context = bool(getattr(args, "use_head_context", True))
        self.task_context = TaskIOContextEncoder(TaskContextV2Config(
            dim=args.dim,
            free_tokens=int(getattr(args, "task_context_tokens", 4)),
            max_head_tokens=int(getattr(args, "head_context_tokens", classes)),
            dropout=args.dropout,
        ))
'''
if old in txt:
    txt = txt.replace(old, new)
elif "self.task_context = TaskIOContextEncoder" not in txt:
    print("[patch] transfer: old task_context block not found; maybe patch_audio_head_context_tokens not applied yet")

old = '''        ctx_parts = []
        if self.task_context_tokens > 0:
            task_ctx = self.task_context[: self.task_context_tokens].to(device=evidence.device, dtype=evidence.dtype)
            ctx_parts.append(task_ctx.unsqueeze(0).expand(evidence.shape[0], -1, -1))
        if self.use_head_context and self.head_context_tokens > 0:
            # The assembler sees the head/query space before it assembles the program.
            # This is not the target label. It is the task/output interface context.
            hq = self.head.query[: min(self.head_context_tokens, self.head.query.shape[0])]
            hq = hq.to(device=evidence.device, dtype=evidence.dtype)
            ctx_parts.append(hq.unsqueeze(0).expand(evidence.shape[0], -1, -1))
        if ctx_parts:
            context = self.context_norm(torch.cat(ctx_parts, dim=1))
            evidence = torch.cat([context, evidence], dim=1)
'''
new = '''        head_query = self.head.query if self.use_head_context else None
        context = self.task_context(
            evidence.shape[0],
            evidence.device,
            evidence.dtype,
            role_id=int(getattr(self, "role_id", ROLE_TASK_HEAD_CORE)),
            input_kind_id=int(getattr(self, "input_kind_id", INPUT_AUDIO_FEATURES)),
            output_kind_id=int(getattr(self, "output_kind_id", OUTPUT_CLASS_LOGITS)),
            loss_kind_id=int(getattr(self, "loss_kind_id", LOSS_CROSS_ENTROPY)),
            readout_kind_id=int(getattr(self, "readout_kind_id", READOUT_CLASS_QUERY)),
            num_outputs=float(self.head.classes),
            sequence_length=float(evidence.shape[1]),
            hidden_dim=float(evidence.shape[-1]),
            head_query=head_query,
        )
        evidence = torch.cat([context, evidence], dim=1)
'''
if old in txt:
    txt = txt.replace(old, new)
elif "context = self.task_context(" not in txt:
    print("[patch] transfer: forward context block not found; maybe patch_audio_head_context_tokens not applied yet")

if txt != orig:
    p.write_text(txt, encoding="utf-8")
    print("[patch] transfer_audio_assembler uses TaskIOContextEncoder v2")
else:
    print("[patch] transfer_audio_assembler already patched or waiting for head-context patch")
PY

grep -n "TaskIOContextEncoder\|layer_step_key\|state_ctx\|role_id\|num_outputs" matrix_program_core/transfer_audio_assembler.py matrix_program_core/assembler_core.py | head -80
