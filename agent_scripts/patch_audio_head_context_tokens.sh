#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python - <<'PY'
from pathlib import Path

p = Path("matrix_program_core/transfer_audio_assembler.py")
txt = p.read_text(encoding="utf-8")
orig = txt

old = '''        self.assembler_core = MatrixProgramAssemblerCore(cfg)
        self.head = AudioAssemblerHead(args.dim, classes, args.head_dropout)

    def forward(self, wav: torch.Tensor):
        evidence = self.input_adapter(wav)
        _cells, aux = self.assembler_core(evidence)
        logits, haux = self.head(aux.slots)
        return logits, aux, haux
'''
new = '''        self.assembler_core = MatrixProgramAssemblerCore(cfg)
        self.head = AudioAssemblerHead(args.dim, classes, args.head_dropout)
        self.task_context_tokens = int(getattr(args, "task_context_tokens", 4))
        self.head_context_tokens = int(getattr(args, "head_context_tokens", classes))
        self.use_head_context = bool(getattr(args, "use_head_context", True))
        self.task_context = nn.Parameter(torch.randn(max(1, self.task_context_tokens), args.dim) * 0.02)
        self.context_norm = nn.LayerNorm(args.dim)

    def forward(self, wav: torch.Tensor):
        evidence = self.input_adapter(wav)
        ctx_parts = []
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
        _cells, aux = self.assembler_core(evidence)
        logits, haux = self.head(aux.slots)
        return logits, aux, haux
'''
if old in txt:
    txt = txt.replace(old, new)
elif "self.task_context_tokens" not in txt:
    raise SystemExit("PATCH_FAIL: AudioAssemblerModel block not found")

old = '''    p.add_argument("--train-mode", choices=["freeze_core", "delta", "full"], default="delta")
    p.add_argument("--train-input-adapter", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--train-head", action=argparse.BooleanOptionalAction, default=True)
'''
new = '''    p.add_argument("--train-mode", choices=["freeze_core", "delta", "full"], default="delta")
    p.add_argument("--task-context-tokens", type=int, default=4)
    p.add_argument("--head-context-tokens", type=int, default=10)
    p.add_argument("--use-head-context", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--train-input-adapter", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--train-head", action=argparse.BooleanOptionalAction, default=True)
'''
if old in txt:
    txt = txt.replace(old, new)
elif "--task-context-tokens" not in txt:
    raise SystemExit("PATCH_FAIL: parser insertion point not found")

if txt != orig:
    p.write_text(txt, encoding="utf-8")
    print("[patch] audio transfer now prepends task/head context tokens before assembler_core")
else:
    print("[patch] already applied")
PY

grep -n "task_context_tokens\|head_context_tokens\|use_head_context\|head/query space\|task-context-tokens" matrix_program_core/transfer_audio_assembler.py
