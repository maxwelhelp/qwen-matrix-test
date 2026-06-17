#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python - <<'PY'
from pathlib import Path
import re

# 1) Make assembler_core context-aware with explicit layer_step_key and split state/memory/global context.
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
    if needle in txt:
        txt = txt.replace(needle, repl)
    else:
        print("[force-patch][WARN] assembler_core layer_step_key insertion point not found")

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
        ctx = (state_ctx + 0.7 * memory_ctx + 0.7 * global_ctx + self.layer_step_key.to(device=cells.device, dtype=cells.dtype).view(1, -1)).float()
        flat = self.ctx_flow(ctx)
'''
if old in txt:
    txt = txt.replace(old, new)
if txt != orig:
    p.write_text(txt, encoding="utf-8")
    print("[force-patch] assembler_core layer_step/context split applied")
else:
    print("[force-patch] assembler_core already ok")

# 2) Force rewrite transfer_audio_assembler.py class/import/parser parts.
p = Path("matrix_program_core/transfer_audio_assembler.py")
txt = p.read_text(encoding="utf-8")
orig = txt

if "from matrix_program_core.task_context_v2 import" not in txt:
    marker = "from matrix_program_core.train_assembler_pretrain import build_flow_targets  # noqa: E402\n"
    insert = marker + '''from matrix_program_core.task_context_v2 import (  # noqa: E402
    TaskContextV2Config,
    TaskIOContextEncoder,
    ROLE_TASK_HEAD_CORE,
    INPUT_AUDIO_FEATURES,
    OUTPUT_CLASS_LOGITS,
    LOSS_CROSS_ENTROPY,
    READOUT_CLASS_QUERY,
)
'''
    if marker not in txt:
        raise SystemExit("PATCH_FAIL force transfer: import marker not found")
    txt = txt.replace(marker, insert)

new_class = '''class AudioAssemblerModel(nn.Module):
    def __init__(self, classes: int, args):
        super().__init__()
        self.input_adapter = MatrixEvidence(args.sample_rate, args.n_mels, args.hop_length, args.evidence_cells, args.dim)
        cfg = AssemblerConfig(
            dim=args.dim,
            evidence_cells=args.evidence_cells,
            layers=args.layers,
            blocks=args.blocks,
            steps=args.steps,
            primitive_slots=args.primitive_slots,
            memory_cells=args.memory_cells,
            global_cells=args.global_cells,
            channel_stages=args.channel_stages,
            dropout=args.dropout,
            use_deltas=True,
        )
        self.assembler_core = MatrixProgramAssemblerCore(cfg)
        self.head = AudioAssemblerHead(args.dim, classes, args.head_dropout)

        # Generic task I/O contract, not a hardcoded classification branch.
        # For attention/layer replacement another script can pass different IDs,
        # but the core still only sees dense context tokens and soft matrices.
        self.role_id = int(getattr(args, "role_id", ROLE_TASK_HEAD_CORE))
        self.input_kind_id = int(getattr(args, "input_kind_id", INPUT_AUDIO_FEATURES))
        self.output_kind_id = int(getattr(args, "output_kind_id", OUTPUT_CLASS_LOGITS))
        self.loss_kind_id = int(getattr(args, "loss_kind_id", LOSS_CROSS_ENTROPY))
        self.readout_kind_id = int(getattr(args, "readout_kind_id", READOUT_CLASS_QUERY))
        self.use_head_context = bool(getattr(args, "use_head_context", True))
        self.task_context = TaskIOContextEncoder(TaskContextV2Config(
            dim=args.dim,
            free_tokens=int(getattr(args, "task_context_tokens", 4)),
            max_head_tokens=int(getattr(args, "head_context_tokens", classes)),
            dropout=args.dropout,
        ))

    def forward(self, wav: torch.Tensor):
        evidence = self.input_adapter(wav)
        head_query = self.head.query if self.use_head_context else None
        context = self.task_context(
            evidence.shape[0],
            evidence.device,
            evidence.dtype,
            role_id=self.role_id,
            input_kind_id=self.input_kind_id,
            output_kind_id=self.output_kind_id,
            loss_kind_id=self.loss_kind_id,
            readout_kind_id=self.readout_kind_id,
            num_outputs=float(self.head.classes),
            sequence_length=float(evidence.shape[1]),
            hidden_dim=float(evidence.shape[-1]),
            head_query=head_query,
        )
        evidence = torch.cat([context, evidence], dim=1)
        _cells, aux = self.assembler_core(evidence)
        logits, haux = self.head(aux.slots)
        return logits, aux, haux


'''
pattern = r'class AudioAssemblerModel\(nn\.Module\):\n.*?\n\ndef trainable_summary\('
if not re.search(pattern, txt, flags=re.S):
    raise SystemExit("PATCH_FAIL force transfer: AudioAssemblerModel block not found")
txt = re.sub(pattern, new_class + 'def trainable_summary(', txt, flags=re.S)

# Add parser args if absent.
parser_marker = '''    p.add_argument("--train-mode", choices=["freeze_core", "delta", "full"], default="delta")
'''
parser_insert = parser_marker + '''    p.add_argument("--task-context-tokens", type=int, default=4)
    p.add_argument("--head-context-tokens", type=int, default=10)
    p.add_argument("--use-head-context", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--role-id", type=int, default=0)
    p.add_argument("--input-kind-id", type=int, default=0)
    p.add_argument("--output-kind-id", type=int, default=0)
    p.add_argument("--loss-kind-id", type=int, default=0)
    p.add_argument("--readout-kind-id", type=int, default=0)
'''
if "--task-context-tokens" not in txt:
    if parser_marker not in txt:
        raise SystemExit("PATCH_FAIL force transfer: parser marker not found")
    txt = txt.replace(parser_marker, parser_insert)

if txt != orig:
    p.write_text(txt, encoding="utf-8")
    print("[force-patch] transfer_audio_assembler uses TaskIOContextEncoder V2")
else:
    print("[force-patch] transfer_audio_assembler already ok")
PY

grep -n "TaskIOContextEncoder\|role_id\|task-context-tokens\|layer_step_key\|state_ctx" \
  matrix_program_core/transfer_audio_assembler.py matrix_program_core/assembler_core.py | head -120
