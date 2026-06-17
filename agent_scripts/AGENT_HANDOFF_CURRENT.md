# Agent handoff: current MatrixProgramAssembler transfer tests

## Goal now

We are testing whether a transferable matrix-program assembly skill exists and transfers.

The test is not about solving synthetic classification anymore. The correct target is:

```text
context/code/input/head/real-matrix-program-sketch
  -> MatrixProgramAssemblerCore
  -> read/primitive/transition/composition/write flow targets
```

Then live transfer tests whether the pretrained assembly skill helps a real task:

```text
load assembler skill checkpoint
freeze base or train delta/full
train on SpeechCommands through head/downstream gradient
compare delta vs freeze_core vs full
```

Main success criterion:

```text
audio_delta best_acc > audio_freeze_core best_acc
```

Secondary criteria:

```text
skill/pretrain val_loss decreases
freeze_core improves when base skill becomes better
full is upper bound, but if full >> delta then base/context is still weak
```

## Key idea

Weights must be split:

1. `assembler_skill_base.pt`
   - primitive bank / base read-flow / primitive-flow / transition-flow / write-flow
   - context-flow base projection
   - layer_step_key / phase keys
   - task_context_base if learned during context/code skill pretrain

2. `assembler_skill_delta.pt`
   - only LoRA-like `*_delta`, `ctx_flow_delta`, etc.

3. `task_adapter_head.pt`
   - audio/text/token/matrix input adapter
   - head/consumer/readout
   - task-side context if trained in live task model

## Important files

### Core / context

- `matrix_program_core/assembler_core.py`
  - `MatrixProgramAssemblerCore`
  - after patches: context-conditioned soft flow, layer_step_key, no hard router

- `matrix_program_core/task_context_v2.py`
  - `TaskIOContextEncoder`
  - generic I/O contract context, not hardcoded classification

- `matrix_program_core/checkpoint_split.py`
  - exports base/delta/task packs
  - now keeps top-level `task_context` as `task_context_base`

### Dataset/pretrain paths

- `matrix_program_core/train_assembler_context_skill_pretrain.py`
  - context-only skill pretrain
  - no CE head, no raw audio

- `matrix_program_core/train_assembler_code_context_pretrain.py`
  - parses code through old v3 AST/StructuredSynthesizer
  - converts code program steps to AssemblerCore flow targets
  - no W->label decoder

- `matrix_program_core/real_decode_to_flow_pack.py`
  - converts old `real_matrix_decodes.jsonl` records into flow targets
  - uses `role_guess`, `selected_terms`, `formula`, `metrics`, `original_shape`

- `neural_matrix_program_dataset_v3/neural_matrix_program_dataset_v3.py`
  - old parser/decoder source
  - useful parts:
    - AST parser and `CodeSkeleton`
    - `StructuredSynthesizer`
    - `SparseProgramDecoder`
    - `run_decode_real`
  - do not use old `BaselineDecoder` for this stage

### Runners

- `agent_scripts/run_context_skill_then_live_debug.sh`
  - context-only skill pretrain -> live audio transfer

- `agent_scripts/run_code_context_skill_live_debug.sh`
  - code AST/skeleton flow pretrain -> live audio transfer

- `agent_scripts/run_code_real_decode_skill_live_debug.sh`
  - code AST/skeleton flow pack + optional real matrix decode pack -> live audio transfer
  - this is the current main runner

## Patch scripts used by runners

- `agent_scripts/patch_assembler_context_softflow.sh`
  - makes aux flow tensors differentiable
  - adds context-conditioned soft-flow

- `agent_scripts/patch_task_context_v2_force.sh`
  - rewrites audio transfer model to prepend TaskIOContextEncoder tokens + head query tokens before assembler

- `agent_scripts/patch_task_context_batch_and_load.sh`
  - makes TaskIOContextEncoder accept batch ids/numerics
  - makes live transfer load `task_context` from assembler checkpoint when present

## What the dataset should contain

Each sample should represent:

```text
context:
  role_id
  input_kind_id
  output_kind_id
  loss_kind_id
  readout_kind_id
  numeric shape/interface fields
  code skeleton / layer role
  input-context sketch
  head/consumer-context sketch
  optional real decoded matrix program sketch

targets:
  read_flow
  primitive_slot_flow
  slot_transition_flow
  primitive_transition_flow
  slot_composition_flow
  write_flow
```

For code parsing, input from previous layer is valid input context:

```text
prev_layer_output / token_states / residual_state -> input/cells context
```

For head/consumer context, use the actual consumer program if available:

```text
head matrix decode / next layer / residual writeback / class-query readout
```

The decoded head program is not the answer; it is consumer-context describing how the result will be read.

## Current commands

### Quick code-only transfer test

```bash
cd /home/maxwelhelp/test/sience/experiments/math_search/WORKING_BEST/test

git pull --ff-only

AUDIO_MODES="delta freeze_core" \
EPOCHS_SKILL=2 \
EPOCHS_AUDIO=2 \
N_CODE=2000 \
LOG_EVERY=25 \
PARSE_DIRS="matrix_program_core simple_butterfly_matrix_v4" \
bash agent_scripts/run_code_real_decode_skill_live_debug.sh

cat agent_reports/latest_code_real_decode_skill_live/REPORT_TO_CHATGPT.txt
```

### Full code + real matrix decode transfer test

Pass real local checkpoint files/directories. The checkpoint files are local and must not be committed.

```bash
cd /home/maxwelhelp/test/sience/experiments/math_search/WORKING_BEST/test

git pull --ff-only

AUDIO_MODES="delta freeze_core full" \
EPOCHS_SKILL=5 \
EPOCHS_AUDIO=3 \
N_CODE=6000 \
LOG_EVERY=50 \
PARSE_DIRS="matrix_program_core simple_butterfly_matrix_v4" \
CHECKPOINT_DIRS="matrix_program_core/runs neural_matrix_program_dataset_v3/runs" \
MAX_MATRICES=128 \
AUDIO_LAMBDA_SKILL=0.001 \
bash agent_scripts/run_code_real_decode_skill_live_debug.sh

cat agent_reports/latest_code_real_decode_skill_live/REPORT_TO_CHATGPT.txt
```

If there are too many checkpoints, use a specific file instead:

```bash
CHECKPOINTS="/path/to/local/best.pt" \
MAX_MATRICES=128 \
bash agent_scripts/run_code_real_decode_skill_live_debug.sh
```

## What to inspect in the report

Look at:

```text
skill_train best_val_loss
skill_metrics tail

audio_delta best_acc
audio_freeze_core best_acc
audio_full best_acc

split reports:
  base_params
  context_base_params
  delta_params
  task_params
```

Interpretation:

```text
delta > freeze_core
  transferable skill + delta adaptation helps

freeze_core close to delta
  base skill itself transfers well

full >> delta
  base/context still weak; full relearns task-specific internals

skill val_loss good but audio bad
  likely InputAdapter/evidence mismatch; next step is AudioEvidenceAdapterV2
```

## Next engineering steps after report

1. If scripts crash: fix runner/shape/API issues first.
2. If skill val_loss does not decrease: improve code/real-decode to flow-target conversion.
3. If skill val_loss decreases but transfer poor: build `AudioEvidenceAdapterV2` with explicit local/global/memory/head/context tokens.
4. If delta > freeze consistently: run longer, more code dirs, more real matrices, and test another task/layer replacement.
5. For layer replacement, use a different TaskIO contract:

```text
role_id = attention_replacement or sequence_mixer
input_kind_id = token_states
output_kind_id = token_states
loss_kind_id = distillation/downstream
readout_kind_id = residual_writeback
head_context = next-layer/residual consumer query/program sketch
```
