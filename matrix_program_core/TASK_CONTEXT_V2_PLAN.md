# Task Context V2 Plan

Problem: a literal `task_type=classification` is too narrow. If the assembler is inserted inside another network layer instead of attention/MLP/mixer, a classification string is the wrong context.

## Core principle

The assembler should not receive a hardcoded task name. It should receive a **task I/O contract** and **consumer/head context**.

```text
flow_logits = base_flow
            + projection(input/state/memory/global cells)
            + projection(task_io_contract_tokens)
            + projection(head_or_consumer_query_tokens)
            + projection(layer_step_key)

flow = softmax(flow_logits)
```

Still no router:

- no top-k path choice;
- no argmax branch;
- no if task == classification inside core;
- all choices are dense soft matrices.

## What replaces `task_type=classification`

Use a compact continuous context built from generic fields:

```text
role_id          what kind of module this core is used as
input_kind_id    what representation enters the core
output_kind_id   what representation the consumer expects
loss_kind_id     what training signal shapes the output
readout_kind_id  how the next module/head reads slots
num_outputs      scalar/numeric output interface size
free_context     learned free tokens for task-specific hints
head_context     actual consumer/head query vectors
```

Examples:

### Audio classification

```text
role_id = task_head_core
input_kind_id = audio_features
output_kind_id = class_logits
loss_kind_id = cross_entropy
readout_kind_id = class_query_readout
num_outputs = 10
head_context = class query vectors
```

### Replacement for attention inside a network

```text
role_id = sequence_mixer_or_attention_replacement
input_kind_id = token_states
output_kind_id = token_states
loss_kind_id = downstream_loss_or_distillation
readout_kind_id = residual_writeback
num_outputs = hidden_dim or token_count proxy
head_context = next layer / writeback / residual consumer queries
```

### Matrix reconstruction / decompiler task

```text
role_id = matrix_program_decompiler
input_kind_id = matrix_features
output_kind_id = program_or_matrix_reconstruction
loss_kind_id = reconstruction_plus_flow
readout_kind_id = program_slots
num_outputs = target matrix/program dimension proxy
```

## What sees what

### Input/cells context

Comes from actual input evidence after `InputAdapter`:

```text
state cells  -> local/block state
memory cells -> accumulated decisions
global cells -> whole-example summary
```

### Task context

Comes from `TaskIOContextEncoder`, not a hardcoded branch. It returns context tokens that are prepended to evidence before `AssemblerCore`.

### Head context

Comes from the actual consumer/head query vectors. This is important because the assembler should know who will read its slots.

### Layer/step context

Every assembler step has a learned `layer_step_key`. It tells the same core where in the program grid it currently is: extract/compare/suppress/aggregate, early/late step, etc.

## Transfer modes

- `freeze_core`: test if pretrained assembly works with new input/head only.
- `delta`: train only `*_delta` and context parameters; LoRA-like adaptation.
- `full`: upper-bound, but may forget the pretrained assembly rule.

## Checkpoint split

Portable checkpoint should contain:

```text
assembler_core.state_dict()
task_context_encoder.state_dict() optional base/free-context
meta: role/input/output/loss/readout ids used in pretrain
```

Task-specific checkpoint contains:

```text
input_adapter
head/consumer
optional task context deltas
```

## Success criteria

1. Multitask synthetic pretrain keeps high execution accuracy.
2. Flow skill loss decreases without collapsing entropy too much.
3. On audio transfer, `delta > freeze_core`.
4. For layer replacement experiments, the same core can receive a different role/io contract without changing core code.
