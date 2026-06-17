# Checkpoint Split and Training Modes

Goal: separate transferable matrix-program assembly skill from task-specific input/head weights.

## 1. What is the transferable skill?

The transferable skill is not the input adapter and not the task head. It is the mechanism that assembles matrix programs:

```text
AssemblerSkillPack =
  primitive implementations / primitive bank
  read_flow base matrices
  primitive_slot_flow base matrices
  slot_transition_flow base matrices
  primitive_transition_flow base matrices
  slot_composition_flow base matrices
  write_flow base matrices
  memory/global flow priors
  context-flow base projections
  layer_step_key / phase keys
  optional base TaskIOContextEncoder embeddings
```

This checkpoint should be saved separately as:

```text
assembler_skill_base.pt
```

It should not include raw audio-specific weights or a classification-only head.

## 2. What is task-specific?

Task-specific weights are:

```text
InputAdapter: audio/text/image/matrix/token-state -> evidence tokens
Task/Consumer Head: class head, reconstruction head, residual writeback, next-layer consumer
TaskIOContext deltas/free tokens: compact hints for this task/interface
```

Save these separately as:

```text
task_adapter_head.pt
```

## 3. What is LoRA-like adaptation?

Delta weights are small task adaptations of the assembly skill:

```text
read_delta
primitive_slot_delta
slot_transition_delta
primitive_transition_delta
slot_composition_delta
write_delta
ctx_flow_delta
optional task_context_delta
```

Save them separately as:

```text
assembler_skill_delta.pt
```

## 4. Why context is not the same as head gradient

Context tells the assembler what kind of program to assemble.

Gradient tells the weights whether the assembled program solved the task.

They are complementary:

```text
context -> conditions flow_logits before execution
gradient -> updates base/delta/context/head/input weights after loss
```

For a frozen pretrained skill, context can select a useful soft program without changing base weights. For a new task, head/downstream gradient still trains the input adapter/head and optionally delta weights.

## 5. What context should the assembler see?

The assembler should see compact structured context, not raw task metadata strings and not all raw data.

### Input context

Not “all audio data”. Instead:

```text
InputAdapter(input) -> evidence tokens
  local tokens
  global summary tokens
  memory seed tokens
  boundary/position tokens if needed
```

For attention replacement:

```text
token states -> evidence tokens
  local token summaries
  global sequence summary
  residual/writeback context
```

### Task I/O context

Generic interface contract, not `classification` hardcode:

```text
role_id
input_kind_id
output_kind_id
loss_kind_id
readout_kind_id
num_outputs / hidden_dim / sequence length
free learned context tokens
```

### Head / consumer context

The actual consumer query vectors should be visible before assembly:

```text
class query vectors
next-layer/residual consumer vectors
program-slot query vectors
reconstruction query vectors
```

### Layer/step context

Each program step should know where it is:

```text
layer_step_key[l, s]
phase: extract / compare / suppress / aggregate / writeback
```

## 6. Two valid training modes

### Mode A: Synthetic pretrain with generated dataset

Used to learn the base skill pack.

```text
synthetic input evidence
+ synthetic TaskIO contract
+ synthetic head/consumer queries
+ flow targets
+ execution target
-> train assembler_skill_base
```

Loss:

```text
task/execution loss
+ flow supervision loss
+ prefix/intermediate loss
+ entropy/anti-collapse regularizer
```

Output:

```text
assembler_skill_base.pt
```

### Mode B: Live task training without synthetic flow dataset

Used when replacing a layer or training on a real task.

The assembler sees:

```text
real input evidence
+ TaskIO contract
+ real head/consumer context
+ layer_step keys
```

It trains from:

```text
head/downstream gradient
+ weak skill anchor to base flow
+ entropy/anti-collapse regularizer
+ optional distillation/reconstruction/prefix loss
```

No flow labels are required.

This is important for replacing attention/MLP/mixer inside a network:

```text
token states -> InputAdapter -> evidence
+ role_id=attention_replacement or sequence_mixer
+ output_kind=token_states
+ readout_kind=residual_writeback
+ consumer context from next layer/residual path
-> AssemblerCore -> token-state output
-> downstream loss/distillation gradient
```

## 7. What to freeze in each mode

### freeze_core

```text
freeze assembler_skill_base
train InputAdapter + Head + TaskIO free/context tokens
```

Tests whether the skill transfers at all.

### delta

```text
freeze assembler_skill_base
train *_delta + InputAdapter + Head + TaskIO context
```

Main LoRA-like mode.

### full

```text
train everything
```

Upper bound, but can forget the skill.

## 8. Correct checkpoint format

```python
torch.save({
  "assembler_skill_base": assembler_core.base_state_dict(),
  "task_context_base": task_context_encoder.state_dict(),
  "meta": {
    "dim": D,
    "layers": L,
    "steps": S,
    "blocks": B,
    "primitive_slots": K,
    "primitives": primitive_names,
    "roles_seen": [...],
    "input_kinds_seen": [...],
    "output_kinds_seen": [...],
  }
}, "assembler_skill_base.pt")
```

For task delta:

```python
torch.save({
  "assembler_skill_delta": only_delta_state_dict(),
  "input_adapter": input_adapter.state_dict(),
  "head_or_consumer": head.state_dict(),
  "task_context_delta": task_context_delta.state_dict(),
  "task_meta": {...}
}, "task_delta.pt")
```

## 9. Interpretation of current result

Current result already shows a useful split:

```text
delta > freeze_core
full > delta
```

Meaning:

- pretrained skill is not useless;
- delta adaptation helps;
- base skill/context are still not universal enough, because full can improve more.

Next engineering step: implement explicit save/load helpers for base vs delta vs task adapter/head so experiments do not mix checkpoints accidentally.
