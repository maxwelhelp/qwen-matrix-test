# MacroStepBank Plan

Date: 2026-06-17

Goal: create the next MatrixProgramAssembler version from the current
mechanism/repair pipeline, but add reusable differentiable macro steps. The
system should not only choose primitive matrices, but also learn higher-level
matrix-program fragments that can be reused across tasks.

## Critical Read

The idea is useful, but it has one dangerous ambiguity:

```text
"save successful repairs"
```

This is only valid if "successful" is measured outside the same training batch.
If we promote a macro just because it reduces train loss once, the macro bank
will memorize dataset shortcuts and may make transfer worse. So the first
version must be controlled:

1. no dynamic tensor shape changes;
2. no new primitive birth;
3. no hard macro selection;
4. macros are soft flow prototypes made from existing recipes;
5. promotion requires repeated gain or heldout gain;
6. rejected edits are saved too, so the editor learns what not to reuse.

## Current Base

Use the current version as base:

```text
code/AST/real decode flow pack
-> repair episode dataset
-> train_assembler_mechanism_skill_pretrain.py
-> MatrixProgramAssemblerCore
-> transfer_audio_assembler.py
```

Current flow contract:

```text
read_flow
primitive_slot_flow
slot_transition_flow
primitive_transition_flow
slot_composition_flow
write_flow
```

These are already differentiable matrix recipes. The macro bank must add to
these flows, not replace them with a router.

## Macro Levels

Use four levels, but implement only the first two in MVP:

```text
primitive category:
  low_rank / gate / phase / channel / block / memory / global

primitive chain macro:
  low_rank -> product_gate -> phase_matrix

step macro:
  read pattern + primitive mix + operator transition + composition + write

layer macro:
  several step macros distributed across a layer/phase
```

The MVP should implement `step macro`. A step macro is one reusable soft recipe
for one layer-step-block region, not a new code function.

## MacroStepBank Contract

The bank stores `M` macro prototypes:

```text
macro_read[M,K,A]
macro_primitive[M,K,P]
macro_slot_transition[M,K,K]
macro_primitive_transition[M,P,P]
macro_composition[M,K]
macro_write[M,A]
macro_context[M,D]
macro_stats[M,*]
```

Where:

```text
M = macro count, for MVP 16 or 32
K = primitive slots
A = address cells
P = primitive count
D = hidden dim
```

No hard choice:

```text
macro_weight[N,T,B,M] = softmax(context_to_macro_logits(...))
macro_flow = macro_weight @ macro_bank
final_flow_logits = base_flow_logits + editor_delta + macro_gain * macro_flow_logits
```

The macro path is a soft additive prior over existing flow matrices.

## Macro Extraction

MVP extraction should not depend on live accepted repairs yet.

Input sources:

```text
repair_episode_dataset_v2.pt
code_context_dataset.pt
merged_flow_pack.pt
```

For every real program row and every step/block, build a fingerprint:

```text
fingerprint =
  read_hist
  primitive_hist
  primitive_transition_hist
  slot_transition_hist
  composition_hist
  write_hist
  phase_id
  task/context ids
```

Cluster or prototype these fingerprints:

```text
simple MVP:
  normalize fingerprints
  kmeans/farthest-point select 16-32 prototypes
  average the full flows inside each prototype

later:
  split by task family, phase, read/write family
```

This gives a macro bank without claiming that the macro is "successful". It is
only "common and syntactically valid".

## Accepted Repair Promotion

This is stage 2, after the macro path works.

For each repair/edit episode record:

```text
context
program_before
feedback / gradient summary
macro weights before
program_after
loss_before
loss_after
heldout_loss_before
heldout_loss_after
accepted
```

Promote to macro only if:

```text
train gain > threshold
heldout gain > threshold
same macro family repeats on multiple examples
not duplicate of existing macro
complexity cost is acceptable
```

Save rejected edits too. They are useful negative data for the editor.

## Differentiable Usage Control

The bank must not collapse to one favorite macro. Use soft penalties:

```text
macro_entropy_band:
  avoid uniform forever, avoid one-hot too early

macro_layer_reuse_penalty:
  if one macro dominates many blocks in same layer, increase its cost

macro_phase_diversity:
  different phases should prefer different macro mixtures

macro_context_diversity:
  different task/context families should not all use the same macro distribution

macro_complexity_penalty:
  expensive macros need to justify their usage
```

Do not hard-ban reuse. Add smooth penalties to logits/loss.

## Dormant Capacity

Do not add/delete tensors during training. Expansion should use dormant capacity:

```text
max_steps fixed
max_blocks fixed
max_memory_cells fixed
max_macros fixed

step_alive
block_alive
memory_alive
macro_alive
```

Activation means gate mass increases. Deletion means gate mass goes to zero.
This keeps the whole model differentiable.

## How It Connects To Editor

Current editor changes flow logits:

```text
read
primitive
transition
composition
write
```

Macro editor adds:

```text
macro_logits
macro_gain
macro_alive
macro_transition_logits
```

The editor still sees feedback/gradient context, but can now say:

```text
"this looks like compare_gate_chain"
```

instead of rebuilding the same primitive chain from scratch every time.

## MVP Implementation Plan

### Phase 0: Ledger And Metrics

Create a version ledger so experiments do not blur together.

Track at least:

```text
version
base_commit
architecture
dataset
training command
macro_count
train mode
best_acc
best_epoch
class_read_div
slot_div
macro_entropy
macro_top1_usage
notes
decision
```

### Phase 1: Offline Macro Dataset

Add:

```text
matrix_program_core/build_macro_step_bank.py
```

It reads an existing flow pack and writes:

```text
macro_step_bank.pt
macro_step_bank_report.json
macro_step_bank_preview.csv
```

No core changes yet.

Validation:

```text
reconstruct heldout flow from macro mixtures
measure KL per flow key
measure macro diversity
```

Success:

```text
macro reconstruction KL clearly below naive mean-flow baseline
no one macro covers most examples
```

### Phase 2: Soft Macro Injection

Add optional macro path to `MatrixProgramAssemblerCore` or a wrapper:

```text
--macro-step-bank path
--macro-count M
--lambda-macro-usage
--lambda-macro-div
```

The macro flow is added as a soft bias to existing flow logits. No hard router.

Compare:

```text
base mechanism skill
base + macro bank frozen
base + macro bank trainable context embeddings
```

Success:

```text
skill val_loss improves or equal
audio best_acc improves
class_read_div and slot_div improve
macro usage is non-collapsed
```

### Phase 3: Repair Recorder

Add:

```text
RepairEpisodeRecorder
```

It records before/after program summaries and audio metrics during transfer.
Do not promote anything online yet.

Success:

```text
records are small
records include accepted and rejected examples
gradient/feedback summaries are present
```

### Phase 4: Macro Promotion

Build new macro bank from accepted repairs:

```text
old bank + accepted repair prototypes - duplicates
```

Promotion is offline between runs, not mid-run.

Success:

```text
new bank improves next run against same base settings
improvement appears on val, not only train
```

## First Experiment Matrix

Run only after Phase 1 and Phase 2:

```text
A: current fixed mechanism pipeline, no macros
B: same pipeline + frozen macro bank M=16
C: same pipeline + frozen macro bank M=32
D: same pipeline + macro bank M=32 + light macro diversity loss
```

Primary metric:

```text
SpeechCommands best_acc
```

Secondary metrics:

```text
class_read_div
slot_div
macro_entropy
macro_top1_usage
skill val_loss
```

Stop condition:

```text
If B/C do not beat A or specialize better by epoch 20, do not add repair
promotion yet. Fix macro extraction first.
```

## Main Risks

1. Macro bank memorizes syntax, not task usefulness.
2. Macro usage collapses to one common recipe.
3. Macro bias overpowers gradient learning and makes transfer worse.
4. Accepted repair without heldout checks promotes noise.
5. Larger bank adds compute but no capacity if evidence/head is the bottleneck.
6. Macro recipes may hide missing primitive capacity; this delays the real fix.

## Decision

Proceed, but only with controlled MVP:

```text
current fixed pipeline
+ offline macro prototypes from existing flow recipes
+ soft macro injection
+ version ledger
```

Do not implement primitive birth or dynamic layer growth until macro prototypes
show measurable benefit.

