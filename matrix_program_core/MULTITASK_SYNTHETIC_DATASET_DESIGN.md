# Multitask Synthetic Dataset Design for Matrix Program Assembler

Core idea: synthetic pretrain must expose the same information the real system sees.

Real task view:

```text
InputAdapter(input) -> evidence/state
TaskHead/task query -> what output is needed
AssemblerCore -> matrix program assembly
Head -> final task output
```

Synthetic pretrain should mirror this:

```text
SyntheticInputAdapter(x, task_id, head_query) -> evidence/state
Synthetic task/head query -> desired output type
AssemblerCore -> matrix program assembly
SyntheticHead -> output
```

## Why this is not overfitting to one task

A checkpoint is good if it learns a conditional rule:

```text
given input evidence + task/head context + layer/step context
choose a valid soft matrix program
```

It is bad if it learns one fixed recipe.

Therefore synthetic data must contain many task families and many valid programs, so the assembler learns how to condition assembly, not one answer.

## What synthetic samples should contain

Each sample should include:

- `input_evidence`: what the input adapter would output.
- `task_key`: what task/head wants, e.g. classify, compare, denoise, aggregate, copy, count.
- `head_query`: class/query vector or output slot request.
- `layer_keys`: layer/phase targets: extract, compare, suppress, aggregate, memory, global.
- `target_y`: final output target.
- `target_prefix`: intermediate outputs per step/layer.
- `flow_targets`: optional soft targets for read/primitive/transition/composition/write.
- `solution_family`: short, medium, long, memory-heavy, global-heavy, local-only, mixed.

## Task families for first version

Use small cheap synthetic families that all require different matrix programs:

1. `local_pattern`
   - detect local evidence pattern.
   - favors local read + channel/phase primitives.

2. `global_summary`
   - aggregate over whole evidence.
   - favors global cells + block/channel composition.

3. `selective_copy`
   - copy value from position selected by a key.
   - favors memory and read/write flow.

4. `count_matching`
   - count evidence tokens matching a query.
   - favors compare + aggregate.

5. `rare_event`
   - suppress common background, keep rare signal.
   - favors suppress phase + product_gate.

6. `segment_summary`
   - summarize one segment from several.
   - favors read by task/head query.

7. `denoise_reconstruct`
   - reconstruct clean signal from corrupted evidence.
   - favors multi-step primitive composition.

8. `program_composition`
   - combine two subprograms, e.g. compare then aggregate.
   - favors slot_transition and primitive_transition.

## Multi-solution principle

For each task family, generate several valid solution families:

```text
short:         2 steps, direct read/write
medium:        4 steps, local + memory
deep:          6-8 steps, prefix states
memory-heavy:  uses memory cells
Global-heavy:  uses global cells
local-only:    avoids memory/global
mixed:         uses multiple primitive slots per step
```

Targets should be soft distributions, not one-hot hard choices.

## Layer-specific tasks

Layer phases should not be generic. They should map to different subgoals:

```text
Layer 0 extract:   clean/read/select evidence
Layer 1 compare:   compare against task/head query
Layer 2 suppress:  remove irrelevant/background signals
Layer 3 aggregate: compose/write final state
```

This is why the assembler needs layer/step context key.

## Model requirement

Assembler flow should depend on:

```text
base_flow
+ projection(state_cells)
+ projection(memory_cells)
+ projection(global_cells)
+ learned_layer_step_key
+ task/head/evidence key
```

Still no routers:

```text
flow = softmax(base_logits + context_logits)
```

## First implementation target

Add a new pretrain mode:

```bash
python matrix_program_core/train_assembler_multitask_pretrain.py \
  --task-families local_pattern,global_summary,selective_copy,count_matching,rare_event,segment_summary,denoise_reconstruct,program_composition \
  --solution-families short,medium,deep,memory_heavy,global_heavy,local_only,mixed
```

It should save:

```text
assembler_multitask_best.pt
final_report.json
metrics.csv
```

And transfer should load the same `assembler_core` state.

## Success criteria

1. Pretrain task accuracy improves.
2. Flow skill loss decreases.
3. Flow entropy does not collapse too early.
4. `delta` transfer beats `freeze_core`.
5. `full` improves but does not destroy flow metrics.
