# Program Central Mind / Analyzer Plan

Goal: add a central analytic + memory layer above the current differentiable MatrixProgramAssembler.

The current system can already change assembly softly through gradients:

- `read_flow`
- `primitive_slot_flow`
- `slot_transition_flow`
- `primitive_transition_flow`
- `slot_composition_flow`
- `write_flow`
- `phase_mix`
- `flow_editor`
- optional `macro_step_bank` selector/gain
- class/head reads
- input/task adapters

But it mostly performs implicit optimization: loss -> gradients -> weights change. It does not yet have a clear central module that records which program components were useful, why they helped, what role they played, what similar past repairs existed, and what edit should be tried next.

This document defines the missing layer.

---

## 1. Problem

Right now the assembler can learn, but it does not fully analyze itself.

Missing questions:

- Which primitive was important at `L/S/B/K`?
- Which transition helped: `low_rank -> product_gate`, `product_gate -> phase_matrix`, etc.?
- Which read/write cell was useful or harmful?
- Which macro was selected and why?
- What was the gradient pressure on each component?
- Did the component improve train only or heldout too?
- Was the same edit useful in similar tasks before?
- What should be replaced, increased, suppressed, moved, or remembered?

The central mind should answer these questions and produce structured edit bias, not hard choices.

---

## 2. Core principle

Do not add a huge token-level LLM-style O(N^2) attention over raw audio/text.

Instead, analyze the compressed matrix-program grid:

```text
T = layers * steps = 8
B = blocks = 4
K = primitive slots = 4
A = address cells = 10
P = primitives = 6
M = macro count, e.g. 16 or 32
```

The analyzer sees program objects, not raw sequence tokens.

This keeps it small and fast:

```text
program objects ~= T*B*(read/write/primitive/slot summaries) + macro summaries + class-slot summaries
```

No raw token O(N^2). All attention is over program components, or factorized over `(step, block, component_type)`.

---

## 3. What already exists

Current differentiable mechanisms:

1. `FlowEditAttention`
   - edits read/primitive/slot_transition/primitive_transition/composition/write logits before softmax.
   - uses mechanism tokens.
   - differentiable.

2. Context flow
   - `ctx_flow` and `ctx_flow_delta` add context-dependent flow biases.

3. Deltas
   - `read_delta`, `primitive_slot_delta`, `slot_transition_delta`, `primitive_transition_delta`, `slot_composition_delta`, `write_delta`.

4. MacroStepBank MVP
   - frozen macro prototypes built from flow recipes.
   - macro selector is softmax over macros.
   - macro contributes additive log-flow bias.
   - differentiable selector/gain.
   - no hard router.

What is missing:

- gradient attribution by component;
- memory of accepted/rejected edits;
- recommendations;
- self-explanatory traces;
- macro variation promotion;
- central edit controller using memory.

---

## 4. Proposed architecture

Add these modules in order.

### 4.1 ProgramTraceAnalyzer v1

Pure analyzer. It should not change the model at first.

Inputs:

- current flows from `aux`:
  - `read_flow`
  - `primitive_slot_flow`
  - `slot_transition_flow`
  - `primitive_transition_flow`
  - `slot_composition_flow`
  - `write_flow`
- slots;
- class-slot attention;
- phase mix;
- macro selector weights/gain, if macro bank enabled;
- task/input/head context;
- optional gradients or approximate pressure.

Outputs:

- JSON traces per epoch/batch;
- component importance tables;
- top helpful/harmful primitives;
- top helpful/harmful transitions;
- top useful/harmful read/write cells;
- macro usage and pressure;
- recommendations.

Example trace:

```json
{
  "where": "L1.S1.B2",
  "phase": "compare",
  "read_top": {"global.G0": 0.31, "memory.M2": 0.22, "state.B2": 0.18},
  "primitive_top": {"product_gate": 0.31, "low_rank": 0.24, "phase_matrix": 0.21},
  "transition_top": {"low_rank->product_gate": 0.27, "product_gate->phase_matrix": 0.19},
  "write_top": {"state.B2": 0.35, "memory.M2": 0.24},
  "macro_top": {"macro_7": 0.44, "macro_2": 0.18},
  "pressure": {
    "product_gate": 0.31,
    "low_rank->product_gate": 0.27,
    "read_memory.M2": 0.22,
    "write_memory.M2": 0.18
  },
  "recommendation": [
    "increase product_gate",
    "increase low_rank->product_gate",
    "shift more read mass to memory.M2",
    "keep macro_7 but reduce ctx_matrix self-loop"
  ]
}
```

### 4.2 GradientAttributor v1

Purpose: turn gradients into compact component pressure.

Fast method:

```text
pressure(component) ~= abs(gradient(component_logit) * current_soft_weight)
```

Signed method:

```text
positive pressure: increasing this component likely lowers loss
negative pressure: component likely harmful
```

Better method later:

```text
small ablation / counterfactual delta loss
```

What to attribute:

- read cell pressure: `L/S/B/K/A`
- primitive pressure: `L/S/B/K/P`
- primitive transition pressure: `L/S/P/P`
- slot transition pressure: `L/S/B/K/K`
- composition pressure: `L/S/B/K`
- write pressure: `L/S/B/A`
- macro pressure: `L/S/B/M`
- phase pressure: `L/S/phase`
- class-slot pressure: `class/slot`

Do not store full gradient tensors by default. Store top-k and summaries.

### 4.3 ProgramMemoryBank v1

Persistent memory of repair experience.

Stores:

```json
{
  "context_signature": {
    "task_family": "audio_classification",
    "input_kind": "mel/evidence",
    "head_kind": "class_query",
    "phase": "compare",
    "where": "L1.S1.B2"
  },
  "program_before_summary": "...",
  "gradient_pressure": "...",
  "edit_action": "increase product_gate + low_rank->product_gate + read memory.M2",
  "program_after_summary": "...",
  "metrics": {
    "train_loss_before": 1.12,
    "train_loss_after": 1.06,
    "heldout_gain": 0.02,
    "accepted": true
  },
  "reuse_count": 3
}
```

Must store both accepted and rejected edits.

Rejected examples are important so the model learns not to repeat shortcuts.

### 4.4 ProgramObjectAttention / CentralProgramAttention v1

This is the actual central mind, but only after analyzer works.

It should use typed object tokens, not raw sequence tokens.

Token types:

1. Step tokens
   - location `L/S`
   - phase mix
   - entropy summaries
   - loss/skill deltas

2. Block-step tokens
   - location `L/S/B`
   - read/write summaries
   - primitive histogram
   - transition fingerprint
   - macro usage

3. Primitive tokens
   - `L/S/B/K/P`
   - current weight
   - gradient pressure
   - historical usefulness

4. Transition tokens
   - `L/S/P->P`
   - weight
   - gradient pressure
   - macro association

5. Read/write tokens
   - `L/S/B/K/A` or summarized top-k
   - current mass
   - pressure
   - cell type: state/memory/global

6. Macro tokens
   - macro id
   - selector weight
   - gain
   - historical gain
   - similarity to memory cases

7. Class/head tokens
   - class id
   - class-slot attention
   - class loss/confusion pressure

8. Memory retrieval tokens
   - top similar past repairs
   - accepted/rejected labels
   - gain stats

Output:

- additive edit biases:
  - `delta_read_logits`
  - `delta_primitive_logits`
  - `delta_slot_transition_logits`
  - `delta_primitive_transition_logits`
  - `delta_composition_logits`
  - `delta_write_logits`
  - `delta_macro_logits`
  - optional `delta_alive_logits`

All outputs are soft and differentiable.

No hard edit decisions in MVP.

---

## 5. Attention head design

Do not create 500 large heads like an LLM over raw tokens. Instead create many small specialized matrix heads over program objects.

Recommended MVP: 16-32 small heads grouped by function.

### Group A: location/role heads

1. phase role head
   - detects whether a step behaves like extract/compare/suppress/aggregate.

2. layer-step order head
   - checks whether early/late steps are doing appropriate work.

3. block specialization head
   - checks whether B0..B3 are different or collapsed.

4. slot specialization head
   - checks whether K slots are different or collapsed.

### Group B: component usefulness heads

5. primitive usefulness head
   - reads primitive weights + pressure.

6. primitive transition usefulness head
   - focuses on chains like `low_rank -> product_gate -> phase_matrix`.

7. read usefulness head
   - state/memory/global read pressure.

8. write usefulness head
   - state/memory/global write pressure.

9. composition usefulness head
   - detects whether slot composition uses useful slots or averages everything.

### Group C: macro heads

10. macro selection head
   - which macro is selected where.

11. macro conflict head
   - detects macro used too often or in wrong phase.

12. macro variation head
   - detects where macro should be modified, not just selected.

13. macro memory retrieval head
   - retrieves similar past macro repairs.

### Group D: task/head interaction heads

14. class-slot specialization head
   - detects whether classes read different slots.

15. class confusion head
   - links class errors to program components.

16. head-consumer pressure head
   - links head gradients to slot/primitive/read/write pressure.

### Group E: stability/regularization heads

17. entropy/collapse head
   - finds uniform collapse or one-hot collapse.

18. complexity budget head
   - checks if too many macros/steps/cells are used.

19. memory/global usage head
   - checks if memory/global are dead or overloaded.

20. heldout-risk head
   - detects edits that improve train but likely harm heldout.

### Later expansion: 64-128 micro-heads

If MVP helps, expand to 64-128 micro-heads, not as raw token attention but as typed factorized heads:

- 16 primitive/transition heads
- 16 read/write/cell heads
- 16 macro/memory heads
- 16 task/head/error heads
- 16 stability/complexity heads
- optional 32 domain-specific heads

This is closer in spirit to “many small Qwen-like heads”, but each head has a clear matrix-program role.

---

## 6. Fast non-O(N^2) design

Central attention must not attend over raw input length.

Use factorized program attention:

```text
Step attention:          O(T^2), T=8 small
Block attention:         O(T * B^2), B=4 small
Slot attention:          O(T * B * K^2), K=4 small
Primitive attention:     O(T * B * K * P), P=6 small
Macro attention:         O(T * B * M), M=16/32 small
Memory retrieval:        top-k over cached embeddings, approximate/offline
```

This is tiny compared to sequence attention.

Do not build a giant flat attention over every possible `(L,S,B,K,A,P,M)` object unless top-k compressed.

---

## 7. What to implement first

### Phase 0: reports only

Add `program_trace_analyzer.py`.

Command:

```bash
python matrix_program_core/program_trace_analyzer.py \
  --checkpoint path/to/best.pt \
  --data-root ../architecture_builder/data/speechcommands \
  --out agent_reports/.../program_trace_analysis.json \
  --max-batches 8
```

Output:

- top primitives per layer/step/block;
- top read/write cells;
- top primitive transitions;
- class-slot attention;
- macro usage;
- entropy/collapse metrics;
- if possible: gradient pressure for top components.

No training change.

### Phase 1: gradient pressure capture

Modify training to optionally retain gradients for flow logits / macro selector.

Add flag:

```bash
--trace-program-pressure
--trace-program-every 1
--trace-top-k 12
```

Store only top-k pressure summaries.

### Phase 2: memory bank

Add:

```text
matrix_program_core/program_memory_bank.py
```

Store compressed repair records:

- context signature;
- before/after summaries;
- pressure;
- accepted/rejected;
- train/heldout gain.

### Phase 3: recommendation engine

Rule-based first:

- if primitive pressure positive and current weight low -> recommend increase.
- if transition pressure positive and current transition low -> recommend add/increase.
- if read pressure positive to memory/global -> recommend read shift.
- if macro pressure positive -> recommend macro gain/selector increase.
- if class-slot collapse -> recommend class-slot diversity/head entropy adjustment.
- if macro top1 usage too high -> recommend macro reuse penalty.

No neural controller yet.

### Phase 4: differentiable central edit bias

Add `ProgramMetaController`:

```text
program object tokens + memory retrieval tokens -> edit bias tensors
```

Inject as:

```text
flow_logits = base + context + editor + macro + central_edit_bias
```

Keep it bounded:

```text
central_edit_scale <= 0.10 initially
```

### Phase 5: macro variation recorder

After training, extract useful modified macro patterns and build macro-bank v2.

Accept only if:

- heldout improves;
- not duplicate;
- not too complex;
- reused across multiple examples.

---

## 8. Files to add/change

### New files

1. `matrix_program_core/program_trace_analyzer.py`
   - offline analyzer for checkpoints and datasets.

2. `matrix_program_core/program_gradient_attributor.py`
   - utilities for grad*weight and ablation pressure.

3. `matrix_program_core/program_memory_bank.py`
   - JSONL/SQLite-like compressed memory of accepted/rejected edits.

4. `matrix_program_core/program_meta_controller.py`
   - later: differentiable central mind.

5. `agent_scripts/run_program_trace_analysis.sh`
   - collect traces from latest runs.

6. `agent_scripts/run_central_mind_mvp.sh`
   - later: train with central edit bias.

### Modify

1. `matrix_program_core/assembler_core.py`
   - expose optional trace hooks for flows/logits/macro selector.
   - expose macro selector weights before detach.
   - optionally accept `central_edit_bias` in `AssemblerStep.forward`.

2. `matrix_program_core/transfer_audio_assembler.py`
   - add analyzer flags.
   - save trace JSON per epoch.
   - optionally compute gradient pressure.

3. `matrix_program_core/train_assembler_mechanism_skill_pretrain.py`
   - use analyzer on synthetic repair episodes.
   - save accepted/rejected repair summaries.

4. `matrix_program_core/build_macro_step_bank.py`
   - later: build macro bank from accepted variation memory, not only static flow recipes.

5. `MATRIX_PROGRAM_VERSION_LEDGER.csv`
   - add rows for central mind phases.

---

## 9. What not to do yet

Do not immediately add:

- free-form Python code generation;
- arbitrary new primitive functions;
- dynamic tensor-shape mutation during training;
- hard top-k macro routing;
- huge raw-token attention;
- saving full gradients for every batch.

First build observation, pressure, memory, and recommendations.

Then make it active.

---

## 10. Success metrics

Analyzer success:

- produces stable top-k component reports;
- identifies class-slot collapse, macro collapse, dead memory/global;
- recommendations match observed improvements after manual tests.

Memory success:

- accepted edits reused across runs;
- rejected edits reduce repeated bad patterns;
- macro-bank v2 built from memory improves over static macro-bank.

Central controller success:

- improves accuracy vs baseline and macro-only;
- improves class/slot specialization;
- does not increase collapse;
- improves or preserves heldout performance;
- keeps complexity controlled.

---

## 11. Immediate next step

Before implementing active central control, run the current MacroStepBank comparison and inspect:

- baseline vs macro16 vs macro32 accuracy;
- macro entropy/top1 usage;
- class_read_div and slot_div;
- skill drift;
- whether macro helps early convergence or final best.

Then implement `program_trace_analyzer.py` and run it on:

- best baseline checkpoint;
- best macro16 checkpoint;
- best macro32 checkpoint.

The analyzer should tell whether macros improved actual program structure or only acted as a small bias.
