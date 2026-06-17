# Program Actor-Critic Mind Plan

This document extends `PROGRAM_CENTRAL_MIND_V2_THESIS.md` with the concrete missing policy layer: an actor-critic system that can act on every level of the MatrixProgramAssembler while staying soft, matrix-based, differentiable, and protected by heldout validation.

The key idea:

```text
Memory alone is data.
Critic turns memory into prediction.
Actor turns prediction into soft program edits.
Counterfactual tests turn predictions into real measured experience.
Trust region keeps the actor from destroying training.
```

---

## 1. Core thesis

A real central mind is not just:

```text
gradient -> explanation
```

It is:

```text
program context + candidate edit + memory -> predicted heldout gain + uncertainty
```

Then:

```text
actor proposes edit bias
critic scores it
counterfactual probe tests it
validator accepts/rejects it
memory stores tested result
actor learns from critic + heldout path
```

This makes the central mind useful beyond SGD because it uses:

- cross-run memory;
- heldout/val gain;
- counterfactual tests;
- uncertainty/exploration;
- tested accepted/rejected edits;
- trust-region control.

---

## 2. What the critic is

The critic is a small model trained on memory records:

```text
(context_embedding, edit_embedding) -> predicted_heldout_gain, uncertainty, risk
```

Do not use a Gaussian Process. The dataset can grow to thousands/millions of edit records, so use a small MLP ensemble or MC-dropout MLP.

Recommended first critic:

```text
CriticInput = concat(
  context_embedding,      # what program/task/place looks like
  edit_embedding,         # proposed action
  current_metrics,        # entropy, skill, loss, acc, collapse flags
  recency_features,       # epoch/progress/source age
  complexity_features     # edit size, macro count, active cells
)

CriticOutput:
  mean_gain               # predicted heldout gain, higher is better
  log_variance            # uncertainty/noise estimate
  risk_score              # probability edit hurts heldout
```

MVP implementation:

```text
critic = ensemble of 3-5 small MLPs
uncertainty = variance across ensemble predictions
```

This is the first real “mind”: it predicts the effect of repairs that were not yet tried in the current run.

---

## 3. What the actor is

The actor is the active ProgramMetaController.

It does not output hard edits. It outputs bounded soft biases:

```text
central_read_bias
central_primitive_bias
central_slot_transition_bias
central_primitive_transition_bias
central_composition_bias
central_write_bias
central_macro_bias
central_phase_bias
optional central_alive_bias
```

These biases enter exactly where current flows are already built:

```text
flow_logits = base + context + flow_editor + macro_bias + central_actor_bias
```

Everything remains differentiable.

The actor can work at multiple levels:

1. primitive-level actor
2. transition-level actor
3. read/write actor
4. macro actor
5. step/layer recipe actor
6. global recipe actor

The early MVP should not make one huge actor. Use a hierarchical actor.

---

## 4. Hierarchy of action levels

### Level 0: component action

Acts on one softmax group.

Examples:

```text
increase primitive product_gate at L1.S1.B2.K0
shift read mass from state.B0 to memory.M2
increase transition low_rank -> product_gate at L1.S1
reduce write to global.G1
```

Outputs:

```text
delta_read_logits[L,S,B,K,A]
delta_primitive_logits[L,S,B,K,P]
delta_transition_logits[L,S,P,P]
```

### Level 1: macro action

Chooses or modifies macro recipes.

Examples:

```text
activate macro_7 at L1.S1.B2
reduce macro_0 because it is background/collapse
replace macro_3 with macro_12 in suppress phase
```

Outputs:

```text
delta_macro_logits[L,S,B,M]
delta_macro_gain[L,S,B]
```

### Level 2: step action

Changes the whole block-step fingerprint.

Examples:

```text
make L1.S1.B2 a compare-gate step
make L2.S0.B0 suppress memory noise
make L3.S1.B3 aggregate global context
```

Outputs grouped biases for read/primitive/transition/write.

### Level 3: layer recipe action

Acts on all steps/blocks in a layer.

Examples:

```text
L0 should be more local/extract
L1 should use more low_rank -> product_gate compare chains
L2 should suppress redundant memory reads
L3 should aggregate global/memory to state
```

Outputs phase/layer-level priors.

### Level 4: global program action

Controls global constraints:

```text
macro reuse budget
primitive load balance
class-slot specialization
memory/global usage floor
complexity budget
soft/discrete temperature
```

---

## 5. Required context: what the actor/critic must see

Simply adding fields like `primitive_id` is weak. Each object token needs identity, location, current role, current weight, pressure, future context, and history.

### 5.1 Program object context

For every important object, build a typed token:

```text
object_type: primitive / transition / read / write / macro / slot / class / step / layer
where: L, S, B, K, A, P, M
phase: extract / compare / suppress / aggregate
current_weight: softmax mass
entropy: group entropy
usage: recent usage / macro reuse count
```

### 5.2 Local data context

What is happening at this location:

```text
slot norm / update norm
read source summary
write destination summary
class attention to this slot
head/query pressure
loss contribution proxy
```

### 5.3 Future/downstream context

A primitive is not useful alone; it is useful because later components read it.

Add summaries:

```text
who reads this slot later
which class/head reads it
which memory/global cell receives it
next-step primitive histogram
next-step write targets
```

MVP approximation:

```text
next_step_read_hist
next_step_write_hist
class_slot_attention_hist
macro_next_usage_hist
```

### 5.4 Task/head context

Use existing task/head context, plus:

```text
class confusion vector
per-class loss
head query embedding
class-slot attention entropy
input evidence summary
```

### 5.5 Memory retrieval context

Retrieve top-k similar tested edit records:

```text
retrieved_edit_embedding
retrieved_gain
retrieved_uncertainty
retrieved_accepted/rejected
retrieved_recency
```

---

## 6. Fixed-size cross-task embeddings

Different projects can have different T/B/K/P/M. Cross-task memory only works if contexts map into a common vector space.

Use typed-token pooling, not raw fixed histograms only.

Recommended encoder:

```text
ProgramContextEncoder:
  object tokens -> DeepSets / small factorized attention -> fixed vector C

EditEncoder:
  edit tokens -> DeepSets / small MLP pooling -> fixed vector E
```

Object token fields:

```text
type embedding
role embedding
location embedding normalized
primitive/category embedding
address cell type embedding
soft weight scalar
grad/counterfactual pressure scalar
usage/entropy scalar
```

Pooling:

```text
mean + max + attention pooling by type
```

Output fixed dimensions:

```text
context_embedding: 128 or 256
edit_embedding: 64 or 128
```

This makes SMC, matrix-water-blockless, TTS/vocoder, and other program grids comparable.

---

## 7. Candidate edit generation

The actor/critic needs candidate edits. Do not generate every possible edit.

Candidate sources:

1. gradient pressure candidates
   - high `grad * weight` components.

2. suppressed revival candidates
   - low weight but high memory/critic potential.

3. macro candidates
   - macro selector alternatives.

4. memory retrieval candidates
   - edits that worked in similar contexts.

5. anti-collapse candidates
   - reduce dominant macro/primitive, spread load.

6. soft-vs-discrete gap candidates
   - edits that reduce relaxation gap.

Candidate edit format:

```json
{
  "edit_type": "shift_read | increase_primitive | replace_macro | increase_transition | suppress_component | layer_prior_shift",
  "where": "L1.S1.B2.K0",
  "target": "product_gate or memory.M2 or macro_7",
  "delta_logit": 0.25,
  "scope": "component | step | layer | global",
  "complexity_delta": 0.01
}
```

---

## 8. Acquisition policy: UCB / Expected Improvement

Do not choose counterfactual candidates only by gradient magnitude.

Use the critic:

```text
score(edit) = predicted_gain + beta * uncertainty - lambda_complexity * complexity_delta - lambda_risk * risk
```

This is UCB-like acquisition.

Interpretation:

- exploitation: predicted gain high;
- exploration: uncertainty high;
- safety: risk/complexity low.

Start values:

```text
beta = 0.5 to 1.5
lambda_complexity = 0.1
lambda_risk = 0.5
```

If no trained critic exists yet, use a bootstrap score:

```text
score = 0.35 * gradient_score + 0.35 * memory_similarity_score + 0.20 * anti_collapse_score + 0.10 * random_exploration
```

---

## 9. Two-tier testing to save compute

P40 budget is limited. Use two-stage testing.

### Tier 1: cheap screen

Forward-only counterfactual on heldout micro-batch:

```text
apply edit bias temporarily
run forward only
measure val loss delta
revert
```

No optimizer steps.

### Tier 2: expensive verify

Only for top candidates after screen:

```text
clone small state or checkpoint pointer
apply edit / train N small steps
measure heldout improvement
revert or keep
```

Memory `accepted=true` requires Tier 2 or repeated Tier 1 with strong significance.

---

## 10. Noise floor and statistical significance

Heldout micro-batches are noisy. Do not accept tiny deltas blindly.

Before accepting edits, estimate baseline noise:

```text
measure unchanged model on several heldout micro-batches
noise_std = std(loss)
```

Acceptance rule:

```text
accept if heldout_gain > max(min_gain, k * noise_std)
```

Recommended:

```text
k = 2.0
min_gain = 0.005 to 0.01 relative/CE units, depending on metric scale
```

Store:

```text
noise_std
z_score = gain / noise_std
```

Reject or mark uncertain if below threshold.

---

## 11. Non-stationarity handling

What was useful at epoch 3 may be harmful at epoch 50.

Memory record must include:

```text
epoch
training_progress = epoch / total_epochs
run_id
model_version
program_version
accuracy/loss range
```

Retrieval weighting:

```text
memory_weight = similarity * recency_weight * reproducibility_weight
```

Periodically re-test old accepted edits in the current model state.

If no longer reproducible:

```text
decay record confidence
or mark stale
```

---

## 12. Compound edits and credit assignment

Compound edits are powerful but hard to attribute.

Rules:

1. Early memory records should change one or two factors max.
2. For compound edits, store sub-edit list.
3. Use small factorial tests later:
   - A only
   - B only
   - A+B
4. Critic should receive edit decomposition tokens.

Avoid recording large compound edits as if every component caused the gain.

---

## 13. Trust region for active actor

Do not use a fixed global `central_edit_scale` only.

Use adaptive trust region:

```text
start scale small, e.g. 0.03
if heldout window improves: scale *= 1.05
if heldout worsens: scale *= 0.5 and revert recent actor bias
cap scale, e.g. <= 0.20
```

The actor outputs proposed bias; trust-region controller decides how much of it to apply.

This prevents destructive interventions in unfamiliar contexts.

---

## 14. How the actor affects the existing matrix program

The actor does not execute Python logic. It writes soft matrix biases.

### Primitive mixing

```text
primitive_logits += central_primitive_bias[L,S,B,K,P]
```

### Primitive transitions

```text
primitive_transition_logits += central_primitive_transition_bias[L,S,P,P]
```

### Read/write

```text
read_logits += central_read_bias[L,S,B,K,A]
write_logits += central_write_bias[L,S,B,A]
```

### Slot transitions/composition

```text
slot_transition_logits += central_slot_transition_bias[L,S,B,K,K]
composition_logits += central_composition_bias[L,S,B,K]
```

### Macro selection

```text
macro_selector_logits += central_macro_bias[L,S,B,M]
```

### Layer recipe

```text
phase_mix_logits += central_phase_bias[L,S,phase]
```

Everything stays differentiable.

---

## 15. Actor-critic training loop

Full loop:

```text
for epoch:
  train normal model / program with bi-level split

  collect traces:
    program objects
    metrics
    macro usage
    soft/discrete gap

  generate candidate edits:
    gradient candidates
    memory candidates
    suppressed revival candidates
    anti-collapse candidates
    macro alternatives

  score candidates by critic UCB

  cheap screen top candidates on heldout micro-batch

  expensive verify best candidates

  store tested records in ProgramMemoryBank

  update critic on memory

  train/distill actor:
    imitate high-gain edits
    avoid rejected edits
    optionally combine with heldout architecture gradient

  apply actor bias through trust region
```

---

## 16. MVP order

### MVP 1: fix metrics and collect usable traces

- macro logging correct;
- primitive/macro/read/write usage summaries;
- class-slot and memory/global summaries.

### MVP 2: counterfactual screen without critic

- generate candidates by mixed heuristic;
- forward-only val screen;
- record deltas with noise floor.

### MVP 3: ProgramMemoryBank records

- JSONL memory;
- embedding keys;
- accepted/rejected tested edits only.

### MVP 4: Critic MLP ensemble

- train on memory;
- predict mean gain and uncertainty.

### MVP 5: UCB candidate selection

- counterfactual tests chosen by critic UCB.

### MVP 6: Rule/critic actor

- no neural actor yet;
- apply best critic edit as bounded soft bias.

### MVP 7: Neural ProgramMetaActor

- typed-token actor;
- outputs central bias tensors;
- trained by heldout path + critic distillation;
- trust-region controlled.

---

## 17. Why this can act on all assembler levels

The same actor-critic scheme applies to every action family:

| Level | Context | Edit | Critic target |
|---|---|---|---|
| primitive | L/S/B/K/P, phase, class pressure | increase/decrease primitive | heldout gain |
| transition | L/S/P->P, chain history | increase/decrease transition | heldout gain |
| read | L/S/B/K/A, cell type, future readers | shift read cell | heldout gain |
| write | L/S/B/A, memory/global usage | shift write target | heldout gain |
| macro | L/S/B/M, macro history | activate/replace/suppress macro | heldout gain |
| step | step fingerprint | convert to compare/suppress/etc. | heldout gain |
| layer | layer phase histogram | change layer recipe prior | heldout gain |
| global | whole program metrics | balance/complexity/temperature | heldout gain |

So yes: one critic/actor family can manage primitives, operators, macros, steps, layers, and whole recipe.

But implement it hierarchically, not as one giant tensor at first.

---

## 18. What to implement next in this repo

1. `program_memory_bank.py`
   - JSONL records with embeddings, gain, uncertainty, accepted/rejected.

2. `program_counterfactual_screen.py`
   - forward-only edit probes on heldout micro-batches.

3. `program_critic.py`
   - MLP ensemble critic.

4. `program_actor_bias.py`
   - bounded bias container for read/primitive/transition/write/macro.

5. `run_actor_critic_mind_mvp.sh`
   - run macro32 + counterfactual screen + critic training + bounded actor bias.

Before that, fix macro metrics and add anti-collapse losses.

---

## 19. Minimal first code target

The first useful code target should not be a huge actor.

Implement:

```text
CounterfactualScreen + MemoryBank + CriticMLP
```

Then let the rule engine choose one bounded edit per epoch:

```text
best_edit = argmax critic_UCB(candidates)
apply best_edit with trust scale 0.03
if heldout improves, keep/increase scale
else revert/decrease scale
```

This gives a real active mind with minimal complexity.
