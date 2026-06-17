# Editor Actor-Critic Policy Plan V3

This is the implementation-safe version of the central-mind plan for the current MatrixProgramAssembler codebase.

The previous V2 skeleton was directionally correct, but it still had architecture ambiguities. V3 fixes those before coding:

1. downstream feedback cannot look into the future inside the same forward pass;
2. feedback_bias and actor_bias must have different safety semantics;
3. feedback bus must not be implemented before tested memory exists;
4. Tier-2 rollback must restore full training state, not only weights;
5. growth/promotion needs local and global tiers;
6. macro/operator embeddings must be content-based, not integer-id lookup.

---

## 1. Core thesis

The central mind is not one magic attention head.

It is a safe actor-critic loop around the existing editor:

```text
Editor = actuator that edits assembly logits.
Probe = counterfactual tester that creates real labels.
Memory = tested accepted/rejected/uncertain edit records.
Critic = model predicting heldout gain + uncertainty + risk.
Actor = sparse policy that applies bounded edits.
Feedback bus = lagged/validated experience sent back into assembly workshops.
Growth loop = promotes repeatedly useful recipes into new macros/operators.
```

Everything online remains soft and matrix-based.  
Everything remembered as truth must be tested on heldout/counterfactual checks.

---

## 2. What is allowed to affect logits

Current assembler logits are conceptually:

```text
flow_logits = base + context + current_editor + macro_bias
```

Future version:

```text
flow_logits = base + context + current_editor + macro_bias + feedback_bias + actor_bias
```

But `feedback_bias` and `actor_bias` are not the same thing.

### 2.1 feedback_bias

`feedback_bias` is a learned conditioning path inside the normal bi-level architecture training.

It receives features from lagged traces / memory summaries / critic summaries, but it is optimized by gradient on the heldout architecture path.

It is not a direct deploy decision.

Therefore:

```text
feedback_bias = trainable arch parameter path
safety = normal heldout bi-level training + regularization
```

### 2.2 actor_bias

`actor_bias` is a direct decision from the actor/policy based on tested edits and critic scoring.

It must be controlled by:

```text
LCB deploy score
sparse top-k mask
trust region
hard rollback
cooldown
```

If rollback disables actor intervention, `actor_bias` is disabled. `feedback_bias` stays as part of normal trained architecture, but must not directly copy the rejected actor edit.

This separation avoids two channels fighting each other.

---

## 3. Causality of feedback

A selector at step `L/S` cannot use information from later steps in the same forward pass unless the model uses a two-pass draft/refine scheme.

Fields like:

```text
who reads this slot later
next-step read/write histograms
downstream consumer
```

are not available causally during the first pass.

### MVP rule: lag-by-one feedback

For MVP, feedback is lagged:

```text
forward pass t:
  assemble program
  record trace, downstream usage, editor actions, head reads
  evaluate/probe if scheduled
  update memory/critic

forward pass t+1 or next epoch:
  use summarized feedback from previous passes as conditioning features
```

This is cheap and does not create a future-to-past causal loop.

### Later option: draft-refine

Optional expensive mode later:

```text
pass 1: draft program, collect downstream context
pass 2: refine same batch using downstream feedback from pass 1
```

This costs roughly two forwards and should not be the first implementation.

---

## 4. Assembly workshops and feedback bus

The assembly workshops are the places where the program is built:

1. primitive workshop;
2. operator-size/depth workshop;
3. primitive-transition workshop;
4. read workshop;
5. write workshop;
6. slot-transition/composition workshop;
7. macro workshop;
8. step/layer recipe workshop;
9. global budget/complexity workshop.

Each workshop eventually receives a projected feedback vector:

```text
primitive_logits += primitive_feedback_bias
operator_variant_logits += operator_feedback_bias
primitive_transition_logits += transition_feedback_bias
read_logits += read_feedback_bias
write_logits += write_feedback_bias
macro_selector_logits += macro_feedback_bias
step_alive_logits += step_feedback_bias
```

But this must be implemented only after memory/probe/critic exist.

### Feedback token content

For a location/component, use lagged features:

```text
location: L/S/B/K/A/P/M
component type
current/previous soft weight
entropy of group
operator-v2 family weights
macro id/content embedding
last editor action at this location
last actor action at this location
train gain / heldout gain from tested probes
noise std / z-score
accepted/rejected/uncertain/stale flag
critic mean gain / uncertainty / risk
retrieved memory accepted/rejected ratio
previous-pass downstream consumer summary
previous-pass class/head pressure
training progress / epoch fraction
```

Do not feed same-pass future information into current-step logits.

---

## 5. Probe policy vs deploy policy

Separate exploration from live deployment.

### 5.1 Probe uses UCB

For deciding what to test:

```text
score_probe(edit) = mean_gain + beta * uncertainty - risk_penalty - complexity_penalty
```

Uncertainty is a reason to explore.

### 5.2 Deploy uses LCB

For deciding what to apply to the live training run:

```text
score_deploy(edit) = mean_gain - beta * uncertainty - risk_penalty - complexity_penalty
```

Uncertainty is a reason to be conservative.

Never deploy UCB-selected edits directly.

---

## 6. Sparse actor, not dense second optimizer

The actor must not output dense tiny biases across the entire program.

Memory/probe labels are sparse, mostly one-factor or two-factor edits. Actor output should match that structure.

Required controls:

```text
L1 penalty on actor bias
top-k mask per action family
max edits per window
max total bias norm
sparse accepted-edit distillation
penalty for many tiny nonzero changes
```

Actor output should look like:

```text
few strong tested edits
not dense noise everywhere
```

---

## 7. Two-tier testing and rollback

### Tier 1: cheap screen

```text
save no state
apply temporary edit bias
forward-only on heldout micro-batches
measure loss delta
revert bias
```

### Tier 2: expensive verify

```text
save full training state
apply edit
run N small train/adapt steps or a short heldout window
measure train + heldout
restore or keep
```

Tier-2 must snapshot and restore:

```text
model.state_dict()
optimizer state
scaler state if AMP is used
scheduler state if used
random seeds / RNG state if needed
BatchNorm/running stats if any exist
```

Restoring only model weights is not enough because Adam momentum/variance buffers would be polluted by test steps.

---

## 8. Noise floor and significance

Noise is not constant across training.

Recompute noise periodically:

```text
every N windows:
  evaluate unchanged model on fresh heldout micro-batches
  noise_std = std(losses)
  loss_scale = mean(losses)
```

Acceptance rule:

```text
threshold = max(k * noise_std, relative_min_gain * loss_scale)
accepted if gain > threshold
rejected if gain < -threshold
otherwise uncertain
```

Recommended start:

```text
k = 2.0
relative_min_gain = 0.002 to 0.005
```

Store:

```text
noise_std
loss_scale
z_score = gain / noise_std
num_batches
```

---

## 9. Critic cold start

Do not start with a high-capacity MLP ensemble when there are only hundreds of records.

Start with:

```text
ridge / Bayesian linear regression over fixed embeddings
```

Then switch gradually:

```text
if n_records < 3000:
  use ridge/Bayesian linear critic
else:
  blend into small MLP ensemble
```

Use smooth blending:

```text
critic_weight = min(n_records / N0, 1)
score = (1 - critic_weight) * bootstrap_score + critic_weight * critic_score
```

No hard cutover.

---

## 10. Closed-loop critic re-verify

Actor can exploit critic errors.

Every scheduled window:

```text
take top edits actually deployed by actor
run Tier-2 verify
compare critic predicted_gain vs real_gain
store prediction error
retrain critic with priority on actor-visited regions
```

The critic must be accurate where the actor wants to act, not only on random old memory.

---

## 11. Memory bank

Memory stores only tested records.

Record fields:

```text
context_embedding
edit_embedding
edit type / location / target / scope
train_gain
heldout_gain
noise_std
loss_scale
z_score
accepted/rejected/uncertain/stale
critic_prediction_at_time
prediction_error_after_reverify
complexity_delta
epoch
training_progress
run_id
model_version
program_version
source_task
recency_weight
reproducibility_weight
full_test_state_hash optional
```

### 11.1 Non-stationarity

Old records can become stale.

Retrieval weight:

```text
memory_weight = similarity * recency_weight * reproducibility_weight
```

Retest old accepted edits periodically.

### 11.2 Compound edits

Early memory should prefer one-factor edits.

For compound edits:

```text
store sub-edits
run A only / B only / A+B when possible
```

---

## 12. Content-based macro/operator embeddings

Growth adds new macros and operators. Integer lookup embeddings are fragile.

Do not encode macro identity only as `macro_id` table lookup.

Instead encode macro/operator content:

```text
macro_content = read prototype + primitive prototype + transition prototype + composition prototype + write prototype
macro_embedding = MacroContentEncoder(macro_content)
```

For operator variants:

```text
operator_embedding = OperatorContentEncoder(type, rank/depth/radius/cost, matrix-operator family stats)
```

This lets newly promoted macros/operators get embeddings immediately without expanding learned ID tables.

---

## 13. Cross-task fixed-size embeddings

For transfer across SMC, matrix-water-blockless, TTS/vocoder, etc., use typed-token pooling:

```text
ProgramContextEncoder:
  typed object tokens -> DeepSets / factorized attention -> context_embedding

EditEncoder:
  edit tokens -> DeepSets / MLP pooling -> edit_embedding
```

Token fields:

```text
object type
role / primitive category
normalized location
soft weight
entropy / usage
grad/counterfactual pressure
accepted/rejected history
operator family
cell type: state/memory/global
```

Output sizes:

```text
context_embedding: 128 or 256
edit_embedding: 64 or 128
```

---

## 14. Growth / promotion loop

Without growth, actor only tunes a fixed search space.

Add two promotion tiers.

### 14.1 Local promotion

For one run/task:

Promote a pattern to local macro/operator if:

```text
heldout gain passes noise gate
reused across multiple L/S/B/K locations in the same run
not duplicate of existing local macro
complexity budget acceptable
retest remains positive
```

This is realistic on one P40.

### 14.2 Global promotion

Promote to global bank if:

```text
useful across multiple runs/tasks
embedding distance from existing global macros > duplicate threshold
mean heldout gain positive
risk/variance acceptable
not stale after retests
```

### 14.3 Promotion record

Every promoted element stores:

```text
source edits
support count
mean heldout gain
variance/risk
complexity cost
contexts where useful
contexts where harmful
local/global status
content embedding
```

Promoted macros become first-class MacroStepBank entries. Promoted operator variants become selectable OperatorBankV2 entries.

---

## 15. Soft-vs-discrete diagnostics

Periodically evaluate:

```text
soft program val loss
argmax program val loss
top-k sparse program val loss
discretization_gap = discrete_val_loss - soft_val_loss
```

If gap is large:

```text
anneal temperature
add path dropout
strengthen entropy band
prefer tested sparse edits
reduce trust in dense actor output
```

---

## 16. Compute budget

The central mind must not dominate training cost.

Start limits:

```text
screen/probe budget <= 10-20% of training time
critic training <= small scheduled window
Tier-2 verify only top candidates
```

Always report:

```text
accuracy gain / extra wall-clock cost
heldout gain / probe budget
```

---

## 17. Correct implementation order for this repo

### Stage 1: reliable observation

- fix macro logging;
- log OperatorBankV2 family choices;
- log step_alive;
- log class-slot reads;
- log memory/global usage;
- log editor actions/results;
- no feedback injection yet.

### Stage 2: anti-collapse + bi-level

- primitive/macro load balance;
- entropy band;
- step/memory budget;
- architecture/program parameters trained on heldout path.

### Stage 3: counterfactual screen + memory + cold-start critic MVP

This is the first real active-mind target.

- generate sparse candidate edits;
- Tier-1 forward-only heldout screen;
- dynamic noise floor;
- JSONL memory bank;
- ridge/Bayesian linear critic;
- choose one sparse LCB-safe edit per window;
- apply with scale 0.03;
- Tier-2 verify if needed;
- hard rollback if harmful;
- store accepted/rejected/uncertain.

No neural feedback bus yet.

### Stage 4: feedback bus MVP

Only after Stage 3 produces tested memory.

- build lag-by-one feedback features;
- feed memory/critic summaries as conditioning into workshops;
- optimize feedback_bias through bi-level heldout path;
- keep actor_bias separate and safety-controlled.

### Stage 5: local growth/promotion

- promote repeated local macros/operators inside one run;
- content-based embeddings;
- local bank only.

### Stage 6: critic actor refinement

- UCB for probe;
- LCB for deploy;
- closed-loop re-verify;
- sparse actor rules.

### Stage 7: neural typed-token actor

Only after enough tested records.

- typed object heads;
- sparse top-k output;
- accepted/rejected imitation;
- critic distillation;
- heldout path training;
- trust-region + rollback.

### Stage 8: global growth/promotion

- promote patterns useful across runs/tasks;
- update global MacroStepBank / OperatorBankV2.

---

## 18. Minimal first code target

Do not implement `ProgramFeedbackEncoder` first.

First implement:

```text
CounterfactualScreen
ProgramMemoryBank
ColdStartCritic
SparseLCBEditApplier
FullTrainingStateRollback
```

Then one safe loop:

```text
once per epoch/window:
  observe current program summaries
  generate sparse candidate edits
  screen on heldout micro-batches
  update memory
  train/update ridge critic
  pick deploy edit by LCB
  apply one sparse bias with scale 0.03
  verify / rollback / store result
```

This is the smallest real active mind in the current codebase.

---

## 19. Short summary

Do not build the feedback bus before memory exists.  
Do not let same-pass downstream information flow backward in the forward pass.  
Do not mix feedback_bias and actor_bias safety semantics.  
Do not deploy UCB edits; deploy LCB edits only.  
Do not rollback only weights; rollback full training state.  
Do not use ID-only macro embeddings; use content embeddings.  
Do not wait for cross-task support before local promotion.  
Start with counterfactual screen + memory + cold-start critic + one sparse safe edit per window.
