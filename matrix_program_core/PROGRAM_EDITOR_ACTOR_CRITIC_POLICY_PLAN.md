# Editor Actor-Critic Policy Plan V2

This plan replaces the idea of a single magic central attention head with a real policy loop around the existing matrix-program editor.

The core correction:

```text
The editor is the actuator.
The critic is the first predictive intelligence.
The memory is tested experience data.
The probe generates labels by real counterfactual tests.
The actor is a sparse policy that sends bounded feedback back into the assembly workshops.
The growth loop promotes repeatedly useful repairs into new reusable macros/operators.
```

Everything that is applied online remains soft, matrix-based, and differentiable.  
Everything that is remembered as truth must be tested on heldout/counterfactual probes.

---

## 1. Why this exists

The current assembler is already differentiable:

```text
loss -> gradient -> flow logits / editor / head / adapters change
```

So a central mind that only reads `grad * weight` and says “increase what gradient already increases” is not intelligence. It is just a logger for SGD.

The central mind must add signals a single backprop step does not have:

1. cross-run and cross-task memory;
2. heldout/generalization-aware validation;
3. counterfactual tests for suppressed components;
4. uncertainty-aware exploration;
5. safe deployment with LCB and rollback;
6. open-ended promotion of new macro/operator recipes.

---

## 2. Main architecture

```text
Assembler components propose a matrix program
        ↓
Editor applies soft edits to assembly logits
        ↓
Task/head computes train + heldout signals
        ↓
Observer records what was assembled and what editor changed
        ↓
Probe tests counterfactual candidate edits
        ↓
Memory stores only tested accepted/rejected/uncertain edits
        ↓
Critic learns (context, edit) -> heldout_gain + uncertainty + risk
        ↓
Actor chooses sparse bounded edit bias
        ↓
Feedback bus sends results back into primitive/operator/macro/read/write selectors
        ↓
Growth loop promotes repeatedly useful patterns into new macros/operators
```

This is not one module. It is a closed loop.

---

## 3. Existing parts to reuse

### 3.1 Editor as actuator

Current `FlowEditAttention` already edits:

```text
read logits
primitive logits
slot transition logits
primitive transition logits
composition logits
write logits
```

It should become the main actuator, not be replaced.

Future actor outputs must enter as extra bounded biases:

```text
flow_logits = base + context + current_editor + macro_bias + feedback_bias + actor_bias
```

### 3.2 MacroStepBank

Macro is a reusable step recipe, not a primitive:

```text
macro = read + primitive mix + transitions + composition + write
```

The actor can:

- activate macro;
- suppress background macro;
- replace macro;
- add small macro delta;
- promote tested macro variation later.

### 3.3 OperatorBankV2

The six external primitive families stay compatible, but internally they can contain selectable variants:

- low-rank rank/size;
- butterfly depth;
- Haar/wavelet-like transform;
- local smooth/diff;
- diagonal gate;
- phase variants;
- future Fourier/DCT/Toeplitz variants.

The actor/critic must see these internal choices too.

---

## 4. Feedback bus into assembly workshops

The most important addition: the editor’s result must return to the components that assemble the program.

Assemblers/selectors should not only see raw context. They should also see:

```text
what editor tried here
whether it helped or hurt
what critic predicts now
what memory says for similar contexts
what downstream slots/classes consumed this component
whether this component is stale/collapsed/overused
```

### 4.1 Assembly workshops

Treat each selection site as a workshop:

1. primitive workshop;
2. operator-size/depth workshop;
3. primitive-transition workshop;
4. read workshop;
5. write workshop;
6. slot-transition/composition workshop;
7. macro workshop;
8. step/layer recipe workshop;
9. global budget/complexity workshop.

Every workshop receives a projected feedback vector:

```text
workshop_logits += workshop_feedback_projection(feedback_token)
```

Examples:

```text
primitive_logits[L,S,B,K,P] += feedback_primitive_bias
operator_variant_logits[L,S,B,K,F,V] += feedback_operator_bias
primitive_transition_logits[L,S,P,P] += feedback_transition_bias
read_logits[L,S,B,K,A] += feedback_read_bias
write_logits[L,S,B,A] += feedback_write_bias
macro_selector_logits[L,S,B,M] += feedback_macro_bias
step_alive_logits[L,S] += feedback_step_bias
```

### 4.2 Feedback token content

For each location/component:

```text
location: L/S/B/K/A/P/M
component type
current soft weight
entropy of its group
operator-v2 family weights
macro id / macro weight
last editor action
last editor scale
train gain
heldout gain
noise std
z-score
accepted/rejected/uncertain/stale
critic mean gain
critic uncertainty
critic risk
retrieved memory accepted/rejected ratio
who reads this slot later
class/head pressure
next-step read/write/primitive histograms
training progress / epoch fraction
```

This is the real “reverse answer” from editor back to the assembly shops.

---

## 5. Typed object heads, not one magic head

Use small matrix-program object heads, not raw-sequence attention.

Suggested heads:

### Location and role

1. phase-role head;
2. layer-order head;
3. block-specialization head;
4. slot-specialization head.

### Component heads

5. primitive-usefulness head;
6. operator-size/depth head;
7. transition-chain head;
8. read-cell head;
9. write-cell head;
10. composition head.

### Macro heads

11. macro-selection head;
12. macro-conflict/collapse head;
13. macro-variation head;
14. macro-memory-retrieval head.

### Task/head heads

15. class-slot head;
16. class-confusion head;
17. downstream-consumer head;
18. head-pressure head.

### Stability/safety heads

19. entropy/collapse head;
20. complexity-budget head;
21. memory/global-usage head;
22. soft-vs-discrete-gap head;
23. heldout-risk head;
24. non-stationarity/recency head.

MVP: 16-32 micro-heads. Later: 64-128.  
Do not build a huge LLM-like head over raw audio tokens.

---

## 6. Critic

The critic predicts the measured effect of an edit.

```text
critic(context_embedding, edit_embedding, metrics, recency, complexity)
    -> mean_heldout_gain
    -> uncertainty
    -> risk_score
```

### 6.1 Cold-start critic

Do not start with a large MLP ensemble when memory has only hundreds of records.

Cold-start options:

1. ridge / Bayesian linear regression on fixed embeddings;
2. small calibrated linear model with posterior variance;
3. only later: MLP ensemble when records > 3k-5k.

Reason: early high-dimensional MLP critic can memorize noise and be badly calibrated out of distribution.

### 6.2 Later critic

When enough records exist:

```text
critic_ensemble = 3-5 small MLPs
uncertainty = variance across ensemble
```

The critic must be trained on tested edits only.

---

## 7. Probe policy vs deploy policy

This must be explicit.

### 7.1 Probe / exploration uses UCB

For deciding what to test:

```text
score_probe(edit) = mean_gain + beta * uncertainty - risk_penalty - complexity_penalty
```

Uncertainty is good here because it means “worth exploring”.

### 7.2 Live deployment uses LCB

For deciding what to actually apply during the real run:

```text
score_deploy(edit) = mean_gain - beta * uncertainty - risk_penalty - complexity_penalty
```

Uncertainty is bad here because live deployment must be conservative.

Never use UCB directly for online actor bias.

---

## 8. Candidate generation

Candidate sources:

1. gradient pressure candidates;
2. suppressed revival candidates;
3. macro alternatives;
4. memory-retrieved edits;
5. anti-collapse edits;
6. soft/discrete-gap edits;
7. operator-size/depth edits;
8. step/layer alive edits;
9. growth/promotion candidates.

Candidate format:

```json
{
  "edit_type": "increase_primitive | shift_read | replace_macro | increase_transition | suppress_component | activate_step | change_operator_size | promote_macro",
  "where": "L1.S1.B2.K0",
  "target": "product_gate / memory.M2 / macro_7 / low_rank_r16",
  "delta_logit": 0.25,
  "scope": "component | step | layer | global",
  "complexity_delta": 0.01
}
```

Before critic exists, bootstrap candidate score:

```text
score_bootstrap =
  0.30 * gradient_score
+ 0.25 * suppressed_revival_score
+ 0.25 * memory_similarity_score
+ 0.15 * anti_collapse_score
+ 0.05 * random_exploration
```

Switch to critic gradually:

```text
critic_weight = min(n_tested_records / N0, 1.0)
score = (1 - critic_weight) * score_bootstrap + critic_weight * score_probe
```

No hard cutover.

---

## 9. Two-tier testing

### Tier 1: cheap screen

Forward-only heldout micro-batch:

```text
apply edit bias temporarily
forward only
measure val loss delta
revert
```

### Tier 2: expensive verify

For candidates that pass screen:

```text
apply edit
train/adapt N small steps or run short heldout window
measure heldout gain
revert or keep
```

Memory `accepted=true` requires Tier 2 or repeated Tier 1 with strong significance.

---

## 10. Noise floor and significance

Estimate baseline noise:

```text
baseline_losses = unchanged model on several heldout micro-batches
noise_std = std(baseline_losses)
```

Acceptance:

```text
accepted if gain > max(min_gain, k * noise_std)
rejected if gain < -max(min_gain, k * noise_std)
otherwise uncertain
```

Recommended:

```text
k = 2.0
min_gain = small nonzero threshold
```

Store:

```text
noise_std
z_score
num_batches
gain_metric
```

---

## 11. Sparse actor, not dense second optimizer

Memory validates mostly sparse one-factor or two-factor edits.  
Therefore the actor must not output unrestricted dense tensors everywhere.

Required constraints:

1. L1 sparsity on actor bias;
2. top-k mask per action family;
3. max edit count per epoch/window;
4. distillation to sparse accepted edits;
5. penalty for many small nonzero changes.

Online actor bias should look like:

```text
few strong tested edits
not tiny noise everywhere
```

Otherwise it becomes another dense optimizer and repeats the mirror-SGD problem.

---

## 12. Hard rollback + soft trust region

Soft trust region:

```text
scale starts at 0.03
if heldout improves: scale *= 1.05
if heldout worsens: scale *= 0.5
max scale: 0.20
```

Hard rollback is also required:

```text
save checkpoint before actor intervention
if heldout window worsens by > k * noise_std:
    restore checkpoint
    disable actor bias for cooldown window
    store edit as rejected/harmful
```

Trust-region scale only protects future steps. Rollback repairs damage already done.

---

## 13. Closed-loop critic re-verify

Actor can exploit critic errors. This is standard model-based RL failure.

Protection:

```text
every N steps/epochs:
    take top edits actor actually deployed
    run Tier-2 verify
    compare critic predicted_gain vs real_gain
    store prediction error
    retrain critic with priority on actor-visited regions
```

The critic must be most accurate where the actor wants to act.

---

## 14. Memory bank

Memory stores only tested records.

Record fields:

```text
context_embedding
edit_embedding
edit type / location / target
train_gain
heldout_gain
noise_std
z_score
accepted/rejected/uncertain
critic_prediction_at_time
prediction_error_after_reverify
complexity_delta
epoch
training_progress
run_id
model_version
program_version
source_task
staleness / recency weight
```

### 14.1 Non-stationarity

Old records can become stale.

Retrieval weight:

```text
memory_weight = similarity * recency_weight * reproducibility_weight
```

Retest old accepted edits periodically. If they stop working, decay confidence or mark stale.

### 14.2 Compound edits

Early memory should use one-factor edits whenever possible.

For compound edits:

```text
store sub-edits
run A only / B only / A+B factorial tests when possible
```

This prevents false credit assignment.

---

## 15. Cross-task fixed-size embeddings

For transfer across SMC, matrix-water-blockless, TTS/vocoder, etc., use fixed-size typed-token encoders.

Do not rely on raw shape-specific histograms only.

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

## 16. Open-ended growth / promotion loop

Without growth, actor only tunes a fixed search space. That is useful, but not open-ended architecture synthesis.

Add Stage 8: tested-and-promoted macro/operator growth.

### 16.1 Macro promotion criteria

Promote a tested pattern into `MacroStepBank` if:

```text
heldout gain passes noise gate
reused across >= N examples/runs/tasks
embedding distance from existing macros > duplicate threshold
complexity budget acceptable
not stale after retest
```

Promoted macro becomes first-class:

```text
new macro id
new macro read/primitive/transition/composition/write prototype
available to actor like any other macro
```

### 16.2 Operator variant promotion

If repeated edits discover a useful operator variant:

```text
low_rank_r16 + product_gate + phase path
specific wavelet/diff/smooth mix
block_butterfly depth2 pattern
```

then promote it as a named operator variant inside OperatorBankV2.

Do not allow arbitrary Python generation first. Promote only combinations of existing safe matrix operators.

### 16.3 Growth memory

Every promoted element stores:

```text
source edits
support count
mean heldout gain
variance/risk
complexity cost
examples where useful
examples where harmful
```

---

## 17. Soft-vs-discrete diagnostics

Periodically evaluate:

```text
soft program val loss
argmax program val loss
top-k sparse program val loss
discretization_gap = discrete_val_loss - soft_val_loss
```

If gap is large:

- anneal temperature;
- add path dropout;
- strengthen entropy band;
- prefer tested sparse edits;
- do not trust dense soft actor too much.

---

## 18. Compute budget

The central mind has a cost. Track it.

Suggested limits:

```text
screen/probe budget <= 10-20% of training time
critic training budget <= small scheduled window
Tier-2 verify only for top candidates
```

Measure net benefit:

```text
accuracy gain / extra wall-clock cost
heldout gain / probe budget
```

A smarter controller that doubles runtime for +0.2% is not useful on one P40.

---

## 19. Implementation order

### Stage 1: reliable observation

- fix macro logging;
- log operator-v2 family choices;
- log step alive;
- log class-slot reads;
- log memory/global usage;
- log editor actions/results.

### Stage 2: anti-collapse + bi-level

- primitive/macro load balance;
- entropy band;
- step/memory budget;
- architecture parameters trained on heldout path.

### Stage 3: feedback bus MVP

- create `ProgramFeedbackEncoder`;
- project feedback into primitive/read/write/transition/macro/step logits;
- initially feed only tested editor results and simple memory summaries.

### Stage 4: counterfactual screen

- generate sparse candidates;
- forward-only heldout screen;
- noise floor;
- store tested records.

### Stage 5: memory bank

- JSONL memory;
- fixed-size embeddings;
- accepted/rejected/uncertain/stale.

### Stage 6: cold-start critic

- Bayesian/ridge linear critic first;
- upgrade to MLP ensemble after enough tested records.

### Stage 7: rule/critic sparse actor

- choose edit by LCB for deploy;
- UCB only for probe;
- apply sparse bounded bias;
- hard rollback + cooldown.

### Stage 8: growth/promotion

- promote repeated useful macro variations;
- promote safe operator variants;
- update MacroStepBank / OperatorBankV2.

### Stage 9: neural typed-token actor

- typed object heads;
- critic distillation;
- accepted/rejected imitation;
- heldout path training;
- sparse top-k output and trust-region deployment.

---

## 20. Minimal first code target

Do not build the neural actor first.

First useful MVP:

```text
ProgramFeedbackEncoder + CounterfactualScreen + MemoryBank + cold-start Critic
```

Then:

```text
one sparse LCB-selected edit per epoch/window
apply with scale 0.03
verify heldout
rollback if harmful
store record
```

This is the smallest real active mind.

---

## 21. Short summary

The editor changes the program.  
The feedback bus teaches the assembly workshops what editor actions worked.  
The critic predicts which future edits will help.  
UCB explores; LCB deploys safely.  
The actor must be sparse.  
Hard rollback is mandatory.  
Memory must store only tested edits.  
Growth/promotion turns repeated useful repairs into new first-class macros/operators.  
That is the path from differentiable tuning to real matrix-program architecture synthesis.
