# Editor Actor-Critic Policy Plan

This is the new central-mind plan after the important correction:

> The mind is not one magic attention head above the model.  
> The mind is a policy loop around the existing editor.

The existing `FlowEditAttention` / editor is already the natural actuator: it edits the matrix-program assembly. The missing piece is a feedback/policy system that teaches the editor where, why, and how strongly to act.

So the real system should be:

```text
Assembler components propose a program
Editor applies soft edits
Heldout/critic/counterfactual loop evaluates edits
Memory stores tested experience
Critic learns to predict edit gain
Actor/policy chooses future edit biases
Components receive editor feedback and memory context
```

Everything remains matrix-based and differentiable in the online path.

---

## 1. What is the central mind?

The central mind is a 5-part loop:

1. **Editor / Actuator**
   - existing flow editor plus future central edit bias.
   - writes soft biases into read/primitive/transition/write/macro logits.

2. **Observer / Trace**
   - records what the program actually assembled:
     - read flows;
     - primitive mixes;
     - primitive transitions;
     - slot transitions;
     - writes;
     - macro selections;
     - step/layer roles;
     - head/class reads;
     - operator-v2 family choices;
     - step alive gates.

3. **Probe / Counterfactual generator**
   - tests edits that normal gradient may miss:
     - revive suppressed primitive;
     - replace macro;
     - shift read/write;
     - reduce dominant self-loop;
     - increase low-rank/product-gate chain;
     - activate/deactivate step.

4. **Critic / Surrogate**
   - predicts heldout gain and uncertainty:

   ```text
   critic(context_embedding, edit_embedding) -> mean_gain, uncertainty, risk
   ```

5. **Memory / Experience Bank**
   - stores only edits that were actually tested:
     - accepted;
     - rejected;
     - uncertain;
     - stale/retested.

The actor/policy chooses what the editor should try next, using critic + memory + heldout signal.

---

## 2. Why this is not just SGD

SGD only sees the current differentiable path and current train/val batch.

This mind adds information SGD does not have:

- cross-run memory;
- cross-task memory;
- counterfactual tests for suppressed paths;
- heldout-gain labels for edits;
- uncertainty/exploration;
- trust-region accept/reject;
- non-stationarity handling;
- soft-vs-discrete diagnostics;
- explicit anti-collapse controls.

Naive `grad * weight` is only a diagnostic. It must not be treated as truth.

---

## 3. Existing parts to reuse

### 3.1 Existing editor

Current editor already modifies:

```text
read logits
primitive logits
slot transition logits
primitive transition logits
composition logits
write logits
```

This becomes the main actuator.

### 3.2 Existing MacroStepBank

Macro bank is not a primitive. It is a reusable step recipe:

```text
macro = read + primitive mix + transitions + composition + write
```

The actor should be able to:

- activate macro;
- suppress background macro;
- replace macro;
- add small macro delta;
- promote tested macro variations later.

### 3.3 Existing OperatorBankV2 idea

Operator families can be selected softly inside the six external primitive slots:

- low-rank size;
- butterfly depth;
- Haar/wavelet-like variant;
- local smooth/diff;
- diagonal gate;
- phase variants.

The actor/critic should see these internal operator choices too.

---

## 4. The right information flow

Current simplified flow:

```text
context -> assembler -> program -> loss -> gradient
```

New flow:

```text
context
  -> assembler components
  -> current program
  -> editor proposes soft edits
  -> task/head loss
  -> trace + counterfactual probes
  -> heldout gain / risk / noise
  -> memory record
  -> critic update
  -> actor policy
  -> feedback bias back into assembler components
```

The key addition is the feedback path:

```text
editor_result -> feedback_encoder -> primitive/read/write/macro selectors
```

The selectors should not only see the current data. They should see what the editor tried before and whether it worked.

---

## 5. Where the actor acts

The actor does not call Python branches. It writes soft matrix biases.

### 5.1 Primitive mix

```text
primitive_logits[L,S,B,K,P] += actor_primitive_bias[L,S,B,K,P]
```

Examples:

- increase `product_gate` at compare step;
- decrease dead `ctx_matrix` self-loop;
- increase `wavelet/haar` variant in extract step via operator-v2 delta.

### 5.2 Primitive transitions

```text
primitive_transition_logits[L,S,P,P] += actor_transition_bias[L,S,P,P]
```

Examples:

- increase `low_rank -> product_gate`;
- increase `product_gate -> phase_matrix`;
- reduce useless `phase_matrix -> phase_matrix` if it is only a lazy self-loop.

### 5.3 Read/write

```text
read_logits[L,S,B,K,A] += actor_read_bias[L,S,B,K,A]
write_logits[L,S,B,A] += actor_write_bias[L,S,B,A]
```

Examples:

- shift read to memory.M2;
- reduce global read if it harms local extraction;
- write useful compare result to memory.

### 5.4 Slot transitions/composition

```text
slot_transition_logits[L,S,B,K,K] += actor_slot_bias[L,S,B,K,K]
composition_logits[L,S,B,K] += actor_composition_bias[L,S,B,K]
```

Examples:

- split collapsed slots;
- make slot K0 carry local feature and K1 carry memory feature;
- stop all slots averaging the same thing.

### 5.5 Macro selection

```text
macro_selector_logits[L,S,B,M] += actor_macro_bias[L,S,B,M]
```

Examples:

- suppress macro_0 background if it dominates;
- activate macro_7 only in L1.S1.B2;
- replace macro_3 with macro_12 in suppress layer.

### 5.6 Step/layer/global recipe

```text
step_alive_logits[L,S] += actor_step_bias[L,S]
phase_logits[L,S,phase] += actor_phase_bias[L,S,phase]
```

Examples:

- activate extra compare step;
- weaken redundant step;
- make L0 more extract/local;
- make L3 more aggregate/global.

---

## 6. What context every selector must see

The selectors need rich context, not just ID embeddings.

### 6.1 Local object context

For every object:

```text
object_type
location: L/S/B/K/A/P/M
phase role
current soft weight
entropy of its softmax group
usage count
step_alive
operator-v2 family weights
macro id / macro weight
```

### 6.2 Downstream context

A component matters because something later uses it. Add:

```text
who reads this slot later
which class/head reads it
which memory/global cell receives it
next-step read histogram
next-step write histogram
next-step primitive histogram
class-slot attention
per-class loss/confusion
```

### 6.3 Editor feedback context

For this location/component:

```text
last edit applied
last edit scale
train gain
heldout gain
noise std
z-score
accepted/rejected/uncertain
staleness/recency
```

### 6.4 Critic/memory context

```text
retrieved similar edits
critic predicted gain
critic uncertainty
critic risk
memory accepted/rejected ratio
retest status
```

### 6.5 Task/input/head context

```text
task embedding
input/evidence summary
head query summary
class confusion vector
train/val loss state
training progress / epoch fraction
```

---

## 7. Typed object tokens and small heads

Do not build huge raw-token attention. The mind attends over program objects.

Suggested small heads:

### Location/role heads

1. phase-role head
2. layer-order head
3. block-specialization head
4. slot-specialization head

### Component heads

5. primitive-usefulness head
6. transition-chain head
7. read-cell head
8. write-cell head
9. composition head
10. operator-size/depth head

### Macro heads

11. macro-selection head
12. macro-conflict/collapse head
13. macro-variation head
14. macro-memory-retrieval head

### Task/head heads

15. class-slot head
16. class-confusion head
17. downstream-consumer head
18. head-pressure head

### Stability heads

19. entropy/collapse head
20. complexity-budget head
21. memory/global-usage head
22. soft-vs-discrete-gap head
23. heldout-risk head
24. non-stationarity/recency head

MVP: 16-32 small heads. Later: 64-128 micro-heads. Not one 500-head LLM-like module.

---

## 8. Critic architecture

Use a small ensemble MLP.

### Input

```text
x = concat(
  context_embedding,      # fixed-size program/task/location summary
  edit_embedding,         # proposed action summary
  metric_embedding,       # loss/acc/entropy/collapse/skill
  recency_embedding,      # epoch/progress/source age
  complexity_embedding    # edit cost and scope
)
```

### Output

```text
mean_gain       # predicted heldout gain
log_variance    # uncertainty/noise
risk_score      # probability of hurting heldout
```

### Uncertainty

Use ensemble variance first:

```text
critic_ensemble = 3-5 MLPs
uncertainty = variance(predicted_gain)
```

MC-dropout can be added later.

---

## 9. Actor / policy

The actor proposes edit candidates or direct soft biases.

MVP actor should be simple:

```text
candidate generator + critic UCB + trust-region application
```

Later neural actor:

```text
typed object tokens + memory retrieval tokens -> bounded central bias tensors
```

The actor should be trained by:

1. heldout architecture gradient;
2. imitation of accepted edits;
3. avoidance of rejected edits;
4. critic distillation:

```text
maximize critic_predicted_gain - risk - complexity
```

---

## 10. Candidate generation

Do not test everything.

Candidate sources:

1. gradient candidates;
2. suppressed revival candidates;
3. macro alternatives;
4. memory-retrieved edits;
5. anti-collapse edits;
6. soft/discrete-gap edits;
7. operator-size/depth edits;
8. step/layer alive edits.

Candidate format:

```json
{
  "edit_type": "increase_primitive | shift_read | replace_macro | increase_transition | suppress_component | activate_step | change_operator_size",
  "where": "L1.S1.B2.K0",
  "target": "product_gate / memory.M2 / macro_7 / low_rank_r16",
  "delta_logit": 0.25,
  "scope": "component | step | layer | global",
  "complexity_delta": 0.01
}
```

---

## 11. Acquisition policy

Do not pick top-k only by gradient magnitude.

Use UCB / Expected Improvement:

```text
score(edit) = predicted_gain + beta * uncertainty - lambda_risk * risk - lambda_complexity * cost
```

Early bootstrap before critic:

```text
score =
  0.30 * gradient_score
+ 0.25 * suppressed_revival_score
+ 0.25 * memory_similarity_score
+ 0.15 * anti_collapse_score
+ 0.05 * random_exploration
```

Then replace with critic UCB after enough tested records.

---

## 12. Two-tier testing

### Tier 1: cheap screen

Forward-only heldout micro-batch:

```text
apply edit bias temporarily
forward only
measure val loss delta
revert
```

### Tier 2: expensive verify

Only for candidates that pass screen:

```text
apply edit
train N small steps or run short adaptation
measure heldout
revert or keep
```

Memory `accepted=true` requires Tier 2 or repeated Tier 1 with high significance.

---

## 13. Noise floor and significance gate

Before accepting edits, estimate noise:

```text
baseline_losses = unchanged model on several heldout micro-batches
noise_std = std(baseline_losses)
```

Acceptance:

```text
accepted if gain > max(min_gain, k * noise_std)
```

Recommended:

```text
k = 2.0
min_gain = small but nonzero
```

Store:

```text
noise_std
z_score
num_batches
```

---

## 14. Non-stationarity

Memory records must include:

```text
epoch
training_progress
model_version
program_version
loss/acc range
run_id
source_task
```

Retrieval weight:

```text
weight = similarity * recency_weight * reproducibility_weight
```

Retest old accepted edits periodically. If they no longer work, decay confidence or mark stale.

---

## 15. Compound edit credit assignment

Avoid large compound edits early.

Rules:

1. First memory should mostly use one-factor edits.
2. Compound edits store sub-edit list.
3. Later use small factorial tests:
   - A only;
   - B only;
   - A+B.

This prevents the critic from learning false correlations.

---

## 16. Cross-task fixed-size embeddings

For transfer across SMC, matrix-water-blockless, TTS/vocoder, etc., use a fixed-size encoder.

Do not rely on raw histograms with task-specific shapes.

Use typed-token pooling:

```text
object tokens -> DeepSets / factorized attention -> fixed context vector
edit tokens -> DeepSets / MLP pooling -> fixed edit vector
```

Output sizes:

```text
context_embedding: 128 or 256
edit_embedding: 64 or 128
```

---

## 17. Trust-region control

Actor bias must be bounded adaptively.

```text
scale starts at 0.03
if heldout improves: scale *= 1.05
if heldout worsens: scale *= 0.5 and revert recent actor bias
max scale: 0.20
```

This allows strong influence only after proven safe.

---

## 18. How it connects to current code

### Add traces from `assembler_core.py`

Expose:

- flow logits or soft flows;
- macro selector weights;
- operator-v2 family weights;
- step alive;
- phase weights;
- entropy summaries.

### Add feedback injection

Extend step forward:

```python
AssemblerStep.forward(cells, feedback_bias=None)
```

Bias contains:

```text
read_bias
primitive_bias
slot_transition_bias
primitive_transition_bias
composition_bias
write_bias
macro_bias
phase_bias
step_alive_bias
```

### Add training hooks in `transfer_audio_assembler.py`

- collect traces every N epochs;
- run counterfactual screen;
- update memory bank;
- train critic;
- optionally apply actor bias with trust region.

---

## 19. Implementation order

### Stage 1: reliable observation

- fix macro logging;
- log operator-v2 family choices;
- log step alive;
- log class-slot reads;
- log memory/global usage.

### Stage 2: anti-collapse + bi-level

- primitive/macro load balance;
- entropy band;
- step/memory budget;
- architecture parameters trained on heldout path.

### Stage 3: counterfactual screen

- generate candidates;
- forward-only screen;
- noise floor;
- save tested records.

### Stage 4: memory bank

- JSONL memory;
- fixed-size embeddings;
- accepted/rejected/uncertain/stale.

### Stage 5: critic

- MLP ensemble;
- predict gain/uncertainty/risk;
- UCB acquisition.

### Stage 6: rule/critic actor

- choose best bounded edit from candidates;
- apply via trust region;
- verify and store result.

### Stage 7: neural actor

- typed-token actor;
- outputs central edit bias;
- trained by heldout gradient + critic distillation + accepted/rejected imitation.

---

## 20. Short summary

The editor is the actuator.  
The critic is the first real intelligence.  
The memory is experience data.  
The actor is the policy.  
The counterfactual probe creates labels.  
The trust region keeps it safe.  
The assembler components must receive feedback from editor results, not only current data.

This is the correct path from differentiable matrix-program assembly to an active self-improving architecture builder.
