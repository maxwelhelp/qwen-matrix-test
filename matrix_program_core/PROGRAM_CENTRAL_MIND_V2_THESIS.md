# Program Central Mind V2: not a mirror of SGD

This file corrects the first Central Mind plan.

The main risk: if Central Mind only reads same-batch `grad * weight` and recommends increasing what SGD already increases, it is not a mind. It is a description of ordinary backprop.

A useful central mind must add signals that a single training step cannot provide.

---

## Core thesis

Central Mind is useful only if it brings at least one of these sources of extra information:

1. **Cross-run / cross-task memory**
   - previous matrices, models, tasks, repairs, accepted and rejected edits.
   - SGD on the current batch cannot know this.

2. **Generalization-aware signal**
   - architecture/program decisions are judged on heldout/val batches, not only train.
   - use DARTS-style bi-level training.

3. **Counterfactual structural tests**
   - temporarily boost/zero/swap suppressed primitives/macros/read/write paths and measure heldout loss.
   - fixes saturated-softmax blind spots.

4. **Explicit anti-collapse losses**
   - primitive/macro/class-slot load balancing, entropy band, macro reuse penalty.
   - do not wait for a meta-controller to rediscover this.

5. **Soft-vs-discrete diagnostics**
   - compare soft mixture program with argmax/top-k discretized program.
   - if soft program works but discrete program fails, the relaxation is lying.

6. **Real tested memory**
   - memory records only edits that were actually applied and measured.
   - no fake `accepted=true` from train-gradient inference alone.

---

## Key failure modes

### 1. Same-batch gradient mirror

`abs(grad(logit) * weight)` is useful as a diagnostic, but it is not new information. It should not be the basis for trusted accepted/rejected memory.

### 2. Saturated softmax blind spot

If a component weight is near zero, its logit gradient can also be near zero. Then attribution says “not important”, even if the component would help if revived.

Required fix:

```text
counterfactual revival:
  boost suppressed component
  evaluate heldout micro-batch
  revert
  record delta
```

### 3. Fake memory

Memory must distinguish:

```text
online continuous bias:
  part of normal gradient training

offline proposal-test-accept:
  apply candidate edit
  measure train and heldout
  accept/reject
  store memory
```

### 4. Differentiable NAS soft/discrete gap

Soft mixtures can look good because they are ensembles. The discrete argmax program can be worse. Track this explicitly.

### 5. Collapse needs treatment, not just diagnosis

Add auxiliary losses now:

- primitive load balance;
- macro load balance;
- class-slot balance;
- memory/global usage floor;
- entropy band;
- macro reuse penalty.

### 6. Memory needs embedding retrieval

Do not use only JSON strings like `task_family=audio`. Store vector keys:

```text
primitive_hist + transition_hist + read_hist + write_hist + phase_hist + macro_hist + task/head summary + error signature
```

Retrieve by cosine top-k.

---

## Revised implementation order

### Phase A — fix macro logging and report truth

Current macro run showed macro variants have more trainable params and different accuracy, but CSV reported:

```text
macro_entropy=0
macro_top1_usage=0
macro_gain=0
```

So metrics are not wired correctly.

Fix first:

- macro selector entropy;
- macro top1 usage;
- macro gain;
- macro contribution norm;
- macro selector grad norm.

### Phase B — anti-collapse auxiliary losses

Add cheap losses before any neural central controller:

- `primitive_load_balance_loss`
- `macro_load_balance_loss`
- `read_write_cell_balance_loss`
- `class_slot_balance_loss`
- `entropy_band_loss`

These should be optional flags with small coefficients.

### Phase C — bi-level program training

Separate parameters:

**W / ordinary weights**

- input adapter;
- primitive internals;
- head;
- normal task weights.

**A / program-architecture weights**

- flow logits/deltas;
- flow editor deltas;
- macro selector/gain;
- phase mix deltas;
- future central edit bias.

Training loop:

```python
for train_batch, arch_batch in paired_loaders:
    # ordinary weights learn train fit
    freeze(A); unfreeze(W)
    loss_train = task_loss(train_batch)
    opt_w.zero_grad(); loss_train.backward(); opt_w.step()

    # program architecture learns heldout/generalization
    freeze(W); unfreeze(A)
    loss_arch = task_loss(arch_batch) + anti_collapse + complexity
    opt_a.zero_grad(); loss_arch.backward(); opt_a.step()
```

This is the highest value/cost change.

### Phase D — soft vs argmax diagnostics

Add periodic evaluation:

- normal soft program val loss;
- argmax primitive/macro program val loss;
- top-k sparse program val loss;
- `discretization_gap`.

If gap is large:

- anneal temperature;
- add path dropout;
- use entropy band;
- test Gumbel-softmax/sparsemax later.

### Phase E — counterfactual attributor

For top-k candidates:

- boost suppressed primitive;
- zero dominant primitive;
- swap macro;
- shift read/write;
- measure heldout delta;
- revert.

Only this can discover components killed by softmax saturation.

### Phase F — planted-solution synthetic tests

Before trusting analyzer on real tasks, create synthetic programs where true answer is known:

- low_rank required;
- product_gate required;
- memory read required;
- global read harmful;
- macro chain required;
- wrong-step repair required;
- suppressed component must be revived.

Analyzer must recover planted component.

### Phase G — ProgramMemoryBank V2

Store only tested edits:

```json
{
  "key_embedding": "primitive/read/write/transition/macro/task/error vector",
  "edit": "increase product_gate + shift read to memory.M2",
  "train_gain": 0.03,
  "heldout_gain": 0.02,
  "accepted": true,
  "complexity_delta": 0.01,
  "source_run": "..."
}
```

Include rejected edits.

### Phase H — rule-based recommendation engine

Before neural controller, use rules with priority:

1. counterfactual heldout gain;
2. retrieved accepted memory;
3. gradient pressure;
4. entropy/collapse diagnostics.

Respect softmax conflicts: recommend reallocations, not “increase everything”.

### Phase I — active ProgramMetaController

Only after A-H.

Inputs:

- program object tokens;
- counterfactual summaries;
- memory retrieval tokens;
- task/head context;
- collapse diagnostics.

Outputs bounded soft edit biases:

- read bias;
- primitive bias;
- transition bias;
- composition bias;
- write bias;
- macro bias.

Train this controller through the heldout/architecture path, not only the train loss.

---

## Immediate next engineering tasks

1. Fix macro metrics logging.
2. Add anti-collapse/load-balance losses.
3. Add bi-level runner for `editor_delta` and macro variants.
4. Add soft-vs-argmax diagnostic.
5. Add counterfactual attributor for top-k components.
6. Only then build active central mind.

This order prevents Central Mind from becoming a decorative hypernetwork on top of SGD.
