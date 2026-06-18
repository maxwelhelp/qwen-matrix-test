# Agent Implementation Master Plan

This is the master execution plan for the next MatrixProgramAssembler version. It summarizes the recent design documents and turns them into an ordered implementation proposal for the coding agent.

Base rule:

```text
Do not rewrite the project from scratch.
Use the current working MatrixProgramAssembler / OperatorBankV2 / latent-role version as the base.
Add the new mechanisms step by step, behind flags, with smoke tests and reports.
```

---

## 0. Synchronize and read first

Run:

```bash
cd /home/maxwelhelp/test/sience/experiments/math_search/WORKING_BEST/test

git fetch origin main
git reset --hard origin/main
git status -sb
git log --oneline -12
```

Read these documents before coding:

```bash
sed -n '1,260p' matrix_program_core/READ_TRANSFORM_WRITE_GRAMMAR_PLAN.md
sed -n '1,260p' matrix_program_core/LATENT_ROLE_SPONTANEOUS_SPECIALIZATION_PLAN.md
sed -n '1,260p' matrix_program_core/INPUT_HEAD_STRUCTURE_PRIOR_PLAN.md
sed -n '1,260p' matrix_program_core/HEAD_MACRO_GROWTH_PLAN.md
sed -n '1,280p' matrix_program_core/PROGRAM_EDITOR_ACTOR_CRITIC_POLICY_PLAN.md
sed -n '1,320p' matrix_program_core/PROGRAM_EDITOR_ACTOR_CRITIC_V3_ADDENDUM.md
sed -n '1,220p' MATRIX_PROGRAM_VERSION_LEDGER.csv
```

Read current implementation files:

```bash
sed -n '1,260p' matrix_program_core/assembler_core.py
sed -n '260,620p' matrix_program_core/assembler_core.py
sed -n '620,980p' matrix_program_core/assembler_core.py
sed -n '1,260p' matrix_program_core/transfer_audio_assembler.py
sed -n '260,760p' matrix_program_core/transfer_audio_assembler.py
sed -n '760,1040p' matrix_program_core/transfer_audio_assembler.py
```

Useful optional idea source:

```bash
find /home/maxwelhelp/all/Aof -maxdepth 4 -type f \
  \( -name '*.py' -o -name '*.md' -o -name '*.txt' \) | head -200
```

AOF is only for ideas: cells, organs, memory, growth, topology, anti-collapse. Do not copy hard routing or non-differentiable logic. Our philosophy is soft, matrix-based, differentiable program assembly.

---

## 1. Current diagnosis

Recent latent-role / phase-free runs show:

```text
best_acc around 58.5% @ 20 epochs
class_read_div improved strongly
slot_div improved strongly
layer_sim and step_sim improved
```

Meaning:

```text
class-head specializes;
slots specialize;
layers/steps specialize by output behavior;
OperatorBankV2 and the read/write grammar work.
```

But:

```text
latent roles remain too uniform;
role_entropy is close to max;
role_similarity remains high;
roles are not yet separate organs.
```

So the next version must not simply add more roles. It must fix the conditions that let roles and layers specialize.

---

## 2. Core design to preserve

The external grammar remains minimal:

```text
read -> transform -> write
```

Do not turn it into a long hard-coded chain.

Instead:

```text
transform = primitive | compose | compare | branch | loop | macro
```

Keep mandatory:

```text
read path
transform/operator path
write path
residual/norm stability
step_alive
memory/global address space
complexity budget
anti-collapse diagnostics
```

Do not hardcode semantic layer roles:

```text
L0 = extract
L1 = compare
L2 = suppress
L3 = aggregate
```

Roles should emerge from task/data pressure, structure hints, sequential curriculum, usage feedback, and later tested edits.

---

## 3. Critical implementation order

### Stage 1: Observation and source-group logging — do first

Problem:

If every layer can freely read raw input/evidence from the start, later layers can shortcut and all layers become similar.

Add explicit read source groups:

```text
evidence/input
state/current cells
memory
global
output/slots if applicable
```

Required logs:

```text
read_mass_by_group[L,S,group]
write_mass_by_group[L,S,group]
memory_read_mass[L,S]
memory_write_mass[L,S]
global_read_mass[L,S]
global_write_mass[L,S]
late_input_read_mass[L,S]
```

Do not change behavior first. Add logging only.

Success:

```text
analysis_epoch_*.json reports where each layer/step reads and writes.
REPORT_TO_CHATGPT includes source-group summaries.
```

Files likely touched:

```text
assembler_core.py
transfer_audio_assembler.py
agent_scripts/run_*.sh
```

---

### Stage 2: Sequential-read curriculum — critical

Implement controlled evidence/input access.

Rationale:

```text
specialization is easier when early training is mostly sequential;
later layers can get input reread only after specialization or uncertainty.
```

Add config flags:

```text
sequential_read_curriculum: bool
input_read_depth_cost: float
input_read_unlock_epoch: int
input_read_unlock_schedule: linear/cosine/metric
confidence_input_reread: bool
late_input_read_max_bias: float
```

Behavior:

Early epochs:

```text
L0/S0 can read evidence/input cheaply.
L1/L2/L3 input read is penalized.
They mainly read state from previous updates plus memory/global.
```

Later:

```text
input read is gradually unlocked;
late layers can reread input if confidence low / margin low / critic later predicts gain.
```

MVP unlock condition:

```text
epoch >= input_read_unlock_epoch
```

Later metric unlock:

```text
layer_sim below threshold
step_sim below threshold
class_read_div below threshold
```

Do not fully remove ability to read input. Make it expensive early.

Required diagnostics:

```text
late_input_read_penalty
late_input_read_mass
input_unlock_value
```

Success:

```text
L0 reads evidence strongly early;
later layers read state/memory/global more early;
late input reread appears only after unlock;
accuracy does not collapse.
```

---

### Stage 3: True phase-free cleanup

Ensure no remaining behavior depends on named phases.

Search:

```bash
grep -R "PHASES\|self.phase\|phase_sets\|phase_prior\|extract\|compare\|suppress\|aggregate" -n matrix_program_core agent_scripts
```

Allowed:

```text
report labels / old compatibility only
```

Not allowed as behavior when latent/phase-free mode is on:

```text
layer index -> phase role
phase_sets primitive prior
if self.phase == ... transition prior
fixed L0/L1/L2/L3 role bias
```

If old phase code remains, gate it behind:

```text
phase_prior_mode = fixed|weak|free
```

Default for new runner:

```text
phase_prior_mode=free
```

Success:

```text
phase-free runner starts with no layer-index phase bias;
analysis proves phase mix is not tied to layer id.
```

---

### Stage 4: Latent role strengthening without hardcoding

Current roles are too uniform. Do not reintroduce named phases. Strengthen anonymous roles.

Add or tune:

```text
role_temperature schedule
role_to_read scale
role_to_primitive scale
role_to_transition scale
role_to_write scale
role_to_step_alive scale
role_usage_balance
role_similarity penalty
role entropy band
role dropout optional
```

Recommended schedule:

```text
epoch 0-3: role_temperature high, self-prior off
epoch 4-10: temperature anneals down, weak self-prior starts
epoch 10+: roles allowed to sharpen, anti-collapse still active
```

Do not force roles to be uniform forever. Use entropy band, not pure uniform balance.

Required logs:

```text
role_mix[L,S,R]
role_usage[R]
role_entropy[L,S]
role_similarity matrix
role_to_primitive top families
role_to_read/write group bias
```

Success:

```text
role_entropy decreases from near-max without collapsing;
role_similarity decreases;
roles show different primitive/read/write behavior;
accuracy stays close to or improves over 58.5% baseline.
```

---

### Stage 5: Self-organizing usage prior — weak and lagged

Purpose:

A layer/step that repeatedly selects a useful component should stabilize that behavior, but not lock in random early mistakes.

Implement:

```text
usage_ema[L,S,group,item]
prior_bias = alpha * log(normalize(usage_ema) + eps)
```

Apply weakly to:

```text
role logits
primitive logits
operator variant logits
transition logits
read/write source group logits
macro logits later
```

Hard rules:

```text
lagged only, never same-forward self-amplification
detach usage_ema
clamp bias, e.g. [-0.25, 0.25]
alpha starts 0, then 0.05-0.10
anti-collapse stays active
```

Do not use accepted/rejected labels yet. This is behavior stabilization, not memory truth.

Success:

```text
roles/components sharpen more than before;
no single primitive/role takes over everywhere;
val accuracy improves or convergence gets faster.
```

---

### Stage 6: Input/head structure priors — weak hints, not hard roles

Implement after Stage 1-5 are stable.

Input side:

```text
AudioStructureProbeBank or generic StructureProbeBank
energy / local_energy / diff/onset / smoothness / frequency summaries / noise / low-rank proxy
```

Head side:

```text
ClassificationHeadTaskProbe
num_classes / class read entropy / class overlap / class confusion from previous epoch / margin stats
```

Convert to universal structure tokens:

```text
LOCAL, GLOBAL, LOW_RANK, DIFF, BOUNDARY, MEMORY, FILTER_GATE, CLASS_SEPARATION, WRITE_HEAVY
```

Inject as weak clamped hints:

```text
primitive logits += alpha_input * hint
role logits += alpha_input/head * hint
read/write group logits += hint
head read logits += alpha_head * hint
```

Rules:

```text
alpha 0.05-0.15
clamp [-0.25,0.25]
hint dropout
no layer-index role assignment
```

A/B tests:

```text
A no structure hints
B input hints only
C head hints only
D input + head hints
E input + head + usage prior
```

Success:

```text
same or better accuracy;
faster convergence;
roles become less uniform;
class_read_div improves without depending only on class_slot_prior.
```

---

### Stage 7: Compare as first-class transform primitive

Do not treat compare only as macro. Add first-class compare family inside transform.

Compare forms:

```text
a-b
a*b
cosine/dot
bilinear score
query-key match
gated contrast
```

Use cases:

```text
attention-like selection
memory retrieval
branch routing
class-pair confusion
head decision programs
```

Constraints:

```text
compare cost
score norm/z-loss
avoid dense all-pairs by default
local/top-k approximation if needed
```

Success:

```text
compare usage appears in attention/memory/head-confusion cases;
no O(N^2) blowup by default;
accuracy or confusion-pair metrics improve.
```

---

### Stage 8: Active mind MVP — candidate generator before actor

Do not build a neural feedback bus first.

Implement minimal active loop from V3 + addendum:

```text
CandidateEditGenerator
WorkshopCoordinator
CounterfactualScreen
ProgramMemoryBank JSONL
ColdStartCritic ridge/Bayesian linear
SparseLCBEditApplier
FullTrainingStateRollback
```

Candidate sources:

```text
gradient/saliency
collapse/entropy
residual/error
memory retrieval
safe exploration
```

Every candidate must be grammar-labeled:

```text
read
transform.primitive
transform.operator_variant
transform.compare
transform.compose
transform.branch
transform.loop
write
memory
head
budget
```

MVP rule:

```text
max 1 sparse edit per window
max 1 workshop per deployed edit
scale 0.03
Tier-1 screen first
LCB deploy only
full rollback if harmful
```

Do not use UCB for live deployment. UCB only selects probes.

Stage order:

```text
critic/actor refinement before local growth/promotion
```

Success:

```text
JSONL memory has accepted/rejected/uncertain records;
critic predicted vs real gain logged;
rollback works;
compute overhead reported;
no dense actor tensor appears.
```

---

### Stage 9: HeadMacroBank growth — after tested memory exists

Do not implement head growth before candidate/memory/critic infrastructure.

Candidate head macros:

```text
class_pair_contrast
class_unique_slot_read
suppress_shared_slot
memory_global_gate
local_global_product_read
background_rejector
confidence_calibrator
rare_class_recall
```

Promotion only after:

```text
heldout gain passes dynamic noise gate
not duplicate
complexity budget ok
global val not worse
retest positive
```

Success:

```text
confusion pairs improve;
head macros are sparse and interpretable;
class_read_div improves without relying purely on class_slot_prior;
no train-only overfit macros.
```

---

### Stage 10: Local macro/operator growth — last, not first

Only after Stages 8-9 are reliable.

Core growth principle:

```text
primitive_A + primitive_B + condition = new transform/operator/macro
```

Examples:

```text
diff + smooth + gate -> OnsetLikeTransform
low_rank + product_gate + residual -> ConditionalLowRankTransform
compare(slot_A, slot_B) + margin_gate -> ClassPairContrastTransform
```

Promotion utility:

```text
utility = heldout_gain
        + functional_gain
        + target_subproblem_gain
        - complexity_cost
        - duplicate_penalty
        - instability_penalty
        - compute_cost
```

Accept only if above dynamic noise threshold.

Success:

```text
promoted macros are reused;
not duplicates;
heldout gain survives retest;
complexity does not explode.
```

---

## 4. New runner names

Add runners without breaking old ones:

```text
agent_scripts/run_rtww_seq_latent_audio.sh
agent_scripts/run_structure_prior_ablation.sh
agent_scripts/run_active_mind_mvp.sh
```

Each runner must create:

```text
agent_reports/<run_name_timestamp>/REPORT_TO_CHATGPT.txt
summary.txt
metrics.csv
analysis_epoch_*.json
```

No weights inside reports.

---

## 5. Required report fields

Each report must include:

```text
git commit
config
best_acc / best_epoch
train/val final
read_mass_by_group per layer/step
write_mass_by_group per layer/step
late_input_read_mass
role_usage / role_entropy / role_similarity
class_read_div
slot_div
layer_sim / step_sim
primitive/operator family usage
compare usage if enabled
memory/global read/write usage
step_alive
complexity costs
if active mind enabled:
  candidate counts by source
  candidate counts by grammar element
  screened/accepted/rejected/uncertain
  deployed edit
  critic predicted vs real gain
  rollback count
  overhead
```

---

## 6. Smoke and main commands

Smoke:

```bash
EPOCHS_AUDIO=1 \
TRAIN_N=512 \
VAL_N=256 \
LOG_EVERY=2 \
bash agent_scripts/run_rtww_seq_latent_audio.sh
```

Short real test:

```bash
EPOCHS_AUDIO=8 \
TRAIN_N=6000 \
VAL_N=1000 \
AUDIO_LR=2e-4 \
LAYERS=4 \
STEPS=3 \
bash agent_scripts/run_rtww_seq_latent_audio.sh
```

Main baseline:

```bash
EPOCHS_AUDIO=20 \
TRAIN_N=12000 \
VAL_N=2000 \
AUDIO_LR=2e-4 \
LAYERS=4 \
STEPS=3 \
bash agent_scripts/run_rtww_seq_latent_audio.sh
```

Structure prior ablation:

```bash
EPOCHS_AUDIO=8 \
TRAIN_N=6000 \
VAL_N=1000 \
RUN_VARIANTS="no_hints input_hints head_hints input_head_hints input_head_usage" \
bash agent_scripts/run_structure_prior_ablation.sh
```

Active mind MVP later:

```bash
EPOCHS_AUDIO=10 \
TRAIN_N=6000 \
VAL_N=1000 \
ACTIVE_MIND=1 \
MAX_DEPLOY_EDITS=1 \
ACTOR_EDIT_SCALE=0.03 \
bash agent_scripts/run_active_mind_mvp.sh
```

---

## 7. What not to do

Do not:

```text
rewrite from scratch
make hard layer roles again
let all layers freely read raw input from epoch 0
build neural feedback bus before tested memory exists
deploy UCB edits
use dense actor bias tensors
promote macros before critic/memory is reliable
accept candidate edits based only on train gradient
rollback only model weights
let memory write be unconstrained
let head macros bypass core and memorize raw input
```

---

## 8. Definition of success for the new version

Minimum success:

```text
matches or exceeds ~58.5% @ 20 epochs with better diagnostics
roles less uniform than previous latent-role run
late input-read controlled, not shortcutting all layers
class_read_div / slot_div / layer_sim / step_sim remain good
```

Strong success:

```text
>=60% @ 20-25 epochs without fixed phase roles
roles become interpretable organs
memory/global used non-trivially
input/head hints improve convergence
compare helps confusion/memory cases
active mind records tested edits without destabilizing training
```

Research success:

```text
tested head/core macros are promoted and reused
critic predictions correlate with real heldout gain
local growth improves accuracy or reduces complexity
```

---

## 9. Short instruction to agent

Implement the next version in this order:

```text
1. source-group logging
2. sequential-read curriculum
3. true phase-free cleanup
4. stronger anonymous latent roles
5. weak lagged self-organizing usage prior
6. input/head structure hints
7. first-class compare transform
8. candidate generator + active mind MVP
9. head macro growth
10. local core macro/operator growth
```

Stop after each stage, run smoke, produce report, and do not move to growth until observation/curriculum/roles are stable.
