# Closed-loop MatrixProgram transfer plan

Current result:

- Code/AST flow pretrain learns syntax of matrix-program flows.
- Live audio transfer does not yet show `delta > freeze_core`.
- Reason: the pretrain dataset teaches `context -> flow`, but live assembly is driven by downstream gradient from input adapter and head/consumer. The dataset has no real input/head closed loop.

What stays fixed:

- No hard router.
- No top-k path choice.
- Primitive/operator choices remain dense soft matrices.
- Transferable weights remain separated into base/delta/task packs.

Next version: closed-loop skill pretrain

Each synthetic/pretrain sample should contain:

- input/evidence tokens;
- task/context tokens;
- head/consumer query tokens;
- teacher or target output;
- differentiable task loss;
- optional flow target as weak regularizer.

Training objective:

- primary: downstream loss through the same `MatrixProgramAssemblerCore -> head` path used in live tasks;
- secondary: flow KL to keep program syntax valid;
- tertiary: entropy/alive/update budget.

This teaches the core not only what a valid program looks like, but how program changes affect a head/readout loss.

Useful variants:

1. Synthetic teacher tasks
   - sample a teacher matrix program;
   - generate evidence and target logits/states from it;
   - train student core to solve output loss and optionally match teacher flow.

2. Gradient-context tokens
   - run a short forward/backward with current head;
   - compress gradients of slots/head/input into low-rank context tokens;
   - feed those tokens to the assembler as matrix context, not as a router.

3. Multi-candidate soft projection
   - keep several projected flow variants alive in low-rank space;
   - blend by differentiable quality weights from loss/gradient summaries;
   - no hard top-k; mass flows toward better variants.

Immediate experiment order:

1. Run current live transfer with `TRAIN_TASK_CONTEXT=1`.
2. If context helps, build `AudioEvidenceAdapterV2` to make live evidence closer to pretrain context.
3. Add closed-loop synthetic skill pretrain and compare:
   - code-flow skill only;
   - closed-loop skill only;
   - code-flow + closed-loop.

Success criterion remains:

```text
audio_delta best_acc > audio_freeze_core best_acc
```
