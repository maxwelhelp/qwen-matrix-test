# Context and Layer-Task Conditioning

Problem found after gradient-fix:

- flow-loss now trains assembly matrices;
- but a global flow is still too weak for datasets where every layer/phase/task needs a different program;
- if the assembler does not know the current task context, it averages programs.

## What context must be visible

Assembler flow should depend on:

1. state cells: local task state for blocks;
2. memory cells: accumulated decisions/history;
3. global cells: whole input/task summary;
4. layer/step key: learned phase/program position;
5. evidence-derived key: what kind of input/task this sample represents.

## Still no router

This is not a router:

```text
flow_logits = base_flow_logits + Linear(context)
flow = softmax(flow_logits)
```

No argmax. No top-k. No discrete branch. All primitive paths stay alive.

## Why layer tasks matter

If layer 0 is extract, layer 1 is compare, layer 2 is suppress, layer 3 is aggregate,
then using one context-free target makes the assembler learn an average program.
Layer/step keys allow the same core to say:

```text
same primitive dictionary
but different read/primitive/transition/write composition per phase
```

## Dataset consequence

The synthetic dataset should not only contain one generic target. It should include:

- multiple layer tasks;
- multiple valid solution lengths;
- local-only / memory-heavy / global-heavy variants;
- short/medium/long program traces;
- prefix/intermediate targets;
- same task with several equivalent programs.

The model should learn a distribution over valid soft programs, not a single hard recipe.
