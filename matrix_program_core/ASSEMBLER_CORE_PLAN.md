# Matrix Program Assembler Core Plan

Цель: переносить не decoder и не task-head, а **одни и те же веса логики сборки матричных программ**.

## Что переносим

`AssemblerCore` должен хранить отдельный checkpoint с матрицами:

- `read_flow`: какие адреса/cells читать на каждом layer/step/block/slot.
- `primitive_slot_flow`: какие примитивы используются в каждом primitive-slot.
- `slot_transition_flow`: операции между primitive-slots внутри шага.
- `primitive_transition_flow`: операции между типами примитивов.
- `slot_composition_flow`: как K primitive-slots собираются в update.
- `write_flow`: куда писать update: state cells, memory cells, global cells.
- `phase_bias`: фазовый prior: extract/compare/suppress/aggregate.
- `*_delta`: LoRA-like матрицы для новой задачи.

Это и есть переносимый навык сборки.

## Что НЕ переносим как основной навык

- InputAdapter: аудио/текст/матрица/картинка. Это задача-специфично.
- TaskHead: классификация/реконструкция/другая цель. Это задача-специфично.
- BaselineDecoder `W -> labels`: только диагностик, не переносимый core.

## Без роутеров

Запрещено:

- hard top-k routing
- argmax выбор пути
- discrete branch/if для выбора операции

Разрешено:

- softmax/sigmoid матрицы смешивания
- dense read/write matrices
- low-rank projections
- LoRA-like delta matrices
- factorized transition matrices

Все пути остаются живыми, градиент идёт через все примитивы.

## Почему не полный огромный тензор

Полная логика вида:

`layer × step × block × slot × read × primitive × transition × write × memory`

слишком дорогая. Поэтому используем факторизацию:

```text
read_flow:              [L,S,B,K,A]
primitive_slot_flow:    [L,S,B,K,P]
slot_transition_flow:   [L,S,B,K,K]
primitive_transition:   [L,S,P,P]
slot_composition_flow:  [L,S,B,K]
write_flow:             [L,S,B,A]
```

Вместо полного `[L,S,B,K,K,P,P,A]` используем:

```text
slot-to-slot K×K + primitive-to-primitive P×P + read/write over A
```

Это быстрее и всё ещё выражает:

- куда ставить примитив;
- в какой step/block/slot;
- какой read нужен;
- куда write;
- как использовать memory/global;
- какие другие slots активны;
- какой phase/program prior.

## Forward ядра

```text
InputAdapter -> evidence [B,E,D]

AssemblerCore:
  cells0 = initialize_cells(evidence)  # state + memory + global

  for layer in L:
    for step in S:
      for block in B vectorized:
        read_ctx[k] = Σ_a read_flow[l,s,b,k,a] * cells[a]
        primitive_out[k,p] = Primitive_p(read_ctx[k], context)
        transitioned[k,p] = slot_transition[k,j] + primitive_transition[p,q]
        slot[k] = Σ_p primitive_slot_flow[l,s,b,k,p] * transitioned[k,p]
        update[b] = Σ_k composition_flow[l,s,b,k] * slot[k]
        cells += write_flow[l,s,b,a] * update[b]

  return slots, aux_flows
```

## Примитивы

Сначала используем текущие матричные примитивы:

- `channel_butterfly`
- `block_butterfly`
- `low_rank`
- `ctx_matrix`
- `product_gate`
- `phase_matrix`

Потом можно расширять словарь.

## Synthetic pretrain должен учить этот же core

Нужен не только final `W`, а задачи:

- evidence/context;
- target output;
- target read_flow;
- target primitive_slot_flow;
- target transition_flow;
- target composition_flow;
- target write_flow;
- prefix/intermediate outputs.

Loss:

```text
loss = task_execution_loss
     + λ_read * read_flow_loss
     + λ_prim * primitive_slot_loss
     + λ_slot_trans * slot_transition_loss
     + λ_prim_trans * primitive_transition_loss
     + λ_comp * composition_loss
     + λ_write * write_loss
     + λ_prefix * prefix_loss
```

## Transfer режимы

1. `freeze_core`
   - заморозить весь AssemblerCore;
   - обучать InputAdapter + Head;
   - проверка реальной переносимости.

2. `delta`
   - заморозить base AssemblerCore;
   - обучать только `*_delta` + InputAdapter + Head;
   - основной LoRA-like режим.

3. `full`
   - обучать всё;
   - контрольная верхняя граница, но риск забывания навыка.

## Метрики

Нужны не только task accuracy:

- `task_acc`
- `read_flow_kl`
- `primitive_slot_kl`
- `slot_transition_kl`
- `primitive_transition_kl`
- `write_flow_kl`
- `prefix_loss`
- `read_entropy`
- `primitive_entropy`
- `slot_transition_entropy`
- `write_entropy`
- `memory_usage`
- `global_usage`
- `trainable_core_params`

## Первый быстрый тест

1. `assembler_core.py` — факторизованный core.
2. `train_assembler_pretrain.py` — synthetic pretrain этого же core.
3. `transfer_audio_assembler.py` — AudioAdapter -> AssemblerCore -> Head.
4. `agent_scripts/run_assembler_core_debug.sh` — pretrain + delta/freeze/full отчёт без `.pt`.

## Главная проверка

Если `freeze_core` уже даёт сигнал, значит core переносит сборку.
Если `delta` заметно лучше `freeze_core`, значит LoRA-like доучивание логики сборки работает.
Если `full` лучше, но ломает flow-метрики, значит full забывает навык и нужно усиливать pretrain/regularizer.
