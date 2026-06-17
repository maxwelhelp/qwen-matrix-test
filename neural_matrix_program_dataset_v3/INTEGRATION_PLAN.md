# Интеграция Neural Matrix Program Dataset v3

## Роль в проекте

Этот пакет надо держать отдельно от `simple_butterfly_matrix_v*`.

`simple_butterfly_matrix_v*` - это сама архитектура для задачи.

`neural_matrix_program_dataset_v3` - это генератор учебного материала: как из последовательных шагов, read-источников, примитивов, переходов и write-операций собирать матричную программу.

Так мы разделяем:

1. Task checkpoint: веса конкретной модели под задачу, например SpeechCommands.
2. Program-builder weights: отдельный "ум сборки", который учится собирать матричные программы.
3. Program dataset: примеры `W`, `step_mats`, `prefix_mats`, soft hist targets и реальные decode-записи из checkpoint.

## Что уже есть в датасете

Скрипт строит четыре уровня:

1. `static_pseudocode`: AST реального кода, без обещания точной математики.
2. `real_structure_synthetic_matrix`: реальная структура кода плюс синтетические, но логичные матричные программы.
3. `real_weight_program_decode`: реальные веса из checkpoint декодятся через словарь операторов.
4. `runtime_jacobian_program_decode`: протокол на будущее для hooks/Jacobian/attention maps.

Для нас самый важный старт:

- `synthetic/dataset.pt`
- `synthetic/programs.jsonl`
- `synthetic/baseline_metrics.json`
- `real_decode/real_matrix_decodes.jsonl`, когда появятся checkpoint-файлы.

## Как это ложится на нашу философию

Никакого hard router и top-k в самой архитектуре.

Датасет может содержать дискретные имена для анализа, но модель должна видеть их как мягкие матрицы:

- `read_hist` -> soft read matrix.
- `primitive_hist` -> soft primitive mass.
- `transition_hist` -> soft transition/program-flow matrix.
- `primitive_transition_pair_hist` -> soft tensor for `primitive_a -> operation -> primitive_b`.
- `step_mats` -> матричные цели для каждого шага.
- `prefix_mats` -> цели для правильной последовательности шагов.
- `W` -> итоговая матричная программа.

То есть выбор не делается резким `argmax`. Он обучается как масса в матрицах:

```text
read_flow
primitive_flow
transition_flow
primitive_transition_pair_flow
phase_flow
write_flow
class_phase_matrix
variant_transport
factorized_operator_basis
```

## Почему это должно помочь

Сейчас `v1/v2` упирались примерно в 53% на SpeechCommands: модель училась признакам, но class/head начинал читать слишком похожие поздние aggregate-слоты.

`v3` чинит class/head через `ClassMatrix` и `ClassPairMatrix`, но ему все еще не хватает предварительной грамматики сборки: что читать, в каком порядке, каким примитивом менять состояние, куда писать.

Этот датасет как раз дает дешевую предтренировку такой грамматики:

```text
layer -> block -> step -> read -> primitive -> transition -> write -> W
```

Авторасширение примитивов включено: parser подхватывает module-level
`PRIMITIVES` из нашего кода и сливает их с логическими dataset-примитивами.
Сейчас получается `P=17`, включая:

```text
channel_butterfly
block_butterfly
low_rank
ctx_matrix
product_gate
phase_matrix
```

Для неизвестных новых имен включается эвристика по названию:
`attention`, `conv`, `gate`, `norm`, `mlp`, `butterfly`, `phase`, `ctx`,
`low_rank`. Если имя совсем новое и не похоже ни на одну категорию, оно пока
получает fallback `Identity + LowRankA`; это место надо усиливать через real
checkpoint decode и runtime/Jacobian.

Главная польза не в baseline decoder из скрипта. Baseline нужен только как sanity check. Настоящая польза - сделать адаптер для нашей следующей версии:

```text
MatrixProgramAssemblyPretrain
  input: W / skeleton / optional checkpoint decode
  target: step_mats, prefix_mats, read_hist, primitive_hist, transition_hist
  losses:
    W reconstruction
    prefix reconstruction
    step reconstruction
    read KL
    primitive KL
    transition KL
    primitive-transition-pair KL
    phase balance
```

## Следующая архитектурная версия

Рекомендуемый следующий файл после проверки датасета:

```text
simple_butterfly_matrix_v4/program_pretrained_matrix_transport.py
```

Что добавить в v4:

1. Общую factorized operator basis:
   `Down[D,r] -> operation space r -> Up[r,D]`.
2. Soft `primitive_flow[P,P]` оставить, но связать с operator basis.
3. Soft `read/write/phase` матрицы учить от dataset targets.
4. Class head оставить как в v3, потому что классы должны быть дифференцируемыми через `ClassMatrix`, а не hard label routing.
5. Добавить pretrain режим на `synthetic/dataset.pt`, потом finetune на SpeechCommands.

## Важное ограничение

Этот пакет сам по себе не делает нашу модель универсальной. Он дает язык сборки. Переносимость появится, если:

1. Program-builder weights предобучены на разных skeleton/checkpoint decode.
2. Task head/последние task-веса меняются под задачу.
3. Основа остается матричной и мягкой, без hard router.
