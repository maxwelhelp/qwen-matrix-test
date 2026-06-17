# Запуск Matrix Program Syntax

## 1. Быстрая проверка

```bash
cd /home/maxwelhelp/test/sience/experiments/math_search/WORKING_BEST/test
bash neural_matrix_program_dataset_v3/commands/run_local_smoke.sh
```

Проверить, что датасет автоматически подхватил примитивы из нашего кода:

```bash
bash neural_matrix_program_dataset_v3/commands/check_auto_primitives.sh
```

Ожидаемо сейчас:

```text
primitive_count 17
notes ['Auto-expanded primitives from PRIMITIVES: 6', ...]
```

## 2. Medium CPU прогон

```bash
cd /home/maxwelhelp/test/sience/experiments/math_search/WORKING_BEST/test
bash neural_matrix_program_dataset_v3/commands/run_syntax_pretrain_cpu_medium.sh
```

Выход:

```text
neural_matrix_program_dataset_v3/runs/syntax_pretrain_cpu_medium/synthetic/dataset.pt
neural_matrix_program_dataset_v3/runs/syntax_pretrain_cpu_medium/synthetic/programs.jsonl
neural_matrix_program_dataset_v3/runs/syntax_pretrain_cpu_medium/synthetic/baseline_metrics.json
```

## 3. Основной CUDA прогон

```bash
cd /home/maxwelhelp/test/sience/experiments/math_search/WORKING_BEST/test
bash neural_matrix_program_dataset_v3/commands/run_syntax_pretrain_cuda.sh
```

Выход:

```text
neural_matrix_program_dataset_v3/runs/syntax_pretrain_cuda/synthetic/dataset.pt
neural_matrix_program_dataset_v3/runs/syntax_pretrain_cuda/synthetic/programs.jsonl
neural_matrix_program_dataset_v3/runs/syntax_pretrain_cuda/synthetic/baseline_metrics.json
```

## 4. P40 прогон операций между примитивами

```bash
cd /home/maxwelhelp/test/sience/experiments/math_search/WORKING_BEST/test
bash neural_matrix_program_dataset_v3/commands/run_syntax_pretrain_p40_pair_ops.sh
```

Главная метрика тут:

```text
primitive_transition_pair_kl
```

Она проверяет не просто частоту примитивов, а мягкую операцию между ними:

```text
primitive_a -> transition_op -> primitive_b
```

Это не router. Это supervised soft tensor target `[primitive, transition, primitive]`.

Парсер автоматически расширяет список примитивов из кода. Для текущих
`simple_butterfly_matrix_v*` он видит `P=17`:

```text
noop, identity, keep_state, small_refine, mlp, matrix_mlp, compare,
memory_read, global_read, normalize, suppress,
channel_butterfly, block_butterfly, low_rank, ctx_matrix, product_gate, phase_matrix
```

Поэтому pair-target имеет форму:

```text
primitive_transition_pair_hist[N,17,6,17]
```

Смотреть:

```bash
cat neural_matrix_program_dataset_v3/runs/syntax_pretrain_p40_pair_ops/synthetic/baseline_metrics.json
```

## 5. Обучение v3 на SpeechCommands

```bash
cd /home/maxwelhelp/test/sience/experiments/math_search/WORKING_BEST/test
bash simple_butterfly_matrix_v3/commands/run_speechcommands.sh
```

Это пока не использует syntax-pretrain веса напрямую. Это контрольный запуск текущей матричной архитектуры на задаче.

Для Tesla P40 лучше запускать fp32:

```bash
bash simple_butterfly_matrix_v3/commands/run_speechcommands_p40_fp32.sh
```

Продолжить от уже обученного `best.pt`:

```bash
bash simple_butterfly_matrix_v3/commands/run_speechcommands_p40_continue.sh
```

Продолжить от `speechcommands_v3_p40_continue/best.pt` аккуратнее, с меньшим LR:

```bash
bash simple_butterfly_matrix_v3/commands/run_speechcommands_p40_continue2_low_lr.sh
```

Более строгий вариант против позднего `aggregate` shortcut:

```bash
bash simple_butterfly_matrix_v3/commands/run_speechcommands_p40_continue2_balanced.sh
```

## 6. Декод реального checkpoint после задачи

```bash
cd /home/maxwelhelp/test/sience/experiments/math_search/WORKING_BEST/test
bash neural_matrix_program_dataset_v3/commands/run_decode_v3_checkpoint.sh
```

По умолчанию декодируется:

```text
simple_butterfly_matrix_v3/runs/speechcommands_v3/best.pt
```

Можно передать другой checkpoint и out-dir:

```bash
bash neural_matrix_program_dataset_v3/commands/run_decode_v3_checkpoint.sh \
  simple_butterfly_matrix_v3/runs/speechcommands_v3/best.pt \
  neural_matrix_program_dataset_v3/runs/real_decode_v3_speechcommands
```

## Что это проверяет

`train-synth` проверяет, что матрица `W` содержит читаемую программу:

```text
read -> primitive -> transition -> write -> step_mats -> prefix_mats -> W
```

Для нашей архитектуры главные поля:

```text
W
step_mats
prefix_mats
read_hist
primitive_hist
transition_hist
primitive_transition_pair_hist
```

`first_op_acc` и `last_op_acc` вторичны. В нашей модели не нужно делать hard argmax первого или последнего оператора. Нам нужны мягкие матричные массы и восстановление последовательности.

## Что еще не соединено

Текущий `train-synth` учит baseline decoder, а не `ClassMatrixTransport`.

Для настоящего переноса нужен следующий файл:

```text
simple_butterfly_matrix_v4/program_pretrained_matrix_transport.py
```

Он должен брать `synthetic/dataset.pt`, учить матричный builder на `W/step_mats/prefix_mats/hist`, потом переносить builder-веса в task модель.
