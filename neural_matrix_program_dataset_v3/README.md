# Neural Matrix Program Dataset v3

Финальная версия инструмента для построения честного датасета матричных программ из реального Python/PyTorch-кода и реальных весов нейронок.

## Что это делает

Инструмент строит 4 уровня данных:

| Уровень | Что означает | Насколько “реально” |
|---|---|---|
| `static_pseudocode` | AST-парсинг кода: слои, блоки, вызовы, `softmax`, `einsum`, `Linear`, `state update` | структура реальная, матрицы нет |
| `real_structure_synthetic_matrix` | реальная структура кода + логичные контролируемые матричные программы | хороший pretrain азбуки/логики |
| `real_weight_program_decode` | реальные матрицы из checkpoint/state_dict -> декод в операторную формулу | не фейк, W реально из модели |
| `runtime_jacobian_program_decode` | будущий уровень: запуск модели, hooks/Jacobian/attention maps -> программа | самый дорогой и самый честный |

Главная идея:

```text
код до матриц
  -> static pseudocode / structure

реальные матрицы из весов
  -> operator dictionary
  -> sparse/product-like program decode
  -> formula + errors

динамические места
  -> runtime trace / Jacobian / frozen attention map
```

## Установка

Минимум:

```bash
pip install torch
```

Опционально для `.safetensors`:

```bash
pip install safetensors
```

## Файлы в пакете

```text
neural_matrix_program_dataset_v3.py   # основной CLI
README.md                             # этот файл
requirements.txt
examples/quick_smoke.sh
examples/p40_project_run.sh
```

## 1. Быстрый smoke test на CPU

```bash
python neural_matrix_program_dataset_v3.py all \
  --parse-dir ./your_project \
  --out ./runs/nmpd_v3_smoke \
  --n 256 \
  --D 16 \
  --layers 2 \
  --blocks 2 \
  --steps 2 \
  --primitive-slots 2 \
  --max-program-steps 4 \
  --device cpu \
  --epochs 4 \
  --batch-size 64 \
  --hidden 128
```

Если checkpoint не передан, `decode-real` будет пропущен. Это нормально.

## 2. Основной запуск на P40 / CUDA

```bash
python neural_matrix_program_dataset_v3.py all \
  --parse-dir ./your_project \
  --checkpoint-dir ./checkpoints \
  --out ./runs/nmpd_v3_project \
  --n 10000 \
  --D 32 \
  --device cuda \
  --epochs 25 \
  --batch-size 512 \
  --hidden 512 \
  --max-parse-files 300 \
  --max-checkpoint-files 20 \
  --max-matrices 500 \
  --decode-topk 12 \
  --mined-svd-atoms 4
```

## 3. Только распарсить код

```bash
python neural_matrix_program_dataset_v3.py parse \
  --parse-dir ./your_project \
  --out ./runs/nmpd_v3_parse \
  --max-parse-files 300
```

Выход:

```text
runs/nmpd_v3_parse/ast/skeleton.json
runs/nmpd_v3_parse/ast/ast_interactions.jsonl
runs/nmpd_v3_parse/ast/ast_summary.json
```

## 4. Только synthetic dataset по реальной структуре кода

```bash
python neural_matrix_program_dataset_v3.py build-synth \
  --parse-dir ./your_project \
  --out ./runs/nmpd_v3_synth \
  --n 5000 \
  --D 32 \
  --device cuda
```

Выход:

```text
synthetic/dataset.pt
synthetic/programs.jsonl
synthetic/build_summary.json
```

Это учит дешево:

```text
слой -> блок -> шаг -> read -> primitive -> transition -> write -> итоговая W
```

Это не “реальные веса”, но структура взята из кода, а матричные программы логично подставлены по типу взаимодействия.

## 5. Только реальные матрицы из checkpoint

```bash
python neural_matrix_program_dataset_v3.py decode-real \
  --checkpoint ./model.pt \
  --out ./runs/nmpd_v3_real \
  --D 64 \
  --device cuda \
  --max-matrices 300 \
  --decode-topk 12 \
  --auto-mined-atoms \
  --mined-svd-atoms 4
```

Выход:

```text
real_decode/real_matrix_decodes.jsonl
real_decode/real_matrix_dataset.pt
real_decode/decode_summary.json
```

Каждая запись содержит:

```json
{
  "truth_level": "real_weight_program_decode",
  "source": "...",
  "name": "model.layers.0.self_attn.q_proj.weight",
  "original_shape": [896, 896],
  "decoded_shape": [64, 64],
  "metrics": {
    "rec_err": 0.12,
    "functional_err_gaussian": 0.13,
    "active": 12,
    "uses_mined_atoms": true
  },
  "selected_terms": [...],
  "formula": "W_real = ..."
}
```

## Важная честность

Если матрица была прямоугольная или conv, она ресайзится в `D x D` для pattern/program decode.

Это честно помечается в `honesty_note`.

То есть:

```text
square D x D:
  ближе к exact decode

rectangular / conv / resized:
  pattern decode, не точная замена исходной формы
```

## Что такое auto dictionary / mined atoms

Базовый словарь:

```text
Identity
ShiftRight/Left
Diff/Laplacian
Blur
DCTLow/Mid/High
PrefixMean
DistanceDecay
BlockAvg
Diag gates
LowRank fixed
Graph diffusion
```

Для реальной матрицы можно автоматически добавить target-specific atoms:

```text
MinedDiagTarget
MinedRowMeanTarget
MinedColMeanTarget
MinedSVD1..k
MinedToeplitzTarget
MinedBlockMean2/4/8
```

Они помечаются как:

```json
"origin": "mined_from_target"
```

Это важно: если ошибка хорошая только из-за mined atoms, значит словарь ещё надо обобщать. Если хороший decode получается базовыми atoms — это сильнее.

## Что смотреть после запуска

### Synthetic

```bash
cat ./runs/nmpd_v3_project/synthetic/baseline_metrics.json
```

Важные поля:

```text
used_ops_f1        # модель понимает какие операторы были в W
first_op_acc       # видит первый оператор
last_op_acc        # видит последний оператор
read_hist_kl       # понимает распределение read-типов
primitive_hist_kl  # понимает primitive-логику
transition_hist_kl # понимает transition-логику
primitive_transition_pair_kl # понимает operation между двумя primitive
```

### Real decode

```bash
cat ./runs/nmpd_v3_project/real_decode/decode_summary.json
head -n 3 ./runs/nmpd_v3_project/real_decode/real_matrix_decodes.jsonl
```

Смотри:

```text
rec_err
functional_err_gaussian
uses_mined_atoms
selected_terms
formula
role_guess
```

## Рекомендованный пайплайн

```text
Шаг 1:
  parse/build-synth/train-synth
  -> учим азбуку/грамматику/структуру дешево.

Шаг 2:
  decode-real на checkpoint/state_dict
  -> реальные веса превращаем в формулы.

Шаг 3:
  отдельно добавить runtime hooks/Jacobian
  -> dynamic ops: attention/softmax/GELU/norm.

Шаг 4:
  merge dataset:
    static_pseudocode
    real_structure_synthetic_matrix
    real_weight_program_decode
    runtime_jacobian_program_decode
```

## Команды для твоего текущего проекта

Из папки проекта:

```bash
python neural_matrix_program_dataset_v3.py all \
  --parse-dir . \
  --out ./runs/nmpd_v3_self \
  --n 5000 \
  --D 32 \
  --device cuda \
  --epochs 25 \
  --batch-size 512 \
  --hidden 512 \
  --max-parse-files 300 \
  --max-program-steps 16
```

Если есть checkpoint:

```bash
python neural_matrix_program_dataset_v3.py all \
  --parse-dir . \
  --checkpoint-dir ./checkpoints \
  --out ./runs/nmpd_v3_self_with_real \
  --n 5000 \
  --D 64 \
  --device cuda \
  --epochs 25 \
  --batch-size 512 \
  --hidden 512 \
  --max-matrices 500 \
  --decode-topk 12 \
  --mined-svd-atoms 4
```

## Ограничения

1. AST без запуска не знает реальные gate choices.
2. Реальные `Linear.weight` можно декодить прямо.
3. `Conv` переводится в im2col-like flat matrix.
4. `softmax/GELU/LayerNorm/attention` глобально не являются одной постоянной матрицей.
5. Для динамики нужен runtime trace/Jacobian.
6. Rectangular weights ресайзятся в `D x D`, поэтому это pattern decode, не exact deploy replacement.

## Что дальше добавить в v4

- `torch.fx` trace для форм и graph-level структуры.
- forward hooks для реальных activation/gates.
- local Jacobian per block.
- attention map bank decode.
- merge всех уровней в один unified training dataset.
