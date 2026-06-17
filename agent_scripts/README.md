# Agent scripts

Скрипты в этой папке не кладут веса в GitHub и не пушат `.pt`/датасеты. Они нужны, чтобы быстро проверить гипотезы, собрать короткий отчёт и вставить его в чат.

## skill debug

```bash
bash agent_scripts/run_skill_debug_3ep.sh
cat agent_reports/latest_skill_debug/REPORT_TO_CHATGPT.txt
```

Что делает скрипт:

1. Локально патчит `neural_matrix_program_dataset_v3.py`, добавляя `--init-decoder` и `--resume-decoder` для `train-synth`.
2. Запускает короткие 3-эпоховые проверки: fresh, resume, masked, masked-resume.
3. Собирает только метрики/логи в `agent_reports/...` без `.pt`.
4. Создаёт `REPORT_TO_CHATGPT.txt`, который можно целиком вставить в чат.
