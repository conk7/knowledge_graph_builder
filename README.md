
# Knowledge Graph Builder & Data Parser

Проект — набор инструментов для построения и анализа графа знаний поверх Markdown-хранилища заметок (например, Obsidian).
В репозитории есть два основных направления:

- `src/kg_builder` — основной модуль: находит и добавляет семантические связи между заметками.
- `src/data_parser` — пайплайн получения данных (скачивание/краулинг Wikipedia и разметка ссылок).
- `src/eval` — набор скриптов для оценки качества (links/retrieval/RAG/Neo4j).

По умолчанию связи записываются в Markdown в секцию `## Related Connections` в виде Dataview-строк:

```md
- Is a:: [[Target]]
```

---

## Требования

- Python >= 3.13 (см. `pyproject.toml`)
- `uv` (менеджер окружения/зависимостей)
- torch+cuda (в некоторых местах `cuda` может быть захардкожена)
- Для Neo4j-оценки: Docker + docker compose

Дополнительно нужны модели spaCy:

- `en_core_web_sm` для английского
- `ru_core_news_sm` для русского

---

## Установка

1) Установить `uv`.

2) Установить зависимости:

```bash
uv sync
```

3) Установить модели spaCy:

```bash
uv run python -m spacy download en_core_web_sm
uv run python -m spacy download ru_core_news_sm
```

---

## Запуск

# Data parser

Hub-n-Spoke пайплайн (Wikipedia → vault):

```bash
uv run python -m src.data_parser.hub_n_spoke.main \
  --seed Internet --lang en --vault "data/test_vaults/gold/hub_n_spoke" \
  --skip-extract --skip-apply --skip-crawl --use-llm-fallback
```

Flex downloader (скачать конкретные статьи; пример `sequence`):

```bash
uv run python -m src.data_parser.flex.main \
  "Geissler_tube" "Cathode_ray" "Electron" "Plum_pudding_model" \
  "Geiger–Marsden_experiments" "Rutherford_model" "Bohr_model" \
  --lang en --out "data/test_vaults/gold/sequence" --retries 3 --filter-links
```

----------
# KG builder

Инициализация (создаёт структуру метаданных в vault):

```bash
uv run python -m src.kg_builder.main init </path/to/vault>
```

Запуск обработки:

```bash
uv run python -m src.kg_builder.main run </path/to/vault> \
  --save-mode export \
  --export-path /path/to/output_vault \
  --retrieval-strategy combined \
  --broad-query-mode chunk \
  --fresh-start \
  --api \
  --lang en
```

Подсказка по флагам:

- `--save-mode`: `inplace | json | export`
- `--fresh-start`: очищает прошлые метаданные/индекс; в `inplace` также удаляет ранее добавленные ссылки
- `--api`: использовать внешнее API для LLM (см. переменные окружения ниже)
- `--ignore-local-config`: игнорировать сохранённый в vault `config.json` и взять параметры из `src/kg_builder/config.py`

### Настройка LLM

Модуль поддерживает два режима:

1) **Локальная модель** через `llama-cpp-python`.
	- Задайте путь к модели (GGUF) через переменную окружения:

```bash
export LLM_MODEL_PATH=/absolute/path/to/model.gguf
```

2) **Внешний API** (активируется флагом `--api`).
	- Переменные окружения читаются из `.env`. Минимальный набор:

```dotenv
LLM_PROVIDER=openai   # openai | google | cerebras | groq
MODEL=...             # название модели у провайдера

# Для OpenAI-совместимого API:
OPENAI_API_KEY=...
OPENAI_BASE_URL=http://localhost:1234/v1 (если запускать локально)

# Для Google:
GOOGLE_API_KEY=...

# Для Cerebras:
CEREBRAS_API_KEY=...

# Для Groq:
GROQ_API_KEY=...
```

---


Пример для датасета `versus`, LLM через API:

```bash
uv run python -m src.kg_builder.main "data/test_vaults/gold/versus/" \
  --save-mode export \
  --export-path "results/links/versus" \
  --retrieval-strategy combined \
  --broad-query-mode chunk \
  --splitter-type recursive \
  --ignore-local-config \
  --fresh-start \
  --api
```

Сохранить найденные связи в JSON:

```bash
uv run python -m src.kg_builder.main "data/test_vaults/gold/sequence" --save-mode json --fresh-start --api
```

Export-режим (копировать vault в новую папку):

```bash
uv run python -m src.kg_builder.main "data/test_vaults/gold/sequence" \
  --save-mode export --export-path "results/links/sequence" \
  --fresh-start --api --retrieval-strategy combined
```

----------

# Оценка

### Оценка ссылок на наличие и тип (vault vs vault)

```bash
uv run python scripts/eval_links_dataset.py \
  --gold-root data/test_vaults/gold \
  --pred-root results/links \
  --output results/evaluated_gold/link_metrics.json
```

### Оценка Retrieval (извлечение + оценка)

```bash
uv run python -m src.eval.retrieval.retrieval_evaluator \
  --vault data/test_vaults/gold \
  --gold-dir data/test_retrieval/gold \
  --output-dir ./results/evaluated_gold \
  --show-inner-progress \
  --strategies Strict
```

### Оценка Link classification

```bash
uv run python -m src.eval.classification.main \
  --gold-dir data/test_retrieval/gold \
  --vault-dir data/test_vaults/gold \
  --output results/classification_results.jsonl
```

### Оценка RAG

Базовый алгоритм

```bash
uv run python -m src.eval.ragas.base_eval \
  --vault results/links/versus \
  --qa-file data/test_qa/versus.json \
  --max-hops 2 --top-k-seed 3 --top-k-context 10 \
  --ignore-local-config
```

Typed алгоритм

```bash
uv run python -m src.eval.ragas.typed_eval \
  --vault results/links/versus \
  --qa-file data/test_qa/versus.json \
  --max-hops 2 --threshold 0.2 --top-k-seed 3 --top-k-context 10 \
  --ignore-local-config
```

Реренсный алгоритм

```bash
uv run python -m src.eval.ragas.reference_eval \
  --vault results/links/versus \
  --qa-file data/test_qa/versus.json \
  --output results/ragas_reference_graphrag.json \
  --top-k-seed 3 --top-k-context 10 \
  --ignore-local-config
```


### Neo4j

Поднять Neo4j:

```bash
docker compose -f docker-compose.neo4j.yml up -d
```

Остановить:

```bash
docker compose -f docker-compose.neo4j.yml stop
```

Оценка Neo4j (hub_n_spoke):

```bash
uv run python -m src.eval.ragas.neo4j_eval \
  --vault data/test_vaults/gold/hub_n_spoke \
  --qa-file data/test_qa/hub_n_spoke.json \
  --output results/ragas_neo4j/hub_n_spoke.json \
  --max-hops 2 --top-k-seed 8 --top-k-context 10
```

---


## Где смотреть результаты

- Для `kg_builder` служебные файлы лежат в `<vault>/.kg_builder/.out/`.
  - логи: `obsidian_ai_linker.log`
  - кандидаты/связи: `candidates.json`, `reranked_candidates.json`, `links.json`
- Для `eval` результаты обычно пишутся в `results/` (пути/файлы зависят от запуска).

