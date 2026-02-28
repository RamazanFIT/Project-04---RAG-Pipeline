# RAG Pipeline Project

> **nFactorial Incubator — Модуль 4**
> Построение RAG-пайплайна поверх годовых отчётов КТЖ и Матен Петролеум

---

## Итоговые результаты

### Лучшая конфигурация: Experiment 9 — Reranking (avg = 0.9119)

| Метрика | Значение |
|---------|----------|
| Faithfulness | 0.9333 |
| Answer Relevancy | 0.8691 |
| Context Recall | 0.9667 |
| Context Precision | 0.8783 |
| **Average** | **0.9119** |

**Параметры:** Chunk Size 1024, Overlap 200, Top-K 5, α=0.5 (hybrid), Reranking ON

### Все эксперименты

| # | Конфигурация | Faith. | Ans.Rel. | Ctx.Recall | Ctx.Prec. | Avg |
|---|-------------|--------|----------|------------|-----------|-----|
| 0 | Baseline (1024/200, K=5, α=0.5) | 0.8000 | 0.8218 | 0.8667 | 0.7053 | 0.7985 |
| 1 | Chunk 512/100 | 0.7556 | 0.7549 | 0.7667 | 0.6327 | 0.7275 |
| 2 | Chunk 2048/200 | 0.8667 | 0.7822 | 0.8333 | 0.7440 | 0.8066 |
| 3 | Top-K=3 | 0.7689 | 0.7571 | 0.7667 | 0.6444 | 0.7343 |
| 4 | Top-K=10 | 0.8500 | 0.8425 | 0.9667 | 0.7706 | 0.8575 |
| 5 | α=0.0 (BM25 only) | 0.6944 | 0.7025 | 0.8000 | 0.5899 | 0.6967 |
| 6 | α=0.3 | 0.7722 | 0.7919 | 0.7667 | 0.7069 | 0.7594 |
| 7 | α=0.7 | 0.8556 | 0.8157 | 0.9000 | 0.7403 | 0.8279 |
| 8 | α=1.0 (Vector only) | 0.8500 | 0.8764 | 0.9000 | 0.7052 | 0.8329 |
| 9 | **Reranking** | **0.9333** | **0.8691** | **0.9667** | **0.8783** | **0.9119** |

---

## Оценка

| Задание | Баллы | Содержание |
|---------|-------|------------|
| Задание 1A | 20 б. | Naive RAG (DocLing + ChromaDB + GPT-4o-mini) |
| Задание 1B | 30 б. | Advanced RAG (Hybrid Search, Reranking, Query Rewriting) |
| Задание 2A | 30 б. | 10 экспериментов (greedy search по параметрам) |
| Задание 2B | 20 б. | RAGAS evaluation + итоговый вывод |
| Бонус | +30 б. | GraphRAG с Neo4j (требуется Docker) |

---

## Технический стек

| Категория | Инструмент |
|-----------|-----------|
| Парсинг PDF | DocLing |
| Оркестратор | LangChain |
| Векторная БД | ChromaDB |
| Embedding | intfloat/multilingual-e5-large |
| Reranker | BAAI/bge-reranker-v2-m3 |
| LLM | GPT-4o-mini |
| Оценка | RAGAS |
| Граф БД (бонус) | Neo4j |

---

## Структура проекта

```
nfactorial_project_4/
├── rag_pipeline.ipynb          ← основной ноутбук (все ячейки с output)
├── README.md                   ← этот файл
├── TZ.md                       ← детальное ТЗ
├── requirements.txt            ← зависимости
├── .env                        ← API ключ (не в git)
├── golden_dataset.json         ← 30 эталонных пар вопрос-ответ
├── ktj.pdf                     ← годовой отчёт КТЖ (~368 стр.)
├── matnp_2024_rus.pdf          ← годовой отчёт Матен Петролеум
├── parsed_ktj.md               ← кэш парсинга КТЖ
├── parsed_matnp.md             ← кэш парсинга Матен Петролеум
├── all_chunks.json             ← кэш чанков (fixed/recursive/layout)
├── experiment_scores.json      ← RAGAS метрики 10 экспериментов
├── experiment_results.json     ← полные результаты экспериментов
├── naive_test_results.json     ← результаты Naive RAG теста
├── ragas_results.png           ← график метрик
├── run_pipeline.py             ← скрипт запуска пайплайна
└── _inject_outputs.py          ← скрипт инъекции output'ов
```

---

## Запуск

### 1. Установка зависимостей
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Переменные окружения
```bash
echo "OPENAI_API_KEY=sk-..." > .env
```

### 3. Запуск ноутбука
```bash
jupyter notebook rag_pipeline.ipynb
```

### 4. (Бонус) Neo4j
```bash
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest
pip install neo4j
```

---

## Источники данных

| Файл | Компания | Страниц |
|------|----------|---------|
| `ktj.pdf` | АО «НК «ҚТЖ» | ~368 |
| `matnp_2024_rus.pdf` | АО «Матен Петролеум» | ~20–30 |
