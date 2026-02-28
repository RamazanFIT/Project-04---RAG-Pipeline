# Детальное Техническое Задание — RAG Pipeline Project

> **Читайте вместе с README.md**
> Здесь подробно расписан каждый шаг, критерии оценки и чек-листы.

---

## Содержание

1. [Задание 1 — Построение RAG-пайплайна (50 б.)](#задание-1)
   - [Часть A: Naive RAG (20 б.)](#часть-a-naive-rag)
   - [Часть B: Advanced RAG (30 б.)](#часть-b-advanced-rag)
2. [Задание 2 — Эксперименты и оценка (50 б.)](#задание-2)
   - [Часть A: Серия экспериментов (30 б.)](#часть-a-серия-экспериментов)
   - [Часть B: RAGAS + итоговый вывод (20 б.)](#часть-b-ragas--итоговый-вывод)
3. [Бонус: GraphRAG с Neo4j (+30 б.)](#бонус-graphrag)
4. [Структура ноутбука](#структура-ноутбука)
5. [Чек-листы по заданиям](#чек-листы)

---

## Задание 1

### Часть A: Naive RAG (20 баллов)

**Цель:** Базовый рабочий RAG-пайплайн поверх двух PDF-файлов.

#### Шаг 1 — Парсинг PDF (Visual Layout)

Использовать **один** из инструментов:

| Инструмент | Установка | Когда выбрать |
|------------|-----------|---------------|
| **DocLing** | `pip install docling` | Рекомендуется, открытый, лёгкий |
| **Unstructured** | `pip install unstructured` | Хорошая поддержка таблиц |
| **LlamaParse** | `pip install llama-parse` (нужен API ключ) | Облачный сервис, лучшее качество |

**Требования к парсингу:**
- Таблицы → Markdown-таблицы или JSON-объекты
- Если таблица разбивается на несколько чанков — **дублировать заголовок** таблицы в каждый чанк
- Сохранять метаданные: `source_file`, `page_number`, `section_title`

**Пример с DocLing:**
```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("ktj.pdf")
markdown_text = result.document.export_to_markdown()
```

**Пример с Unstructured:**
```python
from unstructured.partition.pdf import partition_pdf

elements = partition_pdf(
    filename="ktj.pdf",
    strategy="hi_res",          # обязательно для таблиц
    infer_table_structure=True,
    include_metadata=True,
)
```

---

#### Шаг 2 — Naive Chunking

```
Размер чанка: 1024 токена
Перекрытие (overlap): 200 токенов
Стратегия: фиксированный размер (Fixed Size)
```

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=200,
    length_function=len,
)
chunks = splitter.split_documents(documents)
```

---

#### Шаг 3 — Создание эмбеддингов и загрузка в векторную БД

**Модели эмбеддингов (мультиязычные):**

| Модель | Размер | Dim | Max Tokens | Примечание |
|--------|--------|-----|------------|------------|
| `intfloat/multilingual-e5-large` | 560M | 1024 | 512 | **Рекомендуется** — SOTA на MIRACL |
| `intfloat/multilingual-e5-base` | 278M | 768 | 512 | Быстрее, чуть слабее |
| `BAAI/bge-m3` | 568M | 1024 | 8192 | Длинный контекст, Dense+Sparse |
| `Alibaba-NLP/gte-multilingual` | 305M | 768 | 8192 | Быстрый, 70+ языков |
| `paraphrase-multilingual-mpnet` | 278M | 768 | 512 | Классика |

> Рекомендация: начните с `intfloat/multilingual-e5-large`.
> Для больших таблиц — попробуйте `BAAI/bge-m3` (8192 токенов).

```python
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
)
```

---

#### Шаг 4 — Dense Retrieval (косинусное сходство)

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},
)
```

---

#### Шаг 5 — Подключение LLM и генерация ответов

```python
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt_template = """Используйте следующий контекст для ответа на вопрос.
Отвечайте только на основе предоставленного контекста.
Если ответа в контексте нет — скажите "Информация не найдена в документах."

Контекст:
{context}

Вопрос: {question}
Ответ:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True,
)
```

---

#### Шаг 6 — Ручное тестирование (5–10 вопросов)

Протестируйте на вопросах из разных категорий и **зафиксируйте ошибки**:

| № | Тип вопроса | Пример | Ожидаемая проблема |
|---|------------|--------|--------------------|
| 1 | Числовой факт | «Доход КТЖ в 2024?» | Таблица разорвана |
| 2 | Точное название | «Что такое Достык – Мойынты?» | Семантический поиск не находит |
| 3 | Сравнение годов | «Чем 2024 лучше 2023?» | Путаются цифры из разных строк |
| 4 | ESG данные | «Выбросы CO2 КТЖ?» | Диаграмма не распарсилась |
| 5 | Дочерняя компания | «Что такое АО Кожан?» | Разный документ |

**Типичные проблемы Naive RAG (зафиксировать в ноутбуке):**
- Таблицы разрываются посередине — цифры из разных строк путаются
- Точные названия (`«Достык – Мойынты»`) не находятся через семантический поиск
- Диаграммы и графики игнорируются
- Потеря контекста при большом документе

---

### Часть B: Advanced RAG (30 баллов)

**Цель:** Устранить проблемы Naive RAG. Каждое улучшение — обосновать и показать разницу.

---

#### Улучшение 1 — Продвинутый чанкинг (сравнить 3 стратегии)

| Стратегия | Описание | Реализация |
|-----------|----------|------------|
| **Naive (Fixed Size)** | 1024 токена, overlap 200 — baseline | `CharacterTextSplitter` |
| **Semantic / Recursive** | По логическим границам: заголовки, концы параграфов | `RecursiveCharacterTextSplitter` |
| **Layout-Aware** | Чанки по визуальным блокам DocLing/Unstructured. Таблица не разрывается. | Из DocLing элементов |

**Recursive Chunking:**
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

recursive_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ".", "!", "?", ",", " "],
    chunk_size=1024,
    chunk_overlap=100,
    length_function=len,
)
```

**Layout-Aware Chunking (DocLing):**
```python
# DocLing сохраняет структуру документа
# Таблицы → отдельные чанки с заголовком
# Параграфы → по секциям

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

converter = DocumentConverter()
result = converter.convert("ktj.pdf")

chunker = HybridChunker(tokenizer="intfloat/multilingual-e5-large")
chunks = list(chunker.chunk(result.document))
```

**Что сравнивать между стратегиями:**
- Сколько чанков содержат целые таблицы vs разорванные
- Качество поиска по 5 тестовым вопросам (до RAGAS)
- Средняя длина чанка

---

#### Улучшение 2 — Hybrid Search (обязательно)

Объединить **Vector Search** (семантика) + **BM25** (ключевые слова) через **Reciprocal Rank Fusion (RRF)**.

> Критично для поиска точных названий (`«Достык – Мойынты»`, `«АО Кожан»`) и точных цифр.

```python
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# BM25 ретривер
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 5

# Vector ретривер
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Ensemble = Hybrid Search с RRF
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5],  # alpha: 0.0=только BM25, 1.0=только Vector
)
```

**RRF формула:**
```
RRF_score(d, R) = Σ 1 / (k + rank_r(d))
где k=60 (константа), rank_r(d) = позиция документа d в ранжировании r
```

---

#### Улучшение 3 — Reranking

Cross-Encoder переранжирует результаты retrieval для лучшей точности.

**Модель:** `BAAI/bge-reranker-v2-m3`

```python
from FlagEmbedding import FlagReranker

reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)

def rerank_documents(query: str, documents: list, top_k: int = 5) -> list:
    """Переранжировать документы с Cross-Encoder"""
    pairs = [[query, doc.page_content] for doc in documents]
    scores = reranker.compute_score(pairs, normalize=True)

    scored_docs = list(zip(scores, documents))
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    return [doc for _, doc in scored_docs[:top_k]]
```

**Что показать:**
- Таблица: топ-5 документов ДО reranking vs ПОСЛЕ
- Пример вопроса где reranking помог (и где не помог)

---

#### Улучшение 4 — Pre-Retrieval техника (одна на выбор)

##### Вариант A: Query Rewriting
```python
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

rewrite_prompt = PromptTemplate(
    template="""Перепишите следующий вопрос для улучшения поиска в базе знаний.
Сделайте вопрос более конкретным и добавьте синонимы ключевых терминов.
Верните только переписанный вопрос.

Исходный вопрос: {question}
Переписанный вопрос:""",
    input_variables=["question"]
)

rewrite_chain = rewrite_prompt | ChatOpenAI(model="gpt-4o-mini")
```

##### Вариант B: HyDE (Hypothetical Document Embeddings)
```python
hyde_prompt = PromptTemplate(
    template="""Напишите гипотетический ответ на следующий вопрос,
как будто он взят из годового отчёта казахстанской компании.
Используйте реалистичные цифры и термины из финансовой отчётности.

Вопрос: {question}
Гипотетический ответ:""",
    input_variables=["question"]
)

# Искать по гипотетическому ответу, а не по вопросу
```

##### Вариант C: Query Routing
```python
routing_prompt = PromptTemplate(
    template="""Определите, из какого источника нужно искать ответ:
- "ktj" — если вопрос про КТЖ, железную дорогу, грузооборот, инфраструктуру
- "matnp" — если вопрос про Матен Петролеум, добычу нефти, месторождения
- "both" — если вопрос касается обоих источников

Вопрос: {question}
Источник (ответь одним словом: ktj/matnp/both):""",
    input_variables=["question"]
)
```

---

## Задание 2

### Часть A: Серия экспериментов (30 баллов)

**Цель:** Исследовать влияние гиперпараметров на качество пайплайна.
**Минимум:** 6 экспериментов. Принцип — **менять один параметр за раз** (greedy search).

#### Гиперпараметры

| Параметр | Что это | Диапазон | Влияние |
|----------|---------|----------|---------|
| **Chunk Size** | Размер чанка в токенах | 256, 512, 1024, 2048 | Маленькие — точнее, но теряют контекст |
| **Chunk Overlap** | Перекрытие между чанками | 0, 50, 100, 200 | Больше overlap — меньше потерь на границах |
| **Стратегия чанкинга** | Способ разбиения | Fixed, Recursive, Layout-Aware | Layout-Aware сохраняет таблицы |
| **Top-K** | Чанков → в LLM | 3, 5, 10, 15, 20 | Больше K — больше контекста, но больше шума |
| **Alpha (RRF)** | Вес Vector vs BM25 | 0.0–1.0 (шаг 0.1–0.2) | 0.0=только BM25, 1.0=только Vector |
| **Reranking** | Переранжирование | Вкл / Выкл | Улучшает точность, добавляет latency |
| **Embedding-модель** | Модель эмбеддингов | e5-large, bge-m3, mpnet и др. | Разное качество для multilingual |

#### Рекомендуемая последовательность экспериментов

| # | Что меняем | Chunk Size/Overlap | Top-K | Alpha | Rerank | Запустить на Golden Dataset |
|---|-----------|-------------------|-------|-------|--------|----------------------------|
| 0 | **Baseline** | 1024 / 200 | 5 | 0.5 | Нет | Да |
| 1 | Chunk Size ↓ | 512 / 100 | 5 | 0.5 | Нет | Да |
| 2 | Chunk Size ↑ | 2048 / 200 | 5 | 0.5 | Нет | Да |
| 3 | Top-K ↓ | best | 3 | 0.5 | Нет | Да |
| 4 | Top-K ↑ | best | 10 | 0.5 | Нет | Да |
| 5 | Alpha = 0.0 | best | best | 0.0 | Нет | Да (только BM25) |
| 6 | Alpha = 0.3 | best | best | 0.3 | Нет | Да |
| 7 | Alpha = 0.7 | best | best | 0.7 | Нет | Да |
| 8 | Alpha = 1.0 | best | best | 1.0 | Нет | Да (только Vector) |
| 9 | Reranking | best | best | best | Да | Да |

> `best` = лучшее значение из предыдущих экспериментов

#### Что фиксировать по каждому эксперименту

```python
experiment_result = {
    "experiment_id": 1,
    "changed_param": "chunk_size",
    "value": 512,
    "config": {
        "chunk_size": 512,
        "chunk_overlap": 100,
        "top_k": 5,
        "alpha": 0.5,
        "reranking": False,
        "embedding_model": "intfloat/multilingual-e5-large",
    },
    "metrics": {
        "faithfulness": 0.0,       # заполнить после RAGAS
        "answer_relevancy": 0.0,
        "context_recall": 0.0,
        "context_precision": 0.0,
    },
    "conclusion": "Опишите: стало лучше или хуже? Почему?"
}
```

#### Как запускать RAGAS на Golden Dataset

```python
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import json

# Загрузить golden dataset
with open("golden_dataset.json", "r", encoding="utf-8") as f:
    golden_data = json.load(f)

def run_rag_on_dataset(qa_chain, golden_data: list) -> dict:
    """Прогнать пайплайн на Golden Dataset и собрать данные для RAGAS"""
    results = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    for item in golden_data:
        question = item["question"]
        ground_truth = item["ground_truth"]

        # Получить ответ и контексты
        response = qa_chain({"query": question})
        answer = response["result"]
        contexts = [doc.page_content for doc in response["source_documents"]]

        results["question"].append(question)
        results["answer"].append(answer)
        results["contexts"].append(contexts)
        results["ground_truth"].append(ground_truth)

    return results

def evaluate_with_ragas(results: dict) -> dict:
    """Оценить результаты с помощью RAGAS"""
    dataset = Dataset.from_dict({
        "user_input": results["question"],
        "response": results["answer"],
        "retrieved_contexts": results["contexts"],
        "reference": results["ground_truth"],
    })

    llm = ChatOpenAI(model="gpt-4o-mini")

    scores = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm,
    )

    return scores
```

---

### Часть B: RAGAS + итоговый вывод (20 баллов)

#### 1. Итоговая таблица метрик

После всех экспериментов собрать в единую таблицу:

| # | Config | Faithfulness | Answer Rel. | Context Recall | Context Prec. |
|---|--------|-------------|-------------|---------------|---------------|
| 0 | Baseline (1024/200, K=5, α=0.5) | ? | ? | ? | ? |
| 1 | Chunk 512/100 | ? | ? | ? | ? |
| ... | ... | ... | ... | ... | ... |
| **Best** | **Лучшая конфигурация** | **?** | **?** | **?** | **?** |

Отметить: максимальное значение по каждой метрике ✅ и минимальное ❌

---

#### 2. Метрики RAGAS — что они означают

| Метрика | Что измеряет | Идеальное значение |
|---------|-------------|-------------------|
| **Faithfulness** | Подтверждены ли утверждения в ответе найденным контекстом? (галлюцинации) | 1.0 |
| **Answer Relevancy** | Отвечает ли ответ на заданный вопрос? | 1.0 |
| **Context Recall** | Найдены ли все нужные факты из эталонного ответа? | 1.0 |
| **Context Precision** | Релевантны ли найденные чанки? (нет ли мусора) | 1.0 |

По каждой метрике написать:
- Числовое значение (среднее по всем вопросам)
- Интерпретация (что это означает для вашего пайплайна)
- 2–3 конкретных примера: вопросы с **высокой** метрикой + вопросы с **низкой** метрикой + объяснение

---

#### 3. Итоговый вывод (Markdown-ячейка в конце ноутбука)

Создать ячейку с заголовком `## Итоговый вывод` и ответить на вопросы:

```markdown
## Итоговый вывод

### Лучшая конфигурация
Chunk Size: ..., Overlap: ..., Top-K: ..., Alpha: ..., Reranking: ..., Embedding: ...
Метрики: Faithfulness=..., Answer Relevancy=..., Context Recall=..., Context Precision=...

### Параметр с наибольшим влиянием
...и почему

### Параметр с наименьшим влиянием
...и почему

### Оптимальное значение Alpha
При Alpha=... результат лучший, потому что...

### Помог ли Reranking?
На каких типах вопросов особенно эффективен...

### Сложные вопросы (остались нерешёнными)
Вопросы, которые даже при лучшей конфигурации давали плохой результат...
Что ещё можно попробовать: ...
```

> **Оценивается качество анализа, а не "правильные" метрики!**
> Низкий Faithfulness — нормально, если вы объяснили почему и что пробовали.

---

## Бонус: GraphRAG

### +30 баллов (необязательно)

**Цель:** Построить Knowledge Graph из тех же PDF и использовать его для ответов на вопросы, которые плохо решаются vector search.

#### Шаг 1 — Развернуть Neo4j

```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 \
  -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest

pip install neo4j langchain-neo4j
```

Веб-интерфейс: http://localhost:7474 (логин: `neo4j`, пароль: `password`)

#### Шаг 2 — Извлечь сущности и связи из документов

Сущности для извлечения:
- **Компании:** КТЖ, Матен Петролеум, АО Кожан, дочерние компании
- **Финансовые показатели:** Доход, EBITDA, Грузооборот, Добыча нефти
- **Проекты:** «Достык – Мойынты», линия в обход Алматы
- **Даты/Годы:** 2024, 2023, 2025 (прогноз)
- **ESG показатели:** Выбросы CO2, несчастные случаи, обучение
- **Месторождения:** Кара-Арна, Восточная Кокарна, Матин, Морское, Каратал

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import json

entity_extraction_prompt = PromptTemplate(
    template="""Извлеките сущности и связи из текста годового отчёта.
Верните JSON в формате:
{{
  "entities": [
    {{"id": "unique_id", "type": "Company|Metric|Project|Person|Location", "name": "название", "properties": {{}}}}
  ],
  "relationships": [
    {{"from": "entity_id", "to": "entity_id", "type": "HAS_METRIC|OWNS|LOCATED_IN|RELATED_TO", "properties": {{}}}}
  ]
}}

Текст:
{text}

JSON:""",
    input_variables=["text"]
)
```

#### Шаг 3 — Загрузить граф в Neo4j

```python
from neo4j import GraphDatabase

class Neo4jLoader:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def create_entity(self, entity: dict):
        with self.driver.session() as session:
            session.run(
                f"MERGE (n:{entity['type']} {{id: $id}}) SET n += $props",
                id=entity["id"],
                props={"name": entity["name"], **entity.get("properties", {})}
            )

    def create_relationship(self, rel: dict):
        with self.driver.session() as session:
            session.run(
                f"""MATCH (a {{id: $from_id}}), (b {{id: $to_id}})
                    MERGE (a)-[r:{rel['type']}]->(b)
                    SET r += $props""",
                from_id=rel["from"],
                to_id=rel["to"],
                props=rel.get("properties", {})
            )

loader = Neo4jLoader("bolt://localhost:7687", "neo4j", "password")
```

#### Шаг 4 — Cypher-запросы для ответов

```cypher
-- Какие инфраструктурные проекты связаны с направлением Достык – Мойынты?
MATCH (p:Project)-[:RELATED_TO]->(r:Route {name: "Достык – Мойынты"})
RETURN p.name, p.status, p.description

-- Какие компании упоминаются в обоих отчётах?
MATCH (c:Company)-[:MENTIONED_IN]->(d1:Document {source: "ktj.pdf"})
MATCH (c)-[:MENTIONED_IN]->(d2:Document {source: "matnp_2024_rus.pdf"})
RETURN c.name

-- ESG-рейтинг КТЖ и экологические проекты
MATCH (ktj:Company {name: "КТЖ"})-[:HAS_METRIC]->(esg:ESGMetric)
MATCH (ktj)-[:IMPLEMENTS]->(proj:Project)-[:TYPE]->(:Category {name: "Экология"})
RETURN esg.name, esg.value, proj.name
```

#### Шаг 5 — Сравнение GraphRAG vs Vector RAG

| Вопрос | Vector RAG ответ | GraphRAG ответ | Победитель |
|--------|-----------------|----------------|------------|
| «Какие инфраструктурные проекты связаны с Достык–Мойынты?» | ... | ... | ? |
| «Какие компании в обоих отчётах?» | ... | ... | ? |
| «Связь ESG-рейтинга КТЖ с экологическими проектами?» | ... | ... | ? |

**Что представить:**
- Код извлечения сущностей и загрузки в Neo4j
- Скриншот графа в Neo4j Browser
- Примеры Cypher-запросов с результатами
- Таблица сравнения
- Вывод: когда GraphRAG лучше, а когда нет

---

## Структура ноутбука

Рекомендуемая структура `rag_pipeline.ipynb`:

```
## 0. Setup & Imports
## 1. Парсинг PDF (DocLing / Unstructured)
   ### 1.1 Парсинг ktj.pdf
   ### 1.2 Парсинг matnp_2024_rus.pdf
   ### 1.3 Пример распарсенных таблиц
## 2. Naive RAG (Часть A — 20 б.)
   ### 2.1 Naive Chunking (1024/200)
   ### 2.2 Embeddings → ChromaDB
   ### 2.3 Dense Retrieval
   ### 2.4 LLM Generation
   ### 2.5 Ручное тестирование + типичные ошибки
## 3. Advanced RAG (Часть B — 30 б.)
   ### 3.1 Сравнение стратегий чанкинга
   ### 3.2 Hybrid Search (BM25 + Vector + RRF)
   ### 3.3 Reranking (bge-reranker-v2-m3)
   ### 3.4 Pre-Retrieval техника (Query Rewriting / HyDE / Routing)
## 4. Golden Dataset
   ### 4.1 Загрузка датасета (30 пар)
## 5. Эксперименты (Часть A — 30 б.)
   ### 5.1 Baseline (Experiment 0)
   ### 5.2 Experiment 1 — Chunk Size
   ### 5.3 Experiment 2 — ...
   ### ... (минимум 6 экспериментов)
## 6. RAGAS Оценка (Часть B — 20 б.)
   ### 6.1 Итоговая таблица метрик
   ### 6.2 Анализ метрик
   ### 6.3 Примеры хороших/плохих ответов
## 7. Итоговый вывод (Markdown ячейка)
## 8. [Бонус] GraphRAG с Neo4j
   ### 8.1 Извлечение сущностей
   ### 8.2 Загрузка в Neo4j
   ### 8.3 Cypher-запросы
   ### 8.4 Сравнение с Vector RAG
```

---

## Чек-листы

### Задание 1: Naive + Advanced RAG ✅

**Naive RAG (20 б.):**
- [ ] PDF распарсены через Visual Layout инструмент (DocLing / Unstructured / LlamaParse)
- [ ] Таблицы → Markdown или JSON
- [ ] Naive chunking: 1024 токена, overlap 200
- [ ] Мультиязычные эмбеддинги (multilingual-e5-large или аналог)
- [ ] Векторная БД (ChromaDB или Qdrant)
- [ ] Dense Retrieval (косинусное сходство)
- [ ] LLM генерация ответов (GPT-4o-mini или аналог)
- [ ] Ручное тестирование 5–10 вопросов с фиксацией ошибок

**Advanced RAG (30 б.):**
- [ ] 3 стратегии чанкинга реализованы и сравнены
- [ ] Hybrid Search (Vector + BM25 + RRF) реализован
- [ ] Reranking (BAAI/bge-reranker-v2-m3) добавлен
- [ ] Результаты до/после reranking показаны
- [ ] Pre-Retrieval техника (одна из трёх) реализована
- [ ] Каждое улучшение обосновано с примерами

### Задание 2: Эксперименты + RAGAS ✅

**Эксперименты (30 б.):**
- [ ] Golden Dataset загружен (30 пар из `golden_dataset.json`)
- [ ] Минимум 6 экспериментов выполнено
- [ ] В каждом эксперименте изменён только один параметр
- [ ] Каждый эксперимент прогнан на Golden Dataset
- [ ] RAGAS метрики записаны для каждого эксперимента
- [ ] Краткий вывод по каждому эксперименту

**RAGAS + вывод (20 б.):**
- [ ] Итоговая таблица всех экспериментов собрана
- [ ] Все 4 метрики проанализированы (числа + интерпретация + примеры)
- [ ] Markdown ячейка «Итоговый вывод» создана
- [ ] Ответы на все 6 вопросов итогового вывода написаны

### Бонус: GraphRAG ✅ (необязательно)

- [ ] Neo4j развёрнут (Docker или Aura Free)
- [ ] Сущности и связи извлечены из обоих PDF
- [ ] Граф загружен в Neo4j
- [ ] Минимум 3 Cypher-запроса с результатами
- [ ] Скриншот / визуализация графа
- [ ] Таблица сравнения GraphRAG vs Vector RAG
- [ ] Вывод: когда какой подход лучше

### Финальная сдача ✅

- [ ] `rag_pipeline.ipynb` — ноутбук с пояснениями, визуализациями, результатами
- [ ] `README.md` — архитектура, схема пайплайна, инструкция запуска
- [ ] Все ячейки запущены (outputs видны без перезапуска)
- [ ] Ключи API не хранятся в коде (использовать `.env` / `os.environ`)
