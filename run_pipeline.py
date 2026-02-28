#!/usr/bin/env python3
"""Полный RAG Pipeline — запуск всех этапов (с retry, кэшированием, resume)"""

import os, json, warnings, time, re, shutil, pickle, math
from dotenv import load_dotenv
load_dotenv()
assert os.getenv('OPENAI_API_KEY'), 'Создайте файл .env с OPENAI_API_KEY=sk-...'
warnings.filterwarnings('ignore')

from openai import APIConnectionError, RateLimitError, APITimeoutError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from FlagEmbedding import FlagReranker

# === Retry decorator for all OpenAI calls ===
llm_retry = retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=2, min=2, max=120),
    retry=retry_if_exception_type((APIConnectionError, RateLimitError, APITimeoutError)),
    reraise=True,
)

# ============================================================
# ЭТАП 1: ПАРСИНГ PDF (с кэшированием)
# ============================================================
print("=" * 70)
print("ЭТАП 1: ПАРСИНГ PDF")
print("=" * 70)

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

# Кэширование: пропускаем парсинг если markdown-файлы уже существуют
if os.path.exists("parsed_ktj.md") and os.path.exists("parsed_matnp.md"):
    print("Кэш найден — загружаем parsed_ktj.md и parsed_matnp.md")
    with open("parsed_ktj.md", "r", encoding="utf-8") as f:
        markdown_ktj = f.read()
    with open("parsed_matnp.md", "r", encoding="utf-8") as f:
        markdown_matnp = f.read()
    print(f"  KTJ: {len(markdown_ktj)} символов (из кэша)")
    print(f"  Матен: {len(markdown_matnp)} символов (из кэша)")
    # Нужны result_ktj/result_matnp для layout-aware чанкинга — парсим заново если нет pickle
    need_docling_results = True
else:
    converter = DocumentConverter()

    t0 = time.time()
    print("Парсинг ktj.pdf...")
    result_ktj = converter.convert("ktj.pdf")
    markdown_ktj = result_ktj.document.export_to_markdown()
    print(f"  KTJ: {len(markdown_ktj)} символов ({time.time()-t0:.0f}s)")

    t0 = time.time()
    print("Парсинг matnp_2024_rus.pdf...")
    result_matnp = converter.convert("matnp_2024_rus.pdf")
    markdown_matnp = result_matnp.document.export_to_markdown()
    print(f"  Матен: {len(markdown_matnp)} символов ({time.time()-t0:.0f}s)")

    with open("parsed_ktj.md", "w", encoding="utf-8") as f:
        f.write(markdown_ktj)
    with open("parsed_matnp.md", "w", encoding="utf-8") as f:
        f.write(markdown_matnp)
    need_docling_results = False

def count_tables(text):
    return len(re.findall(r'\|.*\|.*\n\|[-: |]+\|', text))

print(f"  Таблиц в КТЖ: {count_tables(markdown_ktj)}")
print(f"  Таблиц в Матен: {count_tables(markdown_matnp)}")

# ============================================================
# ЭТАП 2: ЧАНКИНГ (с кэшированием)
# ============================================================
print("\n" + "=" * 70)
print("ЭТАП 2: ЧАНКИНГ (3 стратегии)")
print("=" * 70)

doc_ktj = Document(page_content=markdown_ktj, metadata={"source_file": "ktj.pdf", "company": "КТЖ"})
doc_matnp = Document(page_content=markdown_matnp, metadata={"source_file": "matnp_2024_rus.pdf", "company": "Матен Петролеум"})

def list_to_chunks(items):
    return [Document(page_content=c["page_content"], metadata=c["metadata"]) for c in items]

if os.path.exists("all_chunks.json"):
    print("Кэш найден — загружаем all_chunks.json")
    with open("all_chunks.json", "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
    fixed_chunks = list_to_chunks(chunks_data["fixed"])
    recursive_chunks = list_to_chunks(chunks_data["recursive"])
    layout_chunks = list_to_chunks(chunks_data["layout"])
    print(f"  Fixed Size: {len(fixed_chunks)} чанков (из кэша)")
    print(f"  Recursive: {len(recursive_chunks)} чанков (из кэша)")
    print(f"  Layout-Aware: {len(layout_chunks)} чанков (из кэша)")
else:
    # 1. Fixed Size
    naive_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200, length_function=len)
    fixed_chunks = naive_splitter.split_documents([doc_ktj, doc_matnp])
    print(f"Fixed Size: {len(fixed_chunks)} чанков")

    # 2. Recursive
    recursive_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", "!", "?", ",", " "],
        chunk_size=1024, chunk_overlap=100, length_function=len,
    )
    recursive_chunks = recursive_splitter.split_documents([doc_ktj, doc_matnp])
    print(f"Recursive: {len(recursive_chunks)} чанков")

    # 3. Layout-Aware (requires docling results)
    if need_docling_results:
        converter = DocumentConverter()
        print("Парсинг для layout-aware чанкинга...")
        result_ktj = converter.convert("ktj.pdf")
        result_matnp = converter.convert("matnp_2024_rus.pdf")

    chunker = HybridChunker(tokenizer="intfloat/multilingual-e5-large")
    layout_chunks = []
    for doc_result, source_file in [(result_ktj, "ktj.pdf"), (result_matnp, "matnp_2024_rus.pdf")]:
        for chunk in chunker.chunk(doc_result.document):
            layout_chunks.append(Document(
                page_content=chunk.text,
                metadata={"source_file": source_file}
            ))
    print(f"Layout-Aware: {len(layout_chunks)} чанков")

    def chunks_to_list(chunks):
        return [{"page_content": c.page_content, "metadata": c.metadata} for c in chunks]

    with open("all_chunks.json", "w", encoding="utf-8") as f:
        json.dump({
            "fixed": chunks_to_list(fixed_chunks),
            "recursive": chunks_to_list(recursive_chunks),
            "layout": chunks_to_list(layout_chunks),
        }, f, ensure_ascii=False)

# Статистика
for name, cs in [("Fixed", fixed_chunks), ("Recursive", recursive_chunks), ("Layout-Aware", layout_chunks)]:
    lengths = [len(c.page_content) for c in cs]
    table_count = sum(1 for c in cs if "|" in c.page_content and "---" in c.page_content)
    print(f"  {name}: средн={sum(lengths)/len(lengths):.0f}, мин={min(lengths)}, макс={max(lengths)}, таблиц={table_count}")

# ============================================================
# ЭТАП 3: ЭМБЕДДИНГИ
# ============================================================
print("\n" + "=" * 70)
print("ЭТАП 3: ЭМБЕДДИНГИ")
print("=" * 70)

t0 = time.time()
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    encode_kwargs={"normalize_embeddings": True},
)
print(f"Модель загружена: intfloat/multilingual-e5-large ({time.time()-t0:.0f}s)")

# ============================================================
# ЭТАП 4: NAIVE RAG — ChromaDB + Тестирование
# ============================================================
print("\n" + "=" * 70)
print("ЭТАП 4: NAIVE RAG — ChromaDB + Тестирование")
print("=" * 70)

if os.path.exists("./chroma_naive"):
    shutil.rmtree("./chroma_naive")

naive_vectorstore = Chroma.from_documents(
    documents=fixed_chunks,
    embedding=embeddings,
    persist_directory="./chroma_naive",
)
print(f"ChromaDB (naive): {naive_vectorstore._collection.count()} чанков")

naive_retriever = naive_vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt_template = """Используйте следующий контекст для ответа на вопрос.
Отвечайте только на основе предоставленного контекста.
Если ответа в контексте нет — скажите "Информация не найдена в документах."

Контекст:
{context}

Вопрос: {question}
Ответ:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Ручное тестирование
test_questions = [
    "Каков был доход от основной деятельности КТЖ в 2024 году?",
    'Что такое проект «Достык – Мойынты»?',
    "На сколько вырос доход КТЖ в 2024 по сравнению с 2023?",
    "Какие стандарты ISO внедрены в КТЖ?",
    "Как называется дочерняя компания Матен Петролеум?",
    "Какой объём добычи нефти был у Матен Петролеум в 2024?",
    "До какого года рассчитана стратегия развития КТЖ?",
    "Сколько средств перечислил Матен Петролеум на благотворительность?",
    "Какую цель по грузообороту ставит КТЖ на 2025 год?",
    "В каком городе базируется Матен Петролеум?",
]

print("\n--- Ручное тестирование Naive RAG ---")
naive_test_results = []
for i, q in enumerate(test_questions):
    docs = naive_retriever.invoke(q)
    context = "\n\n".join([d.page_content for d in docs])
    answer = llm_retry(llm.invoke)(prompt_template.format(context=context, question=q)).content
    naive_test_results.append({"question": q, "answer": answer})
    print(f"\nQ{i+1}: {q}")
    print(f"A: {answer[:200]}")

# ============================================================
# ЭТАП 5: ADVANCED RAG
# ============================================================
print("\n" + "=" * 70)
print("ЭТАП 5: ADVANCED RAG")
print("=" * 70)

# Hybrid Search
if os.path.exists("./chroma_advanced"):
    shutil.rmtree("./chroma_advanced")

advanced_vectorstore = Chroma.from_documents(
    documents=layout_chunks,
    embedding=embeddings,
    persist_directory="./chroma_advanced",
)
vector_retriever = advanced_vectorstore.as_retriever(search_kwargs={"k": 5})

bm25_retriever = BM25Retriever.from_documents(layout_chunks)
bm25_retriever.k = 5

hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5],
)
print("Hybrid Search настроен (BM25 + Vector, alpha=0.5)")

# Reranker
print("Загрузка reranker...")
t0 = time.time()
reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
print(f"Reranker загружен ({time.time()-t0:.0f}s)")

def rerank_documents(query, documents, top_k=5):
    if not documents:
        return []
    pairs = [[query, doc.page_content] for doc in documents]
    scores = reranker.compute_score(pairs, normalize=True)
    if isinstance(scores, float):
        scores = [scores]
    scored_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs[:top_k]]

# Query Rewriting
rewrite_prompt = PromptTemplate(
    template="""Перепишите следующий вопрос для улучшения поиска в базе знаний.
Сделайте вопрос более конкретным и добавьте синонимы ключевых терминов.
Верните только переписанный вопрос.

Исходный вопрос: {question}
Переписанный вопрос:""",
    input_variables=["question"]
)
rewrite_chain = rewrite_prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Демо reranking
demo_q = "Каков был доход от основной деятельности КТЖ в 2024 году?"
retrieved = hybrid_retriever.invoke(demo_q)
reranked = rerank_documents(demo_q, retrieved, top_k=5)
print(f"\nДемо: '{demo_q[:50]}...'")
print(f"  До reranking (топ-1): {retrieved[0].page_content[:80]}...")
print(f"  После reranking (топ-1): {reranked[0].page_content[:80]}...")

# Демо Query Rewriting
for q in ["Доход КТЖ?", "Что про экологию?", "Нефть Матен?"]:
    rewritten = llm_retry(rewrite_chain.invoke)({"question": q}).content
    print(f"  Rewrite: '{q}' -> '{rewritten[:80]}'")

# ============================================================
# ЭТАП 6: ЭКСПЕРИМЕНТЫ + RAGAS (с retry + resume)
# ============================================================
print("\n" + "=" * 70)
print("ЭТАП 6: ЭКСПЕРИМЕНТЫ + RAGAS")
print("=" * 70)

# Загрузка Golden Dataset
with open("golden_dataset.json", "r", encoding="utf-8") as f:
    golden_data = json.load(f)
print(f"Golden Dataset: {len(golden_data)} пар")

def make_chunks(chunk_size=1024, chunk_overlap=200, strategy="fixed"):
    if strategy == "layout":
        return layout_chunks
    seps = ["\n\n", "\n", ".", "!", "?", ",", " "] if strategy == "recursive" else ["\n\n", "\n", " ", ""]
    splitter = RecursiveCharacterTextSplitter(
        separators=seps, chunk_size=chunk_size,
        chunk_overlap=chunk_overlap, length_function=len,
    )
    return splitter.split_documents([doc_ktj, doc_matnp])


def run_experiment(chunks, golden_data, alpha=0.5, top_k=5, use_reranking=False, use_rewrite=False):
    vs = Chroma.from_documents(chunks, embeddings)
    v_ret = vs.as_retriever(search_kwargs={"k": top_k})

    bm25_ret = BM25Retriever.from_documents(chunks)
    bm25_ret.k = top_k

    if alpha == 0.0:
        retriever = bm25_ret
    elif alpha == 1.0:
        retriever = v_ret
    else:
        retriever = EnsembleRetriever(
            retrievers=[bm25_ret, v_ret],
            weights=[1 - alpha, alpha],
        )

    results = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for item in golden_data:
        question = item["question"]
        search_query = question
        if use_rewrite:
            search_query = llm_retry(rewrite_chain.invoke)({"question": question}).content

        retrieved_docs = retriever.invoke(search_query)

        if use_reranking and retrieved_docs:
            final_docs = rerank_documents(question, retrieved_docs, top_k=top_k)
        else:
            final_docs = retrieved_docs[:top_k]

        context = "\n\n".join([d.page_content for d in final_docs])
        answer = llm_retry(llm.invoke)(prompt_template.format(context=context, question=question)).content
        contexts = [d.page_content for d in final_docs]

        results["question"].append(question)
        results["answer"].append(answer)
        results["contexts"].append(contexts)
        results["ground_truth"].append(item["ground_truth"])

    vs.delete_collection()
    return results


def evaluate_with_ragas(results):
    from datasets import Dataset
    from ragas import evaluate, RunConfig
    from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

    dataset = Dataset.from_dict({
        "user_input": results["question"],
        "response": results["answer"],
        "retrieved_contexts": results["contexts"],
        "reference": results["ground_truth"],
    })

    run_config = RunConfig(timeout=300, max_workers=4, max_wait=180)

    scores = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=ChatOpenAI(model="gpt-4o-mini"),
        run_config=run_config,
    )
    df = scores.to_pandas()
    return {
        "faithfulness": float(df["faithfulness"].dropna().mean()) if not df["faithfulness"].dropna().empty else 0.0,
        "answer_relevancy": float(df["answer_relevancy"].dropna().mean()) if not df["answer_relevancy"].dropna().empty else 0.0,
        "context_recall": float(df["context_recall"].dropna().mean()) if not df["context_recall"].dropna().empty else 0.0,
        "context_precision": float(df["context_precision"].dropna().mean()) if not df["context_precision"].dropna().empty else 0.0,
    }


# === Запуск экспериментов ===
experiments = [
    ("0: Baseline (1024/200, K=5, α=0.5)", {"chunk_size": 1024, "chunk_overlap": 200, "strategy": "fixed", "alpha": 0.5, "top_k": 5, "use_reranking": False}),
    ("1: Chunk 512/100", {"chunk_size": 512, "chunk_overlap": 100, "strategy": "fixed", "alpha": 0.5, "top_k": 5, "use_reranking": False}),
    ("2: Chunk 2048/200", {"chunk_size": 2048, "chunk_overlap": 200, "strategy": "fixed", "alpha": 0.5, "top_k": 5, "use_reranking": False}),
    ("3: Top-K=3", {"chunk_size": 1024, "chunk_overlap": 200, "strategy": "fixed", "alpha": 0.5, "top_k": 3, "use_reranking": False}),
    ("4: Top-K=10", {"chunk_size": 1024, "chunk_overlap": 200, "strategy": "fixed", "alpha": 0.5, "top_k": 10, "use_reranking": False}),
    ("5: α=0.0 (BM25 only)", {"chunk_size": 1024, "chunk_overlap": 200, "strategy": "fixed", "alpha": 0.0, "top_k": 5, "use_reranking": False}),
    ("6: α=0.3", {"chunk_size": 1024, "chunk_overlap": 200, "strategy": "fixed", "alpha": 0.3, "top_k": 5, "use_reranking": False}),
    ("7: α=0.7", {"chunk_size": 1024, "chunk_overlap": 200, "strategy": "fixed", "alpha": 0.7, "top_k": 5, "use_reranking": False}),
    ("8: α=1.0 (Vector only)", {"chunk_size": 1024, "chunk_overlap": 200, "strategy": "fixed", "alpha": 1.0, "top_k": 5, "use_reranking": False}),
    ("9: Reranking", {"chunk_size": 1024, "chunk_overlap": 200, "strategy": "fixed", "alpha": 0.5, "top_k": 5, "use_reranking": True}),
]

# Resume: загружаем уже завершённые эксперименты
all_scores = {}
all_results = {}
if os.path.exists("experiment_scores.json"):
    with open("experiment_scores.json", "r", encoding="utf-8") as f:
        all_scores = json.load(f)
    print(f"Resume: найдено {len(all_scores)} завершённых экспериментов")
if os.path.exists("experiment_results.json"):
    with open("experiment_results.json", "r", encoding="utf-8") as f:
        all_results = json.load(f)

for name, config in experiments:
    if name in all_scores:
        print(f"\n--- {name} --- ПРОПУСК (уже завершён)")
        continue

    print(f"\n--- {name} ---")
    t0 = time.time()
    chunks = make_chunks(config["chunk_size"], config["chunk_overlap"], config["strategy"])
    results = run_experiment(
        chunks, golden_data,
        alpha=config["alpha"], top_k=config["top_k"],
        use_reranking=config["use_reranking"],
    )
    scores = evaluate_with_ragas(results)

    # Replace NaN with 0
    for k, v in scores.items():
        if math.isnan(v):
            scores[k] = 0.0

    all_scores[name] = scores
    all_results[name] = results
    elapsed = time.time() - t0
    print(f"  Faithfulness={scores.get('faithfulness', 0):.4f}, "
          f"AnswerRel={scores.get('answer_relevancy', 0):.4f}, "
          f"CtxRecall={scores.get('context_recall', 0):.4f}, "
          f"CtxPrec={scores.get('context_precision', 0):.4f} "
          f"({elapsed:.0f}s)")

    # Partial save after each experiment
    with open("experiment_scores.json", "w", encoding="utf-8") as f:
        json.dump(all_scores, f, ensure_ascii=False, indent=2)
    with open("experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 70)
print("ИТОГОВАЯ ТАБЛИЦА")
print("=" * 70)
print(f"{'Experiment':<40} {'Faith':>8} {'AnsRel':>8} {'CtxRec':>8} {'CtxPrc':>8} {'Avg':>8}")
print("-" * 80)
best_avg = 0
best_name = ""
for name, scores in all_scores.items():
    f = scores.get("faithfulness", 0)
    a = scores.get("answer_relevancy", 0)
    cr = scores.get("context_recall", 0)
    cp = scores.get("context_precision", 0)
    avg = (f + a + cr + cp) / 4
    if avg > best_avg:
        best_avg = avg
        best_name = name
    print(f"{name:<40} {f:>8.4f} {a:>8.4f} {cr:>8.4f} {cp:>8.4f} {avg:>8.4f}")

print(f"\nЛучшая конфигурация: {best_name} (avg={best_avg:.4f})")

# Naive test results
with open("naive_test_results.json", "w", encoding="utf-8") as f:
    json.dump(naive_test_results, f, ensure_ascii=False, indent=2)

print("\n=== ВСЕ ЭТАПЫ ЗАВЕРШЕНЫ ===")
print("Файлы:")
print("  parsed_ktj.md, parsed_matnp.md — распарсенные PDF")
print("  all_chunks.json — все чанки (3 стратегии)")
print("  experiment_scores.json — метрики RAGAS")
print("  experiment_results.json — ответы по всем экспериментам")
print("  naive_test_results.json — результаты ручного тестирования")
