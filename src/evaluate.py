"""
evaluate.py
-----------
Benchmark harness that measures retrieval + answer relevance against a
hand-labeled set of queries (benchmark/benchmark_queries.json).

Two complementary signals are combined, since keyword matching alone
is brittle and pure embedding similarity alone can be too lenient:

1. Keyword coverage   - fraction of expected keywords for a query that
                         appear somewhere in the retrieved chunks.
2. Semantic similarity - cosine similarity (via the same embedding model
                         used for indexing) between retrieved chunks and
                         a short reference answer, using the maximum
                         similarity across the top-k chunks per query.

A query counts as "relevant" if EITHER signal clears its threshold.
The overall relevance score is the percentage of benchmark queries
that clear this bar -- this is the number reported as the project's
">90% relevance on benchmark queries" metric.

This is intentionally a lightweight, explainable evaluation rather
than an LLM-as-judge approach, so results are fully reproducible.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from src import config
from src.rag_pipeline import RAGPipeline
from src.vector_store import get_embedding_model

KEYWORD_COVERAGE_THRESHOLD = 0.6   # >=60% of expected keywords present
SIMILARITY_THRESHOLD = 0.45        # cosine similarity of normalized embeddings


@dataclass
class QueryResult:
    question: str
    keyword_coverage: float
    max_similarity: float
    is_relevant: bool
    top_source: str


def _load_benchmark(path: Path = config.BENCHMARK_PATH) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _keyword_coverage(text: str, keywords: List[str]) -> float:
    if not keywords:
        return 0.0
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text_lower)
    return hits / len(keywords)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def run_benchmark(pipeline: RAGPipeline, benchmark_path: Path = config.BENCHMARK_PATH):
    """
    Run every benchmark query through the retriever, score relevance,
    and print a summary report. Returns the list of per-query results
    plus the overall relevance percentage.
    """
    benchmark = _load_benchmark(benchmark_path)
    embed_model = get_embedding_model()

    results: List[QueryResult] = []

    for item in benchmark:
        question = item["question"]
        keywords = item.get("expected_keywords", [])
        reference_answer = item.get("reference_answer", "")

        retrieved_docs = pipeline.retrieve(question)
        combined_text = "\n".join(d.page_content for d in retrieved_docs)

        kw_score = _keyword_coverage(combined_text, keywords)

        max_sim = 0.0
        if reference_answer:
            ref_vec = np.array(embed_model.embed_query(reference_answer))
            for doc in retrieved_docs:
                chunk_vec = np.array(embed_model.embed_query(doc.page_content))
                sim = _cosine_sim(ref_vec, chunk_vec)
                max_sim = max(max_sim, sim)

        is_relevant = (
            kw_score >= KEYWORD_COVERAGE_THRESHOLD
            or max_sim >= SIMILARITY_THRESHOLD
        )

        top_source = (
            retrieved_docs[0].metadata.get("source", "unknown")
            if retrieved_docs else "none"
        )

        results.append(
            QueryResult(
                question=question,
                keyword_coverage=round(kw_score, 3),
                max_similarity=round(max_sim, 3),
                is_relevant=is_relevant,
                top_source=top_source,
            )
        )

    relevance_pct = 100 * sum(r.is_relevant for r in results) / len(results)

    _print_report(results, relevance_pct)
    return results, relevance_pct


def _print_report(results: List[QueryResult], relevance_pct: float) -> None:
    print(f"{'Question':<55} {'KW Cov.':>8} {'Sim.':>7} {'Relevant?':>10}")
    print("-" * 85)
    for r in results:
        q_display = (r.question[:52] + "...") if len(r.question) > 52 else r.question
        print(f"{q_display:<55} {r.keyword_coverage:>8.2f} {r.max_similarity:>7.2f} "
              f"{'YES' if r.is_relevant else 'no':>10}")
    print("-" * 85)
    print(f"Overall relevance across {len(results)} benchmark queries: "
          f"{relevance_pct:.1f}%")


if __name__ == "__main__":
    from src.rag_pipeline import build_pipeline_from_index

    pipeline = build_pipeline_from_index()
    run_benchmark(pipeline)
