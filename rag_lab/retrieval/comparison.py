"""All retrieval strategies in one list, plus running one with a timer and printing its results.

`build_strategies()` is the single place that lists the strategies and their settings.
examples.compare_retrievers and examples.evaluate_retrieval both run this list, so a new
strategy added here appears in the side-by-side comparison AND in the Hit@k/MRR evaluation.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from rag_lab.retrieval import hybrid, metadata_filter, mmr, score_threshold, similarity

Results = list[tuple[Document, float | None]]


@dataclass(frozen=True)
class Strategy:
    name: str
    score_label: str                   # what the score means - never compare scores across strategies
    search: Callable[[str], Results]   # question -> ranked (document, score) pairs, best first


def build_strategies(vector_store: VectorStore, k: int, category_of: Callable[[str], str]) -> list[Strategy]:
    """The five strategies, each bound to the same vector store and k.

    category_of(question) gives the category for the metadata filter.
    """
    return [
        Strategy("similarity", similarity.SCORE_LABEL,
                 lambda q: similarity.similarity_search(vector_store, q, k=k)),
        Strategy("mmr", mmr.SCORE_LABEL,
                 lambda q: mmr.mmr_search(vector_store, q, k=k)),
        Strategy("score_threshold (0.4)", score_threshold.SCORE_LABEL,
                 lambda q: score_threshold.score_threshold_search(vector_store, q, score_threshold=0.4, k=k)),
        Strategy("metadata_filter", metadata_filter.SCORE_LABEL,
                 lambda q: metadata_filter.metadata_filter_search(vector_store, q, category_of(q), k=k)),
        Strategy("hybrid (RRF)", hybrid.SCORE_LABEL,
                 lambda q: hybrid.hybrid_search(vector_store, q, k=k)),
    ]


@dataclass
class StrategyRun:
    name: str
    score_label: str
    results: Results
    duration_ms: float


def run_strategy(strategy: Strategy, question: str) -> StrategyRun:
    start = time.perf_counter()
    results = strategy.search(question)
    duration_ms = (time.perf_counter() - start) * 1000
    return StrategyRun(strategy.name, strategy.score_label, results, duration_ms)


def print_run(run: StrategyRun, preview_chars: int = 90) -> None:
    print(f"\n=== {run.name}  ({len(run.results)} chunks, {run.duration_ms:.0f} ms)")
    print(f"    score: {run.score_label}")
    if not run.results:
        print("    (no results)")
    for document, score in run.results:
        score_text = "   -  " if score is None else f"{score:6.3f}"
        preview = document.page_content.replace("\n", " ")[:preview_chars]
        print(f"    {score_text}  [{document.metadata.get('source')}]  {preview}...")
