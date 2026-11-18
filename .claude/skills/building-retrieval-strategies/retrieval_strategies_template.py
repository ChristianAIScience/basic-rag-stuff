"""Five retrieval strategies for a langchain-postgres PGVectorStore - self-contained template.

Dependencies:  pip install langchain-postgres langchain-core asyncpg

Every function returns [(Document, score | None), ...]. Scores of different strategies mean
different things - always print them with their SCORE_LABELS entry, never compare across strategies.

Usage:
    run = run_strategy("similarity", lambda: similarity_search(store, "What is error TK-407?"))
    print(run.name, run.duration_ms, [(d.metadata["source"], s) for d, s in run.results])
"""

import time
from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

SCORE_LABELS = {
    "similarity": "cosine distance (lower = more similar)",
    "mmr": "no score (MMR re-ranks candidates)",
    "score_threshold": "relevance score (1 - cosine distance, higher = better)",
    "metadata_filter": "cosine distance (lower = more similar)",
    "hybrid": "RRF score (rank-based, higher = better)",
}

Results = list[tuple[Document, float | None]]


def similarity_search(store: VectorStore, query: str, k: int = 4) -> Results:
    """k nearest vectors: ORDER BY embedding <=> query_vector."""
    return store.similarity_search_with_score(query, k=k)


def mmr_search(store: VectorStore, query: str, k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5) -> Results:
    """Fetch fetch_k candidates, then pick k balancing relevance (lambda_mult=1) and diversity (0)."""
    documents = store.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult)
    return [(document, None) for document in documents]


def score_threshold_search(store: VectorStore, query: str, score_threshold: float = 0.4, k: int = 4) -> Results:
    """Drop results below the relevance threshold - can return nothing. Tune the threshold per embedding model."""
    return store.similarity_search_with_relevance_scores(query, k=k, score_threshold=score_threshold)


def metadata_filter_search(store: VectorStore, query: str, category: str, k: int = 4) -> Results:
    """SQL WHERE on a real metadata column before ranking (PGVectorStore metadata_columns)."""
    return store.similarity_search_with_score(query, k=k, filter={"category": {"$eq": category}})


def hybrid_search(store, query: str, k: int = 4, candidates_per_search: int = 10) -> Results:
    """Postgres full-text search + vector search merged with Reciprocal Rank Fusion (1 / (rank + 60)).

    A FRESH HybridSearchConfig per call: langchain-postgres stores the query text inside it.
    plainto_tsquery needs EVERY word to match - keep keyword queries short.
    """
    from langchain_postgres.v2.hybrid_search_config import HybridSearchConfig, reciprocal_rank_fusion

    config = HybridSearchConfig(
        fusion_function=reciprocal_rank_fusion,
        primary_top_k=candidates_per_search,
        secondary_top_k=candidates_per_search,
    )
    return store.similarity_search_with_score(query, k=k, hybrid_search_config=config)


@dataclass
class StrategyRun:
    name: str
    score_label: str
    results: Results
    duration_ms: float


def run_strategy(name: str, search: Callable[[], Results]) -> StrategyRun:
    start = time.perf_counter()
    results = search()
    return StrategyRun(name, SCORE_LABELS.get(name, ""), results, (time.perf_counter() - start) * 1000)
