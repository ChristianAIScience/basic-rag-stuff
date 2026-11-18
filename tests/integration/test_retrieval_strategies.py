"""Run every strategy against a real Postgres (fake embeddings, so rankings are arbitrary)."""

import pytest

from rag_lab.ingestion.pipeline import ingest_directory
from rag_lab.retrieval.hybrid import hybrid_search
from rag_lab.retrieval.metadata_filter import metadata_filter_search
from rag_lab.retrieval.mmr import mmr_search
from rag_lab.retrieval.score_threshold import score_threshold_search
from rag_lab.retrieval.similarity import similarity_search
from tests.integration.conftest import DATA_DIR

pytestmark = pytest.mark.integration


@pytest.fixture
def store(empty_vector_store, embeddings):
    ingest_directory(DATA_DIR, empty_vector_store, embeddings, 500, 50)
    return empty_vector_store


def test_similarity_returns_k_results_sorted_by_distance(store):
    results = similarity_search(store, "battery", k=4)
    distances = [score for _, score in results]
    assert len(results) == 4
    assert distances == sorted(distances)


def test_mmr_returns_k_distinct_chunks(store):
    results = mmr_search(store, "battery", k=4)
    assert len({doc.page_content for doc, _ in results}) == 4


def test_score_threshold_drops_everything_above_max_relevance(store):
    assert score_threshold_search(store, "battery", score_threshold=1.01) == []


def test_metadata_filter_only_returns_requested_category(store):
    results = metadata_filter_search(store, "battery", category="products")
    assert results
    assert {doc.metadata["category"] for doc, _ in results} == {"products"}


def test_hybrid_ranks_exact_keyword_match_first(store):
    # Fake embeddings rank randomly; only the keyword search can find "TK-407".
    # 50 candidates > number of chunks, so every chunk gets a vector rank.
    results = hybrid_search(store, "TK-407", k=4, candidates_per_search=50)
    top_document, _ = results[0]
    assert "TK-407" in top_document.page_content
