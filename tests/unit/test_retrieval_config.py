"""Check that every strategy calls the vector store with the right parameters (no database needed)."""

from unittest.mock import MagicMock

from langchain_core.documents import Document
from langchain_postgres.v2.hybrid_search_config import reciprocal_rank_fusion

from rag_lab.retrieval.comparison import run_strategy
from rag_lab.retrieval.hybrid import hybrid_search
from rag_lab.retrieval.metadata_filter import metadata_filter_search
from rag_lab.retrieval.mmr import mmr_search
from rag_lab.retrieval.score_threshold import score_threshold_search
from rag_lab.retrieval.similarity import similarity_search

DOC = Document(page_content="text")


def test_similarity_search_returns_documents_with_distance():
    store = MagicMock()
    store.similarity_search_with_score.return_value = [(DOC, 0.2)]
    assert similarity_search(store, "q", k=3) == [(DOC, 0.2)]
    store.similarity_search_with_score.assert_called_once_with("q", k=3)


def test_mmr_passes_diversity_parameters_and_has_no_score():
    store = MagicMock()
    store.max_marginal_relevance_search.return_value = [DOC]
    assert mmr_search(store, "q", k=2, fetch_k=10, lambda_mult=0.3) == [(DOC, None)]
    store.max_marginal_relevance_search.assert_called_once_with("q", k=2, fetch_k=10, lambda_mult=0.3)


def test_score_threshold_passes_threshold():
    store = MagicMock()
    score_threshold_search(store, "q", score_threshold=0.7, k=5)
    store.similarity_search_with_relevance_scores.assert_called_once_with("q", k=5, score_threshold=0.7)


def test_metadata_filter_builds_category_filter():
    store = MagicMock()
    metadata_filter_search(store, "q", category="hr")
    store.similarity_search_with_score.assert_called_once_with("q", k=4, filter={"category": {"$eq": "hr"}})


def test_hybrid_uses_reciprocal_rank_fusion_with_fresh_config():
    store = MagicMock()
    hybrid_search(store, "q", k=3, candidates_per_search=7)
    hybrid_search(store, "q", k=3, candidates_per_search=7)

    first_config = store.similarity_search_with_score.call_args_list[0].kwargs["hybrid_search_config"]
    second_config = store.similarity_search_with_score.call_args_list[1].kwargs["hybrid_search_config"]
    assert first_config.fusion_function is reciprocal_rank_fusion
    assert first_config.primary_top_k == first_config.secondary_top_k == 7
    assert first_config is not second_config


def test_run_strategy_measures_time_and_keeps_results():
    run = run_strategy("name", "label", lambda: [(DOC, 0.5)])
    assert run.results == [(DOC, 0.5)]
    assert run.duration_ms >= 0
