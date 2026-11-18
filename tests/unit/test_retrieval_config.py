"""Check that every strategy calls the vector store with the right parameters (no database needed)."""

from unittest.mock import MagicMock

from langchain_core.documents import Document
from langchain_postgres.v2.hybrid_search_config import reciprocal_rank_fusion

from rag_lab.retrieval import hybrid, metadata_filter, mmr, score_threshold, similarity
from rag_lab.retrieval.comparison import Strategy, build_strategies, run_strategy
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


def test_build_strategies_lists_every_strategy_once_with_its_score_label():
    strategies = build_strategies(MagicMock(), k=4, category_of=lambda question: "hr")

    assert [s.name for s in strategies] == [
        "similarity", "mmr", "score_threshold (0.4)", "metadata_filter", "hybrid (RRF)"
    ]
    assert [s.score_label for s in strategies] == [
        similarity.SCORE_LABEL, mmr.SCORE_LABEL, score_threshold.SCORE_LABEL,
        metadata_filter.SCORE_LABEL, hybrid.SCORE_LABEL,
    ]


def test_build_strategies_binds_store_k_and_the_category_of_each_question():
    store = MagicMock()
    strategies = build_strategies(store, k=5, category_of=lambda question: f"category of {question}")

    for strategy in strategies:
        strategy.search("q")

    calls = (store.similarity_search_with_score.call_args_list
             + store.max_marginal_relevance_search.call_args_list
             + store.similarity_search_with_relevance_scores.call_args_list)
    assert len(calls) == 5
    assert all(call.args == ("q",) and call.kwargs["k"] == 5 for call in calls)
    store.similarity_search_with_relevance_scores.assert_called_once_with("q", k=5, score_threshold=0.4)
    assert {"category": {"$eq": "category of q"}} in [c.kwargs.get("filter") for c in calls]


def test_run_strategy_measures_time_and_keeps_results():
    run = run_strategy(Strategy("name", "label", lambda question: [(DOC, 0.5)]), "q")
    assert (run.name, run.score_label, run.results) == ("name", "label", [(DOC, 0.5)])
    assert run.duration_ms >= 0
