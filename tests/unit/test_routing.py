import pytest

from rag_lab.agentic_rag.routing import route_after_analysis, route_after_grading
from tests.unit.fakes import RELEVANT_DOC


def test_analysis_routes_to_retrieval_when_needed():
    assert route_after_analysis({"needs_retrieval": True}) == "retrieve_documents"


def test_analysis_skips_retrieval_when_not_needed():
    assert route_after_analysis({"needs_retrieval": False}) == "generate_answer"


def test_grading_routes_to_answer_when_relevant_documents_exist():
    state = {"relevant_documents": [RELEVANT_DOC], "retry_count": 0, "max_retries": 2}
    assert route_after_grading(state) == "generate_answer"


@pytest.mark.parametrize("retry_count", [0, 1])
def test_grading_routes_to_rewrite_while_retries_are_left(retry_count):
    state = {"relevant_documents": [], "retry_count": retry_count, "max_retries": 2}
    assert route_after_grading(state) == "rewrite_question"


@pytest.mark.parametrize("retry_count", [2, 3])
def test_grading_stops_retrying_at_max_retries(retry_count):
    state = {"relevant_documents": [], "retry_count": retry_count, "max_retries": 2}
    assert route_after_grading(state) == "generate_answer"


def test_max_retries_zero_never_rewrites():
    state = {"relevant_documents": [], "retry_count": 0, "max_retries": 0}
    assert route_after_grading(state) == "generate_answer"


def test_default_max_retries_is_used_when_not_given():
    assert route_after_grading({"relevant_documents": [], "retry_count": 2}) == "generate_answer"
