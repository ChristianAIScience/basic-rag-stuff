"""Run the whole agentic graph with fakes and check the path it takes."""

from rag_lab.agentic_rag.graph import build_agentic_graph
from rag_lab.agentic_rag.prompts import NO_RELEVANT_DOCUMENTS_ANSWER
from tests.unit.fakes import RELEVANT_DOC, fake_llm, fake_retriever


def run(graph, question: str, max_retries: int = 2) -> tuple[list[str], dict]:
    """Return the visited node names and the final state."""
    visited, state = [], {"question": question, "max_retries": max_retries}
    for step in graph.stream(state, stream_mode="updates"):
        for node_name, update in step.items():
            visited.append(node_name)
            state.update(update)
    return visited, state


def test_relevant_documents_lead_straight_to_an_answer():
    graph = build_agentic_graph(fake_retriever(), fake_llm(relevant=True, text="Renew it."))
    visited, state = run(graph, "TK-407?")

    assert visited == ["analyze_question", "retrieve_documents", "grade_documents", "generate_answer"]
    assert state["answer"] == "Renew it."
    assert RELEVANT_DOC in state["relevant_documents"]


def test_question_without_retrieval_skips_the_vector_store():
    retriever = fake_retriever()
    graph = build_agentic_graph(retriever, fake_llm(needs_retrieval=False, text="84"))
    visited, state = run(graph, "12*7?")

    assert visited == ["analyze_question", "generate_answer"]
    assert state["answer"] == "84"
    retriever.invoke.assert_not_called()


def test_irrelevant_documents_trigger_rewrites_until_max_retries_then_fallback():
    retriever = fake_retriever()
    graph = build_agentic_graph(retriever, fake_llm(relevant=False, text="rewritten query"))
    visited, state = run(graph, "dogs in the office?", max_retries=2)

    assert visited == [
        "analyze_question",
        "retrieve_documents", "grade_documents", "rewrite_question",
        "retrieve_documents", "grade_documents", "rewrite_question",
        "retrieve_documents", "grade_documents",
        "generate_answer",
    ]
    assert state["retry_count"] == 2
    assert retriever.invoke.call_count == 3
    assert state["answer"] == NO_RELEVANT_DOCUMENTS_ANSWER


def test_empty_retrieval_result_also_ends_in_fallback():
    graph = build_agentic_graph(fake_retriever(documents=[]), fake_llm())
    _, state = run(graph, "anything", max_retries=1)
    assert state["answer"] == NO_RELEVANT_DOCUMENTS_ANSWER
