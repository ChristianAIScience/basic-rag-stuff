from unittest.mock import MagicMock

from rag_lab.agentic_rag.nodes.analyze_question import analyze_question
from rag_lab.agentic_rag.nodes.generate_answer import generate_answer
from rag_lab.agentic_rag.nodes.grade_documents import grade_documents
from rag_lab.agentic_rag.nodes.retrieve_documents import retrieve_documents
from rag_lab.agentic_rag.nodes.rewrite_question import rewrite_question
from rag_lab.agentic_rag.prompts import NO_RELEVANT_DOCUMENTS_ANSWER, RelevanceGrade
from tests.unit.fakes import OTHER_DOC, RELEVANT_DOC, fake_llm, fake_retriever


def test_analyze_question_stores_decision_and_initializes_state():
    update = analyze_question({"question": "TK-407?"}, llm=fake_llm(needs_retrieval=True))
    assert update == {
        "needs_retrieval": True,
        "search_query": "TK-407?",
        "retry_count": 0,
        "relevant_documents": [],
    }


def test_retrieve_documents_uses_the_search_query_not_the_question():
    retriever = fake_retriever()
    retrieve_documents({"question": "original", "search_query": "rewritten"}, retriever=retriever)
    retriever.invoke.assert_called_once_with("rewritten")


def test_grade_documents_keeps_only_relevant_documents():
    grader = MagicMock()
    grader.invoke.side_effect = [RelevanceGrade(is_relevant=True), RelevanceGrade(is_relevant=False)]
    llm = MagicMock()
    llm.with_structured_output.return_value = grader

    update = grade_documents({"question": "q", "documents": [RELEVANT_DOC, OTHER_DOC]}, llm=llm)

    assert update == {"relevant_documents": [RELEVANT_DOC]}
    assert grader.invoke.call_count == 2  # one LLM call per document


def test_rewrite_question_sets_new_query_and_counts_retry():
    update = rewrite_question(
        {"question": "q", "search_query": "q", "retry_count": 1}, llm=fake_llm(text="  better query \n")
    )
    assert update == {"search_query": "better query", "retry_count": 2}


def test_generate_answer_without_retrieval_answers_directly():
    update = generate_answer({"question": "12*7?", "needs_retrieval": False}, llm=fake_llm(text="84"))
    assert update == {"answer": "84"}


def test_generate_answer_falls_back_without_calling_llm_when_nothing_relevant():
    llm = fake_llm()
    update = generate_answer({"question": "q", "needs_retrieval": True, "relevant_documents": []}, llm=llm)
    assert update == {"answer": NO_RELEVANT_DOCUMENTS_ANSWER}
    llm.invoke.assert_not_called()


def test_generate_answer_uses_only_relevant_documents_as_context():
    llm = fake_llm(text="Renew the certificate.")
    state = {"question": "TK-407?", "needs_retrieval": True, "relevant_documents": [RELEVANT_DOC]}

    update = generate_answer(state, llm=llm)

    assert update == {"answer": "Renew the certificate."}
    prompt_text = llm.invoke.call_args.args[0].to_string()
    assert RELEVANT_DOC.page_content in prompt_text
    assert OTHER_DOC.page_content not in prompt_text
