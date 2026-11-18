"""LLM-as-a-judge without API calls: the judge LLM is a MagicMock returning scripted grades."""

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from rag_lab.evaluation.dataset import CALIBRATION_CASES, EVAL_EXAMPLES, EvalExample
from rag_lab.evaluation.judges import (
    JUDGE_NAMES,
    JudgeResult,
    judge_correctness,
    judge_groundedness,
    judge_relevance,
    judge_retrieval_relevance,
    run_all_judges,
)
from rag_lab.evaluation.prompts import (
    CorrectnessGrade,
    GroundednessGrade,
    RelevanceGrade,
    RetrievalRelevanceGrade,
)
from rag_lab.evaluation.runner import evaluate_target, pass_rates
from rag_lab.evaluation.targets import agentic_target, langchain_target, langgraph_target

DOC = Document(page_content="Employees get 27 vacation days.", metadata={"source": "hr/vacation.md"})


def judge_llm(grade) -> MagicMock:
    """A fake judge whose structured output always returns `grade`."""
    llm = MagicMock()
    llm.with_structured_output.return_value.invoke.return_value = grade
    return llm


def sent_prompt(llm: MagicMock) -> str:
    return llm.with_structured_output.return_value.invoke.call_args.args[0].to_string()


def test_correctness_judge_sends_question_answer_and_reference():
    llm = judge_llm(CorrectnessGrade(explanation="matches", correct=True))

    result = judge_correctness(llm, "How many days?", "27 days", "27 vacation days")

    assert result == JudgeResult("correctness", True, "matches")
    prompt = sent_prompt(llm)
    assert "QUESTION: How many days?" in prompt
    assert "GROUND TRUTH ANSWER: 27 vacation days" in prompt
    assert "STUDENT ANSWER: 27 days" in prompt


def test_judges_request_strict_structured_output():
    llm = judge_llm(RelevanceGrade(explanation="ok", relevant=True))
    judge_relevance(llm, "q", "a")
    llm.with_structured_output.assert_called_once_with(RelevanceGrade, method="json_schema", strict=True)


def test_groundedness_judge_sees_the_documents_as_facts():
    llm = judge_llm(GroundednessGrade(explanation="invented bonus", grounded=False))

    result = judge_groundedness(llm, "27 days and a bonus", [DOC])

    assert result.passed is False
    assert "[Source: hr/vacation.md]" in sent_prompt(llm)


def test_retrieval_relevance_judge_maps_verdict():
    llm = judge_llm(RetrievalRelevanceGrade(explanation="about vacation", relevant=True))
    assert judge_retrieval_relevance(llm, "vacation?", [DOC]).passed is True


@pytest.mark.parametrize("judge", [judge_groundedness, judge_retrieval_relevance])
def test_document_judges_are_not_applicable_without_documents(judge):
    llm = MagicMock()
    result = judge(llm, "text", [])
    assert result.passed is None
    llm.with_structured_output.assert_not_called()  # no LLM call, no cost


def scripted_judge_llm(grades: dict) -> MagicMock:
    """A fake judge that returns a different scripted grade per schema."""
    llm = MagicMock()
    llm.with_structured_output.side_effect = lambda schema, **kwargs: MagicMock(
        invoke=MagicMock(return_value=grades[schema])
    )
    return llm


def test_run_all_judges_returns_one_result_per_judge_in_fixed_order():
    llm = scripted_judge_llm({
        CorrectnessGrade: CorrectnessGrade(explanation="x", correct=True),
        RelevanceGrade: RelevanceGrade(explanation="x", relevant=True),
        GroundednessGrade: GroundednessGrade(explanation="x", grounded=False),
        RetrievalRelevanceGrade: RetrievalRelevanceGrade(explanation="x", relevant=True),
    })
    results = run_all_judges(llm, "q", "a", [DOC], "reference")
    assert [r.key for r in results] == JUDGE_NAMES
    assert [r.passed for r in results] == [True, True, False, True]


def test_evaluate_target_and_pass_rates_ignore_not_applicable_verdicts():
    examples = [EvalExample("q1", "r1"), EvalExample("q2", "r2")]
    answers = {"q1": {"answer": "a1", "documents": [DOC]}, "q2": {"answer": "a2", "documents": []}}
    llm = scripted_judge_llm({
        CorrectnessGrade: CorrectnessGrade(explanation="x", correct=True),
        RelevanceGrade: RelevanceGrade(explanation="x", relevant=True),
        GroundednessGrade: GroundednessGrade(explanation="x", grounded=True),
        RetrievalRelevanceGrade: RetrievalRelevanceGrade(explanation="x", relevant=True),
    })

    results = evaluate_target(lambda q: answers[q], examples, llm)

    assert [r.answer for r in results] == ["a1", "a2"]
    assert pass_rates(results) == {
        "correctness": (2, 2), "relevance": (2, 2),
        "groundedness": (1, 1), "retrieval_relevance": (1, 1),  # q2 had no documents -> not counted
    }


def test_targets_map_each_implementation_to_answer_and_documents():
    chain = MagicMock(invoke=MagicMock(return_value={"question": "q", "answer": "a", "sources": [DOC]}))
    graph = MagicMock(invoke=MagicMock(return_value={"question": "q", "answer": "a", "documents": [DOC]}))
    agentic = MagicMock(invoke=MagicMock(return_value={"answer": "a", "documents": [DOC], "relevant_documents": []}))

    assert langchain_target(chain)("q") == {"answer": "a", "documents": [DOC]}
    assert langgraph_target(graph)("q") == {"answer": "a", "documents": [DOC]}
    # agentic: only the documents graded relevant were used for the answer
    assert agentic_target(agentic, max_retries=1)("q") == {"answer": "a", "documents": []}
    agentic.invoke.assert_called_once_with({"question": "q", "max_retries": 1})


def test_datasets_are_complete():
    assert all(e.question and e.reference_answer for e in EVAL_EXAMPLES)
    assert {c.judge for c in CALIBRATION_CASES} == set(JUDGE_NAMES)
    # one good and one bad case per judge
    assert sorted(c.expected for c in CALIBRATION_CASES) == [False] * 4 + [True] * 4
