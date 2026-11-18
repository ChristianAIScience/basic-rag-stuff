"""LLM-as-a-judge: a second LLM grades the answers of the RAG system.

Four judges, each checking ONE aspect, each returning a yes/no verdict plus its reasoning:

    judge                 | looks at                       | needs a reference answer?
    ----------------------|--------------------------------|--------------------------
    correctness           | question, answer, reference    | yes
    relevance             | question, answer               | no
    groundedness          | answer, retrieved documents    | no
    retrieval_relevance   | question, retrieved documents  | no

The three reference-free judges can also grade live traffic, where no reference exists.
Judges without documents to look at return passed=None ("not applicable").
"""

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from rag_lab.evaluation.prompts import (
    CORRECTNESS_PROMPT,
    GROUNDEDNESS_PROMPT,
    RELEVANCE_PROMPT,
    RETRIEVAL_RELEVANCE_PROMPT,
    CorrectnessGrade,
    GroundednessGrade,
    RelevanceGrade,
    RetrievalRelevanceGrade,
)
from rag_lab.utils.formatting import format_docs

JUDGE_NAMES = ["correctness", "relevance", "groundedness", "retrieval_relevance"]


@dataclass(frozen=True)
class JudgeResult:
    key: str             # one of JUDGE_NAMES
    passed: bool | None  # None = not applicable (e.g. no documents to check against)
    explanation: str


def _grade(llm: BaseChatModel, prompt: ChatPromptTemplate, schema: type[BaseModel], values: dict) -> BaseModel:
    """One judge call: fill the prompt, force the answer into the schema (strict JSON schema)."""
    grader = llm.with_structured_output(schema, method="json_schema", strict=True)
    return grader.invoke(prompt.invoke(values))


def judge_correctness(llm: BaseChatModel, question: str, answer: str, reference_answer: str) -> JudgeResult:
    grade = _grade(llm, CORRECTNESS_PROMPT, CorrectnessGrade,
                   {"question": question, "answer": answer, "reference_answer": reference_answer})
    return JudgeResult("correctness", grade.correct, grade.explanation)


def judge_relevance(llm: BaseChatModel, question: str, answer: str) -> JudgeResult:
    grade = _grade(llm, RELEVANCE_PROMPT, RelevanceGrade, {"question": question, "answer": answer})
    return JudgeResult("relevance", grade.relevant, grade.explanation)


def judge_groundedness(llm: BaseChatModel, answer: str, documents: list[Document]) -> JudgeResult:
    if not documents:
        return JudgeResult("groundedness", None, "No documents were used for this answer.")
    grade = _grade(llm, GROUNDEDNESS_PROMPT, GroundednessGrade, {"facts": format_docs(documents), "answer": answer})
    return JudgeResult("groundedness", grade.grounded, grade.explanation)


def judge_retrieval_relevance(llm: BaseChatModel, question: str, documents: list[Document]) -> JudgeResult:
    if not documents:
        return JudgeResult("retrieval_relevance", None, "No documents were retrieved.")
    grade = _grade(llm, RETRIEVAL_RELEVANCE_PROMPT, RetrievalRelevanceGrade,
                   {"facts": format_docs(documents), "question": question})
    return JudgeResult("retrieval_relevance", grade.relevant, grade.explanation)


def run_judge(
    name: str, llm: BaseChatModel, question: str, answer: str, documents: list[Document], reference_answer: str
) -> JudgeResult:
    """Run one judge by name. Every judge gets the same inputs and uses only those it grades."""
    match name:
        case "correctness":
            return judge_correctness(llm, question, answer, reference_answer)
        case "relevance":
            return judge_relevance(llm, question, answer)
        case "groundedness":
            return judge_groundedness(llm, answer, documents)
        case "retrieval_relevance":
            return judge_retrieval_relevance(llm, question, documents)
    raise ValueError(f"Unknown judge: {name!r} (known: {', '.join(JUDGE_NAMES)})")


def run_all_judges(
    llm: BaseChatModel, question: str, answer: str, documents: list[Document], reference_answer: str
) -> list[JudgeResult]:
    """All four judges for one answer, in the order of JUDGE_NAMES."""
    return [run_judge(name, llm, question, answer, documents, reference_answer) for name in JUDGE_NAMES]
