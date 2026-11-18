"""Node: keep only documents that are relevant to the question.   [LLM decision]

Reads:  question, documents
Writes: relevant_documents

Vector search always returns the k closest chunks - even if none of them is
useful. The LLM grades each chunk with a yes/no decision, one call per chunk.
Grading uses the ORIGINAL question, not the rewritten search query.
"""

from langchain_core.language_models import BaseChatModel

from rag_lab.agentic_rag.prompts import GRADE_PROMPT, RelevanceGrade
from rag_lab.agentic_rag.state import AgenticRAGState


def grade_documents(state: AgenticRAGState, llm: BaseChatModel) -> dict:
    grader = llm.with_structured_output(RelevanceGrade)
    relevant_documents = []
    for document in state["documents"]:
        messages = GRADE_PROMPT.invoke(
            {"question": state["question"], "document": document.page_content}
        )
        grade: RelevanceGrade = grader.invoke(messages)
        if grade.is_relevant:
            relevant_documents.append(document)
    return {"relevant_documents": relevant_documents}
