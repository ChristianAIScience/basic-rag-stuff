"""Adapters: give the three RAG implementations one common interface for the evaluation.

A target is a function   question -> {"answer": str, "documents": list[Document]}
where "documents" are the chunks the answer was generated from (what groundedness checks).
"""

from collections.abc import Callable

from langchain_core.documents import Document
from langchain_core.runnables import Runnable

from rag_lab.agentic_rag.state import DEFAULT_MAX_RETRIES

Target = Callable[[str], dict]


def langchain_target(chain: Runnable) -> Target:
    def run(question: str) -> dict:
        result = chain.invoke(question)
        return {"answer": result["answer"], "documents": result["sources"]}
    return run


def langgraph_target(graph: Runnable) -> Target:
    def run(question: str) -> dict:
        state = graph.invoke({"question": question})
        return {"answer": state["answer"], "documents": state["documents"]}
    return run


def agentic_target(graph: Runnable, max_retries: int = DEFAULT_MAX_RETRIES) -> Target:
    def run(question: str) -> dict:
        state = graph.invoke({"question": question, "max_retries": max_retries})
        # The agentic graph answers only from the documents it graded as relevant
        # (none when it answered directly or fell back).
        documents: list[Document] = state.get("relevant_documents") or []
        return {"answer": state["answer"], "documents": documents}
    return run
