"""Conditional edges: plain Python functions that pick the next node.

They contain NO LLM calls - they only read decisions already stored in the state.
That makes the control flow predictable and easy to unit test.
"""

from typing import Literal

from rag_lab.agentic_rag.state import DEFAULT_MAX_RETRIES, AgenticRAGState


def route_after_analysis(state: AgenticRAGState) -> Literal["retrieve_documents", "generate_answer"]:
    """Skip retrieval for questions that don't need the knowledge base."""
    if state["needs_retrieval"]:
        return "retrieve_documents"
    return "generate_answer"


def route_after_grading(state: AgenticRAGState) -> Literal["generate_answer", "rewrite_question"]:
    """Answer if something relevant was found; otherwise retry - but only max_retries times."""
    if state["relevant_documents"]:
        return "generate_answer"
    if state["retry_count"] >= state.get("max_retries", DEFAULT_MAX_RETRIES):
        return "generate_answer"  # gives up -> generate_answer returns the fallback message
    return "rewrite_question"
