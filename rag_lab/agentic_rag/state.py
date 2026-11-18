"""State of the agentic RAG graph. Every decision the graph makes is visible here.

Input:  {"question": "...", "max_retries": 2}
The analyze_question node initializes the remaining fields.
"""

from typing import TypedDict

from langchain_core.documents import Document

DEFAULT_MAX_RETRIES = 2


class AgenticRAGState(TypedDict, total=False):
    question: str                        # original user question (never changed)
    max_retries: int                     # how often the query may be rewritten
    needs_retrieval: bool                # decided by the LLM in analyze_question
    search_query: str                    # query sent to the retriever (rewritten on retries)
    documents: list[Document]            # result of the latest retrieval
    relevant_documents: list[Document]   # documents the LLM graded as relevant
    retry_count: int                     # number of rewrites done so far
    answer: str
