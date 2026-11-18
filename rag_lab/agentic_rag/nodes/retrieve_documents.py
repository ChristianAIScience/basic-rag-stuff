"""Node: search the vector store.   [deterministic - no LLM]

Reads:  search_query (the original question, or a rewritten version on retries)
Writes: documents
"""

from langchain_core.retrievers import BaseRetriever

from rag_lab.agentic_rag.state import AgenticRAGState


def retrieve_documents(state: AgenticRAGState, retriever: BaseRetriever) -> dict:
    return {"documents": retriever.invoke(state["search_query"])}
