"""Node 1: look up chunks relevant to the question.

Reads:  state["question"]
Writes: state["documents"]

The retriever is passed in as an argument (bound with functools.partial in
graph.py), so the node is a plain function that is easy to test with a fake.
"""

from langchain_core.retrievers import BaseRetriever

from rag_lab.langgraph_rag.state import RAGState


def retrieve_documents(state: RAGState, retriever: BaseRetriever) -> dict:
    documents = retriever.invoke(state["question"])
    return {"documents": documents}
