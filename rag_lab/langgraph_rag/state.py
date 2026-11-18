"""The graph state: a typed dict that travels through the graph.

Every node receives the CURRENT state and returns a dict with only the keys it
wants to change. LangGraph merges that update into the state before the next node runs:

    invoke({"question": "..."})
      -> retrieve_documents returns {"documents": [...]}    state: question, documents
      -> generate_answer    returns {"answer": "..."}       state: question, documents, answer
"""

from typing import TypedDict

from langchain_core.documents import Document


class RAGState(TypedDict):
    question: str
    documents: list[Document]
    answer: str
