"""Basic LangGraph RAG workflow (retrieve -> generate) - self-contained template.

Dependencies:  pip install langgraph langchain-core  (+ a chat model package)

Conventions: typed state, nodes are plain functions returning only the keys they change,
dependencies bound with functools.partial, the graph function only wires.

Usage:
    graph = build_rag_graph(retriever, llm)
    final_state = graph.invoke({"question": "How many vacation days do I get?"})
    for step in graph.stream({"question": "..."}, stream_mode="updates"):   # watch the state flow
        print(step)                                                          # {node_name: update}
"""

from functools import partial
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "Use ONLY the context below. If it does not contain the answer, say you don't know.\n\n"
                   "Context:\n{context}"),
        ("human", "{question}"),
    ]
)


class RAGState(TypedDict):
    question: str
    documents: list[Document]
    answer: str


def format_docs(documents: list[Document]) -> str:
    return "\n\n".join(f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in documents)


def retrieve_documents(state: RAGState, retriever) -> dict:
    """Reads: question. Writes: documents."""
    return {"documents": retriever.invoke(state["question"])}


def generate_answer(state: RAGState, llm) -> dict:
    """Reads: question, documents. Writes: answer. Explicit calls (no | pipe) work with MagicMock LLMs."""
    messages = RAG_PROMPT.invoke({"context": format_docs(state["documents"]), "question": state["question"]})
    return {"answer": llm.invoke(messages).text}


def build_rag_graph(retriever, llm):
    graph = StateGraph(RAGState)
    graph.add_node("retrieve_documents", partial(retrieve_documents, retriever=retriever))
    graph.add_node("generate_answer", partial(generate_answer, llm=llm))
    graph.add_edge(START, "retrieve_documents")
    graph.add_edge("retrieve_documents", "generate_answer")
    graph.add_edge("generate_answer", END)
    return graph.compile()


def visited_nodes(graph, state: dict) -> list[str]:
    """Node names in execution order - the basis for graph-path tests."""
    return [node for step in graph.stream(state, stream_mode="updates") for node in step]
