"""Build the basic RAG graph:  START -> retrieve_documents -> generate_answer -> END

Building a graph has three steps:
  1. StateGraph(RAGState)  - declare which state the graph manages
  2. add_node(...)         - register the functions that do the work
  3. add_edge(...)         - define the order in which they run
`compile()` checks the graph and returns a Runnable with invoke()/stream().
"""

from functools import partial

from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from rag_lab.langgraph_rag.nodes.generate_answer import generate_answer
from rag_lab.langgraph_rag.nodes.retrieve_documents import retrieve_documents
from rag_lab.langgraph_rag.state import RAGState


def build_rag_graph(retriever: BaseRetriever, llm: BaseChatModel) -> CompiledStateGraph:
    graph = StateGraph(RAGState)

    # partial() binds the dependencies; LangGraph then calls node(state).
    graph.add_node("retrieve_documents", partial(retrieve_documents, retriever=retriever))
    graph.add_node("generate_answer", partial(generate_answer, llm=llm))

    graph.add_edge(START, "retrieve_documents")
    graph.add_edge("retrieve_documents", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile()
