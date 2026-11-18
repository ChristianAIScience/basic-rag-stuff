"""Build the agentic RAG graph.

    START -> analyze_question
               |-- no retrieval needed --------------------------> generate_answer -> END
               '-- retrieval needed --> retrieve_documents
                                          -> grade_documents
                                               |-- relevant docs found ---------> generate_answer
                                               |-- retries exhausted -----------> generate_answer (fallback)
                                               '-- nothing relevant -> rewrite_question
                                                                         -> retrieve_documents (loop)

Normal edges (add_edge) always go to the same node.
Conditional edges (add_conditional_edges) call a routing function that returns the next node name.
"""

from functools import partial

from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from rag_lab.agentic_rag.nodes.analyze_question import analyze_question
from rag_lab.agentic_rag.nodes.generate_answer import generate_answer
from rag_lab.agentic_rag.nodes.grade_documents import grade_documents
from rag_lab.agentic_rag.nodes.retrieve_documents import retrieve_documents
from rag_lab.agentic_rag.nodes.rewrite_question import rewrite_question
from rag_lab.agentic_rag.routing import route_after_analysis, route_after_grading
from rag_lab.agentic_rag.state import AgenticRAGState


def build_agentic_graph(retriever: BaseRetriever, llm: BaseChatModel) -> CompiledStateGraph:
    graph = StateGraph(AgenticRAGState)

    graph.add_node("analyze_question", partial(analyze_question, llm=llm))
    graph.add_node("retrieve_documents", partial(retrieve_documents, retriever=retriever))
    graph.add_node("grade_documents", partial(grade_documents, llm=llm))
    graph.add_node("rewrite_question", partial(rewrite_question, llm=llm))
    graph.add_node("generate_answer", partial(generate_answer, llm=llm))

    graph.add_edge(START, "analyze_question")
    graph.add_conditional_edges("analyze_question", route_after_analysis)
    graph.add_edge("retrieve_documents", "grade_documents")
    graph.add_conditional_edges("grade_documents", route_after_grading)
    graph.add_edge("rewrite_question", "retrieve_documents")
    graph.add_edge("generate_answer", END)

    return graph.compile()
