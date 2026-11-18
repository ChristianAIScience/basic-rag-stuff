"""Run the agentic RAG graph and show every decision it makes.

Run:  uv run python -m examples.agentic_rag                 (three demo questions)
      uv run python -m examples.agentic_rag "Your question"
(Requires: docker compose up -d  and  uv run python -m examples.ingest_documents)
"""

import sys

from rag_lab.agentic_rag.graph import build_agentic_graph
from rag_lab.config import load_settings
from rag_lab.db.connection import get_engine
from rag_lab.db.vector_store import get_vector_store
from rag_lab.langchain_rag.retriever import build_retriever
from rag_lab.models import get_chat_model, get_embeddings
from rag_lab.tracking import enable_mlflow

DEMO_QUESTIONS = [
    "What do I do when TunnelKite shows error TK-407?",  # knowledge base -> retrieve and answer
    "What is 12 times 7?",                                # general -> skips retrieval
    "What is Lumora's policy on bringing dogs to the office?",  # not in the docs -> retries, then fallback
]
MAX_RETRIES = 2


def describe(node_name: str, update: dict) -> str:
    """One readable line per node, showing the decision it made."""
    match node_name:
        case "analyze_question":
            return f"needs_retrieval={update['needs_retrieval']}"
        case "retrieve_documents":
            return f"retrieved {len(update['documents'])} chunks"
        case "grade_documents":
            return f"{len(update['relevant_documents'])} chunks graded relevant"
        case "rewrite_question":
            return f"retry {update['retry_count']}: new query {update['search_query']!r}"
        case _:
            return ""


def run(graph, question: str) -> None:
    print(f"\n{'=' * 70}\nQuestion: {question}")
    state: dict = {"question": question, "max_retries": MAX_RETRIES}
    for step in graph.stream(state, stream_mode="updates"):
        for node_name, update in step.items():
            print(f"  -> {node_name:<20} {describe(node_name, update)}")
            state.update(update)

    print(f"\nAnswer: {state['answer']}")
    if state.get("relevant_documents"):
        print("Sources:", sorted({d.metadata["source"] for d in state["relevant_documents"]}))


def main() -> None:
    settings = load_settings()
    enable_mlflow(settings, experiment="rag-examples")  # trace every run (optional)
    vector_store = get_vector_store(get_engine(settings), get_embeddings(settings), settings)
    graph = build_agentic_graph(build_retriever(vector_store), get_chat_model(settings))

    questions = sys.argv[1:] or DEMO_QUESTIONS
    for question in questions:
        run(graph, question)


if __name__ == "__main__":
    main()
