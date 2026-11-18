"""Ask a question with the basic LangGraph RAG workflow.

Run:  uv run python -m examples.basic_langgraph_rag
      uv run python -m examples.basic_langgraph_rag "How long is the Glowbeam X2 warranty?"
(Requires: docker compose up -d  and  uv run python -m examples.ingest_documents)

Unlike the LangChain example, this one also shows HOW the state changes after each node.
"""

import sys

from rag_lab.config import load_settings
from rag_lab.db.connection import get_engine
from rag_lab.db.vector_store import get_vector_store
from rag_lab.langchain_rag.retriever import build_retriever
from rag_lab.langgraph_rag.graph import build_rag_graph
from rag_lab.models import get_chat_model, get_embeddings
from rag_lab.utils.formatting import print_answer

DEFAULT_QUESTION = "How many vacation days do I get, and how many can I carry over?"


def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION
    settings = load_settings()

    vector_store = get_vector_store(get_engine(settings), get_embeddings(settings), settings)
    graph = build_rag_graph(build_retriever(vector_store), get_chat_model(settings))

    print("Graph structure (Mermaid):")
    print(graph.get_graph().draw_mermaid())

    print(f"Question: {question}\n")
    # stream_mode="updates" yields {node_name: state_update} after every node.
    final_state = {"question": question}
    for step in graph.stream({"question": question}, stream_mode="updates"):
        for node_name, update in step.items():
            print(f"-> node '{node_name}' updated state keys: {list(update)}")
            final_state.update(update)

    print_answer(final_state["answer"], final_state["documents"])


if __name__ == "__main__":
    main()
