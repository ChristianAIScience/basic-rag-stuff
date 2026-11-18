"""Ask a question with the basic LangChain RAG chain.

Run:  uv run python -m examples.basic_langchain_rag
      uv run python -m examples.basic_langchain_rag "How long is the Glowbeam X2 warranty?"
(Requires: docker compose up -d  and  uv run python -m examples.ingest_documents)
"""

import sys

from rag_lab.config import load_settings
from rag_lab.db.connection import get_engine
from rag_lab.db.vector_store import get_vector_store
from rag_lab.langchain_rag.chain import build_rag_chain
from rag_lab.langchain_rag.retriever import build_retriever
from rag_lab.models import get_chat_model, get_embeddings
from rag_lab.utils.formatting import print_answer

DEFAULT_QUESTION = "How many vacation days do I get, and how many can I carry over?"


def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION
    settings = load_settings()

    vector_store = get_vector_store(get_engine(settings), get_embeddings(settings), settings)
    chain = build_rag_chain(build_retriever(vector_store), get_chat_model(settings))

    print(f"Question: {question}")
    result = chain.invoke(question)
    print_answer(result["answer"], result["sources"])


if __name__ == "__main__":
    main()
