"""Strategy 1: plain vector similarity search.

How it works:
  1. The question is embedded with the same model that embedded the chunks.
  2. Postgres computes the cosine distance between the question vector and every
     chunk vector (`embedding <=> question_vector`) and returns the k smallest.

Cosine distance = 1 - cosine similarity. It ranges from 0 (same direction, very
similar) to 2 (opposite). LOWER IS BETTER. The absolute value depends on the
embedding model, so compare distances only within one model.
"""

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

SCORE_LABEL = "cosine distance (lower = more similar)"


def similarity_search(vector_store: VectorStore, query: str, k: int = 4) -> list[tuple[Document, float]]:
    return vector_store.similarity_search_with_score(query, k=k)
