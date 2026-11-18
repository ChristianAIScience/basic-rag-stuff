"""Strategy 3: similarity search with a minimum relevance score.

Plain similarity search ALWAYS returns k chunks - even for a question the
documents can't answer. A threshold drops chunks that are not similar enough,
so "nothing relevant" can become an empty result.

LangChain converts the cosine distance into a relevance score:
    relevance = 1 - cosine distance        (HIGHER IS BETTER, roughly 0..1)

Limitations:
- Scores are not probabilities. 0.4 does not mean "40 % relevant".
- Typical values differ between embedding models. With text-embedding-3-small,
  related texts often score only 0.3-0.6. Tune the threshold on your own data.
"""

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

SCORE_LABEL = "relevance score (1 - cosine distance, higher = better)"


def score_threshold_search(
    vector_store: VectorStore,
    query: str,
    score_threshold: float = 0.4,
    k: int = 4,
) -> list[tuple[Document, float]]:
    return vector_store.similarity_search_with_relevance_scores(
        query, k=k, score_threshold=score_threshold
    )
