"""Strategy 2: Maximum Marginal Relevance (MMR).

Problem with plain similarity search: the top-k chunks are often near-duplicates
(e.g. three overlapping chunks of the same paragraph). They waste prompt space.

MMR fixes this:
  1. Fetch `fetch_k` candidates by similarity (e.g. 20).
  2. Pick k of them one by one. Each pick maximizes
         lambda_mult * similarity(question, chunk)
       - (1 - lambda_mult) * max similarity(chunk, already picked chunks)

lambda_mult = 1.0 -> pure relevance (same as similarity search)
lambda_mult = 0.0 -> pure diversity
MMR re-ranks the candidates, so it returns no meaningful single score.
"""

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

SCORE_LABEL = "no score (MMR re-ranks candidates)"


def mmr_search(
    vector_store: VectorStore,
    query: str,
    k: int = 4,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
) -> list[tuple[Document, None]]:
    documents = vector_store.max_marginal_relevance_search(
        query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
    )
    return [(document, None) for document in documents]
