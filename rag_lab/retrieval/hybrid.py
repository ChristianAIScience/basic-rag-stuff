"""Strategy 5: hybrid search = keyword search + vector search, merged with RRF.

Vector search understands meaning ("time off" ~ "vacation"), but can miss exact
terms like error codes ("TK-407") or product names. Keyword search is the opposite.
Hybrid search runs both and merges the two ranked lists.

Both searches run inside the same Postgres - no extra search engine:
  - vector search:  ORDER BY embedding <=> :question_vector
  - keyword search: Postgres full-text search
        WHERE to_tsvector('english', content) @@ plainto_tsquery('english', :question)
        ORDER BY ts_rank_cd(...)

Reciprocal Rank Fusion (RRF) merges them using only the RANK in each list:
    rrf_score(chunk) = sum over both lists of 1 / (rank + 60)
A chunk ranked high in both lists wins. RRF ignores the raw scores, which is
why it can combine a distance (vector) with a text rank (keyword) safely.

Note: plainto_tsquery requires ALL words of the question to match, so keyword
search works best with short keyword-like queries.
"""

from langchain_core.documents import Document
from langchain_postgres import PGVectorStore
from langchain_postgres.v2.hybrid_search_config import HybridSearchConfig, reciprocal_rank_fusion

SCORE_LABEL = "RRF score (rank-based, higher = better)"


def hybrid_search(
    vector_store: PGVectorStore,
    query: str,
    k: int = 4,
    candidates_per_search: int = 10,
) -> list[tuple[Document, float]]:
    # A fresh config per call: langchain-postgres stores the query text inside it.
    config = HybridSearchConfig(
        fusion_function=reciprocal_rank_fusion,
        primary_top_k=candidates_per_search,    # vector search candidates
        secondary_top_k=candidates_per_search,  # keyword search candidates
    )
    return vector_store.similarity_search_with_score(query, k=k, hybrid_search_config=config)
