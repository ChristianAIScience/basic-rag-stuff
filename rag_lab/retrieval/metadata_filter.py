"""Strategy 4: similarity search restricted by metadata.

`source` and `category` are real table columns (see db/vector_store.py), so the
filter becomes a SQL WHERE clause that runs BEFORE ranking:

    WHERE category = 'hr' ORDER BY embedding <=> :question_vector LIMIT k

Filter syntax (MongoDB-like): {"category": {"$eq": "hr"}}
Other operators: $ne, $in, $nin, $gt, $lt, $like, $and, $or, ...
"""

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

SCORE_LABEL = "cosine distance (lower = more similar)"


def metadata_filter_search(
    vector_store: VectorStore,
    query: str,
    category: str,
    k: int = 4,
) -> list[tuple[Document, float]]:
    return vector_store.similarity_search_with_score(
        query, k=k, filter={"category": {"$eq": category}}
    )
