"""Step 4 of ingestion: turn chunk texts into embedding vectors.

An embedding is a list of floats (1536 of them for text-embedding-3-small).
The model is trained so that texts with similar meaning get vectors that point in
similar directions. Later, the question is embedded with the SAME model and
Postgres finds the chunks whose vectors are closest to it.

The vector store could embed for us inside `add_documents()`, but doing it here
makes this step visible.
"""

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


def embed_chunks(chunks: list[Document], embeddings: Embeddings) -> list[list[float]]:
    # One batched API call (internally split into batches) instead of one call per chunk.
    return embeddings.embed_documents([chunk.page_content for chunk in chunks])
