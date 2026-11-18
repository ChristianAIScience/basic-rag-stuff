"""Embeddings wrapper that embeds each distinct query only once.

Comparing five retrieval strategies on the same question would otherwise call the
embedding API five times for identical text. The vector store's async search path uses
`aembed_query`, whose default implementation calls `embed_query` - so caching here covers both.
"""

from langchain_core.embeddings import Embeddings


class QueryCachedEmbeddings(Embeddings):
    def __init__(self, inner: Embeddings) -> None:
        self.inner = inner
        self._query_cache: dict[str, list[float]] = {}

    def embed_query(self, text: str) -> list[float]:
        if text not in self._query_cache:
            self._query_cache[text] = self.inner.embed_query(text)
        return self._query_cache[text]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.inner.embed_documents(texts)
