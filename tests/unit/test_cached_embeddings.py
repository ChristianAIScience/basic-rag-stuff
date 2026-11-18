import asyncio

from langchain_core.embeddings import DeterministicFakeEmbedding

from rag_lab.utils.cached_embeddings import QueryCachedEmbeddings


class CountingEmbeddings(DeterministicFakeEmbedding):
    calls: int = 0

    def embed_query(self, text: str) -> list[float]:
        self.calls += 1
        return super().embed_query(text)


def test_each_distinct_query_is_embedded_once_sync_and_async():
    inner = CountingEmbeddings(size=4)
    cached = QueryCachedEmbeddings(inner)

    first = cached.embed_query("vacation?")
    assert cached.embed_query("vacation?") == first
    assert asyncio.run(cached.aembed_query("vacation?")) == first   # the vector store's search path
    cached.embed_query("vpn?")

    assert inner.calls == 2


def test_documents_are_passed_through():
    cached = QueryCachedEmbeddings(DeterministicFakeEmbedding(size=4))
    assert len(cached.embed_documents(["a", "b"])) == 2
