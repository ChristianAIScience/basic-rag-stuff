"""Hit@k/MRR plumbing against the real database (fake embeddings, no API calls)."""

import pytest

from rag_lab.db.chunks import fetch_all_chunks
from rag_lab.evaluation.question_generation import RetrievalExample, check_dataset, content_hash, sample_chunks
from rag_lab.evaluation.retrieval_metrics import evaluate_retriever
from rag_lab.ingestion.pipeline import ingest_directory
from rag_lab.retrieval.similarity import similarity_search
from tests.integration.conftest import DATA_DIR

pytestmark = pytest.mark.integration


@pytest.fixture
def stored_chunks(empty_vector_store, embeddings, settings):
    chunks = ingest_directory(DATA_DIR, empty_vector_store, embeddings, 500, 50)
    return chunks, fetch_all_chunks(settings)


def test_fetch_all_chunks_returns_every_stored_chunk_with_its_id(stored_chunks, empty_vector_store):
    ingested, fetched = stored_chunks
    assert len(fetched) == len(ingested)
    some_id = fetched[0].id
    assert empty_vector_store.get_by_ids([some_id])[0].page_content == fetched[0].page_content


def test_question_equal_to_chunk_text_is_found_at_rank_one(stored_chunks, empty_vector_store):
    # DeterministicFakeEmbedding maps identical text to an identical vector (distance 0),
    # so using the chunk text as the question must rank the gold chunk first.
    _, fetched = stored_chunks
    examples = [
        RetrievalExample(chunk.page_content, chunk.id, chunk.metadata["source"], chunk.metadata["category"])
        for chunk in sample_chunks(fetched, 10, seed=42)
    ]

    scores = evaluate_retriever(
        lambda q: [doc.id for doc, _ in similarity_search(empty_vector_store, q, k=5)], examples
    )

    assert scores.hit_at[1] == 1.0
    assert scores.mrr == 1.0


def test_reingesting_with_other_chunk_size_is_detected_as_stale(stored_chunks, empty_vector_store, embeddings, settings):
    _, fetched = stored_chunks
    examples = [RetrievalExample(f"question {i}", c.id, c.metadata["source"], c.metadata["category"],
                                 content_hash(c.page_content))
                for i, c in enumerate(sample_chunks(fetched, 10, seed=42))]
    assert check_dataset(examples, fetched) == []

    # Same files, other chunk size: chunk ids are reused (uuid5 of source#position) for different text.
    ingest_directory(DATA_DIR, empty_vector_store, embeddings, 300, 30)
    refetched = fetch_all_chunks(settings)

    assert {e.chunk_id for e in examples} <= {c.id for c in refetched}   # ids alone would look fine ...
    assert any("different text" in p for p in check_dataset(examples, refetched))  # ... the hash catches it
