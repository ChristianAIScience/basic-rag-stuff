import pytest

from rag_lab.db.vector_store import ensure_table
from rag_lab.ingestion.pipeline import ingest_directory
from tests.integration.conftest import DATA_DIR

pytestmark = pytest.mark.integration


def count_rows(vector_store) -> int:
    return len(vector_store.similarity_search("anything", k=1000))


def test_ensure_table_is_idempotent(engine, settings, empty_vector_store):
    assert ensure_table(engine, settings) is False  # table exists -> nothing happens


def test_ingestion_stores_chunks_and_rerun_does_not_duplicate(empty_vector_store, embeddings):
    chunks = ingest_directory(DATA_DIR, empty_vector_store, embeddings, 500, 50)
    assert count_rows(empty_vector_store) == len(chunks) > 0

    ingest_directory(DATA_DIR, empty_vector_store, embeddings, 500, 50)
    assert count_rows(empty_vector_store) == len(chunks)


def test_stored_chunks_keep_metadata_and_can_be_filtered(empty_vector_store, embeddings):
    ingest_directory(DATA_DIR, empty_vector_store, embeddings, 500, 50)

    results = empty_vector_store.similarity_search("vpn", k=50, filter={"category": {"$eq": "hr"}})

    assert results
    assert {doc.metadata["category"] for doc in results} == {"hr"}
    assert all(doc.metadata["source"].startswith("hr/") for doc in results)
