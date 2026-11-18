"""Load the example documents from data/ into PostgreSQL.

Run:  uv run python -m examples.ingest_documents
      uv run python -m examples.ingest_documents --fake-embeddings   (no OpenAI key needed)
Safe to run repeatedly - existing chunks are updated, not duplicated.

--fake-embeddings stores random-but-stable vectors instead of real ones. Table, storage,
metadata and keyword search work, but similarity search returns arbitrary results.
Running again WITHOUT the flag overwrites the same rows with real embeddings.
"""

import sys
from pathlib import Path

from langchain_core.embeddings import DeterministicFakeEmbedding

from rag_lab.config import load_settings
from rag_lab.db.connection import get_engine
from rag_lab.db.vector_store import ensure_table, get_vector_store
from rag_lab.ingestion.pipeline import ingest_directory
from rag_lab.models import get_embeddings

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    settings = load_settings()
    engine = get_engine(settings)

    created = ensure_table(engine, settings)
    print(f"Table '{settings.vector_table}': {'created' if created else 'already exists'}")

    if "--fake-embeddings" in sys.argv:
        print("Using FAKE embeddings - similarity search results will be meaningless.")
        embeddings = DeterministicFakeEmbedding(size=settings.vector_size)
    else:
        embeddings = get_embeddings(settings)
    vector_store = get_vector_store(engine, embeddings, settings)

    chunks = ingest_directory(
        DATA_DIR, vector_store, embeddings, settings.chunk_size, settings.chunk_overlap
    )

    sources = sorted({chunk.metadata["source"] for chunk in chunks})
    print(f"Stored {len(chunks)} chunks from {len(sources)} files "
          f"(chunk_size={settings.chunk_size}, chunk_overlap={settings.chunk_overlap}):")
    for source in sources:
        count = sum(1 for chunk in chunks if chunk.metadata["source"] == source)
        print(f"  {source:<32} {count} chunks")

    sample_vector = embeddings.embed_query("How many vacation days do I get?")
    print(f"\nAn embedding has {len(sample_vector)} dimensions, e.g. {[round(float(x), 4) for x in sample_vector[:5]]} ...")


if __name__ == "__main__":
    main()
