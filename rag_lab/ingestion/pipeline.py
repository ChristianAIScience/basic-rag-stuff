"""Step 5: the complete ingestion pipeline.

    load -> clean -> split -> assign ids -> embed -> store

Re-running is safe: every chunk gets a deterministic id derived from its source
file and its position. Storing a chunk whose id already exists updates that row
(an "upsert") instead of creating a duplicate.
"""

import uuid
from collections import defaultdict
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGVectorStore

from rag_lab.ingestion.cleaner import clean_documents
from rag_lab.ingestion.embedder import embed_chunks
from rag_lab.ingestion.loader import load_documents
from rag_lab.ingestion.splitter import split_documents


def make_chunk_ids(chunks: list[Document]) -> list[str]:
    """Same file + same chunk position -> same id, on every run."""
    position_per_source: dict[str, int] = defaultdict(int)
    ids = []
    for chunk in chunks:
        source = chunk.metadata["source"]
        ids.append(str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}#{position_per_source[source]}")))
        position_per_source[source] += 1
    return ids


def ingest_directory(
    data_dir: Path,
    vector_store: PGVectorStore,
    embeddings: Embeddings,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Ingest all .txt/.md files below data_dir. Returns the stored chunks."""
    documents = clean_documents(load_documents(data_dir))
    chunks = split_documents(documents, chunk_size, chunk_overlap)
    if not chunks:
        return []

    vectors = embed_chunks(chunks, embeddings)
    vector_store.add_embeddings(
        texts=[chunk.page_content for chunk in chunks],
        embeddings=vectors,
        metadatas=[chunk.metadata for chunk in chunks],
        ids=make_chunk_ids(chunks),
    )
    return chunks
