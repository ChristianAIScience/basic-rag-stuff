"""Idempotent document ingestion into PostgreSQL/pgvector - self-contained template.

Dependencies:  pip install langchain-postgres langchain-text-splitters langchain-core asyncpg sqlalchemy
               (+ an embeddings package, e.g. langchain-openai)

Pipeline:  load -> clean -> split -> deterministic ids -> embed -> upsert

Usage:
    engine = PGEngine.from_connection_string(url=connection_url(...))     # postgresql+asyncpg://
    ensure_table(engine, "documents", vector_size=1536)
    store = PGVectorStore.create_sync(engine=engine, embedding_service=embeddings, table_name="documents",
                                      metadata_columns=METADATA_COLUMNS)
    chunks = ingest_directory(Path("data"), store, embeddings, chunk_size=500, chunk_overlap=50)
Re-running updates the same rows (same file + same chunk position -> same id) instead of duplicating.
"""

import re
import uuid
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote_plus

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED_SUFFIXES = {".txt", ".md"}
# Real SQL columns -> filterable with WHERE; all other metadata goes into a JSON column.
METADATA_COLUMNS = ["source", "category"]


def connection_url(user: str, password: str, host: str, port: int, database: str) -> str:
    """asyncpg works with PGEngine's background event loop on every OS (psycopg async fails on Windows)."""
    return f"postgresql+asyncpg://{user}:{quote_plus(password)}@{host}:{port}/{database}"


def load_documents(data_dir: Path) -> list[Document]:
    """data/hr/policy.md -> source="hr/policy.md", category="hr" (first folder)."""
    documents = []
    for path in sorted(data_dir.rglob("*")):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        relative = path.relative_to(data_dir)
        documents.append(Document(
            page_content=path.read_text(encoding="utf-8"),
            metadata={"source": relative.as_posix(),
                      "category": relative.parts[0] if len(relative.parts) > 1 else "general"},
        ))
    return documents


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)   # trailing whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)   # keep paragraph breaks for the splitter
    return text.strip()


def clean_documents(documents: list[Document]) -> list[Document]:
    return [Document(page_content=clean_text(d.page_content), metadata=d.metadata)
            for d in documents if clean_text(d.page_content)]


def split_documents(documents: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True)
    return splitter.split_documents(documents)


def make_chunk_ids(chunks: list[Document]) -> list[str]:
    """Deterministic: same source + same chunk position -> same uuid on every run."""
    position: dict[str, int] = defaultdict(int)
    ids = []
    for chunk in chunks:
        source = chunk.metadata["source"]
        ids.append(str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}#{position[source]}")))
        position[source] += 1
    return ids


def ensure_table(engine, table_name: str, vector_size: int) -> bool:
    """Create the vector extension + table once. True if created, False if it already existed."""
    from langchain_postgres import Column
    from sqlalchemy.exc import ProgrammingError

    try:
        engine.init_vectorstore_table(table_name=table_name, vector_size=vector_size,
                                      metadata_columns=[Column(name, "TEXT") for name in METADATA_COLUMNS])
        return True
    except ProgrammingError as error:
        if "already exists" in str(error):
            return False
        raise


def ingest_directory(data_dir: Path, store, embeddings: Embeddings, chunk_size: int, chunk_overlap: int) -> list[Document]:
    chunks = split_documents(clean_documents(load_documents(data_dir)), chunk_size, chunk_overlap)
    if not chunks:
        return []
    vectors = embeddings.embed_documents([chunk.page_content for chunk in chunks])  # the visible embedding step
    store.add_embeddings(
        texts=[chunk.page_content for chunk in chunks],
        embeddings=vectors,
        metadatas=[chunk.metadata for chunk in chunks],
        ids=make_chunk_ids(chunks),
    )
    return chunks
