"""Vector database operations on PostgreSQL + pgvector.

The table created here looks like this:

    langchain_id        UUID PRIMARY KEY   -- chunk id (deterministic, see ingestion/pipeline.py)
    content             TEXT               -- the chunk text
    embedding           vector(1536)       -- pgvector column holding the embedding
    source              TEXT               -- metadata column, e.g. "hr/vacation_policy.md"
    category            TEXT               -- metadata column, e.g. "hr"
    langchain_metadata  JSON               -- all other metadata

A similarity search embeds the question and asks Postgres for the rows with the
smallest cosine distance:  ORDER BY embedding <=> :question_vector LIMIT k
"""

from langchain_core.embeddings import Embeddings
from langchain_postgres import Column, PGEngine, PGVectorStore
from sqlalchemy.exc import ProgrammingError

from rag_lab.config import Settings

# Metadata stored in real columns, so filters become plain SQL WHERE clauses.
METADATA_COLUMNS = ["source", "category"]


def ensure_table(engine: PGEngine, settings: Settings) -> bool:
    """Create the pgvector extension and the table if they don't exist.

    Returns True if the table was created, False if it already existed.
    Existing data is never dropped.
    """
    try:
        engine.init_vectorstore_table(
            table_name=settings.vector_table,
            vector_size=settings.vector_size,
            metadata_columns=[Column(name, "TEXT") for name in METADATA_COLUMNS],
        )
        return True
    except ProgrammingError as error:
        if "already exists" in str(error):
            return False
        raise


def get_vector_store(engine: PGEngine, embeddings: Embeddings, settings: Settings) -> PGVectorStore:
    """Connect a LangChain vector store to the existing table."""
    return PGVectorStore.create_sync(
        engine=engine,
        embedding_service=embeddings,
        table_name=settings.vector_table,
        metadata_columns=METADATA_COLUMNS,
    )
