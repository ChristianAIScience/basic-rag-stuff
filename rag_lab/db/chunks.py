"""Read all stored chunks (with their ids) - needed to sample chunks for labelling.

LangChain's vector store has no "list everything" method, so this is one plain SQL query.
It uses the same connection URL as the rest of the project (rag_lab/db/connection.py), and
SQLAlchemy quotes the table name. ORDER BY makes the row order - and therefore a seeded
random sample - independent of how Postgres happens to store the rows.
"""

import asyncio

from langchain_core.documents import Document
from sqlalchemy import column, select, table
from sqlalchemy.ext.asyncio import create_async_engine

from rag_lab.config import Settings
from rag_lab.db.connection import build_connection_url

COLUMNS = ["langchain_id", "content", "source", "category"]


def chunks_query(table_name: str):
    return (
        select(*[column(name) for name in COLUMNS])
        .select_from(table(table_name))
        .order_by(column("source"), column("langchain_id"))
    )


async def _fetch_rows(settings: Settings) -> list:
    engine = create_async_engine(build_connection_url(settings))
    try:
        async with engine.connect() as connection:
            result = await connection.execute(chunks_query(settings.vector_table))
            return list(result.mappings())
    finally:
        await engine.dispose()


def fetch_all_chunks(settings: Settings) -> list[Document]:
    """Every chunk as a Document whose `id` is the chunk id stored in the table."""
    return [
        Document(
            id=str(row["langchain_id"]),
            page_content=row["content"],
            metadata={"source": row["source"], "category": row["category"]},
        )
        for row in asyncio.run(_fetch_rows(settings))
    ]
