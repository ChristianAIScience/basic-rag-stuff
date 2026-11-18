"""Read all stored chunks (with their ids) - needed to sample chunks for labelling.

LangChain's vector store has no "list everything" method, so this is one plain SQL query.
"""

import asyncio

import asyncpg
from langchain_core.documents import Document

from rag_lab.config import Settings


async def _fetch_rows(settings: Settings) -> list[asyncpg.Record]:
    connection = await asyncpg.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        user=settings.postgres_user,
        password=settings.postgres_password,
        database=settings.postgres_db,
    )
    try:
        return await connection.fetch(
            f'SELECT langchain_id, content, source, category FROM "{settings.vector_table}" ORDER BY source, langchain_id'
        )
    finally:
        await connection.close()


def fetch_all_chunks(settings: Settings) -> list[Document]:
    """Every chunk as a Document whose `id` is the chunk id stored in the table."""
    rows = asyncio.run(_fetch_rows(settings))
    return [
        Document(
            id=str(row["langchain_id"]),
            page_content=row["content"],
            metadata={"source": row["source"], "category": row["category"]},
        )
        for row in rows
    ]
