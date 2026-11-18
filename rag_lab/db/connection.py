"""Database connection: builds the connection URL and the PGEngine (a connection pool).

We use the `asyncpg` driver. PGEngine runs its queries on a background asyncio
event loop; asyncpg works with that loop on every OS, including Windows.
"""

from urllib.parse import quote_plus

from langchain_postgres import PGEngine

from rag_lab.config import Settings


def build_connection_url(settings: Settings) -> str:
    # quote_plus protects passwords that contain special characters like "@" or "/".
    password = quote_plus(settings.postgres_password)
    return (
        f"postgresql+asyncpg://{settings.postgres_user}:{password}"
        f"@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
    )


def get_engine(settings: Settings) -> PGEngine:
    return PGEngine.from_connection_string(url=build_connection_url(settings))
