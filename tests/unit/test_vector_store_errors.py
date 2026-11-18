"""Error handling of ensure_table() without a database."""

from unittest.mock import MagicMock

import pytest
from sqlalchemy.exc import ProgrammingError

from rag_lab.config import Settings
from rag_lab.db.connection import build_connection_url
from rag_lab.db.vector_store import ensure_table

SETTINGS = Settings(
    postgres_host="db", postgres_port=5433, postgres_db="rag", postgres_user="user",
    postgres_password="p@ss/word", llm_api_key="sk", embedding_model="e", llm_model="m", judge_model="m",
    chunk_size=500, chunk_overlap=50, vector_table="documents", vector_size=1536, mlflow_tracking_uri="",
)


def engine_raising(message: str) -> MagicMock:
    engine = MagicMock()
    engine.init_vectorstore_table.side_effect = ProgrammingError("CREATE TABLE", {}, Exception(message))
    return engine


def test_ensure_table_returns_true_when_created():
    assert ensure_table(MagicMock(), SETTINGS) is True


def test_ensure_table_returns_false_when_table_already_exists():
    assert ensure_table(engine_raising('relation "documents" already exists'), SETTINGS) is False


def test_ensure_table_reraises_other_database_errors():
    with pytest.raises(ProgrammingError):
        ensure_table(engine_raising("permission denied"), SETTINGS)


def test_connection_url_escapes_special_characters_in_password():
    url = build_connection_url(SETTINGS)
    assert url == "postgresql+asyncpg://user:p%40ss%2Fword@db:5433/rag"
