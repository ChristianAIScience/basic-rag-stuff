"""Shared fixtures for integration tests: a real Postgres, but fake embeddings.

Needs:  docker compose up -d
No OpenAI key is required - DeterministicFakeEmbedding produces stable random vectors.
The tests use their own table, so your real `documents` table is never touched.
"""

import dataclasses
from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from rag_lab.config import load_settings
from rag_lab.db.connection import get_engine
from rag_lab.db.vector_store import METADATA_COLUMNS, get_vector_store
from langchain_postgres import Column

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
TEST_VECTOR_SIZE = 32


@pytest.fixture(scope="session")
def settings():
    return dataclasses.replace(load_settings(), vector_table="test_documents", vector_size=TEST_VECTOR_SIZE)


@pytest.fixture(scope="session")
def engine(settings):
    return get_engine(settings)


@pytest.fixture
def embeddings():
    return DeterministicFakeEmbedding(size=TEST_VECTOR_SIZE)


@pytest.fixture
def empty_vector_store(engine, embeddings, settings):
    """A fresh, empty test table for every test."""
    engine.init_vectorstore_table(
        table_name=settings.vector_table,
        vector_size=settings.vector_size,
        metadata_columns=[Column(name, "TEXT") for name in METADATA_COLUMNS],
        overwrite_existing=True,
    )
    return get_vector_store(engine, embeddings, settings)
