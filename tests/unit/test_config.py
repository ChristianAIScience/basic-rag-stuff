import pytest

from rag_lab import config
from rag_lab.config import load_settings


@pytest.fixture(autouse=True)
def no_dotenv(monkeypatch):
    """Make sure a developer's real .env file never leaks into these tests."""
    monkeypatch.setattr(config, "load_dotenv", lambda: None)


@pytest.fixture
def minimal_env(monkeypatch):
    for name in ["POSTGRES_HOST", "POSTGRES_PORT", "CHUNK_SIZE", "CHUNK_OVERLAP", "LLM_MODEL"]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("POSTGRES_PASSWORD", "secret")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")


def test_defaults_are_applied(minimal_env):
    settings = load_settings()
    assert settings.postgres_host == "localhost"
    assert settings.postgres_port == 5432
    assert settings.chunk_size == 500
    assert settings.chunk_overlap == 50


def test_values_are_read_from_environment(minimal_env, monkeypatch):
    monkeypatch.setenv("POSTGRES_PORT", "6543")
    monkeypatch.setenv("CHUNK_SIZE", "800")
    settings = load_settings()
    assert settings.postgres_port == 6543
    assert settings.chunk_size == 800


@pytest.mark.parametrize("missing", ["POSTGRES_PASSWORD", "LLM_API_KEY"])
def test_missing_required_variable_raises(minimal_env, monkeypatch, missing):
    monkeypatch.delenv(missing)
    with pytest.raises(ValueError, match=missing):
        load_settings()


def test_overlap_must_be_smaller_than_chunk_size(minimal_env, monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "100")
    monkeypatch.setenv("CHUNK_OVERLAP", "100")
    with pytest.raises(ValueError, match="CHUNK_OVERLAP"):
        load_settings()
