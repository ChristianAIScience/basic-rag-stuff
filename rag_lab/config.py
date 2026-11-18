"""Central configuration, loaded from environment variables (and the .env file).

Every other module asks `load_settings()` for its configuration instead of
reading os.environ directly. That keeps credentials in one place and makes
the configuration easy to replace in tests.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    postgres_host: str
    postgres_port: int
    postgres_db: str
    postgres_user: str
    postgres_password: str
    llm_api_key: str
    embedding_model: str
    llm_model: str
    judge_model: str
    chunk_size: int
    chunk_overlap: int
    vector_table: str
    vector_size: int
    mlflow_tracking_uri: str  # empty = MLflow disabled


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name} (see .env.example)")
    return value


def load_settings() -> Settings:
    """Read settings from the environment. Values in .env never override real env vars."""
    load_dotenv()

    llm_model = os.getenv("LLM_MODEL", "gpt-5-mini")
    settings = Settings(
        postgres_host=os.getenv("POSTGRES_HOST", "localhost"),
        postgres_port=int(os.getenv("POSTGRES_PORT", "5432")),
        postgres_db=os.getenv("POSTGRES_DB", "rag_learning"),
        postgres_user=os.getenv("POSTGRES_USER", "postgres"),
        postgres_password=_require("POSTGRES_PASSWORD"),
        llm_api_key=_require("LLM_API_KEY"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        llm_model=llm_model,
        judge_model=os.getenv("JUDGE_MODEL", llm_model),  # the LLM that grades answers (LLM-as-a-judge)
        chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
        vector_table=os.getenv("VECTOR_TABLE", "documents"),
        vector_size=int(os.getenv("VECTOR_SIZE", "1536")),
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", ""),
    )

    if settings.chunk_overlap >= settings.chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
    return settings
