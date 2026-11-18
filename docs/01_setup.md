# 1. Setup & Configuration

## What

- `pyproject.toml` – Python ≥ 3.14, dependencies, pytest configuration.
- `docker-compose.yml` – PostgreSQL 17 with the pgvector extension.
- `.env` – all credentials and settings (copied from `.env.example`, never committed).
- `rag_lab/config.py` – reads `.env` into one `Settings` object.
- `rag_lab/models.py` – creates the OpenAI chat model and embedding model.

## Why

Every module needs the same database credentials and model names. Reading them in one
place (`load_settings()`) avoids hard-coded secrets and makes tests easy: a test can
build its own `Settings` without touching the environment.

## The code

```python
from rag_lab.config import load_settings
from rag_lab.models import get_chat_model, get_embeddings

settings = load_settings()          # raises ValueError if POSTGRES_PASSWORD or LLM_API_KEY is missing
llm = get_chat_model(settings)      # ChatOpenAI(model=LLM_MODEL)
embeddings = get_embeddings(settings)  # OpenAIEmbeddings(model=EMBEDDING_MODEL)
```

| Variable | Default | Meaning |
|---|---|---|
| `POSTGRES_HOST/PORT/DB/USER` | localhost / 5432 / rag_learning / postgres | connection |
| `POSTGRES_PASSWORD` | – (required) | also used by Docker Compose to initialize the DB (only on first start - see below) |
| `LLM_API_KEY` | – (required) | OpenAI API key |
| `EMBEDDING_MODEL` | text-embedding-3-small | 1536-dimensional embeddings |
| `LLM_MODEL` | gpt-5-mini | chat model |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 500 / 50 | text splitting |
| `VECTOR_TABLE` / `VECTOR_SIZE` | documents / 1536 | table name, vector dimension (must match the embedding model) |

**Where the data lives:** the database files are stored in `postgres-data/` in the project folder
(a bind mount, see `docker-compose.yml`). They survive `docker compose down`, deleting the container
and reinstalling Docker. The folder is in `.gitignore`. To start with an empty database:
`docker compose down`, delete `postgres-data/`, `docker compose up -d`.
Stop the container before copying the folder as a backup.

**Changing the password later:** the container uses `POSTGRES_PASSWORD` only when `postgres-data/` is
created. After changing it in `.env`, update the existing database too:

```bash
docker compose exec postgres psql -U postgres -c "ALTER USER postgres PASSWORD 'new_password';"
```

## How it connects

`Settings` flows into `db/connection.py` (connection URL), `models.py` (models),
and the ingestion pipeline (chunk size). Nothing else reads environment variables.

## Run

```bash
cp .env.example .env     # edit POSTGRES_PASSWORD and LLM_API_KEY
docker compose up -d
uv sync
uv run python -m pytest
```

## Expected output

`docker compose up -d` reports the container as started; `pytest` reports all unit tests passed.
Check the database with:

```bash
docker compose exec postgres psql -U postgres -d rag_learning -c "select version();"
```
