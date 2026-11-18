# Quickstart

From zero to a first RAG answer in about ten minutes. Every step ends with a check, so you know it worked
before you continue. The chapters after this one explain *why* everything works the way it does.

## Requirements

| Tool | Used for | Check |
|---|---|---|
| [uv](https://docs.astral.sh/uv/) | Python 3.14 and all dependencies | `uv --version` |
| [Docker](https://www.docker.com/) | PostgreSQL with pgvector | `docker info` |
| OpenAI API key | embeddings and answers (optional for steps 1-4) | – |
| [Node.js](https://nodejs.org/) ≥ 20 | only for this documentation website | `node --version` |

## 1. Get the code

```bash
git clone https://github.com/ChristianAIScience/basic-rag-stuff.git
cd basic-rag-stuff
```

## 2. Configure

```bash
cp .env.example .env
```

Open `.env` and set at least:

```dotenv
POSTGRES_PASSWORD=choose_a_password
LLM_API_KEY=sk-...            # your OpenAI key (a placeholder is fine until step 5)
```

If another PostgreSQL already runs on your computer, also change `POSTGRES_PORT` (e.g. to `5433`).

## 3. Start the database and install Python

```bash
docker compose up -d
uv sync
```

**Check:** `docker compose ps` shows the `postgres` container as `healthy`, and

```bash
uv run python -m pytest
```

ends with `... passed, ... deselected`. These unit tests need neither the database nor an API key
(see [Testing](09_testing.md)).

## 4. Load the example documents

Without an OpenAI key, use fake embeddings - storage, metadata and keyword search work, similarity search does not:

```bash
uv run python -m examples.ingest_documents --fake-embeddings
```

With a real key in `.env`:

```bash
uv run python -m examples.ingest_documents
```

**Check:** the output ends with `Stored 25 chunks from 7 files`. Running it again gives the same number -
existing chunks are updated, not duplicated.

## 5. Ask questions (needs a real OpenAI key)

```bash
uv run python -m examples.basic_langchain_rag "How many vacation days do I get?"
uv run python -m examples.basic_langgraph_rag "How long is the Glowbeam X2 warranty?"
uv run python -m examples.agentic_rag
uv run python -m examples.compare_retrievers
```

**Check:** the answers quote facts from `data/` (e.g. *27 vacation days*) and list their source files.
The sample company "Lumora Systems" is fictional, so a correct answer can only come from retrieval.

If you used `--fake-embeddings` in step 4, run step 4 again without the flag first.

## 6. Optional: PDF chunking and this website

```bash
uv run python -m examples.pdf_chunking         # no database or API key needed

cd website
npm install
npm start                                      # http://localhost:3000
```

## Where to go next

| Goal | Chapter |
|---|---|
| Understand the configuration | [1. Setup & Configuration](01_setup.md) |
| See how documents become vectors | [2. Document Ingestion](02_ingestion.md) |
| Compare search methods | [3. Retrieval Strategies](03_retrieval.md) |
| Build the RAG pipeline | [4. LangChain](04_langchain_rag.md), [5. LangGraph](05_langgraph_rag.md), [6. Agentic RAG](06_agentic_rag.md) |
| Understand and write tests | [9. Testing](09_testing.md) |

## Common problems

| Symptom | Fix |
|---|---|
| `Missing required environment variable: POSTGRES_PASSWORD` | `.env` is missing or the value is empty (step 2) |
| `Nothing accepts connections` / `connection refused` | Docker is not running, or `POSTGRES_PORT` in `.env` differs from the container |
| `password authentication failed` after changing the password | the container keeps its first password - see [1. Setup](01_setup.md) |
| `AuthenticationError` from OpenAI | `LLM_API_KEY` is still the placeholder |
| Answers are unrelated to the question | the data was ingested with `--fake-embeddings`; ingest again without it |
