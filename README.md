# RAG Learning Lab — LangChain & LangGraph

A small, runnable project for learning Retrieval-Augmented Generation (RAG).
All examples share one PostgreSQL + pgvector database and one configuration.

| Area | What you learn | Run |
|---|---|---|
| Ingestion | load → clean → split → embed → store | `uv run python -m examples.ingest_documents` |
| PDF chunking | split PDFs into chapters/sections by font size (pdfplumber) | `uv run python -m examples.pdf_chunking` |
| Retrieval strategies | similarity, MMR, score threshold, metadata filter, hybrid | `uv run python -m examples.compare_retrievers` |
| LangChain RAG | a fixed pipeline built with LCEL | `uv run python -m examples.basic_langchain_rag` |
| LangGraph RAG | the same pipeline as a stateful graph | `uv run python -m examples.basic_langgraph_rag` |
| Agentic RAG | a graph that decides, grades, rewrites and retries | `uv run python -m examples.agentic_rag` |

The sample documents in `data/` describe the fictional company **Lumora Systems**.
Because the facts are made up, a correct answer *must* come from retrieval — the LLM can't know them.

## Quick start

Requirements: [uv](https://docs.astral.sh/uv/), Docker, an OpenAI API key.

```bash
cp .env.example .env              # then set POSTGRES_PASSWORD and LLM_API_KEY
docker compose up -d              # PostgreSQL 17 + pgvector
uv sync                           # installs Python 3.14 and all dependencies
uv run python -m examples.ingest_documents
uv run python -m examples.basic_langchain_rag
```

Tests:

```bash
uv run python -m pytest                  # unit tests: offline, no API key, no database
uv run python -m pytest -m integration   # needs `docker compose up -d`; OpenAI tests skip without a real key
```

## Documentation website

The explanations in `docs/` can also be read as a website (React, [Docusaurus](https://docusaurus.io)),
with chapter navigation, table of contents, readable URLs, dark mode and rendered diagrams.
It reads `docs/*.md` directly - edit the Markdown files, not copies. Requires Node.js >= 20.

```bash
cd website
npm install
npm start          # development server with live reload: http://localhost:3000
npm run build      # static site in website/build/ (npm run serve to preview it)
```

**German version:** the language switch (top right) shows the site in German under `/de/`.
The translations live in `website/i18n/de/`: the chapters in `docusaurus-plugin-content-docs/current/`
(same file names as `docs/`), the start page and navigation texts in the `*.json` files.
When you change a chapter in `docs/`, update its German copy as well - a missing translation falls
back to the English text.
`npm start` serves one language at a time: use `npm start -- --locale de` for German.
The switch between both languages works in the built site (`npm run build` + `npm run serve`).

## Learning path

Read the docs in order — each one explains what a component does, why it exists,
how it connects to the rest, how to run it and what output to expect.

1. [Setup & configuration](docs/01_setup.md)
2. [Document ingestion & how embeddings are stored](docs/02_ingestion.md)
3. [Retrieval strategies](docs/03_retrieval.md)
4. [LangChain RAG chain](docs/04_langchain_rag.md)
5. [LangGraph RAG workflow](docs/05_langgraph_rag.md)
6. [Agentic RAG with LangGraph](docs/06_agentic_rag.md)
7. [Comparison: LangChain vs. LangGraph, deterministic vs. agentic](docs/07_comparison.md)
8. [PDF chunking by font size](docs/08_pdf_chunking.md)

## Project layout

```
rag_lab/
  config.py            settings from .env
  models.py            OpenAI chat model + embeddings factories
  db/                  connection (PGEngine) and vector store table
  ingestion/           loader, pdf_loader (font-size chunking), cleaner, splitter, embedder, pipeline
  retrieval/           one module per retrieval strategy + comparison helper
  langchain_rag/       retriever, prompt, chain
  langgraph_rag/       state, nodes/, graph
  agentic_rag/         state, prompts, nodes/, routing, graph
  utils/formatting.py  shared formatting helpers
examples/              runnable entry points (python -m examples.<name>)
tests/unit/            fast tests with fakes
tests/integration/     tests against the Docker database
data/                  sample knowledge base (hr/, products/, it_support/, handbook/*.pdf)
postgres-data/         PostgreSQL database files (created by Docker, not in git)
website/               documentation website (Docusaurus/React), reads docs/
.claude/skills/        Claude Code skills: one per implementation area (how to extend it, pitfalls)
.claude/skill-evals/   test scenarios (3 per skill) for checking the skills with Claude
```
