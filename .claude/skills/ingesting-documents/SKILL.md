---
name: ingesting-documents
description: Loads, cleans, splits, embeds and stores documents in the pgvector table through the rag_lab ingestion pipeline. Use when changing chunking, adding a file type, re-ingesting with real or fake embeddings, or debugging duplicate or missing chunks.
---

# Ingesting documents into pgvector

Reference implementation, one module per step:
`rag_lab/ingestion/loader.py` → `cleaner.py` → `splitter.py` → `pipeline.py` (chunk ids) → `embedder.py` → `rag_lab/db/vector_store.py`.
Entry point: `uv run python -m examples.ingest_documents [--fake-embeddings]`. Learner-facing explanation: `docs/02_ingestion.md`.

## Change or extend the pipeline

Run the unit tests after every edit. Copy this checklist:

```
- [ ] 1. Edit the one module that owns the step (new file type -> dispatch in loader.py)
- [ ] 2. Add a unit test in tests/unit/test_ingestion.py (tmp_path, no database)
- [ ] 3. uv run python -m pytest            -> fix and repeat until green
- [ ] 4. uv run python -m pytest -m integration   (Docker running)
- [ ] 5. Run examples.ingest_documents --fake-embeddings twice; row count must not change
```

Every new document type sets the metadata keys `source` (path relative to `data/`, forward slashes) and `category` (first folder name).
PDFs are split by headings before the text splitter - see the `chunking-pdfs-by-font-size` skill.

## Decisions the code does not explain

- **Idempotency**: chunk id = `uuid5(NAMESPACE_URL, f"{source}#{position}")`; storing an existing id updates the row. When a file shrinks to fewer chunks, rows with the old higher positions remain - reset the database (`managing-pgvector-docker`).
- **Visible embedding step**: the pipeline calls `embed_chunks()` and then `vector_store.add_embeddings(...)` rather than `add_documents()`, so learners see where vectors are created.
- **Table creation**: `PGEngine.init_vectorstore_table()` raises when the table exists; `ensure_table()` swallows only "already exists" and re-raises everything else. It also creates the `vector` extension.
- **Filterable metadata**: only `source` and `category` are SQL columns (`METADATA_COLUMNS`). Other keys land in the JSON column; filtering on them needs a new column and a recreated table.
- **Driver**: `postgresql+asyncpg://`. PGEngine runs a background asyncio loop, and psycopg's async mode fails on the Windows Proactor loop.
- **Fake embeddings**: `DeterministicFakeEmbedding(size=settings.vector_size)` keeps the real dimension (1536), so a later real run overwrites the same rows. Similarity results are random with them; keyword search and filters still work.
- **Chunk defaults**: `CHUNK_SIZE=500` characters with k = 4 retrieved chunks keeps the prompt context near 2,000 characters; `CHUNK_OVERLAP=50` (10 %) keeps a sentence cut at a boundary whole in the neighbour. `load_settings()` rejects overlap >= size.
