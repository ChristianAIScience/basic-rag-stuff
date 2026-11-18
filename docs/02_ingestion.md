# 2. Document Ingestion

## What

The pipeline in `rag_lab/ingestion/` turns files into searchable rows in Postgres:

```
data/*.md, *.txt, *.pdf
   │ loader.py     -> Document(page_content, metadata={source, category})
   │                  (PDFs: one Document per chapter/section, see 03_pdf_chunking.md)
   │ cleaner.py    -> normalized whitespace, empty docs removed
   │ splitter.py   -> chunks of ≤ CHUNK_SIZE characters, CHUNK_OVERLAP shared
   │ pipeline.py   -> deterministic id per chunk
   │ embedder.py   -> one vector (1536 floats) per chunk
   ▼ vector_store  -> one table row per chunk
```

## Why

- **Chunks** – an LLM prompt should contain a few precise passages, not whole files.
- **Overlap** – a sentence cut at a chunk boundary still appears complete in the neighbour.
- **Metadata** – `source` lets the answer cite files; `category` (the folder name) enables filtering.
- **Deterministic ids** – `uuid5(source + chunk position)`. Re-running ingestion *updates*
  the same rows instead of inserting duplicates.

## How embeddings are stored and searched

`db/vector_store.py` creates this table once (`ensure_table()`; an existing table is kept):

```sql
CREATE TABLE documents (
  langchain_id        UUID PRIMARY KEY,
  content             TEXT NOT NULL,
  embedding           vector(1536) NOT NULL,   -- pgvector type
  source              TEXT,
  category            TEXT,
  langchain_metadata  JSON
);
```

At query time the question is embedded with the same model, and Postgres sorts by
cosine distance using pgvector's `<=>` operator:

```sql
SELECT content, embedding <=> :question_vector AS distance
FROM documents ORDER BY distance LIMIT 4;
```

With a handful of rows this is a full scan. For large tables you would add an
approximate index (HNSW or IVFFlat) – not needed for learning.

## Run

```bash
uv run python -m examples.ingest_documents
uv run python -m examples.ingest_documents --fake-embeddings   # no OpenAI key needed
```

`--fake-embeddings` stores random-but-stable 1536-dim vectors. Table, metadata and keyword search
work, but similarity search is meaningless. A later run without the flag overwrites the same rows
(same chunk ids) with real embeddings.

## Expected output

```
Table 'documents': created              # "already exists" on the second run
Stored 25 chunks from 7 files (chunk_size=500, chunk_overlap=50):
  hr/remote_work.txt               3 chunks
  ...
An embedding has 1536 dimensions, e.g. [0.0123, -0.0456, ...] ...
```

Run it twice and count rows – the number stays the same:

```bash
docker compose exec postgres psql -U postgres -d rag_learning -c "select count(*) from documents;"
```

Try `CHUNK_SIZE=200` in `.env`: more, smaller chunks. (If chunks get fewer, old rows with
higher positions remain – start fresh with `docker compose down` and deleting the `postgres-data/` folder.)
