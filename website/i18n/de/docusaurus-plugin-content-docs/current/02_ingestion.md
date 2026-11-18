# 2. Dokumente einlesen (Ingestion)

## Was

Die Pipeline in `rag_lab/ingestion/` macht aus Dateien durchsuchbare Zeilen in Postgres:

```
data/*.md, *.txt, *.pdf
   │ loader.py     -> Document(page_content, metadata={source, category})
   │                  (PDFs: one Document per chapter/section, see 08_pdf_chunking.md)
   │ cleaner.py    -> normalized whitespace, empty docs removed
   │ splitter.py   -> chunks of ≤ CHUNK_SIZE characters, CHUNK_OVERLAP shared
   │ pipeline.py   -> deterministic id per chunk
   │ embedder.py   -> one vector (1536 floats) per chunk
   ▼ vector_store  -> one table row per chunk
```

## Warum

- **Chunks** – ein LLM-Prompt sollte wenige, präzise Textstellen enthalten, keine ganzen Dateien.
- **Überlappung (Overlap)** – ein Satz, der an einer Chunk-Grenze zerschnitten wird, steht im Nachbar-Chunk trotzdem vollständig.
- **Metadaten** – mit `source` kann die Antwort ihre Quellen nennen; `category` (der Ordnername) ermöglicht Filter.
- **Deterministische IDs** – `uuid5(Quelle + Chunk-Position)`. Ein erneuter Lauf *aktualisiert*
  dieselben Zeilen, statt Duplikate anzulegen.

## Wie Embeddings gespeichert und durchsucht werden

`db/vector_store.py` legt diese Tabelle einmalig an (`ensure_table()`; eine bestehende Tabelle bleibt erhalten):

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

Bei einer Anfrage wird die Frage mit demselben Modell eingebettet, und Postgres sortiert nach
der Kosinus-Distanz mit dem pgvector-Operator `<=>`:

```sql
SELECT content, embedding <=> :question_vector AS distance
FROM documents ORDER BY distance LIMIT 4;
```

Bei wenigen Zeilen ist das ein vollständiger Durchlauf über die Tabelle. Bei großen Tabellen würde man
einen Näherungsindex (HNSW oder IVFFlat) anlegen – zum Lernen nicht nötig.

## Ausführen

```bash
uv run python -m examples.ingest_documents
uv run python -m examples.ingest_documents --fake-embeddings   # no OpenAI key needed
```

`--fake-embeddings` speichert zufällige, aber stabile Vektoren mit 1536 Dimensionen. Tabelle, Metadaten
und Stichwortsuche funktionieren, die Ähnlichkeitssuche liefert aber keine sinnvollen Ergebnisse. Ein späterer
Lauf ohne den Schalter überschreibt dieselben Zeilen (gleiche Chunk-IDs) mit echten Embeddings.

## Erwartete Ausgabe

```
Table 'documents': created              # "already exists" on the second run
Stored 25 chunks from 7 files (chunk_size=500, chunk_overlap=50):
  hr/remote_work.txt               3 chunks
  ...
An embedding has 1536 dimensions, e.g. [0.0123, -0.0456, ...] ...
```

Zweimal ausführen und die Zeilen zählen – die Anzahl bleibt gleich:

```bash
docker compose exec postgres psql -U postgres -d rag_learning -c "select count(*) from documents;"
```

Probiere `CHUNK_SIZE=200` in `.env`: mehr, kleinere Chunks. (Werden es weniger Chunks, bleiben alte Zeilen
mit höheren Positionen stehen – dann neu anfangen mit `docker compose down` und Löschen des Ordners `postgres-data/`.)
