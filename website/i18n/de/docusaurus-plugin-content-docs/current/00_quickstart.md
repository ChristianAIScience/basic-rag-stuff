# Schnellstart

In etwa zehn Minuten von null zur ersten RAG-Antwort. Jeder Schritt endet mit einer Prüfung, damit du weißt,
dass er funktioniert hat, bevor du weitermachst. Die folgenden Kapitel erklären, *warum* alles so funktioniert.

## Voraussetzungen

| Werkzeug | Wofür | Prüfen |
|---|---|---|
| [uv](https://docs.astral.sh/uv/) | Python 3.14 und alle Abhängigkeiten | `uv --version` |
| [Docker](https://www.docker.com/) | PostgreSQL mit pgvector | `docker info` |
| OpenAI-API-Key | Embeddings und Antworten (für Schritt 1–4 optional) | – |
| [Node.js](https://nodejs.org/) ≥ 20 | nur für diese Doku-Website | `node --version` |

## 1. Code holen

```bash
git clone https://github.com/ChristianAIScience/basic-rag-stuff.git
cd basic-rag-stuff
```

## 2. Konfigurieren

```bash
cp .env.example .env
```

Öffne `.env` und setze mindestens:

```dotenv
POSTGRES_PASSWORD=choose_a_password
LLM_API_KEY=sk-...            # your OpenAI key (a placeholder is fine until step 5)
```

Läuft auf deinem Rechner schon ein anderes PostgreSQL, ändere zusätzlich `POSTGRES_PORT` (z. B. auf `5433`);
ist Port 5000 belegt, ändere `MLFLOW_PORT` und `MLFLOW_TRACKING_URI` (z. B. auf `5001`).

## 3. Datenbank starten und Python installieren

```bash
docker compose up -d
uv sync
```

**Prüfen:** `docker compose ps` zeigt die Container `postgres` und `mlflow` als `healthy`
(MLflow-Oberfläche: http://localhost:5000, siehe [10. MLflow](10_mlflow.md)), und

```bash
uv run python -m pytest
```

endet mit `... passed, ... deselected`. Diese Unit-Tests brauchen weder Datenbank noch API-Key
(siehe [Tests](09_testing.md)).

## 4. Beispieldokumente einlesen

Ohne OpenAI-Key mit Fake-Embeddings – Speichern, Metadaten und Stichwortsuche funktionieren, die Ähnlichkeitssuche nicht:

```bash
uv run python -m examples.ingest_documents --fake-embeddings
```

Mit echtem Key in `.env`:

```bash
uv run python -m examples.ingest_documents
```

**Prüfen:** Die Ausgabe endet mit `Stored 25 chunks from 7 files`. Ein zweiter Lauf ergibt dieselbe Zahl –
vorhandene Chunks werden aktualisiert, nicht verdoppelt.

## 5. Fragen stellen (braucht einen echten OpenAI-Key)

```bash
uv run python -m examples.basic_langchain_rag "How many vacation days do I get?"
uv run python -m examples.basic_langgraph_rag "How long is the Glowbeam X2 warranty?"
uv run python -m examples.agentic_rag
uv run python -m examples.compare_retrievers
```

**Prüfen:** Die Antworten nennen Fakten aus `data/` (z. B. *27 vacation days*) und ihre Quelldateien.
Die Beispielfirma „Lumora Systems" ist erfunden – eine richtige Antwort kann also nur aus der Suche stammen.

Hast du in Schritt 4 `--fake-embeddings` verwendet, führe Schritt 4 vorher noch einmal ohne den Schalter aus.

## 6. Optional: PDF-Chunking und diese Website

```bash
uv run python -m examples.pdf_chunking         # no database or API key needed

cd website
npm install
npm start                                      # http://localhost:3000
```

## Wie geht es weiter?

| Ziel | Kapitel |
|---|---|
| Die Konfiguration verstehen | [1. Setup & Konfiguration](01_setup.md) |
| Sehen, wie aus Dokumenten Vektoren werden | [2. Dokumente einlesen](02_ingestion.md) |
| Suchverfahren vergleichen | [4. Retrieval-Strategien](04_retrieval.md) |
| Die RAG-Pipeline bauen | [5. LangChain](05_langchain_rag.md), [6. LangGraph](06_langgraph_rag.md), [7. Agentic RAG](07_agentic_rag.md) |
| Tests verstehen und schreiben | [9. Tests](09_testing.md) |

## Häufige Probleme

| Symptom | Lösung |
|---|---|
| `Missing required environment variable: POSTGRES_PASSWORD` | `.env` fehlt oder der Wert ist leer (Schritt 2) |
| `Nothing accepts connections` / `connection refused` | Docker läuft nicht, oder `POSTGRES_PORT` in `.env` passt nicht zum Container |
| `password authentication failed` nach einer Passwortänderung | der Container behält sein erstes Passwort – siehe [1. Setup](01_setup.md) |
| `AuthenticationError` von OpenAI | `LLM_API_KEY` ist noch der Platzhalter |
| Antworten passen nicht zur Frage | die Daten wurden mit `--fake-embeddings` eingelesen; ohne den Schalter neu einlesen |
