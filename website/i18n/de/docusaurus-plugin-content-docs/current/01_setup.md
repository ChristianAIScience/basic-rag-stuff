# 1. Setup & Konfiguration

## Was

- `pyproject.toml` – Python ≥ 3.14, Abhängigkeiten, pytest-Konfiguration.
- `docker-compose.yml` – PostgreSQL 17 mit der Erweiterung pgvector.
- `.env` – alle Zugangsdaten und Einstellungen (Kopie von `.env.example`, wird nie eingecheckt).
- `rag_lab/config.py` – liest `.env` in ein einziges `Settings`-Objekt.
- `rag_lab/models.py` – erzeugt das OpenAI-Chatmodell und das Embedding-Modell.

## Warum

Jedes Modul braucht dieselben Datenbank-Zugangsdaten und Modellnamen. Werden sie an einer
Stelle gelesen (`load_settings()`), stehen keine Geheimnisse im Code, und Tests werden einfach:
Ein Test kann sein eigenes `Settings`-Objekt bauen, ohne die Umgebung anzufassen.

## Der Code

```python
from rag_lab.config import load_settings
from rag_lab.models import get_chat_model, get_embeddings

settings = load_settings()          # raises ValueError if POSTGRES_PASSWORD or LLM_API_KEY is missing
llm = get_chat_model(settings)      # ChatOpenAI(model=LLM_MODEL)
embeddings = get_embeddings(settings)  # OpenAIEmbeddings(model=EMBEDDING_MODEL)
```

| Variable | Standardwert | Bedeutung |
|---|---|---|
| `POSTGRES_HOST/PORT/DB/USER` | localhost / 5432 / rag_learning / postgres | Verbindung |
| `POSTGRES_PASSWORD` | – (Pflicht) | wird auch von Docker Compose zum Anlegen der DB genutzt (nur beim ersten Start – siehe unten) |
| `LLM_API_KEY` | – (Pflicht) | OpenAI-API-Key |
| `EMBEDDING_MODEL` | text-embedding-3-small | Embeddings mit 1536 Dimensionen |
| `LLM_MODEL` | gpt-5-mini | Chatmodell |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 500 / 50 | Aufteilen der Texte |
| `VECTOR_TABLE` / `VECTOR_SIZE` | documents / 1536 | Tabellenname, Vektordimension (muss zum Embedding-Modell passen) |

**Wo die Daten liegen:** Die Datenbankdateien liegen im Ordner `postgres-data/` im Projekt
(ein Bind-Mount, siehe `docker-compose.yml`). Sie überstehen `docker compose down`, das Löschen
des Containers und eine Neuinstallation von Docker. Der Ordner steht in `.gitignore`. Für eine leere
Datenbank: `docker compose down`, `postgres-data/` löschen, `docker compose up -d`.
Vor dem Kopieren des Ordners als Sicherung den Container stoppen.

**Passwort später ändern:** Der Container nutzt `POSTGRES_PASSWORD` nur, wenn `postgres-data/`
neu angelegt wird. Nach einer Änderung in `.env` muss auch die bestehende Datenbank angepasst werden:

```bash
docker compose exec postgres psql -U postgres -c "ALTER USER postgres PASSWORD 'new_password';"
```

## Wie es zusammenhängt

`Settings` fließt in `db/connection.py` (Verbindungs-URL), `models.py` (Modelle)
und in die Ingestion-Pipeline (Chunk-Größe). Kein anderes Modul liest Umgebungsvariablen.

## Ausführen

```bash
cp .env.example .env     # edit POSTGRES_PASSWORD and LLM_API_KEY
docker compose up -d
uv sync
uv run python -m pytest
```

## Erwartete Ausgabe

`docker compose up -d` meldet den Container als gestartet; `pytest` meldet alle Unit-Tests als bestanden.
Die Datenbank prüfen:

```bash
docker compose exec postgres psql -U postgres -d rag_learning -c "select version();"
```
