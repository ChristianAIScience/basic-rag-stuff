# 12. MLflow: Tracing und Experiment-Tracking

## Was

[MLflow](https://mlflow.org/docs/latest/genai/) ist eine Open-Source-Plattform, um ML- und GenAI-Arbeit nachzuverfolgen.
Das Projekt betreibt den **MLflow-Server in Docker** (neben PostgreSQL) und nutzt zwei seiner Funktionen:

| Funktion | Was man bekommt | Genutzt von |
|---|---|---|
| **Tracing** | jeder LangChain-/LangGraph-Lauf als Trace: ein Span pro Chain-Schritt, Retriever- und LLM-Aufruf, mit Ein- und Ausgaben, Dauer und Tokens | die RAG-Beispiele und `evaluate_rag` |
| **Experiment-Tracking** | jede Auswertung als *Run*: Konfiguration als **Parameter**, Ergebnisse als **Metriken**, Details als **Artefakte** | `evaluate_rag`, `evaluate_retrieval` |

Der Python-Code braucht nur das schlanke Client-Paket **`mlflow-skinny`** (kein Server, keine Oberfläche,
kein pandas/scipy). Server, Oberfläche und Datenbank laufen im Container `mlflow`.

## Warum

- **Tracing** zeigt, *was in einem Lauf wirklich passiert ist*: welche Chunks der Retriever geliefert hat, den genauen
  Prompt an das LLM, wie oft der agentische Graph die Anfrage umformuliert hat – ohne zusätzliche `print()`-Aufrufe.
- **Tracking** macht aus Auswertungen eine Historie: `CHUNK_SIZE`, LLM oder Prompt ändern, erneut auswerten und die
  Runs nebeneinander vergleichen, statt Zahlen aus dem Terminal abzuschreiben.

## Einrichtung

`docker-compose.yml` enthält den Dienst:

```yaml
mlflow:
  image: ghcr.io/mlflow/mlflow:v3.18.0
  command: >
    mlflow server --host 0.0.0.0 --port 5000
    --backend-store-uri sqlite:////mlflow/mlflow.db
    --artifacts-destination /mlflow/artifacts
  ports:
    - "127.0.0.1:${MLFLOW_PORT:-5000}:5000"   # reachable from this PC only
  volumes:
    - ./mlflow-data:/mlflow
```

- `--host 0.0.0.0` ist im Container nötig (Standard des Servers ist `127.0.0.1`). Weil der Port nur auf `127.0.0.1`
  veröffentlicht wird, bleibt die Oberfläche auf deinem Rechner.
- Runs, Metriken und Traces liegen in SQLite, Artefakte als Dateien – beides in `./mlflow-data/` (von Git ignoriert,
  übersteht `docker compose down`). Zurücksetzen: Container stoppen und den Ordner löschen.
- Das offizielle Image enthält keinen PostgreSQL-Treiber; deshalb nutzt MLflow eine eigene SQLite-Datenbank statt der
  pgvector-Datenbank.

In `.env`:

```dotenv
MLFLOW_PORT=5000                          # use e.g. 5001 if port 5000 is taken
MLFLOW_TRACKING_URI=http://localhost:5000 # empty = run everything without MLflow
```

## Der Code

`rag_lab/tracking.py` ist das einzige Modul, das mit MLflow spricht:

```python
def enable_mlflow(settings, experiment: str, tracing: bool = True) -> bool:
    if not settings.mlflow_tracking_uri:              # MLflow is optional
        return False
    if not mlflow_server_available(settings.mlflow_tracking_uri):
        return False                                  # examples keep working
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(experiment)
    if tracing:
        mlflow.langchain.autolog()                    # traces LangChain AND LangGraph
    return True
```

Eine Auswertung loggen:

```python
with mlflow_run(use_mlflow, run_name=f"judges-{name}"):           # traces inside are linked to the run
    results = evaluate_target(targets[name], examples, judge_llm)
    log_results(
        params={**rag_params(settings), "target": name},          # models, chunk size, ...
        metrics=judge_metrics(results),                           # correctness_pass_rate, ...
        details={"results": [asdict(r) for r in results]},       # every verdict with reasoning
        details_file="judgments.json",
    )
```

| Experiment | Erzeugt von | Inhalt |
|---|---|---|
| `rag-examples` | `basic_langchain_rag`, `basic_langgraph_rag`, `agentic_rag` | ein Trace pro Frage |
| `rag-evaluation` | `evaluate_rag` | ein Run pro RAG-Umsetzung: Judge-Bestehensquoten + `judgments.json` + alle Traces |
| `retrieval-evaluation` | `evaluate_retrieval` | ein Eltern-Run mit einem verschachtelten Run pro Strategie: `hit_at_1/3/5`, `mrr_at_5` |

MLflow-Metriknamen dürfen kein `@` enthalten, deshalb wird Hit@1 als `hit_at_1` geloggt.

## Wie es zusammenhängt

MLflow beobachtet den vorhandenen Code – an den RAG-Chains und -Graphen ändert sich nichts. `autolog()` hängt sich in
das Callback-System von LangChain, das auch LangGraph nutzt. Die Auswertungen aus Kapitel 10 und 11 bestimmen, *was*
gemessen wird; MLflow speichert die Ergebnisse, damit man sie vergleichen kann.

## Ausführen

```bash
docker compose up -d                                  # starts postgres AND mlflow
uv run python -m examples.agentic_rag                 # -> traces in experiment "rag-examples"
uv run python -m examples.evaluate_rag --limit 3      # -> runs in "rag-evaluation"
uv run python -m examples.evaluate_retrieval          # -> runs in "retrieval-evaluation"
```

Die Oberfläche öffnen unter `http://localhost:<MLFLOW_PORT>`.

- Reiter **Traces** eines Experiments: Ein Klick auf einen Trace zeigt den Span-Baum – Graph oder Chain als Wurzel,
  darunter jeder Node bzw. Chain-Schritt, der Retriever-Aufruf und jeder LLM-Aufruf mit Prompt und Antwort.
- **Runs**: mehrere Runs auswählen → *Compare*, um Parameter und Metriken nebeneinander zu sehen.

## Erwartete Ausgabe

Jedes Beispiel gibt zu Beginn eine MLflow-Zeile aus:

```
MLflow: experiment 'rag-evaluation' -> http://localhost:5000/#/experiments/2
```

Ohne laufenden Server arbeiten die Beispiele ohne MLflow weiter:

```
MLflow: server not reachable at http://localhost:5000 - continuing without MLflow (start it with: docker compose up -d)
```

## Ein typisches Experiment

1. Den aktuellen Stand auswerten (`evaluate_rag`, `evaluate_retrieval`).
2. Eine Sache ändern, z. B. `CHUNK_SIZE=300` in `.env`.
3. Neu einlesen und den Retrieval-Datensatz neu erzeugen (die Chunk-IDs ändern sich, Kapitel 11).
4. Erneut auswerten.
5. In MLflow beide Runs vergleichen: Der Parameter `chunk_size` erklärt den Unterschied in den Metriken.

## Hinweise

- Weil auch jeder Judge-Aufruf aufgezeichnet wird, sind die Traces in `rag-evaluation` groß – nützlich, um ein zweifelhaftes
  Urteil zu untersuchen.
- MLflow hat auch eigene LLM-Judges (`mlflow.genai.evaluate` mit Scorern wie `RetrievalGroundedness`). Sie brauchen
  lokal das volle Paket `mlflow`; dieses Projekt behält seine eigenen Judges (Kapitel 10) und nutzt MLflow zum Speichern
  und Vergleichen.
- `rag_lab/tracking.py` setzt `MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT`, weil MLflows eigene Run-Links Emojis enthalten,
  die in Windows-Konsolen mit alter Codepage fehlschlagen; die Beispiele geben den Link selbst aus.
