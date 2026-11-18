# 10. MLflow: Tracing and Experiment Tracking

## What

[MLflow](https://mlflow.org/docs/latest/genai/) is an open-source platform for tracking ML and GenAI work.
The project runs the **MLflow server in Docker** (next to PostgreSQL) and uses two of its features:

| Feature | What you get | Used by |
|---|---|---|
| **Tracing** | every LangChain/LangGraph run as a trace: one span per chain step, retriever and LLM call, with inputs, outputs, timing and tokens | the RAG examples and `evaluate_rag` |
| **Experiment tracking** | every evaluation as a *run*: configuration as **params**, scores as **metrics**, details as **artifacts** | `evaluate_rag`, `evaluate_retrieval` |

The Python code only needs the lightweight client package **`mlflow-skinny`** (no server, no UI,
no pandas/scipy). The server, its UI and its database live in the `mlflow` container.

## Why

- **Tracing** shows *what actually happened* inside a run: which chunks the retriever returned, the exact
  prompt the LLM received, how often the agentic graph rewrote the query - without adding `print()` calls.
- **Tracking** turns evaluations into a history: change `CHUNK_SIZE`, the LLM or the prompt, evaluate again,
  and compare the runs side by side instead of copying numbers from the terminal.

## Setup

`docker-compose.yml` contains the service:

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

- `--host 0.0.0.0` is needed inside a container (the server's default is `127.0.0.1`). Publishing the port
  on `127.0.0.1` keeps the UI private to your computer.
- Runs, metrics and traces are stored in SQLite, artifacts as files - both in `./mlflow-data/`
  (git-ignored, survives `docker compose down`). Reset: stop the stack and delete the folder.
- The official image has no PostgreSQL driver, which is why MLflow keeps its own SQLite database instead of
  using the pgvector database.

In `.env`:

```dotenv
# Port 5000 taken? Change BOTH lines to the same free port, e.g. 5001.
MLFLOW_PORT=5000                          # where Docker publishes the server
MLFLOW_TRACKING_URI=http://localhost:5000 # where the code connects; empty = no MLflow
```

## The code

`rag_lab/tracking.py` is the only module that talks to MLflow:

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

Logging an evaluation:

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

| Experiment | Created by | Content |
|---|---|---|
| `rag-examples` | `basic_langchain_rag`, `basic_langgraph_rag`, `agentic_rag` | one trace per question |
| `rag-evaluation` | `evaluate_rag` | one run per RAG implementation: judge pass rates + `judgments.json` + all traces |
| `retrieval-evaluation` | `evaluate_retrieval` | a parent run with one nested run per strategy: `hit_at_1/3/5`, `mrr_at_5` |

MLflow metric names cannot contain `@`, so Hit@1 is logged as `hit_at_1`.

## How it connects

MLflow observes the existing code - nothing in the RAG chains or graphs changes. `autolog()` hooks into
LangChain's callback system, which LangGraph uses too. The evaluations of the next two chapters (11 and 12) decide *what*
is measured; MLflow stores the results so they can be compared.

## Run

```bash
docker compose up -d                                  # starts postgres AND mlflow
uv run python -m examples.agentic_rag                 # -> traces in experiment "rag-examples"
uv run python -m examples.evaluate_rag --limit 3      # -> runs in "rag-evaluation"
uv run python -m examples.evaluate_retrieval          # -> runs in "retrieval-evaluation"
```

Open the UI at `http://localhost:<MLFLOW_PORT>`.

- **Traces** tab of an experiment: click a trace to see the span tree - the graph or chain as root span,
  below it each node or chain step, the retriever call and every LLM call with its prompt and answer.
- **Runs**: select several runs → *Compare* to see params and metrics side by side.

## Expected output

Every example prints one MLflow line at the start:

```
MLflow: experiment 'rag-evaluation' -> http://localhost:5000/#/experiments/2
```

Without a running server the examples continue without MLflow:

```
MLflow: server not reachable at http://localhost:5000 - continuing without MLflow (start it with: docker compose up -d)
```

## A typical experiment

1. Evaluate the current setup (`evaluate_rag`, `evaluate_retrieval`).
2. Change one thing, e.g. `CHUNK_SIZE=300` in `.env`.
3. Re-ingest and regenerate the retrieval dataset (the labelled chunk texts change, chapter 11).
4. Evaluate again.
5. In MLflow, compare both runs: the `chunk_size` param explains the difference in the metrics.

## Notes

- Tracing every judge call too makes `rag-evaluation` traces large - useful for debugging a doubtful verdict.
- MLflow also has its own LLM judges (`mlflow.genai.evaluate` with scorers such as `RetrievalGroundedness`).
  They need the full `mlflow` package locally; this project keeps its own judges (chapter 12) and uses MLflow
  for storage and comparison.
- `rag_lab/tracking.py` sets `MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT` because MLflow's own run links contain
  emoji that fail on Windows consoles with a legacy code page; the examples print the link themselves.
