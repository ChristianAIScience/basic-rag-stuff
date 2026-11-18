---
name: tracking-with-mlflow
description: Traces LangChain/LangGraph runs and logs evaluation runs (params, metrics, artifacts) to the MLflow server that runs as a Docker service. Use when starting or debugging MLflow, adding tracing or logging to an example, comparing evaluation runs, or when MLflow output, ports or traces misbehave.
---

# Tracking with MLflow

Reference implementation: `rag_lab/tracking.py` (the only module that imports `mlflow`), service `mlflow` in `docker-compose.yml`;
used by `examples/evaluate_rag.py`, `examples/evaluate_retrieval.py` and the three RAG examples. Explanation: `docs/12_mlflow.md`.
Copyable, self-contained version for other projects: [mlflow_tracking_template.py](mlflow_tracking_template.py) (includes the docker-compose service).

## Rules

- **Server in Docker, client skinny**: image `ghcr.io/mlflow/mlflow` (pinned tag), SQLite backend + artifacts in `./mlflow-data` (git-ignored), port published on `127.0.0.1` only. The project depends on `mlflow-skinny` - no server, UI, pandas or scipy locally. `mlflow.genai.evaluate` needs the full package and is not used.
- **Import mlflow only through `rag_lab/tracking.py`**: it sets `MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT` (MLflow's emoji run links crash Windows consoles with a legacy code page) and `MLFLOW_DISABLE_AGENT_HINT` before the import. Examples use `enable_mlflow()`, `mlflow_run()`, `log_results()`, `rag_params()`.
- **MLflow is optional**: empty `MLFLOW_TRACKING_URI` or an unreachable server → `enable_mlflow()` returns False and the example runs unchanged. Keep every MLflow call behind that flag.
- **Tracing** = `mlflow.langchain.autolog()` (covers LangGraph too); traces created inside `mlflow_run(...)` are linked to that run. Use `tracing=False` where no LangChain code runs (retrieval metrics).
- **Metric names** contain no `@`: `hit_at_1`, `mrr_at_5`, `<judge>_pass_rate`. Log the config with `rag_params(settings)` so runs are comparable.
- Experiments: `rag-examples` (traces), `rag-evaluation` (one run per target), `retrieval-evaluation` (parent run + nested run per strategy).

## Add MLflow to an example

```
- [ ] 1. use_mlflow = enable_mlflow(settings, experiment="...")  right after load_settings()
- [ ] 2. Wrap each comparable unit in  with mlflow_run(use_mlflow, run_name=...):
- [ ] 3. if use_mlflow: log_results(params=..., metrics=..., details=...)
- [ ] 4. Unit test with mlflow mocked (tests/unit/test_tracking.py pattern: monkeypatch tracking.mlflow)
- [ ] 5. uv run python -m pytest, then -m integration with docker compose up -d -> fix and repeat until green
- [ ] 6. Run the example and check the run in the UI; document the experiment in docs/12_mlflow.md + German copy
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| "server not reachable" | `docker compose up -d`; `MLFLOW_TRACKING_URI` must use the port in `MLFLOW_PORT` - change both together |
| non-http URI (`sqlite:///...`, `file:...`) | valid: `enable_mlflow` only probes http(s) servers and passes other URIs to MLflow unchanged |
| `ports are not available ... 5000` | another MLflow or app uses the port: set `MLFLOW_PORT` and `MLFLOW_TRACKING_URI` to a free port; leave other projects' containers alone |
| `UnicodeEncodeError` with `\U0001f3c3` | something imported `mlflow` before `rag_lab.tracking` - import via `rag_lab.tracking` |
| Traces missing right after a run | export is asynchronous: `mlflow.flush_trace_async_logging()` (the integration test does this) |
| Later runs fail with "experiment is deleted" | a deleted experiment was still active: `mlflow.set_experiment(experiment_id="0")` after deleting (test teardown) |
| `FutureWarning ... experiment_ids` in `search_traces` | use `locations=[experiment_id]` |
