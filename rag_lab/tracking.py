"""MLflow integration: experiment tracking and tracing.

MLflow runs as a Docker service (docker-compose.yml, UI on http://localhost:<MLFLOW_PORT>).
The project uses the lightweight client package `mlflow-skinny` and two MLflow features:

- Tracing: `mlflow.langchain.autolog()` records every LangChain/LangGraph run as a trace -
  one span per chain step, retriever call and LLM call, with inputs, outputs and timing.
- Tracking: evaluations are logged as runs with params (models, chunk size, ...) and
  metrics (judge pass rates, Hit@k, MRR), so experiments can be compared in the UI.

MLflow is optional: with MLFLOW_TRACKING_URI empty or the server unreachable, the examples
print a note and run as before.
"""

import os
import urllib.request
from contextlib import AbstractContextManager, nullcontext

# Set before importing mlflow:
# - MLflow prints run links with emoji, which fails on Windows consoles with a legacy code page;
#   enable_mlflow() prints the experiment link itself.
# - MLflow prints a hint for coding agents on import; it is noise for this project.
os.environ.setdefault("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402

from rag_lab.config import Settings  # noqa: E402


def mlflow_server_available(tracking_uri: str, timeout_s: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(f"{tracking_uri.rstrip('/')}/health", timeout=timeout_s) as response:
            return response.status == 200
    except OSError:
        return False


def enable_mlflow(settings: Settings, experiment: str, tracing: bool = True) -> bool:
    """Connect to the MLflow server and select an experiment. Returns False if MLflow is not used."""
    if not settings.mlflow_tracking_uri:
        print("MLflow: disabled (MLFLOW_TRACKING_URI is not set)")
        return False
    if not mlflow_server_available(settings.mlflow_tracking_uri):
        print(f"MLflow: server not reachable at {settings.mlflow_tracking_uri} - continuing without MLflow "
              "(start it with: docker compose up -d)")
        return False

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    experiment_info = mlflow.set_experiment(experiment)
    if tracing:
        mlflow.langchain.autolog()
    print(f"MLflow: experiment '{experiment}' -> "
          f"{settings.mlflow_tracking_uri}/#/experiments/{experiment_info.experiment_id}")
    return True


def mlflow_run(enabled: bool, run_name: str, nested: bool = False) -> AbstractContextManager:
    """`with mlflow_run(...):` - an MLflow run when enabled, otherwise does nothing.

    Traces created inside the block are linked to the run.
    """
    return mlflow.start_run(run_name=run_name, nested=nested) if enabled else nullcontext()


def log_results(params: dict, metrics: dict[str, float], details: dict | None = None,
                details_file: str = "details.json") -> None:
    """Log one evaluation into the active run: config as params, scores as metrics, details as JSON artifact."""
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    if details is not None:
        mlflow.log_dict(details, details_file)


def rag_params(settings: Settings) -> dict[str, str | int]:
    """The configuration that decides RAG quality - logged with every evaluation run."""
    return {
        "llm_model": settings.llm_model,
        "judge_model": settings.judge_model,
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "vector_table": settings.vector_table,
    }
