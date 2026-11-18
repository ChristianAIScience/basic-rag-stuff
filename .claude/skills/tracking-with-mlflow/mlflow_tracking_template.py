"""MLflow tracing + run logging for LangChain/LangGraph - self-contained template to copy into other projects.

Dependencies:  pip install mlflow-skinny langchain-core      (server: Docker image ghcr.io/mlflow/mlflow)

Server (docker-compose.yml):
    mlflow:
      image: ghcr.io/mlflow/mlflow:v3.18.0
      command: >
        mlflow server --host 0.0.0.0 --port 5000
        --backend-store-uri sqlite:////mlflow/mlflow.db --artifacts-destination /mlflow/artifacts
      ports: ["127.0.0.1:5000:5000"]     # change host port AND the tracking URI together
      volumes: ["./mlflow-data:/mlflow"]

Usage:
    use_mlflow = enable_mlflow("http://localhost:5000", experiment="rag-evaluation")
    with mlflow_run(use_mlflow, run_name="judges-langchain"):      # traces inside are linked to the run
        ...run the chain / graph...
        if use_mlflow:
            log_results({"chunk_size": 500}, {"correctness_pass_rate": 0.9}, {"rows": [...]}, "details.json")

Import mlflow ONLY through this module: the environment variables below must be set before the import.
"""

import os
import urllib.request
from contextlib import AbstractContextManager, nullcontext

# MLflow prints run links with emoji -> UnicodeEncodeError on Windows consoles with a legacy code page.
os.environ.setdefault("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", "true")
# MLflow prints a hint for coding agents on import.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402


def mlflow_server_available(tracking_uri: str, timeout_s: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(f"{tracking_uri.rstrip('/')}/health", timeout=timeout_s) as response:
            return response.status == 200
    except (OSError, ValueError):  # ValueError: not a URL at all - never crash the caller
        return False


def enable_mlflow(tracking_uri: str, experiment: str, tracing: bool = True) -> bool:
    """Connect and select the experiment. False (and the caller continues without MLflow) if unavailable."""
    if not tracking_uri:
        print("MLflow: disabled (no tracking URI)")
        return False
    # Only http(s) servers can be probed; sqlite:///..., file:... or databricks URIs go to MLflow unchanged.
    if tracking_uri.startswith(("http://", "https://")) and not mlflow_server_available(tracking_uri):
        print(f"MLflow: server not reachable at {tracking_uri} - continuing without MLflow")
        return False
    mlflow.set_tracking_uri(tracking_uri)
    experiment_info = mlflow.set_experiment(experiment)
    if tracing:
        mlflow.langchain.autolog()  # traces LangChain AND LangGraph runs
    print(f"MLflow: experiment '{experiment}' -> {tracking_uri}/#/experiments/{experiment_info.experiment_id}")
    return True


def mlflow_run(enabled: bool, run_name: str, nested: bool = False) -> AbstractContextManager:
    """An MLflow run when enabled, otherwise a no-op context manager."""
    return mlflow.start_run(run_name=run_name, nested=nested) if enabled else nullcontext()


def log_results(params: dict, metrics: dict[str, float], details: dict | None = None,
                details_file: str = "details.json") -> None:
    """Into the active run. Metric names must not contain '@' (use hit_at_1, not hit@1)."""
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    if details is not None:
        mlflow.log_dict(details, details_file)


def traces_of_experiment(experiment_name: str) -> list:
    """Read traces back (e.g. in tests). Trace export is asynchronous - flush first."""
    mlflow.flush_trace_async_logging()
    experiment = mlflow.get_experiment_by_name(experiment_name)
    return mlflow.MlflowClient().search_traces(locations=[experiment.experiment_id])
