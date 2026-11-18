"""MLflow against the real Docker server (docker compose up -d). Fake LLM - no API calls."""

import dataclasses
import uuid

import pytest
from langchain_core.documents import Document
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from rag_lab import tracking
from rag_lab.langchain_rag.chain import build_rag_chain

pytestmark = pytest.mark.integration


@pytest.fixture
def mlflow_experiment(settings):
    if not settings.mlflow_tracking_uri or not tracking.mlflow_server_available(settings.mlflow_tracking_uri):
        pytest.skip("MLflow server not reachable (MLFLOW_TRACKING_URI / docker compose up -d)")
    name = f"integration-test-{uuid.uuid4().hex[:8]}"
    assert tracking.enable_mlflow(settings, name) is True
    experiment = tracking.mlflow.get_experiment_by_name(name)
    yield experiment
    tracking.mlflow.langchain.autolog(disable=True)
    tracking.mlflow.MlflowClient().delete_experiment(experiment.experiment_id)


def test_run_logs_params_metrics_artifact_and_linked_trace(mlflow_experiment, settings):
    retriever = RunnableLambda(lambda q: [Document(page_content="27 days", metadata={"source": "hr/v.md"})])
    chain = build_rag_chain(retriever, GenericFakeChatModel(messages=iter([AIMessage(content="27 days.")])))

    with tracking.mlflow_run(True, run_name="test-run") as run:
        chain.invoke("How many vacation days?")
        tracking.log_results(tracking.rag_params(settings), {"correctness_pass_rate": 0.9}, {"rows": [1]}, "r.json")
    tracking.mlflow.flush_trace_async_logging()

    client = tracking.mlflow.MlflowClient()
    logged = client.get_run(run.info.run_id)
    assert logged.data.metrics == {"correctness_pass_rate": 0.9}
    assert logged.data.params["chunk_size"] == str(settings.chunk_size)
    assert [artifact.path for artifact in client.list_artifacts(run.info.run_id)] == ["r.json"]

    traces = client.search_traces(locations=[mlflow_experiment.experiment_id])
    assert len(traces) == 1
    assert traces[0].info.trace_metadata["mlflow.sourceRun"] == run.info.run_id
    assert traces[0].data.spans[0].name == "RunnableSequence"  # the whole chain is the root span
