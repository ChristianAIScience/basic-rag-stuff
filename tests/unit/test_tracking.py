"""MLflow helpers without a server: mlflow calls are replaced by mocks."""

import dataclasses
from unittest.mock import MagicMock

import pytest

from rag_lab import tracking
from rag_lab.evaluation.judges import JudgeResult
from rag_lab.evaluation.retrieval_metrics import RetrievalScores
from rag_lab.evaluation.runner import ExampleResult, judge_metrics
from tests.unit.test_vector_store_errors import SETTINGS


@pytest.fixture
def fake_mlflow(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr(tracking, "mlflow", mock)
    return mock


def test_mlflow_is_disabled_without_tracking_uri(fake_mlflow):
    assert tracking.enable_mlflow(SETTINGS, "exp") is False
    fake_mlflow.set_tracking_uri.assert_not_called()


def test_examples_continue_when_server_is_unreachable(fake_mlflow, monkeypatch):
    monkeypatch.setattr(tracking, "mlflow_server_available", lambda uri: False)
    settings = dataclasses.replace(SETTINGS, mlflow_tracking_uri="http://localhost:5999")

    assert tracking.enable_mlflow(settings, "exp") is False
    fake_mlflow.set_experiment.assert_not_called()


def test_enable_mlflow_sets_uri_experiment_and_tracing(fake_mlflow, monkeypatch):
    monkeypatch.setattr(tracking, "mlflow_server_available", lambda uri: True)
    settings = dataclasses.replace(SETTINGS, mlflow_tracking_uri="http://localhost:5001")

    assert tracking.enable_mlflow(settings, "rag-evaluation") is True
    fake_mlflow.set_tracking_uri.assert_called_once_with("http://localhost:5001")
    fake_mlflow.set_experiment.assert_called_once_with("rag-evaluation")
    fake_mlflow.langchain.autolog.assert_called_once_with()


def test_tracing_can_be_switched_off(fake_mlflow, monkeypatch):
    monkeypatch.setattr(tracking, "mlflow_server_available", lambda uri: True)
    settings = dataclasses.replace(SETTINGS, mlflow_tracking_uri="http://localhost:5001")
    tracking.enable_mlflow(settings, "exp", tracing=False)
    fake_mlflow.langchain.autolog.assert_not_called()


def test_server_check_is_false_for_closed_port():
    assert tracking.mlflow_server_available("http://127.0.0.1:9", timeout_s=0.5) is False


def test_mlflow_run_does_nothing_when_disabled(fake_mlflow):
    with tracking.mlflow_run(False, run_name="x"):
        pass
    fake_mlflow.start_run.assert_not_called()


def test_log_results_logs_params_metrics_and_details(fake_mlflow):
    tracking.log_results({"chunk_size": 500}, {"hit_at_1": 0.7}, {"rows": []}, "out.json")
    fake_mlflow.log_params.assert_called_once_with({"chunk_size": 500})
    fake_mlflow.log_metrics.assert_called_once_with({"hit_at_1": 0.7})
    fake_mlflow.log_dict.assert_called_once_with({"rows": []}, "out.json")


def test_rag_params_contain_the_quality_relevant_config():
    assert tracking.rag_params(SETTINGS) == {
        "llm_model": "m", "judge_model": "m", "embedding_model": "e",
        "chunk_size": 500, "chunk_overlap": 50, "vector_table": "documents",
    }


def test_judge_metrics_are_pass_rates_without_empty_judges():
    results = [
        ExampleResult("q1", "a", [JudgeResult("correctness", True, ""), JudgeResult("groundedness", None, "")], 0.1),
        ExampleResult("q2", "a", [JudgeResult("correctness", False, ""), JudgeResult("groundedness", None, "")], 0.1),
    ]
    assert judge_metrics(results) == {"correctness_pass_rate": 0.5}  # groundedness: 0 applicable -> left out


def test_retrieval_scores_as_metrics_use_names_mlflow_accepts():
    scores = RetrievalScores(hit_at={1: 0.5, 3: 0.75, 5: 1.0}, mrr=0.7, gold_ranks=[])
    assert scores.as_metrics() == {"hit_at_1": 0.5, "hit_at_3": 0.75, "hit_at_5": 1.0, "mrr_at_5": 0.7}
