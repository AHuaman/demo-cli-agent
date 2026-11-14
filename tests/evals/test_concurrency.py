import os

from tests.evals.concurrency import cap_eval_workers


def test_cap_eval_workers_defaults_to_two(monkeypatch):
    # delenv records the variable as absent, so monkeypatch removes it again
    # after the test even though cap_eval_workers sets it.
    monkeypatch.delenv("MLFLOW_GENAI_EVAL_MAX_WORKERS", raising=False)
    cap_eval_workers()
    assert os.environ["MLFLOW_GENAI_EVAL_MAX_WORKERS"] == "2"


def test_cap_eval_workers_keeps_explicit_value(monkeypatch):
    monkeypatch.setenv("MLFLOW_GENAI_EVAL_MAX_WORKERS", "8")
    cap_eval_workers()
    assert os.environ["MLFLOW_GENAI_EVAL_MAX_WORKERS"] == "8"
