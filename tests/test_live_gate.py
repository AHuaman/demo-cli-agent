import pytest

from tests.live import live_evals_enabled


@pytest.mark.parametrize(
    "value,expected", [("1", True), ("0", False), ("true", False), ("", False)]
)
def test_live_gate_only_exact_one(monkeypatch, value, expected):
    monkeypatch.setenv("RUN_LIVE_EVALS", value)
    assert live_evals_enabled() is expected


def test_live_gate_unset(monkeypatch):
    monkeypatch.delenv("RUN_LIVE_EVALS", raising=False)
    assert live_evals_enabled() is False
