import pytest
from pydantic import ValidationError

from src.settings import Settings


def test_settings_load_from_env(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:9999")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///test.db")

    settings = Settings(_env_file=None)
    assert settings.LLM_API_KEY == "test-key"
    assert settings.MLFLOW_TRACKING_URI == "http://localhost:9999"
    assert settings.DATABASE_URL == "sqlite:///test.db"


def test_settings_default_values(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    # Clear other env vars to use defaults
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)

    settings = Settings(_env_file=None)
    # Just verify they are populated (not empty)
    assert settings.MLFLOW_TRACKING_URI
    assert settings.DATABASE_URL
    assert settings.LLM_BASE_URL is None


def test_judge_and_temperature_defaults(monkeypatch):
    monkeypatch.delenv("JUDGE_MODEL_NAME", raising=False)
    monkeypatch.delenv("MODEL_TEMPERATURE", raising=False)
    s = Settings(LLM_API_KEY="test-key", _env_file=None)
    assert s.JUDGE_MODEL_NAME == "gpt-5.2"
    assert s.MODEL_TEMPERATURE == 0.0


def test_legacy_openai_api_key_still_accepted(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-key")
    assert Settings(_env_file=None).LLM_API_KEY == "legacy-key"


def test_llm_api_key_wins_over_legacy(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "new-key")
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-key")
    assert Settings(_env_file=None).LLM_API_KEY == "new-key"


def test_base_url_read_from_env(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
    assert (
        Settings(_env_file=None).LLM_BASE_URL == "https://integrate.api.nvidia.com/v1"
    )


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_base_url_means_default(monkeypatch, blank):
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_BASE_URL", blank)
    assert Settings(_env_file=None).LLM_BASE_URL is None


def test_missing_key_fails(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
