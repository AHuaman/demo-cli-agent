from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Any OpenAI-compatible service (OpenAI, NVIDIA NIM, ...). OPENAI_API_KEY is
    # still accepted so existing .env files keep working.
    LLM_API_KEY: str = Field(
        validation_alias=AliasChoices("LLM_API_KEY", "OPENAI_API_KEY")
    )
    # None means OpenAI's default endpoint. OPENAI_BASE_URL is the legacy name.
    LLM_BASE_URL: str | None = Field(
        default=None, validation_alias=AliasChoices("LLM_BASE_URL", "OPENAI_BASE_URL")
    )
    MODEL_NAME: str = "gpt-4.1-mini"
    JUDGE_MODEL_NAME: str = "gpt-5.2"
    MODEL_TEMPERATURE: float = 0.0
    MLFLOW_TRACKING_URI: str = "sqlite:///mlflow.db"
    DATABASE_URL: str = "sqlite+aiosqlite:///database.db"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    @field_validator("LLM_BASE_URL", mode="before")
    @classmethod
    def _blank_base_url_is_default(cls, value: str | None) -> str | None:
        # `LLM_BASE_URL=` copied from .env.example must not become an empty URL.
        if value is None:
            return None
        return value.strip() or None


settings = Settings()
