"""Application settings loaded from environment variables.

All secrets are read from the environment / a local `.env` file.
See `.env.example` for the list of available variables.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Model
    model_provider: str = Field(default="groq", alias="MODEL_PROVIDER")
    model_name: str = Field(default="llama-3.3-70b-versatile", alias="MODEL_NAME")

    # API keys (kept secret; never log these)
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")

    # Ollama (fully-local option)
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")

    # Retrieval
    retriever_top_k: int = Field(default=4, alias="RETRIEVER_TOP_K")
    retrieval_min_score: float = Field(default=2.0, alias="RETRIEVAL_MIN_SCORE")

    # Data
    knowledge_base_path: Path = Field(
        default=BASE_DIR / "data" / "knowledge_base.csv", alias="KNOWLEDGE_BASE_PATH"
    )

    # Observability
    langchain_tracing: bool = Field(default=True, alias="LANGCHAIN_TRACING_V2")
    langchain_api_key: str = Field(default="", alias="LANGCHAIN_API_KEY")
    langchain_project: str = Field(default="kb-assistant", alias="LANGCHAIN_PROJECT")

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_file: Path = Field(default=BASE_DIR / "logs" / "assistant.log", alias="LOG_FILE")

    @property
    def has_langsmith(self) -> bool:
        """LangSmith is only enabled when credentials are actually present."""
        return bool(self.langchain_tracing and self.langchain_api_key)

    @property
    def api_key(self) -> str:
        """The API key matching the configured provider."""
        return {
            "groq": self.groq_api_key,
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
        }.get(self.model_provider, "")


@lru_cache
def get_settings() -> Settings:
    return Settings()
