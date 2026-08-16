from os import environ
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values
from pydantic import Field, HttpUrl, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .models import JsonMode, ProviderName


class WorkerSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: ProviderName = ProviderName.OLLAMA
    llm_base_url: HttpUrl | None = None
    llm_model: str | None = None
    llm_api_key: SecretStr | None = None
    llm_api_key_env: str | None = None
    llm_timeout_seconds: float | None = Field(default=None, gt=0)
    llm_connect_timeout_seconds: float = Field(default=30, gt=0)
    llm_read_timeout_seconds: float | None = Field(default=None, gt=0)
    llm_num_ctx: int = Field(default=16_384, gt=0)
    # Parallel request slots for the Ollama runner. Ollama sizes the runner
    # context as num_ctx * num_parallel, so large models need 1 (the default);
    # small matching models can use several slots.
    llm_num_parallel: int = Field(default=1, gt=0)
    llm_max_output_characters: int = Field(default=100_000, gt=0)
    llm_max_output_tokens: int = Field(default=4_096, gt=0)
    llm_temperature: float = Field(default=0, ge=0)
    llm_stream: bool = True
    llm_json_mode: JsonMode = JsonMode.AUTO
    llm_keep_alive: str = "30m"
    # Per-request "don't think" flag for reasoning models (qwen3.x). None = do not
    # send the think parameter, leaving the model's default (thinking enabled).
    llm_think: bool | None = None
    # Whether to surface the model's reasoning trace to the client. None or True
    # forward it (reasoning_content / reasoning); False hides it while the model
    # still thinks internally.
    llm_show_reasoning: bool | None = None
    # Thinking intensity level for Ollama thinking models (low/medium/high/max).
    # None = model default (or plain `think: true`); a level forces thinking at
    # that depth, budgeting how much of the output goes to the chain-of-thought.
    llm_think_level: Literal["low", "medium", "high", "max"] | None = None
    llm_unload_policy: str = "immediate"  # "immediate", "never", or minutes like "5", "10", "30"
    ollama_base_url: HttpUrl | None = None
    ollama_model: str | None = None
    ollama_timeout_seconds: float | None = Field(default=None, gt=0)
    worker_command_timeout_seconds: float = Field(default=300, gt=0)
    worker_reports_directory: Path = Path(".local-worker/reports")
    worker_backups_directory: Path = Path(".local-worker/backups")
    worker_state_directory: Path = Path(".local-worker/state")

    @model_validator(mode="after")
    def resolve_legacy_ollama_settings(self) -> "WorkerSettings":
        default_base_url = HttpUrl("http://localhost:11434")
        base_url = self.llm_base_url or self.ollama_base_url or default_base_url
        model = self.llm_model or self.ollama_model or "qwen2.5-coder:3b"
        timeout = self.llm_timeout_seconds or self.ollama_timeout_seconds or 600
        read_timeout = self.llm_read_timeout_seconds or timeout
        object.__setattr__(self, "llm_base_url", base_url)
        object.__setattr__(self, "llm_model", model)
        object.__setattr__(self, "llm_timeout_seconds", timeout)
        object.__setattr__(self, "llm_read_timeout_seconds", read_timeout)
        if self.ollama_base_url is None and self.llm_provider is ProviderName.OLLAMA:
            object.__setattr__(self, "ollama_base_url", base_url)
        if self.ollama_model is None and self.llm_provider is ProviderName.OLLAMA:
            object.__setattr__(self, "ollama_model", model)
        if self.ollama_timeout_seconds is None and self.llm_provider is ProviderName.OLLAMA:
            object.__setattr__(self, "ollama_timeout_seconds", timeout)
        if self.llm_provider is ProviderName.OLLAMA and base_url.host not in {
            "localhost",
            "127.0.0.1",
            "::1",
            "host.docker.internal",
        }:
            raise ValueError("Ollama LLM_BASE_URL must use a loopback host")
        return self

    def resolve_api_key(self) -> tuple[str | None, str | None]:
        if self.llm_api_key is not None and self.llm_api_key.get_secret_value():
            return self.llm_api_key.get_secret_value(), self.llm_api_key_env
        variable_name = self.llm_api_key_env
        if variable_name is None and self.llm_base_url and self.llm_base_url.host:
            variable_name = (
                "OPENROUTER_API_KEY"
                if self.llm_base_url.host.lower() == "openrouter.ai"
                else "OPENAI_COMPATIBLE_API_KEY"
            )
        if variable_name is None:
            return None, None
        value = environ.get(variable_name)
        if value is None:
            dotenv_value = dotenv_values(".env").get(variable_name)
            value = str(dotenv_value) if dotenv_value is not None else None
        return value, variable_name

    @field_validator("ollama_base_url")
    @classmethod
    def validate_local_ollama_url(cls, url: HttpUrl | None) -> HttpUrl | None:
        if url is None:
            return None
        if url.host not in {"localhost", "127.0.0.1", "::1", "host.docker.internal"}:
            raise ValueError("OLLAMA_BASE_URL must use a loopback host")
        return url

    @field_validator(
        "worker_reports_directory",
        "worker_backups_directory",
        "worker_state_directory",
    )
    @classmethod
    def validate_worker_directory(cls, path: Path) -> Path:
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(
                "Worker output directories must be relative and remain in the repository"
            )
        return path
