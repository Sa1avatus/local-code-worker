from os import environ
from pathlib import Path

from dotenv import dotenv_values, set_key, unset_key
from pydantic import SecretStr

from .config import WorkerSettings
from .models import ProviderName
from .web_models import ProviderSettingsInput

WEB_API_KEY_ENV = "LOCAL_CODE_WORKER_UI_API_KEY"


def public_settings(settings: WorkerSettings) -> dict[str, object]:
    if settings.llm_provider is ProviderName.OLLAMA:
        api_key, api_key_env = None, None
    else:
        api_key, api_key_env = settings.resolve_api_key()
    return {
        "provider": settings.llm_provider.value,
        "base_url": str(settings.llm_base_url),
        "model": settings.llm_model,
        "api_key_configured": bool(api_key),
        "api_key_env": api_key_env,
    }


def load_public_settings(env_path: Path = Path(".env")) -> dict[str, object]:
    return public_settings(load_web_worker_settings(env_path))


def load_web_worker_settings(env_path: Path = Path(".env")) -> WorkerSettings:
    settings = WorkerSettings(_env_file=env_path)
    if settings.llm_api_key is not None:
        return settings
    variable = settings.llm_api_key_env
    if variable is None:
        return settings
    file_values = dotenv_values(env_path)
    secret = environ.get(variable) or file_values.get(variable)
    if not secret:
        return settings
    return WorkerSettings(_env_file=env_path, llm_api_key=SecretStr(str(secret)))


def initialize_container_settings(env_path: Path) -> None:
    if env_path.exists() or environ.get("LOCAL_CODE_WORKER_CONTAINER") != "1":
        return
    save_provider_settings(
        ProviderSettingsInput.model_validate(
            {
                "provider": "ollama",
                "base_url": "http://host.docker.internal:11434",
                "model": "qwen2.5-coder:3b",
            }
        ),
        env_path,
    )


def save_provider_settings(
    value: ProviderSettingsInput,
    env_path: Path = Path(".env"),
) -> dict[str, object]:
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.touch(exist_ok=True)
    path = str(env_path)
    set_key(path, "LLM_PROVIDER", value.provider.value)
    set_key(path, "LLM_BASE_URL", str(value.base_url).rstrip("/"))
    set_key(path, "LLM_MODEL", value.model)

    if value.api_key_action == "clear":
        unset_key(path, "LLM_API_KEY")
        unset_key(path, "LLM_API_KEY_ENV")
        unset_key(path, WEB_API_KEY_ENV)
    elif value.api_key_action == "replace":
        assert value.api_key is not None
        unset_key(path, "LLM_API_KEY")
        set_key(path, "LLM_API_KEY_ENV", WEB_API_KEY_ENV)
        set_key(path, WEB_API_KEY_ENV, value.api_key.get_secret_value())

    api_key_configured = False
    api_key_env: str | None = None
    if value.provider is ProviderName.OPENAI_COMPATIBLE:
        if value.api_key_action == "replace":
            api_key_configured = True
            api_key_env = WEB_API_KEY_ENV
        elif value.api_key_action == "keep":
            current = load_web_worker_settings(env_path)
            key, api_key_env = current.resolve_api_key()
            api_key_configured = bool(key)
    return {
        "provider": value.provider.value,
        "base_url": str(value.base_url).rstrip("/"),
        "model": value.model,
        "api_key_configured": api_key_configured,
        "api_key_env": api_key_env,
    }
