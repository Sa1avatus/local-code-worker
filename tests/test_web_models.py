from pathlib import Path

import pytest
from pydantic import ValidationError

from local_code_worker.web_config import (
    initialize_container_settings,
    load_public_settings,
    save_provider_settings,
)
from local_code_worker.web_models import ProviderSettingsInput


def input_settings(**overrides: object) -> ProviderSettingsInput:
    values: dict[str, object] = {
        "provider": "ollama",
        "base_url": "http://localhost:11434",
        "model": "qwen:test",
        "context_length": 8192,
    }
    values.update(overrides)
    return ProviderSettingsInput.model_validate(values)


def test_provider_settings_accepts_local_and_external_providers() -> None:
    assert input_settings().model == "qwen:test"
    external = input_settings(
        provider="openai-compatible",
        base_url="https://example.test/v1",
        model=" vendor/model ",
        api_key_action="replace",
        api_key="secret",
    )
    assert external.model == "vendor/model"


@pytest.mark.parametrize(
    "base_url",
    ["http://example.test:11434", "https://ollama.example.test"],
)
def test_provider_settings_rejects_remote_ollama(base_url: str) -> None:
    with pytest.raises(ValidationError, match="loopback"):
        input_settings(base_url=base_url)


@pytest.mark.parametrize("model", ["", "   ", "bad\nmodel", "x" * 201])
def test_provider_settings_rejects_invalid_model(model: str) -> None:
    with pytest.raises(ValidationError):
        input_settings(model=model)


def test_provider_settings_requires_explicit_key_action() -> None:
    with pytest.raises(ValidationError):
        input_settings(api_key="secret")
    with pytest.raises(ValidationError):
        input_settings(api_key_action="replace", api_key="")
    with pytest.raises(ValidationError):
        input_settings(api_key_action="clear", api_key="secret")


def test_provider_settings_never_serializes_secret() -> None:
    sentinel = "UNIQUE-WEB-SECRET"
    value = input_settings(
        provider="openai-compatible",
        base_url="https://example.test/v1",
        api_key_action="replace",
        api_key=sentinel,
    )
    assert sentinel not in repr(value)
    assert sentinel not in str(value)
    assert sentinel not in value.model_dump_json()

    with pytest.raises(ValidationError) as captured:
        input_settings(api_key_action="replace", api_key=f"{sentinel}\n")
    assert sentinel not in str(captured.value)


def test_save_provider_settings_persists_key_without_returning_it(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    sentinel = "SAVED-SECRET-VALUE"
    value = input_settings(
        provider="openai-compatible",
        base_url="https://example.test/v1",
        model="vendor/model",
        api_key_action="replace",
        api_key=sentinel,
    )
    result = save_provider_settings(value, env_path)
    assert result["api_key_configured"] is True
    assert sentinel not in str(result)

    public = load_public_settings(env_path)
    assert public["provider"] == "openai-compatible"
    assert public["model"] == "vendor/model"
    assert public["context_length"] == 8192
    assert public["api_key_configured"] is True
    assert sentinel not in str(public)


def test_context_length_must_be_within_safe_bounds() -> None:
    with pytest.raises(ValueError):
        input_settings(context_length=256)


def test_save_provider_settings_clears_web_key(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    save_provider_settings(
        input_settings(
            provider="openai-compatible",
            base_url="https://example.test/v1",
            api_key_action="replace",
            api_key="secret",
        ),
        env_path,
    )
    result = save_provider_settings(
        input_settings(
            provider="openai-compatible",
            base_url="https://example.test/v1",
            api_key_action="clear",
        ),
        env_path,
    )
    assert result["api_key_configured"] is False
    assert load_public_settings(env_path)["api_key_configured"] is False


def test_switching_to_ollama_preserves_external_key(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    save_provider_settings(
        input_settings(
            provider="openai-compatible",
            base_url="https://example.test/v1",
            api_key_action="replace",
            api_key="preserved-secret",
        ),
        env_path,
    )
    save_provider_settings(input_settings(), env_path)
    restored = save_provider_settings(
        input_settings(
            provider="openai-compatible",
            base_url="https://example.test/v1",
        ),
        env_path,
    )
    assert restored["api_key_configured"] is True


def test_container_defaults_use_host_ollama(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_path = tmp_path / ".env"
    monkeypatch.setenv("LOCAL_CODE_WORKER_CONTAINER", "1")
    initialize_container_settings(env_path)
    public = load_public_settings(env_path)
    assert public["provider"] == "ollama"
    assert public["base_url"] == "http://host.docker.internal:11434/"
    assert public["api_key_configured"] is False
    assert public["api_key_env"] is None
