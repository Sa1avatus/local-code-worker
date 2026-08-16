from pathlib import Path

import pytest
from pydantic import ValidationError

from local_code_worker.web_config import (
    initialize_container_settings,
    load_public_settings,
    public_gateway_settings,
    save_gateway_settings,
    save_provider_settings,
)
from local_code_worker.web_models import GatewaySettingsInput, ProviderSettingsInput


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


def test_save_gateway_settings_round_trips_three_tiers_without_exposing_secret(
    tmp_path: Path,
) -> None:
    env_path = tmp_path / ".env"
    secret = "STRONG-CLOUD-SECRET"
    value = GatewaySettingsInput.model_validate(
        {
            "mode": "router",
            "tiers": {
                "local": {
                    "provider": "ollama",
                    "base_url": "http://localhost:11434",
                    "model": "local-reasoner",
                    "context_length": 16384,
                },
                "mid": {
                    "provider": "ollama",
                    "base_url": "http://localhost:11434",
                    "model": "local-executor",
                    "context_length": 32768,
                    "num_parallel": 1,
                },
                "strong": {
                    "provider": "openai-compatible",
                    "base_url": "https://cloud.example/v1",
                    "model": "cloud-strong",
                    "context_length": 65536,
                    "api_key_action": "replace",
                    "api_key": secret,
                },
            },
        }
    )

    result = save_gateway_settings(value, env_path)

    assert set(result["tiers"]) == {"local", "mid", "strong"}
    assert result["tiers"]["strong"]["api_key_configured"] is True
    assert result["tiers"]["mid"]["num_parallel"] == 1
    assert result["tiers"]["local"]["num_parallel"] == 1  # default
    assert secret not in str(result)
    assert secret not in str(public_gateway_settings(env_path))
    assert secret in env_path.read_text(encoding="utf-8")


def test_save_gateway_settings_round_trips_think(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    value = GatewaySettingsInput.model_validate(
        {
            "mode": "router",
            "tiers": {
                "local": {
                    "provider": "ollama",
                    "base_url": "http://localhost:11434",
                    "model": "local",
                    "context_length": 16384,
                    "think": False,
                },
                "mid": {
                    "provider": "ollama",
                    "base_url": "http://localhost:11434",
                    "model": "mid",
                    "context_length": 16384,
                    "think": True,
                },
                "strong": {
                    "provider": "openai-compatible",
                    "base_url": "https://cloud.example/v1",
                    "model": "strong",
                    "context_length": 16384,
                },
            },
        }
    )

    result = save_gateway_settings(value, env_path)

    assert result["tiers"]["local"]["think"] is False
    assert result["tiers"]["mid"]["think"] is True
    assert result["tiers"]["strong"]["think"] is None  # default = model default
    text = env_path.read_text(encoding="utf-8")
    assert "GATEWAY_LOCAL_THINK" in text
    assert "GATEWAY_MID_THINK" in text
    assert "GATEWAY_STRONG_THINK" not in text


def test_save_gateway_settings_round_trips_show_reasoning(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    value = GatewaySettingsInput.model_validate(
        {
            "mode": "router",
            "tiers": {
                "local": {
                    "provider": "ollama",
                    "base_url": "http://localhost:11434",
                    "model": "local",
                    "context_length": 16384,
                    "show_reasoning": False,
                },
                "mid": {
                    "provider": "ollama",
                    "base_url": "http://localhost:11434",
                    "model": "mid",
                    "context_length": 16384,
                    "show_reasoning": True,
                },
                "strong": {
                    "provider": "openai-compatible",
                    "base_url": "https://cloud.example/v1",
                    "model": "strong",
                    "context_length": 16384,
                },
            },
        }
    )

    result = save_gateway_settings(value, env_path)

    assert result["tiers"]["local"]["show_reasoning"] is False
    assert result["tiers"]["mid"]["show_reasoning"] is True
    assert result["tiers"]["strong"]["show_reasoning"] is None  # default = show
    text = env_path.read_text(encoding="utf-8")
    assert "GATEWAY_LOCAL_SHOW_REASONING" in text
    assert "GATEWAY_MID_SHOW_REASONING" in text
    assert "GATEWAY_STRONG_SHOW_REASONING" not in text


def test_gateway_settings_reject_invalid_num_parallel() -> None:
    base = {
        "mode": "router",
        "tiers": {
            "local": {
                "provider": "ollama",
                "base_url": "http://localhost:11434",
                "model": "local",
                "context_length": 16384,
            },
            "mid": {
                "provider": "ollama",
                "base_url": "http://localhost:11434",
                "model": "mid",
                "context_length": 16384,
            },
            "strong": {
                "provider": "openai-compatible",
                "base_url": "https://cloud.example/v1",
                "model": "strong",
                "context_length": 16384,
            },
        },
    }
    for invalid in (0, 65, -1):
        tiers = dict(base["tiers"])
        tiers["local"] = {**tiers["local"], "num_parallel": invalid}
        with pytest.raises(ValidationError):
            GatewaySettingsInput.model_validate({**base, "tiers": tiers})


def test_gateway_settings_reject_invalid_think() -> None:
    base = {
        "mode": "router",
        "tiers": {
            "local": {
                "provider": "ollama",
                "base_url": "http://localhost:11434",
                "model": "local",
                "context_length": 16384,
            },
            "mid": {
                "provider": "ollama",
                "base_url": "http://localhost:11434",
                "model": "mid",
                "context_length": 16384,
            },
            "strong": {
                "provider": "openai-compatible",
                "base_url": "https://cloud.example/v1",
                "model": "strong",
                "context_length": 16384,
            },
        },
    }
    for invalid in (2, "maybe", [True]):
        tiers = dict(base["tiers"])
        tiers["mid"] = {**tiers["mid"], "think": invalid}
        with pytest.raises(ValidationError):
            GatewaySettingsInput.model_validate({**base, "tiers": tiers})


def test_gateway_settings_require_all_three_tiers() -> None:
    with pytest.raises(ValidationError, match="LOCAL, MID, and STRONG"):
        GatewaySettingsInput.model_validate(
            {
                "mode": "router",
                "tiers": {
                    "local": {
                        "provider": "ollama",
                        "base_url": "http://localhost:11434",
                        "model": "local",
                        "context_length": 16384,
                    }
                },
            }
        )
