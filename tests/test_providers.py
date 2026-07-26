import json
from pathlib import Path

import httpx
import pytest

from local_code_worker.cli import create_parser, provider_check, settings_from_arguments
from local_code_worker.config import WorkerSettings
from local_code_worker.exceptions import ProviderConfigurationError, ProviderError
from local_code_worker.models import JsonMode, ProviderName
from local_code_worker.providers.factory import create_provider
from local_code_worker.providers.ollama import OllamaProvider
from local_code_worker.providers.openai_compatible import OpenAICompatibleProvider


def ollama_settings(**overrides: object) -> WorkerSettings:
    values: dict[str, object] = {
        "llm_provider": "ollama",
        "llm_model": "qwen:test",
        "llm_stream": True,
    }
    values.update(overrides)
    return WorkerSettings(_env_file=None, **values)


def openai_settings(**overrides: object) -> WorkerSettings:
    values: dict[str, object] = {
        "llm_provider": "openai-compatible",
        "llm_base_url": "https://example.test/v1",
        "llm_model": "qwen/test:free",
        "llm_api_key": "secret-value",
    }
    values.update(overrides)
    return WorkerSettings(_env_file=None, **values)


def test_factory_selects_configured_provider() -> None:
    assert isinstance(create_provider(ollama_settings()), OllamaProvider)
    assert isinstance(create_provider(openai_settings()), OpenAICompatibleProvider)


def test_ollama_tags_and_missing_model() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"models": [{"name": "other:model"}]})
    )
    provider = OllamaProvider(ollama_settings(), transport)
    health = provider.check_connection()
    assert provider.list_models() == ["other:model"]
    assert health.reachable is True
    assert health.model_available is False


def test_ollama_stream_collects_chunks_and_done_reason() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["format"] == {"type": "object"}
        assert body["stream"] is True
        return httpx.Response(
            200,
            text="\n".join(
                [
                    '{"message":{"content":"{\\"ok\\":"},"done":false}',
                    '{"message":{"content":"true}"},"done":true,"done_reason":"stop"}',
                ]
            ),
        )

    provider = OllamaProvider(ollama_settings(), httpx.MockTransport(handler))
    assert provider.chat([], {"type": "object"}, 100) == '{"ok":true}'
    assert provider.last_generation_metadata
    assert provider.last_generation_metadata.finish_reason == "stop"


@pytest.mark.parametrize(
    ("payload", "category"),
    [
        ("not-json\n", "invalid_stream_chunk"),
        ('{"message":{"content":"x"},"done":false}\n', "truncated_stream"),
    ],
)
def test_ollama_rejects_invalid_or_unfinished_stream(payload: str, category: str) -> None:
    provider = OllamaProvider(
        ollama_settings(),
        httpx.MockTransport(lambda request: httpx.Response(200, text=payload)),
    )
    with pytest.raises(ProviderError) as captured:
        provider.chat([], None, 100)
    assert captured.value.category == category


def test_ollama_stream_enforces_output_limit() -> None:
    provider = OllamaProvider(
        ollama_settings(),
        httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                text='{"message":{"content":"too long"},"done":true}\n',
            )
        ),
    )
    with pytest.raises(ProviderError) as captured:
        provider.chat([], None, 3)
    assert captured.value.category == "output_limit"


def test_ollama_read_error_is_classified_as_transport_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadError("broken TLS record", request=request)

    provider = OllamaProvider(ollama_settings(), httpx.MockTransport(handler))
    with pytest.raises(ProviderError) as captured:
        provider.chat([], None, 100)
    assert captured.value.category == "transport_error"
    assert "broken TLS record" not in str(captured.value)


def test_openai_key_loads_from_named_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUSTOM_LLM_KEY", "loaded-secret")
    settings = openai_settings(llm_api_key=None, llm_api_key_env="CUSTOM_LLM_KEY")
    provider = OpenAICompatibleProvider(settings)
    assert provider.api_key == "loaded-secret"
    assert provider.api_key_env == "CUSTOM_LLM_KEY"


def test_openai_key_loads_from_dotenv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DOTENV_LLM_KEY", raising=False)
    (tmp_path / ".env").write_text("DOTENV_LLM_KEY=dotenv-secret\n", encoding="utf-8")
    settings = openai_settings(llm_api_key=None, llm_api_key_env="DOTENV_LLM_KEY")
    provider = OpenAICompatibleProvider(settings)
    assert provider.api_key == "dotenv-secret"
    assert provider.api_key_env == "DOTENV_LLM_KEY"


def test_openai_missing_key_names_required_variable() -> None:
    settings = openai_settings(
        llm_api_key=None,
        llm_api_key_env="MISSING_TEST_KEY",
    )
    provider = OpenAICompatibleProvider(settings)
    with pytest.raises(
        ProviderConfigurationError,
        match="Environment variable MISSING_TEST_KEY is not set",
    ):
        provider.list_models()


def test_openai_report_base_url_removes_query_and_userinfo() -> None:
    provider = OpenAICompatibleProvider(
        openai_settings(llm_base_url="https://username:password@example.test/v1?token=secret")
    )
    assert provider.base_url == "https://example.test/v1"


def test_openai_stream_collects_sse_usage_and_finish_reason() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer secret-value"
        body = json.loads(request.content)
        assert body["model"] == "qwen/test:free"
        assert body["response_format"] == {"type": "json_object"}
        return httpx.Response(
            200,
            text="\n".join(
                [
                    'data: {"choices":[{"delta":{"content":"{\\"ok\\":"},"finish_reason":null}]}',
                    'data: {"choices":[{"delta":{"content":"true}"},'
                    '"finish_reason":"stop"}],"usage":{"prompt_tokens":3,'
                    '"completion_tokens":2,"total_tokens":5}}',
                    "data: [DONE]",
                ]
            ),
        )

    provider = OpenAICompatibleProvider(openai_settings(), httpx.MockTransport(handler))
    assert provider.chat([], {"type": "object"}, 100) == '{"ok":true}'
    assert provider.last_generation_metadata
    assert provider.last_generation_metadata.finish_reason == "stop"
    assert provider.last_generation_metadata.usage["total_tokens"] == 5


def test_openai_non_stream_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": '{"ok":true}'},
                        "finish_reason": "length",
                    }
                ],
                "usage": {"total_tokens": 9},
            },
        )

    provider = OpenAICompatibleProvider(
        openai_settings(llm_stream=False),
        httpx.MockTransport(handler),
    )
    assert provider.chat([], None, 100) == '{"ok":true}'
    assert provider.last_generation_metadata
    assert provider.last_generation_metadata.finish_reason == "length"


@pytest.mark.parametrize("status", [401, 402, 404, 429, 500])
def test_openai_http_errors_have_stable_category(status: int) -> None:
    provider = OpenAICompatibleProvider(
        openai_settings(llm_stream=False),
        httpx.MockTransport(lambda request: httpx.Response(status)),
    )
    with pytest.raises(ProviderError) as captured:
        provider.chat([], None, 100)
    assert captured.value.category == f"http_{status}"
    assert "secret-value" not in str(captured.value)


def test_openai_malformed_sse_is_rejected() -> None:
    provider = OpenAICompatibleProvider(
        openai_settings(),
        httpx.MockTransport(lambda request: httpx.Response(200, text="bad line\n")),
    )
    with pytest.raises(ProviderError) as captured:
        provider.chat([], None, 100)
    assert captured.value.category == "invalid_stream_chunk"


def test_openai_read_error_is_classified_as_transport_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadError("broken TLS record", request=request)

    provider = OpenAICompatibleProvider(
        openai_settings(),
        httpx.MockTransport(handler),
    )
    with pytest.raises(ProviderError) as captured:
        provider.chat([], None, 100)
    assert captured.value.category == "transport_error"
    assert "broken TLS record" not in str(captured.value)


def test_auto_json_mode_falls_back_only_for_unsupported_response_format() -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            return httpx.Response(
                400,
                text="response_format is not supported",
                request=request,
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]},
        )

    provider = OpenAICompatibleProvider(
        openai_settings(llm_stream=False),
        httpx.MockTransport(handler),
    )
    assert provider.chat([], {"type": "object"}, 100) == "{}"
    assert "response_format" in requests[0]
    assert "response_format" not in requests[1]


def test_provider_does_not_change_selected_model() -> None:
    seen_models: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_models.append(json.loads(request.content)["model"])
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]},
        )

    provider = OpenAICompatibleProvider(
        openai_settings(llm_stream=False, llm_json_mode="prompt-only"),
        httpx.MockTransport(handler),
    )
    provider.chat([], None, 100)
    assert seen_models == ["qwen/test:free"]


def test_cli_values_override_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MODEL", "environment-model")
    parsed = create_parser().parse_args(
        [
            "run",
            "--task",
            str(Path("task.json")),
            "--provider",
            "openai-compatible",
            "--model",
            "cli-model",
            "--json-mode",
            "prompt-only",
        ]
    )
    settings = settings_from_arguments(parsed)
    assert settings.llm_provider is ProviderName.OPENAI_COMPATIBLE
    assert settings.llm_model == "cli-model"
    assert settings.llm_json_mode is JsonMode.PROMPT_ONLY


def test_legacy_cli_model_defaults_to_ollama(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    parsed = create_parser().parse_args(
        ["run", "--task", str(Path("task.json")), "--model", "legacy-model"]
    )
    settings = settings_from_arguments(parsed)
    assert settings.llm_provider is ProviderName.OLLAMA
    assert settings.llm_model == "legacy-model"


def test_provider_check_output_does_not_contain_api_key(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={"data": [{"id": "qwen/test:free"}]},
        )
    )
    settings = openai_settings()
    provider = OpenAICompatibleProvider(settings, transport)
    monkeypatch.setattr("local_code_worker.cli.create_provider", lambda configured: provider)
    assert provider_check(settings) == 0
    output = capsys.readouterr().out
    assert "secret-value" not in output
