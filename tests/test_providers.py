import json
from pathlib import Path

import httpx
import pytest

from local_code_worker.cli import create_parser, provider_check, settings_from_arguments
from local_code_worker.config import WorkerSettings
from local_code_worker.exceptions import ProviderError
from local_code_worker.models import JsonMode, ProviderName
from local_code_worker.providers.base import (
    ProviderCapability,
    ProviderFunctionTool,
    ProviderFunctionToolChoice,
    ProviderMessage,
    ProviderReasoningDeltaEvent,
    ProviderRequest,
    ProviderTextDeltaEvent,
    ProviderUsageEvent,
    validate_event_sequence,
)
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
    ollama = create_provider(ollama_settings())
    openai = create_provider(openai_settings())
    assert isinstance(ollama, OllamaProvider)
    assert isinstance(openai, OpenAICompatibleProvider)
    for provider in (ollama, openai):
        assert provider.capabilities.supports(ProviderCapability.STREAMING)
        assert provider.capabilities.supports(ProviderCapability.JSON_SCHEMA)
        assert provider.capabilities.supports(ProviderCapability.USAGE)
        assert provider.capabilities.supports(ProviderCapability.FUNCTION_TOOLS)


def test_ollama_tags_and_missing_model() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"models": [{"name": "other:model"}]})
    )
    provider = OllamaProvider(ollama_settings(), transport)
    health = provider.check_connection()
    assert provider.list_models() == ["other:model"]
    assert health.reachable is True
    assert health.model_available is False


def test_ollama_running_models_returns_safe_runtime_fields() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/ps"
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "name": "qwen:test",
                        "size": 12_000,
                        "size_vram": 10_000,
                        "context_length": 16_384,
                        "expires_at": "2026-08-01T12:00:00Z",
                        "digest": "must-not-be-exposed",
                    }
                ]
            },
        )

    provider = OllamaProvider(ollama_settings(), httpx.MockTransport(handler))
    assert provider.running_models() == [
        {
            "name": "qwen:test",
            "size": 12_000,
            "size_vram": 10_000,
            "context_length": 16_384,
            "expires_at": "2026-08-01T12:00:00Z",
        }
    ]


def test_ollama_unload_uses_zero_keep_alive() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/generate"
        assert json.loads(request.content) == {"model": "qwen:test", "keep_alive": 0}
        return httpx.Response(200, json={"done": True})

    provider = OllamaProvider(ollama_settings(), httpx.MockTransport(handler))

    provider.unload_model()


def test_ollama_stream_collects_chunks_and_done_reason() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["format"] == "json"
        assert body["stream"] is True
        assert body["options"]["num_predict"] == 256
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
    assert provider.chat([], {"type": "object"}, 100, 256) == '{"ok":true}'
    assert provider.last_generation_metadata
    assert provider.last_generation_metadata.finish_reason == "stop"


def test_ollama_stream_emits_ordered_events_and_ttft() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            text="\n".join(
                [
                    '{"message":{"content":"hel"},"done":false}',
                    '{"message":{"content":"lo"},"done":false}',
                    '{"message":{"content":""},"done":true,"done_reason":"stop",'
                    '"prompt_eval_count":2,"eval_count":1}',
                ]
            ),
        )
    )
    provider = OllamaProvider(ollama_settings(), transport)
    request = ProviderRequest(
        messages=[ProviderMessage(role="user", content="hello")],
        response_schema={"type": "object"},
        max_output_characters=100,
        max_output_tokens=20,
        json_mode=JsonMode.AUTO,
        stream=True,
    )

    events = list(provider.stream(request))

    validate_event_sequence(events)
    assert [event.delta for event in events if isinstance(event, ProviderTextDeltaEvent)] == [
        "hel",
        "lo",
    ]
    usage_event = next(event for event in events if isinstance(event, ProviderUsageEvent))
    assert usage_event.usage.total_tokens == 3
    assert provider.last_generation_metadata is not None
    assert provider.last_generation_metadata.time_to_first_token_ms is not None


def test_ollama_stream_emits_reasoning_before_content() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            text="\n".join(
                [
                    '{"message":{"thinking":"Let me "},"done":false}',
                    '{"message":{"thinking":"think."},"done":false}',
                    '{"message":{"content":"answer"},"done":false}',
                    '{"message":{"content":""},"done":true,"done_reason":"stop"}',
                ]
            ),
        )
    )
    provider = OllamaProvider(ollama_settings(), transport)
    request = ProviderRequest(
        messages=[ProviderMessage(role="user", content="hello")],
        max_output_characters=100,
        json_mode=JsonMode.AUTO,
        stream=True,
    )

    events = list(provider.stream(request))

    validate_event_sequence(events)
    reasoning = [event.delta for event in events if isinstance(event, ProviderReasoningDeltaEvent)]
    content = [event.delta for event in events if isinstance(event, ProviderTextDeltaEvent)]
    assert reasoning == ["Let me ", "think."]
    assert content == ["answer"]
    reasoning_positions = [
        index for index, event in enumerate(events)
        if isinstance(event, ProviderReasoningDeltaEvent)
    ]
    content_positions = [
        index for index, event in enumerate(events)
        if isinstance(event, ProviderTextDeltaEvent)
    ]
    assert max(reasoning_positions) < min(content_positions)
    assert provider.last_generation_metadata is not None
    assert provider.last_generation_metadata.reasoning == "Let me think."


def test_ollama_stream_hides_reasoning_when_show_reasoning_false() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            text="\n".join(
                [
                    '{"message":{"thinking":"hidden"},"done":false}',
                    '{"message":{"content":"answer"},"done":false}',
                    '{"message":{"content":""},"done":true,"done_reason":"stop"}',
                ]
            ),
        )
    )
    provider = OllamaProvider(ollama_settings(llm_show_reasoning=False), transport)
    request = ProviderRequest(
        messages=[ProviderMessage(role="user", content="hello")],
        max_output_characters=100,
        json_mode=JsonMode.AUTO,
        stream=True,
    )

    events = list(provider.stream(request))

    assert not any(isinstance(event, ProviderReasoningDeltaEvent) for event in events)
    content = [event.delta for event in events if isinstance(event, ProviderTextDeltaEvent)]
    assert content == ["answer"]
    assert provider.last_generation_metadata is not None
    assert provider.last_generation_metadata.reasoning is None


def test_ollama_stream_cancellation_closes_http_stream() -> None:
    class TrackingStream(httpx.SyncByteStream):
        def __init__(self) -> None:
            self.closed = False

        def __iter__(self):
            yield b'{"message":{"content":"first"},"done":false}\n'
            yield b'{"message":{"content":"second"},"done":true}\n'

        def close(self) -> None:
            self.closed = True

    tracking_stream = TrackingStream()
    provider = OllamaProvider(
        ollama_settings(),
        httpx.MockTransport(lambda request: httpx.Response(200, stream=tracking_stream)),
    )
    request = ProviderRequest(
        messages=[ProviderMessage(role="user", content="hello")],
        max_output_characters=100,
        json_mode=JsonMode.AUTO,
        stream=True,
    )
    events = provider.stream(request)

    next(events)
    delta = next(events)
    events.close()

    assert isinstance(delta, ProviderTextDeltaEvent)
    assert tracking_stream.closed is True
    assert provider.last_generation_metadata is None


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


def test_ollama_explicit_json_schema_mode_sends_schema() -> None:
    schema = {"type": "object"}

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["format"] == schema
        return httpx.Response(200, text='{"message":{"content":"{}"},"done":true}\n')

    provider = OllamaProvider(
        ollama_settings(llm_json_mode="json-schema"),
        httpx.MockTransport(handler),
    )
    assert provider.chat([], schema, 100, 32) == "{}"


def test_ollama_chat_disables_thinking() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        # Reasoning models (qwen3.x) must not drain the token budget on
        # thinking when the client asked for it.
        assert payload["think"] is False
        return httpx.Response(200, text='{"message":{"content":"ok"},"done":true}\n')

    provider = OllamaProvider(
        ollama_settings(llm_think=False),
        httpx.MockTransport(handler),
    )
    assert provider.chat([{"role": "user", "content": "Say OK"}], None, 100, 32) == "ok"


def test_ollama_chat_omits_think_when_not_configured() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        # Absent llm_think: leave the model's default (thinking enabled) —
        # the gateway also serves interactive clients that want reasoning.
        assert "think" not in payload
        return httpx.Response(200, text='{"message":{"content":"ok"},"done":true}\n')

    provider = OllamaProvider(
        ollama_settings(),
        httpx.MockTransport(handler),
    )
    assert provider.chat([{"role": "user", "content": "Say OK"}], None, 100, 32) == "ok"


def test_ollama_chat_sends_think_level() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["think"] == "high"
        return httpx.Response(200, text='{"message":{"content":"ok"},"done":true}\n')

    provider = OllamaProvider(
        ollama_settings(llm_think_level="high"),
        httpx.MockTransport(handler),
    )
    assert provider.chat([{"role": "user", "content": "Say OK"}], None, 100, 32) == "ok"


def test_ollama_chat_think_false_wins_over_level() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["think"] is False
        return httpx.Response(200, text='{"message":{"content":"ok"},"done":true}\n')

    provider = OllamaProvider(
        ollama_settings(llm_think=False, llm_think_level="high"),
        httpx.MockTransport(handler),
    )
    assert provider.chat([{"role": "user", "content": "Say OK"}], None, 100, 32) == "ok"


def test_ollama_chat_sends_repeat_penalty_and_seed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        options = json.loads(request.content)["options"]
        # .env-only sampling knobs are forwarded to Ollama when set.
        assert options["repeat_penalty"] == 1.1
        assert options["seed"] == 42
        return httpx.Response(200, text='{"message":{"content":"ok"},"done":true}\n')

    provider = OllamaProvider(
        ollama_settings(llm_repeat_penalty=1.1, llm_seed=42),
        httpx.MockTransport(handler),
    )
    assert provider.chat([{"role": "user", "content": "Say OK"}], None, 100, 32) == "ok"


def test_ollama_chat_omits_sampling_knobs_by_default() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        options = json.loads(request.content)["options"]
        assert "repeat_penalty" not in options
        assert "seed" not in options
        return httpx.Response(200, text='{"message":{"content":"ok"},"done":true}\n')

    provider = OllamaProvider(
        ollama_settings(),
        httpx.MockTransport(handler),
    )
    assert provider.chat([{"role": "user", "content": "Say OK"}], None, 100, 32) == "ok"


def test_ollama_chat_captures_thinking() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text='{"message":{"content":"ok","thinking":"step 1 step 2"},'
            '"done":true,"eval_count":5,"prompt_eval_count":10}\n',
        )

    provider = OllamaProvider(
        ollama_settings(llm_think=True),
        httpx.MockTransport(handler),
    )
    assert provider.chat([{"role": "user", "content": "Say OK"}], None, 100, 32) == "ok"
    assert provider.last_generation_metadata is not None
    assert provider.last_generation_metadata.reasoning == "step 1 step 2"


def test_ollama_chat_uses_nonzero_temperature_when_thinking_enabled() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        options = json.loads(request.content)["options"]
        # Greedy decoding makes qwen3.x thinking loop; a non-zero temperature is
        # required whenever thinking is not explicitly disabled.
        assert options["temperature"] == 0.6
        return httpx.Response(200, text='{"message":{"content":"ok"},"done":true}\n')

    provider = OllamaProvider(
        ollama_settings(llm_think=True, llm_temperature=0),
        httpx.MockTransport(handler),
    )
    assert provider.chat([{"role": "user", "content": "Say OK"}], None, 100, 32) == "ok"


def test_ollama_chat_keeps_zero_temperature_when_thinking_disabled() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        options = json.loads(request.content)["options"]
        assert options["temperature"] == 0
        return httpx.Response(200, text='{"message":{"content":"ok"},"done":true}\n')

    provider = OllamaProvider(
        ollama_settings(llm_think=False, llm_temperature=0),
        httpx.MockTransport(handler),
    )
    assert provider.chat([{"role": "user", "content": "Say OK"}], None, 100, 32) == "ok"


def test_ollama_chat_sends_num_parallel_when_configured() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        options = json.loads(request.content)["options"]
        assert options["num_parallel"] == 4
        assert options["num_ctx"] == 4096
        return httpx.Response(200, text='{"message":{"content":"ok"},"done":true}\n')

    provider = OllamaProvider(
        ollama_settings(llm_num_ctx=4096, llm_num_parallel=4),
        httpx.MockTransport(handler),
    )
    assert provider.chat([{"role": "user", "content": "Say OK"}], None, 100, 32) == "ok"


def test_ollama_chat_sends_num_parallel_one_by_default() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        # Default parallelism is 1 slot: large models must not have their
        # runner context multiplied (Ollama sizes context as num_ctx * num_parallel).
        assert json.loads(request.content)["options"]["num_parallel"] == 1
        return httpx.Response(200, text='{"message":{"content":"ok"},"done":true}\n')

    provider = OllamaProvider(
        ollama_settings(),
        httpx.MockTransport(handler),
    )
    assert provider.chat([{"role": "user", "content": "Say OK"}], None, 100, 32) == "ok"


@pytest.mark.parametrize(
    ("json_mode", "category"),
    [
        ("auto", "provider_stream_error"),
        ("json-schema", "structured_output_error"),
    ],
)
def test_ollama_stream_error_chunk_is_classified_without_body(
    json_mode: str,
    category: str,
) -> None:
    provider = OllamaProvider(
        ollama_settings(llm_json_mode=json_mode),
        httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                text='{"error":"SENSITIVE peg-native failure"}\n',
            )
        ),
    )

    with pytest.raises(ProviderError) as captured:
        provider.chat([], {"type": "object"}, 100)

    assert captured.value.category == category
    assert "SENSITIVE" not in str(captured.value)


def test_ollama_non_stream_error_is_classified_without_body() -> None:
    provider = OllamaProvider(
        ollama_settings(llm_stream=False),
        httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={"error": "SENSITIVE provider failure"},
            )
        ),
    )

    with pytest.raises(ProviderError) as captured:
        provider.chat([], None, 100)

    assert captured.value.category == "provider_stream_error"
    assert "SENSITIVE" not in str(captured.value)


def test_ollama_non_stream_function_call_is_normalized() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["tools"][0]["function"]["name"] == "lookup"
        assert body["tool_choice"]["function"]["name"] == "lookup"
        return httpx.Response(
            200,
            json={
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_ollama",
                            "function": {
                                "name": "lookup",
                                "arguments": {"key": "value"},
                            },
                        }
                    ],
                },
                "done_reason": "stop",
            },
        )

    provider = OllamaProvider(
        ollama_settings(llm_stream=False, llm_json_mode="none"),
        httpx.MockTransport(handler),
    )
    provider.chat(
        [],
        None,
        100,
        tools=[ProviderFunctionTool(name="lookup", parameters={"type": "object"})],
        tool_choice=ProviderFunctionToolChoice(name="lookup"),
    )

    assert provider.last_generation_metadata is not None
    assert provider.last_generation_metadata.function_calls[0].model_dump() == {
        "call_id": "call_ollama",
        "name": "lookup",
        "arguments": '{"key":"value"}',
    }


def test_ollama_text_tool_envelope_is_normalized_for_configured_function() -> None:
    provider = OllamaProvider(
        ollama_settings(llm_stream=False, llm_json_mode="none"),
        httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "message": {
                        "content": '<tools>{"name":"lookup","arguments":{"key":"alpha"}}</tools>'
                    },
                    "done": True,
                },
            )
        ),
    )

    content = provider.chat(
        [],
        None,
        200,
        tools=[ProviderFunctionTool(name="lookup", parameters={"type": "object"})],
        tool_choice=ProviderFunctionToolChoice(name="lookup"),
    )

    assert content == ""
    assert provider.last_generation_metadata is not None
    assert provider.last_generation_metadata.function_calls[0].model_dump() == {
        "call_id": "call_ollama_text_0",
        "name": "lookup",
        "arguments": '{"key":"alpha"}',
    }


def test_ollama_text_tool_envelope_does_not_allow_unknown_function() -> None:
    provider = OllamaProvider(
        ollama_settings(llm_stream=False, llm_json_mode="none"),
        httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "message": {"content": '<tools>{"name":"unknown","arguments":{}}</tools>'},
                    "done": True,
                },
            )
        ),
    )

    content = provider.chat(
        [],
        None,
        200,
        tools=[ProviderFunctionTool(name="lookup", parameters={"type": "object"})],
    )

    assert content.startswith("<tools>")
    assert provider.last_generation_metadata is not None
    assert provider.last_generation_metadata.function_calls == []


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


def test_openai_list_models_without_key_sends_no_authorization() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "Authorization" not in request.headers
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"object": "list", "data": [{"id": "free-model"}]})

    settings = openai_settings(llm_api_key=None, llm_api_key_env="MISSING_TEST_KEY")
    provider = OpenAICompatibleProvider(settings, httpx.MockTransport(handler))
    assert provider.list_models() == ["free-model"]


@pytest.mark.parametrize(
    ("base_url", "expected_path"),
    [
        ("https://example.test/v1", "/v1/models"),
        ("https://example.test/v1/", "/v1/models"),
        ("https://example.test", "/v1/models"),
    ],
)
def test_openai_models_endpoint_normalizes_v1(
    base_url: str, expected_path: str
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == expected_path
        assert request.headers["Authorization"] == "Bearer secret-value"
        return httpx.Response(200, json={"object": "list", "data": [{"id": "m1"}]})

    provider = OpenAICompatibleProvider(
        openai_settings(llm_base_url=base_url), httpx.MockTransport(handler)
    )
    assert provider.list_models() == ["m1"]


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
        assert body["max_tokens"] == 256
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
    assert provider.chat([], {"type": "object"}, 100, 256) == '{"ok":true}'
    assert provider.last_generation_metadata
    assert provider.last_generation_metadata.finish_reason == "stop"
    assert provider.last_generation_metadata.usage["total_tokens"] == 5


def test_openai_stream_emits_ordered_events_and_ttft() -> None:
    provider = OpenAICompatibleProvider(
        openai_settings(),
        httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                text="\n".join(
                    [
                        'data: {"choices":[{"delta":{"content":"hel"},"finish_reason":null}]}',
                        'data: {"choices":[{"delta":{"content":"lo"},'
                        '"finish_reason":"stop"}],"usage":{"prompt_tokens":2,'
                        '"completion_tokens":1}}',
                        "data: [DONE]",
                    ]
                ),
            )
        ),
    )
    request = ProviderRequest(
        messages=[ProviderMessage(role="user", content="hello")],
        max_output_characters=100,
        json_mode=JsonMode.JSON_OBJECT,
        stream=True,
    )

    events = list(provider.stream(request))

    validate_event_sequence(events)
    assert [event.delta for event in events if isinstance(event, ProviderTextDeltaEvent)] == [
        "hel",
        "lo",
    ]
    usage_event = next(event for event in events if isinstance(event, ProviderUsageEvent))
    assert usage_event.usage.total_tokens == 3
    assert provider.last_generation_metadata is not None
    assert provider.last_generation_metadata.time_to_first_token_ms is not None


def test_openai_stream_cancellation_closes_http_stream() -> None:
    class TrackingStream(httpx.SyncByteStream):
        def __init__(self) -> None:
            self.closed = False

        def __iter__(self):
            yield b'data: {"choices":[{"delta":{"content":"first"}}]}\n\n'
            yield b'data: {"choices":[{"delta":{"content":"second"}}]}\n\n'
            yield b"data: [DONE]\n\n"

        def close(self) -> None:
            self.closed = True

    tracking_stream = TrackingStream()
    provider = OpenAICompatibleProvider(
        openai_settings(),
        httpx.MockTransport(lambda request: httpx.Response(200, stream=tracking_stream)),
    )
    request = ProviderRequest(
        messages=[ProviderMessage(role="user", content="hello")],
        max_output_characters=100,
        json_mode=JsonMode.JSON_OBJECT,
        stream=True,
    )
    events = provider.stream(request)

    next(events)
    delta = next(events)
    events.close()

    assert isinstance(delta, ProviderTextDeltaEvent)
    assert tracking_stream.closed is True
    assert provider.last_generation_metadata is None


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


def test_openai_non_stream_function_call_is_normalized() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["tools"][0]["function"]["strict"] is True
        assert body["tool_choice"]["function"]["name"] == "lookup"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_openai",
                                    "type": "function",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": '{"key":"value"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        )

    provider = OpenAICompatibleProvider(
        openai_settings(llm_stream=False, llm_json_mode="none"),
        httpx.MockTransport(handler),
    )
    provider.chat(
        [],
        None,
        100,
        tools=[ProviderFunctionTool(name="lookup", parameters={"type": "object"})],
        tool_choice=ProviderFunctionToolChoice(name="lookup"),
    )

    assert provider.last_generation_metadata is not None
    assert provider.last_generation_metadata.function_calls[0].call_id == "call_openai"
    assert provider.last_generation_metadata.function_calls[0].arguments == '{"key":"value"}'


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


def test_ollama_pull_model_streams_progress_and_success() -> None:
    handler_called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal handler_called
        handler_called = True
        assert request.url.path == "/api/pull"
        body = json.loads(request.content)
        assert body["name"] == "qwen:test"
        assert body["stream"] is True
        return httpx.Response(
            200,
            text="\n".join(
                [
                    '{"status": "pulling manifest"}',
                    "",
                    '{"status": "success"}',
                ]
            ),
        )

    provider = OllamaProvider(ollama_settings(), httpx.MockTransport(handler))
    chunks = list(provider.pull_model("qwen:test"))
    assert handler_called is True
    assert chunks == [
        {"status": "pulling manifest"},
        {"status": "success"},
    ]


@pytest.mark.parametrize("name", ["", "   ", "foo\nbar"])
def test_ollama_pull_model_rejects_invalid_names(name: str) -> None:
    handler_called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal handler_called
        handler_called = True
        return httpx.Response(200, text="")

    provider = OllamaProvider(ollama_settings(), httpx.MockTransport(handler))
    with pytest.raises(ValueError):
        list(provider.pull_model(name))
    assert handler_called is False


@pytest.mark.parametrize(
    ("payload", "category"),
    [
        ("not-json\n", "invalid_stream_chunk"),
        ("[]\n", "invalid_stream_chunk"),
        ('{"status":"pulling"}\n', "truncated_stream"),
    ],
)
def test_ollama_pull_model_rejects_invalid_or_truncated_stream(
    payload: str,
    category: str,
) -> None:
    provider = OllamaProvider(
        ollama_settings(),
        httpx.MockTransport(lambda request: httpx.Response(200, text=payload)),
    )
    with pytest.raises(ProviderError) as captured:
        list(provider.pull_model("qwen:test"))
    assert captured.value.category == category


def test_ollama_pull_model_http_error_does_not_expose_body() -> None:
    provider = OllamaProvider(
        ollama_settings(),
        httpx.MockTransport(lambda request: httpx.Response(500, text="SENSITIVE-RESPONSE-BODY")),
    )
    with pytest.raises(ProviderError) as captured:
        list(provider.pull_model("qwen:test"))
    assert captured.value.category == "http_error"
    assert "SENSITIVE-RESPONSE-BODY" not in str(captured.value)


# --- Streaming tool-call tests ---


def test_ollama_stream_yields_tool_calls_event() -> None:
    """Ollama streaming should yield ProviderToolCallsEvent instead of raising."""
    from local_code_worker.providers.base import ProviderCompletedEvent, ProviderToolCallsEvent

    def handler(request: httpx.Request) -> httpx.Response:
        tool_chunk = json.dumps(
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "tc1",
                            "function": {
                                "name": "web_search",
                                "arguments": {"query": "fastapi version"},
                            },
                        }
                    ],
                },
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 10,
                "eval_count": 5,
            }
        )
        return httpx.Response(
            200,
            text="\n".join(
                [
                    '{"message":{"content":"Let me look that up."},"done":false}',
                    tool_chunk,
                ]
            ),
        )

    provider = OllamaProvider(ollama_settings(llm_json_mode="none"), httpx.MockTransport(handler))
    request = ProviderRequest(
        messages=[ProviderMessage(role="user", content="What is the latest FastAPI version?")],
        max_output_characters=500,
        stream=True,
        json_mode=JsonMode.NONE,
        tools=[
            ProviderFunctionTool(
                name="web_search",
                parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            )
        ],
    )
    events = list(provider.stream(request))

    tool_events = [e for e in events if isinstance(e, ProviderToolCallsEvent)]
    assert len(tool_events) == 1
    assert tool_events[0].function_calls[0].name == "web_search"
    assert tool_events[0].function_calls[0].call_id == "tc1"
    assert '"query"' in tool_events[0].function_calls[0].arguments
    # Should still have completed event
    assert isinstance(events[-1], ProviderCompletedEvent)


def test_ollama_stream_text_tool_call_fallback() -> None:
    """When the model encodes a tool call as text, stream should still detect it."""
    from local_code_worker.providers.base import ProviderToolCallsEvent

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text="\n".join(
                [
                    '{"message":{"content":"{\\"name\\":\\"web_search\\",\\"arguments\\":{\\"query\\":\\"fastapi\\"}}"},"done":true,"done_reason":"stop"}',
                ]
            ),
        )

    provider = OllamaProvider(
        ollama_settings(llm_json_mode="none"),
        httpx.MockTransport(handler),
    )
    request = ProviderRequest(
        messages=[ProviderMessage(role="user", content="search")],
        max_output_characters=500,
        stream=True,
        tools=[ProviderFunctionTool(name="web_search", parameters={"type": "object"})],
    )
    events = list(provider.stream(request))

    tool_events = [e for e in events if isinstance(e, ProviderToolCallsEvent)]
    assert len(tool_events) == 1
    assert tool_events[0].function_calls[0].name == "web_search"


def test_openai_stream_yields_tool_calls_event() -> None:
    """OpenAI-compatible streaming should accumulate tool-call deltas."""
    from local_code_worker.providers.base import ProviderCompletedEvent, ProviderToolCallsEvent

    def handler(request: httpx.Request) -> httpx.Response:
        def sse(obj: object) -> str:
            return f"data: {json.dumps(obj, separators=(',', ':'))}"

        lines = [
            sse({"choices": [{"delta": {"role": "assistant", "content": ""}}]}),
            sse(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_abc",
                                        "type": "function",
                                        "function": {"name": "web_search", "arguments": ""},
                                    },
                                ]
                            }
                        }
                    ]
                }
            ),
            sse(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {"index": 0, "function": {"arguments": '{"query":'}},
                                ]
                            }
                        }
                    ]
                }
            ),
            sse(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {"index": 0, "function": {"arguments": '"fastapi"}'}},
                                ]
                            }
                        }
                    ]
                }
            ),
            sse({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
            "data: [DONE]",
        ]
        return httpx.Response(200, text="\n".join(lines) + "\n")

    provider = OpenAICompatibleProvider(
        openai_settings(llm_json_mode="none"),
        httpx.MockTransport(handler),
    )
    request = ProviderRequest(
        messages=[ProviderMessage(role="user", content="What is the latest FastAPI?")],
        max_output_characters=500,
        stream=True,
        tools=[
            ProviderFunctionTool(
                name="web_search",
                parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            )
        ],
    )
    events = list(provider.stream(request))

    tool_events = [e for e in events if isinstance(e, ProviderToolCallsEvent)]
    assert len(tool_events) == 1
    tc = tool_events[0].function_calls[0]
    assert tc.name == "web_search"
    assert tc.call_id == "call_abc"
    assert tc.arguments == '{"query":"fastapi"}'
    assert isinstance(events[-1], ProviderCompletedEvent)


def test_openai_stream_no_tool_calls_yields_no_event() -> None:
    """OpenAI streaming with plain text should not emit ProviderToolCallsEvent."""
    from local_code_worker.providers.base import ProviderToolCallsEvent

    def handler(request: httpx.Request) -> httpx.Response:
        lines = [
            'data: {"choices":[{"delta":{"content":"hello"}}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
            "data: [DONE]",
        ]
        return httpx.Response(200, text="\n".join(lines) + "\n")

    provider = OpenAICompatibleProvider(
        openai_settings(llm_json_mode="none"),
        httpx.MockTransport(handler),
    )
    request = ProviderRequest(
        messages=[ProviderMessage(role="user", content="hi")],
        max_output_characters=500,
        stream=True,
    )
    events = list(provider.stream(request))

    tool_events = [e for e in events if isinstance(e, ProviderToolCallsEvent)]
    assert len(tool_events) == 0
