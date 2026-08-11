import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

import pytest

from local_code_worker import web_app
from local_code_worker.config import WorkerSettings
from local_code_worker.exceptions import ProviderError
from local_code_worker.models import (
    FunctionCallMetadata,
    GenerationMetadata,
    JsonMode,
    ProviderName,
)
from local_code_worker.providers.base import (
    ProviderCompletedEvent,
    ProviderStartedEvent,
    ProviderTextDeltaEvent,
    ProviderUsageEvent,
)
from local_code_worker.responses.state import ResponseStateStore
from local_code_worker.telemetry.models import TokenUsage


class FakeProvider:
    calls = []
    failing_models: set[str] = set()
    attempted_models: list[str] = []
    streamed_models: list[str] = []

    def __init__(self, settings: WorkerSettings):
        self.settings = settings
        self.last_generation_metadata = None

    def list_models(self) -> list[str]:
        return ["qwen:test", "qwen:other"]

    def chat(
        self,
        messages,
        response_schema,
        max_output_characters,
        max_output_tokens=None,
        tools=None,
        tool_choice="auto",
    ):
        self.attempted_models.append(self.settings.llm_model)
        if self.settings.llm_model in self.failing_models:
            raise ProviderError("model failed", category="test_error")
        self.calls.append(messages)
        assert messages[-1]["role"] == "user"
        assert response_schema is None
        assert max_output_tokens == self.settings.llm_max_output_tokens
        content = "" if tools else "ok"
        function_calls = (
            [
                FunctionCallMetadata(
                    call_id="call_test",
                    name=tools[0].name,
                    arguments='{"key":"value"}',
                )
            ]
            if tools
            else []
        )
        self.last_generation_metadata = GenerationMetadata(
            provider=ProviderName.OLLAMA,
            model=self.settings.llm_model,
            base_url="http://localhost:11434",
            started_at="2026-08-01T00:00:00Z",
            completed_at="2026-08-01T00:00:01Z",
            duration_seconds=1,
            prompt_characters=5,
            output_characters=len(content),
            streaming=False,
            response_format_mode=JsonMode.NONE,
            finish_reason="stop",
            function_calls=function_calls,
        )
        return content

    def stream(self, request):
        self.streamed_models.append(self.settings.llm_model)
        if self.settings.llm_model in self.failing_models:
            raise ProviderError("model failed", category="test_error")
        yield ProviderStartedEvent(
            sequence=0,
            provider=ProviderName.OLLAMA,
            model=self.settings.llm_model,
        )
        if request.messages[-1].content == "trigger error":
            raise ProviderError("stream failed", category="test_error")
        yield ProviderTextDeltaEvent(sequence=1, delta="ok")
        yield ProviderUsageEvent(sequence=2, usage=TokenUsage(input_tokens=1, output_tokens=1))
        self.last_generation_metadata = GenerationMetadata(
            provider=ProviderName.OLLAMA,
            model=self.settings.llm_model,
            base_url="http://localhost:11434",
            started_at="2026-08-01T00:00:00Z",
            completed_at="2026-08-01T00:00:01Z",
            duration_seconds=1,
            prompt_characters=5,
            output_characters=2,
            streaming=True,
            response_format_mode=self.settings.llm_json_mode,
            finish_reason="stop",
        )
        yield ProviderCompletedEvent(sequence=3, finish_reason="stop")


@pytest.fixture
def api_server(monkeypatch: pytest.MonkeyPatch, tmp_path):
    settings = WorkerSettings(llm_model="qwen:test")
    monkeypatch.setattr(web_app.WorkerWebHandler, "env_path", tmp_path / ".env")
    monkeypatch.setattr(web_app.WorkerWebHandler, "_settings", lambda self: settings)
    monkeypatch.setattr(web_app, "create_provider", lambda value: FakeProvider(value))
    monkeypatch.setattr(web_app, "RESPONSE_STATE", ResponseStateStore())
    FakeProvider.calls.clear()
    FakeProvider.failing_models.clear()
    FakeProvider.attempted_models.clear()
    FakeProvider.streamed_models.clear()
    monkeypatch.setattr(web_app, "record_model_call", lambda *args, **kwargs: None)
    monkeypatch.setattr(web_app, "record_routing_plan", lambda *args, **kwargs: None)
    monkeypatch.setattr(web_app, "record_route_lease", lambda *args, **kwargs: None)
    monkeypatch.setattr(web_app, "record_escalation", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        web_app,
        "summarize_routing",
        lambda: {"router_decisions_total": 2, "escalations_total": 1},
    )
    monkeypatch.setattr(web_app, "summarize_model_calls", lambda: {"models": []})
    monkeypatch.setattr(
        web_app,
        "summarize_v2_statistics",
        lambda baseline=None: {
            "version": 2,
            "requests": {"request_count": 3},
            "token_savings": (
                {"baseline_cloud_tokens": baseline, "estimated": True}
                if baseline is not None
                else None
            ),
        },
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), web_app.WorkerWebHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def request(port: int, method: str, path: str, payload=None):
    connection = HTTPConnection("127.0.0.1", port)
    body = json.dumps(payload) if payload is not None else None
    headers = {"Content-Type": "application/json"} if body is not None else {}
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse()
    content = response.read().decode()
    content_type = response.getheader("Content-Type")
    connection.close()
    return response.status, content_type, content


def test_openai_models_endpoint(api_server: int) -> None:
    status, _, content = request(api_server, "GET", "/v1/models")
    payload = json.loads(content)

    assert status == 200, content
    assert payload["object"] == "list"
    assert [model["id"] for model in payload["data"]] == [
        "local-code-worker/auto",
        "local-code-worker/local",
        "local-code-worker/mid",
        "local-code-worker/strong",
    ]
    assert {model["owned_by"] for model in payload["data"]} == {"local-code-worker"}


def test_admin_models_endpoint_keeps_physical_discovery(api_server: int) -> None:
    status, _, content = request(api_server, "GET", "/api/models")

    assert status == 200
    assert json.loads(content) == {"models": ["qwen:test", "qwen:other"]}


def test_router_status_and_process_health_endpoints(api_server: int) -> None:
    health_status, _, health_content = request(api_server, "GET", "/health")
    ready_status, _, ready_content = request(api_server, "GET", "/ready")
    router_status, _, router_content = request(api_server, "GET", "/api/v2/router/status")

    assert health_status == 200
    assert json.loads(health_content) == {"status": "ok"}
    assert ready_status == 200
    assert json.loads(ready_content)["router_mode"] == "legacy"
    assert router_status == 200
    assert json.loads(router_content)["metrics"]["escalations_total"] == 1


def test_gateway_settings_endpoint_has_editable_defaults(api_server: int) -> None:
    status, _, content = request(api_server, "GET", "/api/v2/settings")
    payload = json.loads(content)

    assert status == 200
    assert set(payload["tiers"]) == {"local", "mid", "strong"}
    assert all(tier["model"] for tier in payload["tiers"].values())


def test_gateway_settings_endpoint_round_trips_without_exposing_secrets(
    api_server: int,
) -> None:
    tiers = {
        "local": {
            "enabled": True,
            "provider": "ollama",
            "base_url": "http://localhost:11434",
            "model": "reasoner:latest",
            "context_length": 32768,
            "api_key_action": "keep",
        },
        "mid": {
            "enabled": True,
            "provider": "ollama",
            "base_url": "http://localhost:11434",
            "model": "executor:latest",
            "context_length": 32768,
            "api_key_action": "keep",
        },
        "strong": {
            "enabled": True,
            "provider": "openai-compatible",
            "base_url": "https://cloud.example/v1",
            "model": "cloud-strong",
            "context_length": 131072,
            "api_key_action": "replace",
            "api_key": "secret-value",
        },
    }
    status, _, content = request(
        api_server,
        "PUT",
        "/api/v2/settings",
        {
            "mode": "router",
            "tiers": tiers,
            "routellm_enabled": False,
            "routellm_threshold": 0.5,
        },
    )
    payload = json.loads(content)

    assert status == 200
    assert payload["mode"] == "router"
    assert payload["tiers"]["strong"]["api_key_configured"] is True
    assert "secret-value" not in content

    status, _, content = request(api_server, "GET", "/api/v2/settings")

    assert status == 200
    assert json.loads(content) == payload
    assert "secret-value" not in content


def test_non_stream_response_falls_back_from_local_to_mid(api_server: int) -> None:
    tiers = {
        name: {
            "enabled": True,
            "provider": "ollama",
            "base_url": "http://localhost:11434",
            "model": model_name,
            "context_length": 32768,
            "api_key_action": "keep",
        }
        for name, model_name in {
            "local": "local-fails",
            "mid": "mid-succeeds",
            "strong": "strong-unused",
        }.items()
    }
    status, _, _ = request(
        api_server,
        "PUT",
        "/api/v2/settings",
        {
            "mode": "router",
            "tiers": tiers,
            "routellm_enabled": False,
            "routellm_threshold": 0.5,
        },
    )
    assert status == 200
    FakeProvider.failing_models.add("local-fails")

    status, _, content = request(
        api_server,
        "POST",
        "/v1/responses",
        {"model": "local-code-worker/local", "input": "hello"},
    )

    assert status == 200, content
    assert json.loads(content)["output_text"] == "ok"
    assert FakeProvider.attempted_models == ["local-fails", "mid-succeeds"]


def test_stream_response_falls_back_before_first_event(api_server: int) -> None:
    tiers = {
        name: {
            "enabled": True,
            "provider": "ollama",
            "base_url": "http://localhost:11434",
            "model": model_name,
            "context_length": 32768,
            "api_key_action": "keep",
        }
        for name, model_name in {
            "local": "local-fails",
            "mid": "mid-succeeds",
            "strong": "strong-unused",
        }.items()
    }
    status, _, _ = request(
        api_server,
        "PUT",
        "/api/v2/settings",
        {
            "mode": "router",
            "tiers": tiers,
            "routellm_enabled": False,
            "routellm_threshold": 0.5,
        },
    )
    assert status == 200
    FakeProvider.failing_models.add("local-fails")

    status, content_type, content = request(
        api_server,
        "POST",
        "/v1/responses",
        {"model": "local-code-worker/local", "input": "hello", "stream": True},
    )

    assert status == 200, content
    assert content_type == "text/event-stream; charset=utf-8"
    assert "event: response.completed" in content
    assert FakeProvider.streamed_models == ["local-fails", "mid-succeeds"]


def test_v2_statistics_accepts_explicit_cloud_baseline(api_server: int) -> None:
    status, _, content = request(
        api_server,
        "GET",
        "/api/v2/statistics?baseline_cloud_tokens=100",
    )
    payload = json.loads(content)

    assert status == 200
    assert payload["version"] == 2
    assert payload["requests"] == {"request_count": 3}
    assert payload["token_savings"] == {
        "baseline_cloud_tokens": 100,
        "estimated": True,
    }


def test_responses_endpoint_returns_non_stream_text_response(api_server: int) -> None:
    status, content_type, content = request(
        api_server,
        "POST",
        "/v1/responses",
        {"model": "local-code-worker/auto", "input": "hello"},
    )
    payload = json.loads(content)

    assert status == 200
    assert content_type == "application/json; charset=utf-8"
    assert payload["object"] == "response"
    assert payload["status"] == "completed"
    assert payload["model"] == "local-code-worker/auto"
    assert payload["output_text"] == "ok"
    assert payload["output"][0]["content"][0]["text"] == "ok"


def test_responses_endpoint_streams_ordered_sse(api_server: int) -> None:
    status, content_type, content = request(
        api_server,
        "POST",
        "/v1/responses",
        {"model": "local-code-worker/auto", "input": "hello", "stream": True},
    )
    event_names = [
        line.removeprefix("event: ") for line in content.splitlines() if line.startswith("event: ")
    ]

    assert status == 200
    assert content_type == "text/event-stream; charset=utf-8"
    assert event_names[0] == "response.created"
    assert "response.output_text.delta" in event_names
    assert event_names[-1] == "response.completed"


def test_responses_stream_reports_provider_failure_as_sse(api_server: int) -> None:
    status, content_type, content = request(
        api_server,
        "POST",
        "/v1/responses",
        {
            "model": "local-code-worker/auto",
            "input": "trigger error",
            "stream": True,
        },
    )
    event_names = [
        line.removeprefix("event: ") for line in content.splitlines() if line.startswith("event: ")
    ]
    data = [
        json.loads(line.removeprefix("data: "))
        for line in content.splitlines()
        if line.startswith("data: ")
    ]

    assert status == 200
    assert content_type == "text/event-stream; charset=utf-8"
    assert event_names[-1] == "response.failed"
    assert data[-1]["response"]["status"] == "failed"
    assert data[-1]["response"]["error"]["type"] == "server_error"


def test_responses_endpoint_returns_non_stream_function_call(api_server: int) -> None:
    status, _, content = request(
        api_server,
        "POST",
        "/v1/responses",
        {
            "model": "local-code-worker/auto",
            "input": "hello",
            "tools": [{"type": "function", "name": "lookup", "parameters": {"type": "object"}}],
        },
    )

    payload = json.loads(content)
    assert status == 200
    assert payload["output_text"] == ""
    assert payload["output"] == [
        {
            "id": payload["output"][0]["id"],
            "type": "function_call",
            "status": "completed",
            "call_id": "call_test",
            "name": "lookup",
            "arguments": '{"key":"value"}',
        }
    ]


def test_responses_endpoint_rejects_streaming_function_tools(api_server: int) -> None:
    status, _, content = request(
        api_server,
        "POST",
        "/v1/responses",
        {
            "model": "local-code-worker/auto",
            "input": "hello",
            "stream": True,
            "tools": [{"type": "function", "name": "lookup", "parameters": {"type": "object"}}],
        },
    )

    assert status == 400
    assert "streaming function tools" in json.loads(content)["error"]["message"]


def test_responses_endpoint_continues_stored_response(api_server: int) -> None:
    first_status, _, first_content = request(
        api_server,
        "POST",
        "/v1/responses",
        {"model": "local-code-worker/auto", "input": "first", "store": True},
    )
    response_id = json.loads(first_content)["id"]

    second_status, _, _ = request(
        api_server,
        "POST",
        "/v1/responses",
        {
            "model": "local-code-worker/auto",
            "input": "second",
            "previous_response_id": response_id,
        },
    )

    assert first_status == 200
    assert second_status == 200
    assert FakeProvider.calls[-1] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "second"},
    ]


def test_response_chain_route_lease_prevents_downgrade(api_server: int) -> None:
    tiers = {
        name: {
            "enabled": True,
            "provider": "ollama",
            "base_url": "http://localhost:11434",
            "model": name,
            "context_length": 32768,
            "api_key_action": "keep",
        }
        for name in ("local", "mid", "strong")
    }
    status, _, _ = request(
        api_server,
        "PUT",
        "/api/v2/settings",
        {
            "mode": "router",
            "tiers": tiers,
            "routellm_enabled": False,
            "routellm_threshold": 0.5,
        },
    )
    assert status == 200
    first_status, _, first_content = request(
        api_server,
        "POST",
        "/v1/responses",
        {"model": "local-code-worker/strong", "input": "first", "store": True},
    )
    first_id = json.loads(first_content)["id"]

    second_status, _, second_content = request(
        api_server,
        "POST",
        "/v1/responses",
        {
            "model": "local-code-worker/auto",
            "input": "small edit",
            "previous_response_id": first_id,
            "store": True,
        },
    )
    second_id = json.loads(second_content)["id"]
    first_lease = web_app.RESPONSE_STATE.get_stored(first_id).route_lease
    second_lease = web_app.RESPONSE_STATE.get_stored(second_id).route_lease

    assert first_status == 200
    assert second_status == 200
    assert FakeProvider.attempted_models == ["strong", "strong"]
    assert first_lease is not None
    assert second_lease is not None
    assert second_lease.lease_id == first_lease.lease_id


def test_responses_endpoint_rejects_unknown_previous_response(api_server: int) -> None:
    status, _, content = request(
        api_server,
        "POST",
        "/v1/responses",
        {
            "model": "local-code-worker/auto",
            "input": "hello",
            "previous_response_id": "resp_missing",
        },
    )

    assert status == 400
    assert "not found or expired" in json.loads(content)["error"]["message"]


def test_legacy_statistics_contract_is_unchanged(api_server: int) -> None:
    status, _, content = request(api_server, "GET", "/api/statistics")

    assert status == 200
    assert json.loads(content) == {"models": []}


def test_chat_completions_preserves_virtual_model(api_server: int) -> None:
    status, _, content = request(
        api_server,
        "POST",
        "/v1/chat/completions",
        {
            "model": "local-code-worker/local",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    payload = json.loads(content)

    assert status == 200
    assert payload["object"] == "chat.completion"
    assert payload["model"] == "local-code-worker/local"
    assert payload["choices"][0]["message"]["content"] == "ok"


def test_chat_completions_supports_sse_shape(api_server: int) -> None:
    status, content_type, content = request(
        api_server,
        "POST",
        "/v1/chat/completions",
        {"messages": [{"role": "user", "content": "hello"}], "stream": True},
    )

    assert status == 200
    assert content_type == "text/event-stream; charset=utf-8"
    assert '"object": "chat.completion.chunk"' in content
    assert content.endswith("data: [DONE]\n\n")


def test_chat_completions_rejects_invalid_messages(api_server: int) -> None:
    status, _, content = request(
        api_server,
        "POST",
        "/v1/chat/completions",
        {"messages": [{"role": "user", "content": ["not", "text"]}]},
    )

    assert status == 400
    assert json.loads(content)["error"]["type"] == "invalid_request_error"


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/responses"])
def test_generation_endpoints_reject_unknown_virtual_model(api_server: int, path: str) -> None:
    payload = (
        {
            "model": "local-code-worker/unknown",
            "messages": [{"role": "user", "content": "hello"}],
        }
        if path.endswith("chat/completions")
        else {"model": "local-code-worker/unknown", "input": "hello"}
    )

    status, _, content = request(api_server, "POST", path, payload)

    assert status == 400
    assert "Unknown virtual model" in json.loads(content)["error"]["message"]
