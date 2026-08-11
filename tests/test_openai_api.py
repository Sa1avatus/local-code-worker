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


def declared_request(port: int, path: str, content_length: int):
    connection = HTTPConnection("127.0.0.1", port)
    connection.putrequest("POST", path)
    connection.putheader("Content-Type", "application/json")
    connection.putheader("Content-Length", str(content_length))
    connection.endheaders()
    response = connection.getresponse()
    content = response.read().decode()
    connection.close()
    return response.status, json.loads(content)


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


@pytest.mark.parametrize("size", [64 * 1024, 256 * 1024, 1024 * 1024, 5 * 1024 * 1024])
def test_responses_accepts_large_request_bodies(api_server: int, size: int) -> None:
    status, _, content = request(
        api_server,
        "POST",
        "/v1/responses",
        {"model": "local-code-worker/local", "input": "x" * size},
    )

    assert status == 200, content


def test_responses_rejects_body_above_limit_with_openai_error(api_server: int) -> None:
    status, payload = declared_request(api_server, "/v1/responses", 16 * 1024 * 1024 + 1)

    assert status == 413
    assert payload["error"]["code"] == "request_too_large"
    assert payload["error"]["details"] == {
        "max_bytes": 16 * 1024 * 1024,
        "received_bytes": 16 * 1024 * 1024 + 1,
    }


def test_ui_request_rejects_body_above_ui_limit(api_server: int) -> None:
    status, payload = declared_request(api_server, "/api/ollama/pull", 1024 * 1024 + 1)

    assert status == 413
    assert payload["error"]["code"] == "request_too_large"
    assert payload["error"]["details"]["max_bytes"] == 1024 * 1024


def test_responses_logging_contains_only_safe_request_metadata(
    api_server: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = "private-prompt-marker"
    messages = []
    monkeypatch.setattr(web_app.HTTP_LOGGER, "info", messages.append)
    status, _, _ = request(
        api_server,
        "POST",
        "/v1/responses",
        {"model": "local-code-worker/local", "input": marker},
    )

    assert status == 200
    records = [json.loads(message) for message in messages]
    summary = records[-1]
    assert summary["method"] == "POST"
    assert summary["path"] == "/v1/responses"
    assert summary["content_length"] > 0
    assert summary["max_request_bytes"] == 16 * 1024 * 1024
    assert summary["status"] == 200
    assert summary["elapsed_ms"] >= 0
    assert marker not in "".join(messages)


def test_large_request_rejection_is_logged_without_body(
    api_server: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    messages = []
    monkeypatch.setattr(web_app.HTTP_LOGGER, "info", messages.append)
    status, _ = declared_request(api_server, "/v1/responses", 16 * 1024 * 1024 + 1)

    assert status == 413
    records = [json.loads(message) for message in messages]
    rejection = next(record for record in records if record.get("event"))
    assert rejection == {
        "event": "request rejected: body too large",
        "request_id": rejection["request_id"],
        "received_bytes": 16 * 1024 * 1024 + 1,
        "max_bytes": 16 * 1024 * 1024,
    }


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


def test_responses_accepts_tool_declarations_during_text_streaming(api_server: int) -> None:
    status, content_type, content = request(
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

    assert status == 200
    assert content_type == "text/event-stream; charset=utf-8"
    assert "event: response.completed" in content


def test_codex_like_request_above_old_64_kib_limit_streams(api_server: int) -> None:
    large_schema = {
        "type": "object",
        "properties": {
            f"field_{index}": {"type": "string", "description": "x" * 128} for index in range(600)
        },
    }
    payload = {
        "model": "local-code-worker/auto",
        "instructions": "Act as a coding agent.",
        "input": [{"type": "message", "role": "user", "content": "Return test"}],
        "tools": [
            {
                "type": "function",
                "name": "shell_command",
                "description": "Run an approved command",
                "parameters": large_schema,
                "strict": True,
            }
        ],
        "reasoning": {"effort": "high", "summary": "auto"},
        "stream": True,
    }
    assert len(json.dumps(payload).encode()) > 64 * 1024

    status, content_type, content = request(api_server, "POST", "/v1/responses", payload)

    assert status == 200
    assert content_type == "text/event-stream; charset=utf-8"
    assert "event: response.created" in content
    assert "event: response.completed" in content
    assert FakeProvider.streamed_models[-1] == "qwen:test"


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


def test_generation_requests_are_serialized(api_server: int, monkeypatch) -> None:
    first_started = threading.Event()
    release_first = threading.Event()
    active_lock = threading.Lock()
    active = 0
    max_active = 0
    original_chat = FakeProvider.chat

    def blocking_chat(self, *args, **kwargs):
        nonlocal active, max_active
        with active_lock:
            active += 1
            max_active = max(max_active, active)
            is_first = active == 1 and not first_started.is_set()
        if is_first:
            first_started.set()
            assert release_first.wait(timeout=2)
        try:
            return original_chat(self, *args, **kwargs)
        finally:
            with active_lock:
                active -= 1

    monkeypatch.setattr(FakeProvider, "chat", blocking_chat)
    results = []

    def send_request():
        results.append(
            request(
                api_server,
                "POST",
                "/v1/chat/completions",
                {"messages": [{"role": "user", "content": "hello"}]},
            )
        )

    first = threading.Thread(target=send_request)
    second = threading.Thread(target=send_request)
    first.start()
    assert first_started.wait(timeout=2)
    second.start()
    assert second.is_alive()
    release_first.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert [result[0] for result in results] == [200, 200]
    assert max_active == 1


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
