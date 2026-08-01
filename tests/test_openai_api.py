import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

import pytest

from local_code_worker import web_app
from local_code_worker.config import WorkerSettings
from local_code_worker.models import GenerationMetadata, JsonMode, ProviderName


class FakeProvider:
    def __init__(self, settings: WorkerSettings):
        self.settings = settings
        self.last_generation_metadata = None

    def list_models(self) -> list[str]:
        return ["qwen:test", "qwen:other"]

    def chat(self, messages, response_schema, max_output_characters):
        assert messages[-1] == {"role": "user", "content": "hello"}
        assert response_schema is None
        self.last_generation_metadata = GenerationMetadata(
            provider=ProviderName.OLLAMA,
            model=self.settings.llm_model,
            base_url="http://localhost:11434",
            started_at="2026-08-01T00:00:00Z",
            completed_at="2026-08-01T00:00:01Z",
            duration_seconds=1,
            prompt_characters=5,
            output_characters=2,
            streaming=False,
            response_format_mode=JsonMode.NONE,
            finish_reason="stop",
        )
        return "ok"


@pytest.fixture
def api_server(monkeypatch: pytest.MonkeyPatch):
    settings = WorkerSettings(llm_model="qwen:test")
    monkeypatch.setattr(web_app.WorkerWebHandler, "_settings", lambda self: settings)
    monkeypatch.setattr(web_app, "create_provider", lambda value: FakeProvider(value))
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

    assert status == 200
    assert payload["object"] == "list"
    assert [model["id"] for model in payload["data"]] == ["qwen:test", "qwen:other"]


def test_chat_completions_uses_requested_model(api_server: int) -> None:
    status, _, content = request(
        api_server,
        "POST",
        "/v1/chat/completions",
        {"model": "qwen:other", "messages": [{"role": "user", "content": "hello"}]},
    )
    payload = json.loads(content)

    assert status == 200
    assert payload["object"] == "chat.completion"
    assert payload["model"] == "qwen:other"
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
