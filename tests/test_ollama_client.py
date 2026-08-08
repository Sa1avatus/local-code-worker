import json
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from local_code_worker.config import WorkerSettings
from local_code_worker.exceptions import OllamaError
from local_code_worker.ollama_client import OllamaClient


def create_client(handler) -> OllamaClient:
    settings = WorkerSettings(
        _env_file=None,
        ollama_timeout_seconds=1,
        llm_stream=False,
    )
    return OllamaClient(settings, transport=httpx.MockTransport(handler))


def test_ollama_client_returns_message_content() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.url.path == "/api/chat"
        assert body["stream"] is False
        assert body["format"] == "json"
        assert body["options"]["num_predict"] == 4_096
        return httpx.Response(
            200,
            json={
                "message": {
                    "content": (
                        '{"summary":"ok","files":[{"path":"a.py","content":"x","reason":"test"}]}'
                    )
                }
            },
        )

    content = create_client(handler).generate("system", "context", 1_000)
    assert '"summary":"ok"' in content


def test_ollama_client_returns_empty_content_for_pipeline_classification() -> None:
    client = create_client(lambda request: httpx.Response(200, json={"message": {"content": ""}}))
    assert client.generate("system", "context", 100) == ""


def test_ollama_client_rejects_oversized_response() -> None:
    client = create_client(
        lambda request: httpx.Response(200, json={"message": {"content": "x" * 11}})
    )
    with pytest.raises(OllamaError, match="limit"):
        client.generate("system", "context", 10)


def test_ollama_client_reports_connection_refused() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(OllamaError, match="Cannot connect"):
        create_client(handler).generate("system", "context", 100)


def test_ollama_client_reports_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(OllamaError, match="timed out"):
        create_client(handler).generate("system", "context", 100)


def test_settings_reject_external_ollama_url() -> None:
    with pytest.raises(ValidationError, match="loopback"):
        WorkerSettings(ollama_base_url="https://example.com")
    with pytest.raises(ValidationError, match="loopback"):
        WorkerSettings(llm_provider="ollama", llm_base_url="https://example.com")


def test_settings_reject_output_directory_outside_repository() -> None:
    with pytest.raises(ValidationError, match="relative"):
        WorkerSettings(worker_reports_directory=Path("../reports"))
