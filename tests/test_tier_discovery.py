"""Tests for per-tier model discovery (web UI "Найти модели" flow).

Covers: endpoint selection per provider, Base URL normalization, per-tier
independence, HTTP error mapping, empty/stored API keys, and the admin
endpoint that proxies discovery through the existing provider layer.
"""

import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from local_code_worker import web_app
from local_code_worker.exceptions import ProviderConfigurationError, ProviderError
from local_code_worker.models import ProviderName
from local_code_worker.virtual_models import ModelTier
from local_code_worker.web_app import (
    _discovery_error_message,
    _resolve_tier_stored_key,
    discover_tier_models,
)
from local_code_worker.web_models import TierModelDiscoveryInput


def test_discover_ollama_uses_api_tags_of_given_base_url() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "host.docker.internal"
        assert request.url.port == 11434
        assert request.url.path == "/api/tags"
        return httpx.Response(
            200,
            json={"models": [{"name": "qwen3.5:4b"}, {"name": "gemma4:12b"}]},
        )

    models = discover_tier_models(
        ProviderName.OLLAMA,
        "http://host.docker.internal:11434",
        None,
        Path(".env"),
        httpx.MockTransport(handler),
    )
    assert models == ["qwen3.5:4b", "gemma4:12b"]


def test_discover_openai_appends_v1_models_and_sends_bearer_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        assert request.headers["Authorization"] == "Bearer STRONG-SECRET"
        return httpx.Response(
            200,
            json={"object": "list", "data": [{"id": "gpt-5"}, {"id": "gpt-5-mini"}]},
        )

    models = discover_tier_models(
        ProviderName.OPENAI_COMPATIBLE,
        "https://cloud.example",
        "STRONG-SECRET",
        Path(".env"),
        httpx.MockTransport(handler),
    )
    assert models == ["gpt-5", "gpt-5-mini"]


def test_discover_openai_without_key_sends_no_authorization() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "Authorization" not in request.headers
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"object": "list", "data": [{"id": "open"}]})

    models = discover_tier_models(
        ProviderName.OPENAI_COMPATIBLE,
        "https://open.example/v1",
        None,
        Path(".env"),
        httpx.MockTransport(handler),
    )
    assert models == ["open"]


def test_discover_openai_401_surfaces_auth_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    with pytest.raises(ProviderError) as captured:
        discover_tier_models(
            ProviderName.OPENAI_COMPATIBLE,
            "https://cloud.example/v1",
            "wrong-key",
            Path(".env"),
            httpx.MockTransport(handler),
        )
    message = _discovery_error_message(captured.value)
    assert message == "Ошибка авторизации: проверьте API-ключ"
    assert "wrong-key" not in message


def test_discover_ollama_connection_error_surfaces_connection_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(ProviderError) as captured:
        discover_tier_models(
            ProviderName.OLLAMA,
            "http://localhost:11434",
            None,
            Path(".env"),
            httpx.MockTransport(handler),
        )
    assert _discovery_error_message(captured.value) == "Не удалось подключиться к серверу моделей"


@pytest.mark.parametrize(
    ("category", "expected"),
    [
        ("connection", "Не удалось подключиться к серверу моделей"),
        ("timeout", "Таймаут подключения: сервер моделей не ответил"),
        ("transport_error", "Ошибка сети: не удалось связаться с сервером моделей"),
        ("http_401", "Ошибка авторизации: проверьте API-ключ"),
        ("http_403", "Ошибка авторизации: проверьте API-ключ"),
        ("http_404", "Endpoint получения моделей не найден (HTTP 404)"),
        ("http_429", "Не удалось получить список моделей (HTTP 429)"),
        ("models_unsupported", "Модели не найдены или сервер вернул некорректный ответ"),
    ],
)
def test_discovery_error_mapping(category: str, expected: str) -> None:
    assert _discovery_error_message(ProviderError("detail", category=category)) == expected


def test_unknown_provider_reports_unsupported() -> None:
    error = ProviderConfigurationError("Unknown LLM provider: custom")
    assert (
        _discovery_error_message(error)
        == "Для этого провайдера автоматический поиск моделей не поддерживается."
    )


def test_resolve_tier_stored_key_reads_only_own_tier_env(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "LOCAL_CODE_WORKER_STRONG_API_KEY=strong-stored\n"
        "LOCAL_CODE_WORKER_LOCAL_API_KEY=local-stored\n",
        encoding="utf-8",
    )
    assert _resolve_tier_stored_key(ModelTier.STRONG, env_path) == "strong-stored"
    assert _resolve_tier_stored_key(ModelTier.LOCAL, env_path) == "local-stored"
    assert _resolve_tier_stored_key(ModelTier.MID, env_path) is None


def test_discovery_input_rejects_remote_ollama() -> None:
    with pytest.raises(Exception, match="loopback"):
        TierModelDiscoveryInput.model_validate(
            {
                "tier": "local",
                "provider": "ollama",
                "base_url": "https://ollama.example.test",
            }
        )


def test_discovery_input_requires_http_url() -> None:
    with pytest.raises(ValidationError):
        TierModelDiscoveryInput.model_validate(
            {
                "tier": "strong",
                "provider": "openai-compatible",
                "base_url": "not-a-url",
            }
        )


class _FakeDiscoveryProvider:
    """Records the settings a discovery request would use."""

    def __init__(self, settings):
        self.settings = settings

    def list_models(self) -> list[str]:
        return ["model-a", "model-b"]


def _discovery_server(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, factory):
    monkeypatch.setattr(web_app.WorkerWebHandler, "env_path", tmp_path / ".env")
    monkeypatch.setattr(
        web_app, "create_provider", lambda settings, transport=None: factory(settings)
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), web_app.WorkerWebHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def test_discover_models_endpoint_uses_only_requested_card_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[str, object] = {}

    class FakeProvider(_FakeDiscoveryProvider):
        def list_models(self) -> list[str]:
            seen["provider"] = self.settings.llm_provider.value
            seen["base_url"] = str(self.settings.llm_base_url)
            seen["api_key"] = self.settings.resolve_api_key()[0]
            return ["strong-only"]

    server = _discovery_server(monkeypatch, tmp_path, FakeProvider)
    try:
        connection = HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        body = json.dumps(
            {
                "tier": "strong",
                "provider": "openai-compatible",
                "base_url": "https://cloud.example/v1",
                "api_key": "TYPED-KEY",
            }
        ).encode("utf-8")
        connection.request(
            "POST",
            "/api/v2/discover-models",
            body=body,
            headers={"Content-Type": "application/json", "Host": "localhost"},
        )
        response = connection.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        connection.close()
        assert response.status == 200
        assert payload == {"models": ["strong-only"]}
        assert seen["provider"] == "openai-compatible"
        assert seen["base_url"] == "https://cloud.example/v1"
        assert seen["api_key"] == "TYPED-KEY"
        assert "TYPED-KEY" not in json.dumps(payload)
    finally:
        server.shutdown()
        server.server_close()


def test_discover_models_endpoint_maps_upstream_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FailingProvider(_FakeDiscoveryProvider):
        def list_models(self) -> list[str]:
            raise ProviderError("authentication failed", category="http_401")

    server = _discovery_server(monkeypatch, tmp_path, FailingProvider)
    try:
        connection = HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        body = json.dumps(
            {
                "tier": "strong",
                "provider": "openai-compatible",
                "base_url": "https://cloud.example/v1",
                "api_key": "WRONG-KEY",
            }
        ).encode("utf-8")
        connection.request(
            "POST",
            "/api/v2/discover-models",
            body=body,
            headers={"Content-Type": "application/json", "Host": "localhost"},
        )
        response = connection.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        connection.close()
        assert response.status == 400
        assert payload == {"error": "Ошибка авторизации: проверьте API-ключ"}
        assert "WRONG-KEY" not in json.dumps(payload)
    finally:
        server.shutdown()
        server.server_close()


def test_discover_models_endpoint_requires_base_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    server = _discovery_server(monkeypatch, tmp_path, _FakeDiscoveryProvider)
    try:
        connection = HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        body = json.dumps({"tier": "local", "provider": "ollama", "base_url": ""}).encode(
            "utf-8"
        )
        connection.request(
            "POST",
            "/api/v2/discover-models",
            body=body,
            headers={"Content-Type": "application/json", "Host": "localhost"},
        )
        response = connection.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        connection.close()
        assert response.status == 400
        assert "Неверный Base URL" in payload["error"]
    finally:
        server.shutdown()
        server.server_close()
