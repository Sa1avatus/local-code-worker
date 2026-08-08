import json
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from urllib.parse import urlsplit, urlunsplit

import httpx

from ..config import WorkerSettings
from ..exceptions import ProviderError
from ..models import GenerationMetadata, JsonMode, ProviderHealth, ProviderName


class OllamaProvider:
    def __init__(self, settings: WorkerSettings, transport: httpx.BaseTransport | None = None):
        self.settings = settings
        self.transport = transport
        self.last_generation_metadata: GenerationMetadata | None = None

    @property
    def base_url(self) -> str:
        parts = urlsplit(str(self.settings.llm_base_url))
        host = parts.hostname or ""
        if ":" in host:
            host = f"[{host}]"
        netloc = f"{host}:{parts.port}" if parts.port else host
        return urlunsplit((parts.scheme, netloc, parts.path.rstrip("/"), "", ""))

    def _client(self) -> httpx.Client:
        timeout = httpx.Timeout(
            self.settings.llm_timeout_seconds,
            connect=self.settings.llm_connect_timeout_seconds,
            read=self.settings.llm_read_timeout_seconds,
        )
        return httpx.Client(timeout=timeout, transport=self.transport)

    def list_models(self) -> list[str]:
        url = f"{self.base_url}/api/tags"
        try:
            with self._client() as client:
                response = client.get(url)
                response.raise_for_status()
                payload = response.json()
        except httpx.ConnectError as error:
            raise ProviderError(
                f"Cannot connect to Ollama at {url}", category="connection"
            ) from error
        except httpx.TimeoutException as error:
            raise ProviderError(f"Ollama request timed out at {url}", category="timeout") from error
        except httpx.TransportError as error:
            raise ProviderError(
                f"Ollama transport failed at {url}",
                category="transport_error",
            ) from error
        except httpx.HTTPStatusError as error:
            raise ProviderError(
                f"Ollama returned HTTP {error.response.status_code}", category="http_error"
            ) from error
        except (ValueError, TypeError) as error:
            raise ProviderError(
                "Ollama /api/tags returned invalid JSON", category="invalid_json"
            ) from error
        models = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(models, list):
            raise ProviderError(
                "Ollama /api/tags response has no models array", category="invalid_json"
            )
        return [model.get("name", "") for model in models if isinstance(model, dict)]

    def running_models(self) -> list[dict[str, object]]:
        """Return safe runtime placement details reported by Ollama's ``/api/ps``."""
        url = f"{self.base_url}/api/ps"
        try:
            with self._client() as client:
                response = client.get(url)
                response.raise_for_status()
                payload = response.json()
        except httpx.ConnectError as error:
            raise ProviderError(
                f"Cannot connect to Ollama at {url}", category="connection"
            ) from error
        except httpx.TimeoutException as error:
            raise ProviderError(f"Ollama request timed out at {url}", category="timeout") from error
        except httpx.TransportError as error:
            raise ProviderError(
                f"Ollama transport failed at {url}", category="transport_error"
            ) from error
        except httpx.HTTPStatusError as error:
            raise ProviderError(
                f"Ollama returned HTTP {error.response.status_code}", category="http_error"
            ) from error
        except (ValueError, TypeError) as error:
            raise ProviderError(
                "Ollama /api/ps returned invalid JSON", category="invalid_json"
            ) from error

        models = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(models, list):
            raise ProviderError(
                "Ollama /api/ps response has no models array", category="invalid_json"
            )
        safe_models: list[dict[str, object]] = []
        for model in models:
            if not isinstance(model, dict) or not isinstance(model.get("name"), str):
                continue
            runtime: dict[str, object] = {"name": model["name"]}
            for field in ("size", "size_vram", "context_length"):
                value = model.get(field)
                if isinstance(value, int) and value >= 0:
                    runtime[field] = value
            expires_at = model.get("expires_at")
            if isinstance(expires_at, str):
                runtime["expires_at"] = expires_at
            safe_models.append(runtime)
        return safe_models

    def check_connection(self) -> ProviderHealth:
        try:
            models = self.list_models()
        except ProviderError as error:
            return ProviderHealth(
                provider=ProviderName.OLLAMA,
                base_url=self.base_url,
                model=self.settings.llm_model,
                reachable=False,
                model_available=False,
                details=str(error),
            )
        available = self.settings.llm_model in models
        return ProviderHealth(
            provider=ProviderName.OLLAMA,
            base_url=self.base_url,
            model=self.settings.llm_model,
            reachable=True,
            model_available=available,
            details="Model is installed" if available else "Configured model is not installed",
        )

    def pull_model(self, model: str) -> Iterator[dict[str, object]]:
        validated = model.strip()
        if not validated:
            raise ValueError("Model name must not be empty")
        for char in validated:
            if ord(char) < 32 or ord(char) == 127:
                raise ValueError("Model name contains invalid control character")

        try:
            with self._client() as client:
                with client.stream(
                    "POST",
                    f"{self.base_url}/api/pull",
                    json={"name": validated, "stream": True},
                ) as response:
                    response.raise_for_status()
                    success_seen = False
                    for line in response.iter_lines():
                        if not line.strip():
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError as error:
                            raise ProviderError(
                                "Ollama pull returned an invalid NDJSON chunk",
                                category="invalid_stream_chunk",
                            ) from error
                        if not isinstance(chunk, dict):
                            raise ProviderError(
                                "Ollama pull chunk is not a dict",
                                category="invalid_stream_chunk",
                            )
                        yield chunk
                        if chunk.get("status") == "success":
                            success_seen = True
                            break
                    if not success_seen:
                        raise ProviderError(
                            "Ollama pull stream ended before a success status",
                            category="truncated_stream",
                        )
        except httpx.ConnectError as error:
            raise ProviderError(
                f"Cannot connect to Ollama at {self.base_url}", category="connection"
            ) from error
        except httpx.TimeoutException as error:
            raise ProviderError("Ollama request timed out", category="timeout") from error
        except httpx.TransportError as error:
            raise ProviderError(
                "Ollama transport failed",
                category="transport_error",
            ) from error
        except httpx.HTTPStatusError as error:
            raise ProviderError(
                f"Ollama returned HTTP {error.response.status_code}", category="http_error"
            ) from error

    def chat(
        self,
        messages: list[dict[str, str]],
        response_schema: dict[str, object] | None,
        max_output_characters: int,
        max_output_tokens: int | None = None,
    ) -> str:
        options: dict[str, object] = {
            "temperature": self.settings.llm_temperature,
            "num_ctx": self.settings.llm_num_ctx,
        }
        if max_output_tokens is not None:
            options["num_predict"] = max_output_tokens
        request_body: dict[str, object] = {
            "model": self.settings.llm_model,
            "stream": self.settings.llm_stream,
            "keep_alive": self.settings.llm_keep_alive,
            "messages": messages,
            "options": options,
        }
        mode = self.settings.llm_json_mode
        if mode is JsonMode.JSON_SCHEMA and response_schema is not None:
            request_body["format"] = response_schema
        elif mode in {JsonMode.AUTO, JsonMode.JSON_OBJECT}:
            request_body["format"] = "json"

        started_at = datetime.now(UTC)
        started = time.monotonic()
        finish_reason: str | None = None
        try:
            with self._client() as client:
                if self.settings.llm_stream:
                    content, finish_reason, usage = self._stream_chat(
                        client, request_body, max_output_characters
                    )
                else:
                    content, finish_reason, usage = self._non_stream_chat(
                        client, request_body, max_output_characters
                    )
        except httpx.ConnectError as error:
            raise ProviderError(
                f"Cannot connect to Ollama at {self.base_url}", category="connection"
            ) from error
        except httpx.TimeoutException as error:
            raise ProviderError("Ollama response timed out", category="timeout") from error
        except httpx.TransportError as error:
            raise ProviderError(
                "Ollama response transport failed",
                category="transport_error",
            ) from error
        except httpx.HTTPStatusError as error:
            raise ProviderError(
                f"Ollama returned HTTP {error.response.status_code}", category="http_error"
            ) from error

        completed_at = datetime.now(UTC)
        self.last_generation_metadata = GenerationMetadata(
            provider=ProviderName.OLLAMA,
            model=self.settings.llm_model,
            base_url=self.base_url,
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
            duration_seconds=time.monotonic() - started,
            prompt_characters=sum(len(message["content"]) for message in messages),
            output_characters=len(content),
            streaming=self.settings.llm_stream,
            response_format_mode=mode,
            finish_reason=finish_reason,
            usage=usage,
        )
        return content

    def _stream_chat(
        self, client: httpx.Client, request_body: dict[str, object], limit: int
    ) -> tuple[str, str | None, dict[str, int]]:
        parts: list[str] = []
        total = 0
        finish_reason: str | None = None
        saw_done = False
        usage: dict[str, int] = {}
        with client.stream("POST", f"{self.base_url}/api/chat", json=request_body) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ProviderError(
                        "Ollama returned an invalid NDJSON chunk",
                        category="invalid_stream_chunk",
                    ) from error
                if chunk.get("error") is not None:
                    raise ProviderError(
                        "Ollama rejected the streamed response",
                        category=_ollama_error_category(request_body),
                    )
                text = chunk.get("message", {}).get("content", "")
                if not isinstance(text, str):
                    raise ProviderError(
                        "Ollama stream chunk has invalid message.content",
                        category="invalid_stream_chunk",
                    )
                total += len(text)
                if total > limit:
                    raise ProviderError(
                        f"Ollama output exceeds the {limit}-character limit",
                        category="output_limit",
                    )
                parts.append(text)
                if chunk.get("done") is True:
                    saw_done = True
                    finish_reason = chunk.get("done_reason")
                    usage = _parse_usage(chunk)
                    break
        if not saw_done:
            raise ProviderError(
                "Ollama stream ended before a done chunk",
                category="truncated_stream",
            )
        return "".join(parts), finish_reason, usage

    def _non_stream_chat(
        self, client: httpx.Client, request_body: dict[str, object], limit: int
    ) -> tuple[str, str | None, dict[str, int]]:
        response = client.post(f"{self.base_url}/api/chat", json=request_body)
        response.raise_for_status()
        try:
            payload = response.json()
        except ValueError as error:
            raise ProviderError("Ollama returned invalid JSON", category="invalid_json") from error
        if isinstance(payload, dict) and payload.get("error") is not None:
            raise ProviderError(
                "Ollama rejected the response",
                category=_ollama_error_category(request_body),
            )
        content = payload.get("message", {}).get("content") if isinstance(payload, dict) else None
        if not isinstance(content, str):
            raise ProviderError(
                "Ollama response has no message.content string", category="invalid_response"
            )
        if len(content) > limit:
            raise ProviderError(
                f"Ollama output exceeds the {limit}-character limit",
                category="output_limit",
            )
        return content, payload.get("done_reason"), _parse_usage(payload)

    def generate(self, system_prompt: str, user_context: str, max_output_characters: int) -> str:
        from ..models import ModelImplementationResponse

        return self.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_context},
            ],
            ModelImplementationResponse.model_json_schema(),
            max_output_characters,
            self.settings.llm_max_output_tokens,
        )


def _parse_usage(payload: object) -> dict[str, int]:
    if not isinstance(payload, dict):
        return {}
    prompt = payload.get("prompt_eval_count")
    completion = payload.get("eval_count")
    if not isinstance(prompt, int) or not isinstance(completion, int):
        return {}
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def _ollama_error_category(request_body: dict[str, object]) -> str:
    return (
        "structured_output_error"
        if isinstance(request_body.get("format"), dict)
        else "provider_stream_error"
    )
