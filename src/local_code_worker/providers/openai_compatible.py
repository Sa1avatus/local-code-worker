import json
import os
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from urllib.parse import urlsplit, urlunsplit

import httpx

from ..config import WorkerSettings
from ..exceptions import ProviderConfigurationError, ProviderError
from ..models import (
    FunctionCallMetadata,
    GenerationMetadata,
    JsonMode,
    ProviderHealth,
    ProviderName,
)
from ..telemetry.models import TokenUsage, UsageProvenance
from .base import (
    ProviderCapabilities,
    ProviderCapability,
    ProviderCompletedEvent,
    ProviderEvent,
    ProviderFunctionTool,
    ProviderFunctionToolChoice,
    ProviderMessage,
    ProviderRequest,
    ProviderStartedEvent,
    ProviderTextDeltaEvent,
    ProviderUsageEvent,
)


class OpenAICompatibleProvider:
    capabilities = ProviderCapabilities(
        supported=frozenset(
            {
                ProviderCapability.STREAMING,
                ProviderCapability.JSON_OBJECT,
                ProviderCapability.JSON_SCHEMA,
                ProviderCapability.USAGE,
                ProviderCapability.FUNCTION_TOOLS,
            }
        )
    )

    def __init__(self, settings: WorkerSettings, transport: httpx.BaseTransport | None = None):
        self.settings = settings
        self.transport = transport
        self.last_generation_metadata: GenerationMetadata | None = None
        self.api_key, self.api_key_env = settings.resolve_api_key()

    @property
    def base_url(self) -> str:
        parts = urlsplit(str(self.settings.llm_base_url))
        host = parts.hostname or ""
        if ":" in host:
            host = f"[{host}]"
        netloc = f"{host}:{parts.port}" if parts.port else host
        return urlunsplit((parts.scheme, netloc, parts.path.rstrip("/"), "", ""))

    def _require_api_key(self) -> str:
        if not self.api_key:
            variable = self.api_key_env or "OPENAI_COMPATIBLE_API_KEY"
            raise ProviderConfigurationError(f"Environment variable {variable} is not set")
        return self.api_key

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._require_api_key()}",
            "Content-Type": "application/json",
        }
        referer = os.environ.get("OPENROUTER_HTTP_REFERER")
        title = os.environ.get("OPENROUTER_APP_TITLE")
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-Title"] = title
        return headers

    def _client(self) -> httpx.Client:
        timeout = httpx.Timeout(
            self.settings.llm_timeout_seconds,
            connect=self.settings.llm_connect_timeout_seconds,
            read=self.settings.llm_read_timeout_seconds,
        )
        return httpx.Client(timeout=timeout, transport=self.transport, headers=self._headers())

    def list_models(self) -> list[str]:
        try:
            with self._client() as client:
                response = client.get(f"{self.base_url}/models")
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPStatusError as error:
            raise self._http_error(error.response) from error
        except httpx.ConnectError as error:
            raise ProviderError(
                f"Cannot connect to OpenAI-compatible endpoint {self.base_url}",
                category="connection",
            ) from error
        except httpx.TimeoutException as error:
            raise ProviderError(
                "OpenAI-compatible models request timed out", category="timeout"
            ) from error
        except httpx.TransportError as error:
            raise ProviderError(
                "OpenAI-compatible models request transport failed",
                category="transport_error",
            ) from error
        except ValueError as error:
            raise ProviderError(
                "Models endpoint returned invalid JSON", category="invalid_json"
            ) from error
        models = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(models, list):
            raise ProviderError("Models endpoint has no data array", category="models_unsupported")
        return [model.get("id", "") for model in models if isinstance(model, dict)]

    def check_connection(self) -> ProviderHealth:
        self._require_api_key()
        try:
            models = self.list_models()
        except ProviderError as error:
            return ProviderHealth(
                provider=ProviderName.OPENAI_COMPATIBLE,
                base_url=self.base_url,
                model=self.settings.llm_model,
                reachable=False,
                model_available=None,
                details=str(error),
            )
        available = self.settings.llm_model in models
        return ProviderHealth(
            provider=ProviderName.OPENAI_COMPATIBLE,
            base_url=self.base_url,
            model=self.settings.llm_model,
            reachable=True,
            model_available=available,
            details="Model is listed" if available else "Configured model is not listed",
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        response_schema: dict[str, object] | None,
        max_output_characters: int,
        max_output_tokens: int | None = None,
        tools: list[ProviderFunctionTool] | None = None,
        tool_choice: str | ProviderFunctionToolChoice = "auto",
    ) -> str:
        mode = self._effective_json_mode()
        try:
            return self._chat_once(
                messages,
                response_schema,
                max_output_characters,
                max_output_tokens,
                mode,
                tools,
                tool_choice,
            )
        except _UnsupportedResponseFormat:
            if self.settings.llm_json_mode is not JsonMode.AUTO:
                raise
            return self._chat_once(
                messages,
                response_schema,
                max_output_characters,
                max_output_tokens,
                JsonMode.PROMPT_ONLY,
                tools,
                tool_choice,
            )

    def _chat_once(
        self,
        messages: list[dict[str, str]],
        response_schema: dict[str, object] | None,
        limit: int,
        max_output_tokens: int | None,
        mode: JsonMode,
        tools: list[ProviderFunctionTool] | None,
        tool_choice: str | ProviderFunctionToolChoice,
    ) -> str:
        if self.settings.llm_stream:
            parts: list[str] = []
            for event in self.stream(
                ProviderRequest(
                    messages=[ProviderMessage.model_validate(message) for message in messages],
                    response_schema=response_schema,
                    max_output_characters=limit,
                    max_output_tokens=max_output_tokens,
                    json_mode=mode,
                    stream=True,
                    tools=tools or [],
                    tool_choice=tool_choice,
                )
            ):
                if isinstance(event, ProviderTextDeltaEvent):
                    parts.append(event.delta)
            return "".join(parts)

        request_body = self._request_body(
            messages,
            response_schema,
            max_output_tokens,
            mode,
            tools,
            tool_choice,
            stream=False,
        )

        started_at = datetime.now(UTC)
        started = time.monotonic()
        try:
            with self._client() as client:
                content, finish_reason, usage, function_calls = self._non_stream_chat(
                    client, request_body, limit
                )
        except httpx.HTTPStatusError as error:
            if self._is_unsupported_response_format(error.response, request_body):
                raise _UnsupportedResponseFormat() from error
            raise self._http_error(error.response) from error
        except httpx.ConnectError as error:
            raise ProviderError(
                f"Cannot connect to OpenAI-compatible endpoint {self.base_url}",
                category="connection",
            ) from error
        except httpx.TimeoutException as error:
            raise ProviderError(
                "OpenAI-compatible response timed out", category="timeout"
            ) from error
        except httpx.TransportError as error:
            raise ProviderError(
                "OpenAI-compatible response transport failed",
                category="transport_error",
            ) from error

        completed_at = datetime.now(UTC)
        self.last_generation_metadata = GenerationMetadata(
            provider=ProviderName.OPENAI_COMPATIBLE,
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
            function_calls=function_calls,
        )
        return content

    def stream(self, request: ProviderRequest) -> Iterator[ProviderEvent]:
        if not request.stream:
            raise ValueError("OpenAI-compatible stream request must enable streaming")
        if request.tools:
            raise ProviderError(
                "OpenAI-compatible streaming function tools are not supported yet",
                category="unsupported_tools",
            )
        messages = [message.model_dump() for message in request.messages]
        request_body = self._request_body(
            messages,
            request.response_schema,
            request.max_output_tokens,
            request.json_mode,
            request.tools,
            request.tool_choice,
            stream=True,
        )
        parts: list[str] = []
        total = 0
        finish_reason: str | None = None
        usage: dict[str, int] = {}
        sequence = 0
        started_at = datetime.now(UTC)
        started = time.monotonic()
        first_token_at: float | None = None
        yield ProviderStartedEvent(
            sequence=sequence,
            provider=ProviderName.OPENAI_COMPATIBLE,
            model=self.settings.llm_model,
        )
        sequence += 1
        try:
            with self._client() as client:
                with client.stream(
                    "POST", f"{self.base_url}/chat/completions", json=request_body
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line.strip() or line.startswith(":"):
                            continue
                        if not line.startswith("data:"):
                            raise ProviderError(
                                "Malformed SSE line", category="invalid_stream_chunk"
                            )
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            choices = chunk.get("choices", [])
                            choice = choices[0] if choices else {}
                            delta = choice.get("delta", {})
                            text = delta.get("content") or delta.get("refusal") or ""
                        except (ValueError, TypeError, AttributeError) as error:
                            raise ProviderError(
                                "Malformed SSE JSON chunk",
                                category="invalid_stream_chunk",
                            ) from error
                        if not isinstance(text, str):
                            raise ProviderError(
                                "Invalid SSE content", category="invalid_stream_chunk"
                            )
                        total += len(text)
                        if total > request.max_output_characters:
                            raise ProviderError(
                                "Provider output exceeds "
                                f"{request.max_output_characters} characters",
                                category="output_limit",
                            )
                        if text:
                            if first_token_at is None:
                                first_token_at = time.monotonic()
                            parts.append(text)
                            yield ProviderTextDeltaEvent(sequence=sequence, delta=text)
                            sequence += 1
                        if choice.get("finish_reason") is not None:
                            finish_reason = str(choice["finish_reason"])
                        usage.update(_parse_usage(chunk.get("usage")))
        except httpx.HTTPStatusError as error:
            if self._is_unsupported_response_format(error.response, request_body):
                raise _UnsupportedResponseFormat() from error
            raise self._http_error(error.response) from error
        except httpx.ConnectError as error:
            raise ProviderError(
                f"Cannot connect to OpenAI-compatible endpoint {self.base_url}",
                category="connection",
            ) from error
        except httpx.TimeoutException as error:
            raise ProviderError(
                "OpenAI-compatible response timed out", category="timeout"
            ) from error
        except httpx.TransportError as error:
            raise ProviderError(
                "OpenAI-compatible response transport failed",
                category="transport_error",
            ) from error
        completed_at = datetime.now(UTC)
        self.last_generation_metadata = GenerationMetadata(
            provider=ProviderName.OPENAI_COMPATIBLE,
            model=self.settings.llm_model,
            base_url=self.base_url,
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
            duration_seconds=time.monotonic() - started,
            prompt_characters=sum(len(message["content"]) for message in messages),
            output_characters=len("".join(parts)),
            streaming=True,
            response_format_mode=request.json_mode,
            finish_reason=finish_reason,
            usage=usage,
            time_to_first_token_ms=(
                (first_token_at - started) * 1000 if first_token_at is not None else None
            ),
        )
        token_usage = TokenUsage(
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            provenance=(UsageProvenance.EXACT if usage else UsageProvenance.UNAVAILABLE),
        )
        yield ProviderUsageEvent(sequence=sequence, usage=token_usage)
        sequence += 1
        yield ProviderCompletedEvent(sequence=sequence, finish_reason=finish_reason)

    def _request_body(
        self,
        messages: list[dict[str, str]],
        response_schema: dict[str, object] | None,
        max_output_tokens: int | None,
        mode: JsonMode,
        tools: list[ProviderFunctionTool] | None,
        tool_choice: str | ProviderFunctionToolChoice,
        *,
        stream: bool,
    ) -> dict[str, object]:
        request_body: dict[str, object] = {
            "model": self.settings.llm_model,
            "messages": messages,
            "temperature": self.settings.llm_temperature,
            "stream": stream,
        }
        if max_output_tokens is not None:
            request_body["max_tokens"] = max_output_tokens
        if mode is JsonMode.JSON_SCHEMA and response_schema is not None:
            request_body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "model_implementation_response",
                    "strict": True,
                    "schema": response_schema,
                },
            }
        elif mode is JsonMode.JSON_OBJECT:
            request_body["response_format"] = {"type": "json_object"}
        if tools:
            request_body["tools"] = [_serialize_function_tool(tool) for tool in tools]
            request_body["tool_choice"] = _serialize_tool_choice(tool_choice)
        return request_body

    def _non_stream_chat(
        self, client: httpx.Client, request_body: dict[str, object], limit: int
    ) -> tuple[str, str | None, dict[str, int], list[FunctionCallMetadata]]:
        response = client.post(f"{self.base_url}/chat/completions", json=request_body)
        response.raise_for_status()
        try:
            payload = response.json()
            choice = payload["choices"][0]
            message = choice["message"]
            content = message.get("content") or message.get("refusal") or ""
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as error:
            raise ProviderError(
                "Invalid chat completion response", category="invalid_response"
            ) from error
        if not isinstance(content, str):
            raise ProviderError("Invalid chat content", category="invalid_response")
        if len(content) > limit:
            raise ProviderError(
                f"Provider output exceeds {limit} characters", category="output_limit"
            )
        return (
            content,
            choice.get("finish_reason"),
            _parse_usage(payload.get("usage")),
            _parse_openai_function_calls(message),
        )

    def _effective_json_mode(self) -> JsonMode:
        return (
            JsonMode.JSON_OBJECT
            if self.settings.llm_json_mode is JsonMode.AUTO
            else self.settings.llm_json_mode
        )

    def _is_unsupported_response_format(
        self, response: httpx.Response, request_body: dict[str, object]
    ) -> bool:
        if response.status_code != 400 or "response_format" not in request_body:
            return False
        body = response.text.lower()
        return "response_format" in body and any(
            marker in body for marker in ("unsupported", "not supported", "unknown", "unrecognized")
        )

    def _http_error(self, response: httpx.Response) -> ProviderError:
        messages = {
            401: "Authentication failed (HTTP 401)",
            402: "Provider requires payment or credits (HTTP 402)",
            404: "Endpoint or model was not found (HTTP 404)",
            429: "Provider rate limit exceeded (HTTP 429)",
        }
        message = messages.get(response.status_code)
        if message is None and response.status_code >= 500:
            message = f"Provider server error (HTTP {response.status_code})"
        if message is None:
            message = f"OpenAI-compatible provider returned HTTP {response.status_code}"
        return ProviderError(message, category=f"http_{response.status_code}")


class _UnsupportedResponseFormat(ProviderError):
    def __init__(self):
        super().__init__("Provider does not support response_format", category="unsupported_format")


def _parse_usage(raw_usage: object) -> dict[str, int]:
    if not isinstance(raw_usage, dict):
        return {}
    usage: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = raw_usage.get(key)
        if isinstance(value, int):
            usage[key] = value
    return usage


def _serialize_function_tool(tool: ProviderFunctionTool) -> dict[str, object]:
    function: dict[str, object] = {
        "name": tool.name,
        "parameters": tool.parameters,
        "strict": tool.strict,
    }
    if tool.description is not None:
        function["description"] = tool.description
    return {"type": "function", "function": function}


def _serialize_tool_choice(choice: str | ProviderFunctionToolChoice) -> object:
    if isinstance(choice, ProviderFunctionToolChoice):
        return {"type": "function", "function": {"name": choice.name}}
    return choice


def _parse_openai_function_calls(message: object) -> list[FunctionCallMetadata]:
    if not isinstance(message, dict) or not isinstance(message.get("tool_calls"), list):
        return []
    calls: list[FunctionCallMetadata] = []
    for raw_call in message["tool_calls"]:
        if not isinstance(raw_call, dict) or not isinstance(raw_call.get("function"), dict):
            raise ProviderError("Invalid provider tool call", category="invalid_response")
        function = raw_call["function"]
        call_id = raw_call.get("id")
        name = function.get("name")
        arguments = function.get("arguments")
        if not all(isinstance(value, str) for value in (call_id, name, arguments)):
            raise ProviderError("Invalid provider tool call", category="invalid_response")
        calls.append(
            FunctionCallMetadata(
                call_id=call_id,
                name=name,
                arguments=arguments,
            )
        )
    return calls
