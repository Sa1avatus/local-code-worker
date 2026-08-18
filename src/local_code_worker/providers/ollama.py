import json
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from urllib.parse import urlsplit, urlunsplit

import httpx

from ..config import WorkerSettings
from ..exceptions import ProviderError
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
    ProviderFunctionCall,
    ProviderFunctionTool,
    ProviderFunctionToolChoice,
    ProviderMessage,
    ProviderReasoningDeltaEvent,
    ProviderRequest,
    ProviderStartedEvent,
    ProviderTextDeltaEvent,
    ProviderToolCallsEvent,
    ProviderUsageEvent,
)


class OllamaProvider:
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

    def unload_model(self, model: str | None = None) -> None:
        url = f"{self.base_url}/api/generate"
        try:
            with self._client() as client:
                response = client.post(
                    url,
                    json={"model": model or self.settings.llm_model, "keep_alive": 0},
                )
                response.raise_for_status()
        except httpx.HTTPError as error:
            raise ProviderError("Ollama model unload failed", category="model_unload") from error

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
        tools: list[ProviderFunctionTool] | None = None,
        tool_choice: str | ProviderFunctionToolChoice = "auto",
    ) -> str:
        mode = self.settings.llm_json_mode
        if self.settings.llm_stream:
            parts: list[str] = []
            for event in self.stream(
                ProviderRequest(
                    messages=[ProviderMessage.model_validate(message) for message in messages],
                    response_schema=response_schema,
                    max_output_characters=max_output_characters,
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
            tools,
            tool_choice,
            stream=False,
        )

        started_at = datetime.now(UTC)
        started = time.monotonic()
        finish_reason: str | None = None
        try:
            with self._client() as client:
                content, finish_reason, usage, function_calls, reasoning = self._non_stream_chat(
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
            # Graceful degradation: if Ollama rejects tools (HTTP 400),
            # retry without them. Some models don't support tools.
            import logging
            logging.getLogger("local_code_worker.ollama").warning(
                "Ollama HTTP %d, tools=%s, retry=%s",
                error.response.status_code,
                bool(tools),
                tools and error.response.status_code == 400,
            )
            if tools and error.response.status_code == 400:
                request_body = self._request_body(
                    messages, response_schema, max_output_tokens,
                    None, "auto", stream=False,
                )
                try:
                    with self._client() as client:
                        content, finish_reason, usage, function_calls, reasoning = (
                            self._non_stream_chat(client, request_body, max_output_characters)
                        )
                except httpx.HTTPStatusError as retry_error:
                    raise ProviderError(
                        f"Ollama returned HTTP {retry_error.response.status_code}",
                        category="http_error",
                    ) from retry_error
            else:
                raise ProviderError(
                    f"Ollama returned HTTP {error.response.status_code}",
                    category="http_error",
                ) from error

        if tools and not function_calls:
            text_call = _parse_ollama_text_function_call(content, tools)
            if text_call is not None:
                function_calls = [text_call]
                content = ""

        completed_at = datetime.now(UTC)
        self.last_generation_metadata = GenerationMetadata(
            provider=ProviderName.OLLAMA,
            model=self.settings.llm_model,
            base_url=self.base_url,
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
            duration_seconds=time.monotonic() - started,
            prompt_characters=sum(
                len(message.get("content") or "") for message in messages
            ),
            output_characters=len(content),
            streaming=self.settings.llm_stream,
            response_format_mode=mode,
            finish_reason=finish_reason,
            usage=usage,
            function_calls=function_calls,
            reasoning=reasoning,
        )
        return content

    def stream(self, request: ProviderRequest) -> Iterator[ProviderEvent]:
        if not request.stream:
            raise ValueError("Ollama stream request must enable streaming")
        if request.json_mode is not self.settings.llm_json_mode:
            raise ValueError("Ollama stream request json_mode must match provider settings")
        messages = [message.model_dump(exclude_none=True) for message in request.messages]
        request_body = self._request_body(
            messages,
            request.response_schema,
            request.max_output_tokens,
            request.tools,
            request.tool_choice,
            stream=True,
        )
        parts: list[str] = []
        reasoning_parts: list[str] = []
        total = 0
        finish_reason: str | None = None
        saw_done = False
        usage: dict[str, int] = {}
        pending_function_calls: list[ProviderFunctionCall] = []
        sequence = 0
        started_at = datetime.now(UTC)
        started = time.monotonic()
        first_token_at: float | None = None
        yield ProviderStartedEvent(
            sequence=sequence,
            provider=ProviderName.OLLAMA,
            model=self.settings.llm_model,
        )
        sequence += 1
        # Retry loop: if Ollama rejects tools (HTTP 400), strip tools
        # and retry so the model still responds.
        retry_without_tools = request.tools and len(request.tools) > 0
        while True:
            try:
                with self._client() as client:
                    with client.stream(
                        "POST", f"{self.base_url}/api/chat", json=request_body
                    ) as response:
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
                            thinking = chunk.get("message", {}).get("thinking", "")
                            if (
                                isinstance(thinking, str)
                                and thinking
                                and self.settings.llm_show_reasoning is not False
                            ):
                                reasoning_parts.append(thinking)
                                # Stream reasoning live so the client can render the
                                # chain-of-thought as it is produced (Ollama emits
                                # `thinking` tokens before the first content token).
                                yield ProviderReasoningDeltaEvent(sequence=sequence, delta=thinking)
                                sequence += 1
                            if chunk.get("message", {}).get("tool_calls"):
                                for tc in _parse_ollama_function_calls(chunk["message"]):
                                    pending_function_calls.append(
                                        ProviderFunctionCall(
                                            call_id=tc.call_id,
                                            name=tc.name,
                                            arguments=tc.arguments,
                                        )
                                    )
                            if not isinstance(text, str):
                                raise ProviderError(
                                    "Ollama stream chunk has invalid message.content",
                                    category="invalid_stream_chunk",
                                )
                            total += len(text)
                            if total > request.max_output_characters:
                                raise ProviderError(
                                    "Ollama output exceeds the "
                                    f"{request.max_output_characters}-character limit",
                                    category="output_limit",
                                )
                            if text:
                                if first_token_at is None:
                                    first_token_at = time.monotonic()
                                parts.append(text)
                                yield ProviderTextDeltaEvent(sequence=sequence, delta=text)
                                sequence += 1
                            if chunk.get("done") is True:
                                saw_done = True
                                finish_reason = chunk.get("done_reason")
                                usage = _parse_usage(chunk)
                                break
                break  # Success — exit retry loop
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
                if retry_without_tools and error.response.status_code == 400:
                    request_body = self._request_body(
                        messages, request.response_schema,
                        request.max_output_tokens, None, "auto", stream=True,
                    )
                    retry_without_tools = False
                    continue
                raise ProviderError(
                    f"Ollama returned HTTP {error.response.status_code}",
                    category="http_error",
                ) from error
        if not saw_done:
            raise ProviderError(
                "Ollama stream ended before a done chunk",
                category="truncated_stream",
            )
        # Fallback: try to parse tool calls from text content for models
        # that don't use native tool_calls during streaming.
        if request.tools and not pending_function_calls:
            content = "".join(parts)
            text_call = _parse_ollama_text_function_call(content, request.tools)
            if text_call is not None:
                pending_function_calls.append(
                    ProviderFunctionCall(
                        call_id=text_call.call_id,
                        name=text_call.name,
                        arguments=text_call.arguments,
                    )
                )
                parts = []
        completed_at = datetime.now(UTC)
        self.last_generation_metadata = GenerationMetadata(
            provider=ProviderName.OLLAMA,
            model=self.settings.llm_model,
            base_url=self.base_url,
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
            duration_seconds=time.monotonic() - started,
            prompt_characters=sum(
                len(message.get("content") or "") for message in messages
            ),
            output_characters=len("".join(parts)),
            streaming=True,
            response_format_mode=request.json_mode,
            finish_reason=finish_reason,
            usage=usage,
            reasoning="".join(reasoning_parts) or None,
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
        if pending_function_calls:
            yield ProviderToolCallsEvent(
                sequence=sequence,
                function_calls=pending_function_calls,
            )
            sequence += 1
        yield ProviderCompletedEvent(sequence=sequence, finish_reason=finish_reason)

    def _request_body(
        self,
        messages: list[dict[str, str]],
        response_schema: dict[str, object] | None,
        max_output_tokens: int | None,
        tools: list[ProviderFunctionTool] | None,
        tool_choice: str | ProviderFunctionToolChoice,
        *,
        stream: bool,
    ) -> dict[str, object]:
        temperature = self.settings.llm_temperature
        if self.settings.llm_think is not False and temperature <= 0:
            # Reasoning models (qwen3.x) degenerate into a looping thinking trace
            # under greedy decoding (temperature 0), exhausting num_predict before
            # any answer. Thinking requires sampling, so fall back to a non-zero
            # temperature whenever thinking is not explicitly disabled.
            temperature = 0.6
        options: dict[str, object] = {
            "temperature": temperature,
            "num_ctx": self.settings.llm_num_ctx,
            "num_parallel": self.settings.llm_num_parallel,
        }
        if max_output_tokens is not None:
            options["num_predict"] = max_output_tokens
        # .env-only sampling knobs (no UI): forwarded only when set.
        if self.settings.llm_repeat_penalty is not None:
            options["repeat_penalty"] = self.settings.llm_repeat_penalty
        if self.settings.llm_seed is not None:
            options["seed"] = self.settings.llm_seed
        request_body: dict[str, object] = {
            "model": self.settings.llm_model,
            "stream": stream,
            "keep_alive": self.settings.llm_keep_alive,
            "messages": messages,
            "options": options,
        }
        # Reasoning models (qwen3.x) drain the token budget on thinking; the
        # gateway lets the client decide per request. Absent = model default.
        # `think_level` (low/medium/high/max) forces thinking at that intensity.
        if self.settings.llm_think is False:
            request_body["think"] = False
        elif self.settings.llm_think_level is not None:
            request_body["think"] = self.settings.llm_think_level
        elif self.settings.llm_think is True:
            request_body["think"] = True
        mode = self.settings.llm_json_mode
        if mode is JsonMode.JSON_SCHEMA and response_schema is not None:
            request_body["format"] = response_schema
        elif mode in {JsonMode.AUTO, JsonMode.JSON_OBJECT}:
            request_body["format"] = "json"
        if tools:
            request_body["tools"] = [_serialize_function_tool(tool) for tool in tools]
            request_body["tool_choice"] = _serialize_tool_choice(tool_choice)
        return request_body

    def _non_stream_chat(
        self, client: httpx.Client, request_body: dict[str, object], limit: int
    ) -> tuple[str, str | None, dict[str, int], list[FunctionCallMetadata], str | None]:
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
        function_calls = _parse_ollama_function_calls(payload.get("message"))
        thinking = (
            payload.get("message", {}).get("thinking") if isinstance(payload, dict) else None
        )
        return (
            content,
            payload.get("done_reason"),
            _parse_usage(payload),
            function_calls,
            (
                thinking
                if isinstance(thinking, str) and self.settings.llm_show_reasoning is not False
                else None
            ),
        )

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


def _serialize_function_tool(tool: ProviderFunctionTool) -> dict[str, object]:
    function: dict[str, object] = {
        "name": tool.name,
        "parameters": tool.parameters,
    }
    if tool.description is not None:
        function["description"] = tool.description
    return {"type": "function", "function": function}


def _serialize_tool_choice(choice: str | ProviderFunctionToolChoice) -> object:
    if isinstance(choice, ProviderFunctionToolChoice):
        return {"type": "function", "function": {"name": choice.name}}
    return choice


def _parse_ollama_function_calls(message: object) -> list[FunctionCallMetadata]:
    if not isinstance(message, dict) or not isinstance(message.get("tool_calls"), list):
        return []
    calls: list[FunctionCallMetadata] = []
    for index, raw_call in enumerate(message["tool_calls"]):
        if not isinstance(raw_call, dict) or not isinstance(raw_call.get("function"), dict):
            raise ProviderError("Invalid Ollama tool call", category="invalid_response")
        function = raw_call["function"]
        name = function.get("name")
        arguments = function.get("arguments", {})
        if not isinstance(name, str) or not isinstance(arguments, dict):
            raise ProviderError("Invalid Ollama tool call", category="invalid_response")
        call_id = raw_call.get("id")
        calls.append(
            FunctionCallMetadata(
                call_id=call_id if isinstance(call_id, str) else f"call_{index}",
                name=name,
                arguments=json.dumps(arguments, separators=(",", ":")),
            )
        )
    return calls


def _parse_ollama_text_function_call(
    content: str,
    tools: list[ProviderFunctionTool],
) -> FunctionCallMetadata | None:
    stripped = content.strip()
    allowed_names = {tool.name for tool in tools}
    # Try <tools>...</tools> XML format first
    opening = "<tools>"
    closing = "</tools>"
    if stripped.startswith(opening) and stripped.endswith(closing):
        encoded = stripped[len(opening) : -len(closing)].strip()
        result = _try_parse_tool_json(encoded, allowed_names)
        if result is not None:
            return result
    # Try bare JSON function call: {"name": "...", "arguments": {...}}
    if stripped.startswith("{") and stripped.endswith("}"):
        result = _try_parse_tool_json(stripped, allowed_names)
        if result is not None:
            return result
    return None


def _try_parse_tool_json(encoded: str, allowed_names: set[str]) -> FunctionCallMetadata | None:
    try:
        payload = json.loads(encoded)
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    name = payload.get("name")
    arguments = payload.get("arguments")
    if not isinstance(name, str) or name not in allowed_names or not isinstance(arguments, dict):
        return None
    return FunctionCallMetadata(
        call_id="call_ollama_text_0",
        name=name,
        arguments=json.dumps(arguments, separators=(",", ":")),
    )


def _ollama_error_category(request_body: dict[str, object]) -> str:
    return (
        "structured_output_error"
        if isinstance(request_body.get("format"), dict)
        else "provider_stream_error"
    )
