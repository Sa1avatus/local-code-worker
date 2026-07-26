import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

from .exceptions import OllamaError, PatchValidationError, ProviderError, SemanticValidationError
from .models import (
    JsonMode,
    ModelImplementationResponse,
    ProviderName,
    RequestMetadata,
    ResponseAttempt,
    ResponseStatus,
    WorkerTask,
)
from .patch_validator import validate_generated_changes
from .providers.base import LlmProvider
from .report_writer import RunReportWriter
from .repository import normalize_relative_path
from .response_parser import ParsedResponse, classify_model_response


@dataclass(frozen=True)
class GenerationResult:
    status: ResponseStatus
    response: ModelImplementationResponse | None
    attempts: tuple[ResponseAttempt, ...]
    error: str | None


def implementation_response_schema(task: WorkerTask) -> dict[str, object]:
    schema = ModelImplementationResponse.model_json_schema()
    definitions = schema.get("$defs", {})
    generated_file_schema = definitions.get("GeneratedFile")
    if isinstance(generated_file_schema, dict):
        properties = generated_file_schema.get("properties")
        if isinstance(properties, dict) and isinstance(properties.get("path"), dict):
            properties["path"]["enum"] = [
                normalize_relative_path(path) for path in task.allowed_files
            ]
    return schema


def build_request_metadata(
    run_id: str,
    task: WorkerTask,
    model: str,
    system_prompt: str,
    context: str,
    context_file_paths: list[str],
    started_at: str,
    provider: ProviderName = ProviderName.OLLAMA,
    base_url: str = "http://localhost:11434",
    api_key_env: str | None = None,
    json_mode: JsonMode = JsonMode.AUTO,
    stream: bool = True,
) -> RequestMetadata:
    prompt_bytes = f"{system_prompt}\0{context}".encode()
    return RequestMetadata(
        run_id=run_id,
        task_id=task.task_id,
        model=model,
        provider=provider,
        base_url=base_url,
        api_key_env=api_key_env,
        json_mode=json_mode,
        stream=stream,
        system_prompt_characters=len(system_prompt),
        task_context_characters=len(context),
        full_prompt_characters=len(system_prompt) + len(context),
        context_file_paths=context_file_paths,
        prompt_sha256=hashlib.sha256(prompt_bytes).hexdigest(),
        started_at=started_at,
    )


def generate_implementation(
    *,
    task: WorkerTask,
    task_path: Path,
    client: LlmProvider,
    system_prompt: str,
    context: str,
    report_writer: RunReportWriter,
    max_repair_attempts: int,
    save_invalid_response: bool,
) -> GenerationResult:
    schema = implementation_response_schema(task)
    provider_settings = getattr(client, "settings", None)
    provider_output_limit = getattr(
        provider_settings,
        "llm_max_output_characters",
        task.max_output_characters,
    )
    output_limit = min(task.max_output_characters, provider_output_limit)
    attempts: list[ResponseAttempt] = []
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": context},
    ]
    if getattr(provider_settings, "llm_json_mode", None) is JsonMode.PROMPT_ONLY:
        messages[0]["content"] += (
            "\n\nThe API will not enforce the response format. Return exactly one JSON "
            "object and no prose or Markdown. It must match this JSON Schema:\n"
            + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        )
    started = time.monotonic()
    try:
        raw_response = client.chat(
            messages,
            schema,
            output_limit,
        )
    except (ProviderError, OllamaError) as error:
        attempts.append(
            _transport_error_attempt(
                client,
                error,
                attempt=1,
                duration_seconds=time.monotonic() - started,
            )
        )
        status = (
            ResponseStatus.PROVIDER_ERROR
            if isinstance(error, ProviderError)
            else ResponseStatus.OLLAMA_ERROR
        )
        return GenerationResult(status, None, tuple(attempts), str(error))

    parsed = _validate_attempt(task, task_path, raw_response)
    parsed = _apply_finish_reason(client, parsed)
    response_saved = parsed.status is ResponseStatus.VALID or save_invalid_response
    if response_saved:
        report_writer.write_raw_response(1, raw_response)
    if parsed.error:
        report_writer.write_parse_error(1, parsed.error)
    attempts.append(
        _response_attempt(
            client,
            attempt=1,
            status=parsed.status,
            error=parsed.error,
            duration_seconds=time.monotonic() - started,
            response_file_path=("model-response-attempt-1.json" if response_saved else None),
        )
    )
    if parsed.status is ResponseStatus.VALID:
        return GenerationResult(parsed.status, parsed.response, tuple(attempts), None)
    if (
        parsed.status
        not in {
            ResponseStatus.INVALID_JSON,
            ResponseStatus.SCHEMA_VALIDATION_FAILED,
            ResponseStatus.SEMANTIC_VALIDATION_FAILED,
            ResponseStatus.PLACEHOLDER_CONTENT,
            ResponseStatus.TRUNCATED_RESPONSE,
        }
        or max_repair_attempts == 0
    ):
        return GenerationResult(parsed.status, None, tuple(attempts), parsed.error)

    repair_prompt = build_repair_prompt(raw_response, parsed.error or "Unknown error", schema)
    repair_messages = [
        {
            "role": "system",
            "content": (
                "Repair a structured JSON response. Return JSON only. Do not execute commands, "
                "access external systems, add files, or change implementation intent."
            ),
        },
        {"role": "assistant", "content": raw_response},
        {"role": "user", "content": repair_prompt},
    ]
    repair_started = time.monotonic()
    try:
        repaired_raw_response = client.chat(
            repair_messages,
            schema,
            output_limit,
        )
    except (ProviderError, OllamaError) as error:
        attempts.append(
            _transport_error_attempt(
                client,
                error,
                attempt=2,
                duration_seconds=time.monotonic() - repair_started,
                status=ResponseStatus.REPAIR_FAILED,
            )
        )
        return GenerationResult(ResponseStatus.REPAIR_FAILED, None, tuple(attempts), str(error))

    repaired = _validate_attempt(task, task_path, repaired_raw_response)
    repaired = _apply_finish_reason(client, repaired)
    repaired_response_saved = repaired.status is ResponseStatus.VALID or save_invalid_response
    if repaired_response_saved:
        report_writer.write_raw_response(2, repaired_raw_response)
    if repaired.error:
        report_writer.write_parse_error(2, repaired.error)
    repair_status = (
        ResponseStatus.VALID
        if repaired.status is ResponseStatus.VALID
        else ResponseStatus.REPAIR_FAILED
    )
    attempts.append(
        _response_attempt(
            client,
            attempt=2,
            status=repair_status,
            error=repaired.error,
            duration_seconds=time.monotonic() - repair_started,
            response_file_path=(
                "model-response-attempt-2.json" if repaired_response_saved else None
            ),
        )
    )
    return GenerationResult(
        repair_status,
        repaired.response if repair_status is ResponseStatus.VALID else None,
        tuple(attempts),
        repaired.error,
    )


def build_repair_prompt(
    previous_response: str, validation_error: str, schema: dict[str, object]
) -> str:
    return "\n\n".join(
        [
            "Your previous response could not be accepted.",
            f"Validation error:\n{validation_error}",
            "Return only one valid JSON object matching the supplied schema.",
            "Do not explain the error. Do not use Markdown. Do not add files. "
            "Do not change the intended implementation.",
            f"JSON SCHEMA:\n{json.dumps(schema, ensure_ascii=False, indent=2)}",
            f"PREVIOUS RESPONSE:\n{previous_response}",
        ]
    )


def _validate_attempt(task: WorkerTask, task_path: Path, raw_response: str) -> ParsedResponse:
    parsed = classify_model_response(raw_response)
    if parsed.response is None:
        return parsed
    try:
        validated = validate_generated_changes(task, parsed.response, task_path)
    except SemanticValidationError as error:
        status = (
            ResponseStatus.PLACEHOLDER_CONTENT
            if error.category == "placeholder_content"
            else ResponseStatus.SEMANTIC_VALIDATION_FAILED
        )
        return ParsedResponse(status, None, str(error))
    except PatchValidationError as error:
        return ParsedResponse(ResponseStatus.SCHEMA_VALIDATION_FAILED, None, str(error))
    return ParsedResponse(ResponseStatus.VALID, validated, None)


def _apply_finish_reason(client: LlmProvider, parsed: ParsedResponse) -> ParsedResponse:
    metadata = getattr(client, "last_generation_metadata", None)
    if metadata and metadata.finish_reason in {"length", "max_tokens"}:
        return ParsedResponse(
            ResponseStatus.TRUNCATED_RESPONSE,
            None,
            f"Provider finish reason: {metadata.finish_reason}",
        )
    if metadata and metadata.finish_reason == "content_filter":
        return ParsedResponse(
            ResponseStatus.PROVIDER_REFUSAL,
            None,
            "Provider finish reason: content_filter",
        )
    if metadata and metadata.finish_reason == "error":
        return ParsedResponse(
            ResponseStatus.PROVIDER_ERROR,
            None,
            "Provider finish reason: error",
        )
    return parsed


def _response_attempt(
    client: LlmProvider,
    *,
    attempt: int,
    status: ResponseStatus,
    error: str | None,
    duration_seconds: float,
    response_file_path: str | None,
) -> ResponseAttempt:
    metadata = getattr(client, "last_generation_metadata", None)
    return ResponseAttempt(
        attempt=attempt,
        status=status,
        error=error,
        duration_seconds=duration_seconds,
        provider=metadata.provider if metadata else None,
        model=metadata.model if metadata else None,
        prompt_characters=metadata.prompt_characters if metadata else None,
        output_characters=metadata.output_characters if metadata else None,
        finish_reason=metadata.finish_reason if metadata else None,
        usage=metadata.usage if metadata else {},
        error_category=status.value if error else None,
        response_file_path=response_file_path,
    )


def _transport_error_attempt(
    client: LlmProvider,
    error: ProviderError | OllamaError,
    *,
    attempt: int,
    duration_seconds: float,
    status: ResponseStatus | None = None,
) -> ResponseAttempt:
    settings = getattr(client, "settings", None)
    provider = getattr(settings, "llm_provider", None)
    model = getattr(settings, "llm_model", None)
    default_status = (
        ResponseStatus.PROVIDER_ERROR
        if isinstance(error, ProviderError)
        else ResponseStatus.OLLAMA_ERROR
    )
    return ResponseAttempt(
        attempt=attempt,
        status=status or default_status,
        error=str(error),
        duration_seconds=duration_seconds,
        provider=provider,
        model=model,
        error_category=getattr(error, "category", default_status.value),
    )
