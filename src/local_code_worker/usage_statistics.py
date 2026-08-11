import json
import os
import sqlite3
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .models import GenerationMetadata, ResponseAttempt, ResponseStatus
from .routing.models import EscalationEvent, RouteLease, RoutingPlan
from .telemetry.database import TelemetryDatabase
from .telemetry.models import ModelRequestTelemetry, TokenUsage, UsageProvenance

_LOCK = threading.Lock()
_MAX_EVENTS = 10_000


def statistics_path() -> Path:
    configured = os.environ.get("LOCAL_CODE_WORKER_STATISTICS_PATH")
    if configured:
        return Path(configured)
    if os.environ.get("LOCAL_CODE_WORKER_CONTAINER") == "1":
        return Path("/data/model-statistics.json")
    return Path(".local-worker/model-statistics.json")


def telemetry_database_path() -> Path:
    configured = os.environ.get("LOCAL_CODE_WORKER_TELEMETRY_PATH")
    if configured:
        return Path(configured)
    if os.environ.get("LOCAL_CODE_WORKER_CONTAINER") == "1":
        return Path("/data/local-code-worker.db")
    return Path(".local-worker/local-code-worker.db")


def record_model_call(
    metadata: GenerationMetadata | None,
    *,
    kind: str,
    outcome: str,
    path: Path | None = None,
    request_id: str | None = None,
    tier: str | None = None,
    escalation_count: int = 0,
    tool_count: int = 0,
) -> None:
    if metadata is None:
        return
    target = path or statistics_path()
    event = {
        "provider": metadata.provider.value,
        "model": metadata.model,
        "kind": kind,
        "outcome": outcome,
        "duration_seconds": metadata.duration_seconds,
        "prompt_tokens": int(metadata.usage.get("prompt_tokens", 0)),
        "completion_tokens": int(metadata.usage.get("completion_tokens", 0)),
    }
    with _LOCK:
        events = _load_events(target)
        events.append(event)
        _write_events(target, events[-_MAX_EVENTS:])
    if path is None:
        _record_generation_telemetry(
            metadata,
            outcome,
            request_id=request_id,
            tier=tier,
            escalation_count=escalation_count,
            tool_count=tool_count,
        )


def summarize_model_calls(path: Path | None = None) -> dict[str, object]:
    grouped: dict[tuple[str, str], dict[str, float | int | str]] = {}
    with _LOCK:
        events = _load_events(path or statistics_path())
    for event in events:
        provider = event.get("provider")
        model = event.get("model")
        if not isinstance(provider, str) or not isinstance(model, str):
            continue
        key = (provider, model)
        item = grouped.setdefault(
            key,
            {
                "provider": provider,
                "model": model,
                "requests": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "duration_seconds": 0.0,
                "code_valid": 0,
                "code_invalid": 0,
                "api_completed": 0,
                "api_failed": 0,
            },
        )
        item["requests"] += 1
        for field in ("prompt_tokens", "completion_tokens"):
            value = event.get(field)
            if isinstance(value, int) and value >= 0:
                item[field] += value
        duration = event.get("duration_seconds")
        if isinstance(duration, (int, float)) and duration > 0:
            item["duration_seconds"] += float(duration)
        if event.get("kind") == "worker":
            if event.get("outcome") == ResponseStatus.VALID.value:
                item["code_valid"] += 1
            else:
                item["code_invalid"] += 1
        else:
            if event.get("outcome") == "completed":
                item["api_completed"] += 1
            else:
                item["api_failed"] += 1
    models = []
    for item in grouped.values():
        duration = float(item.pop("duration_seconds"))
        completion_tokens = int(item["completion_tokens"])
        item["tokens_per_second"] = round(completion_tokens / duration, 2) if duration else 0
        models.append(item)
    return {"models": sorted(models, key=lambda item: str(item["model"]))}


def summarize_v2_statistics(
    baseline_cloud_tokens: int | None = None,
    *,
    path: Path | None = None,
) -> dict[str, object]:
    database = TelemetryDatabase(path or telemetry_database_path())
    database.initialize()
    savings = (
        database.estimate_cloud_token_savings(baseline_cloud_tokens)
        if baseline_cloud_tokens is not None
        else None
    )
    return {
        "version": 2,
        "requests": database.summarize_requests().model_dump(mode="json"),
        "token_savings": savings.model_dump(mode="json") if savings is not None else None,
    }


def record_routing_plan(request_id: str, plan: RoutingPlan) -> None:
    try:
        database = TelemetryDatabase(telemetry_database_path())
        database.initialize()
        database.record_routing_plan(request_id, plan)
    except (OSError, sqlite3.Error):
        return


def record_route_lease(lease: RouteLease) -> None:
    try:
        database = TelemetryDatabase(telemetry_database_path())
        database.initialize()
        database.record_route_lease(lease)
    except (OSError, sqlite3.Error):
        return


def record_escalation(event: EscalationEvent) -> None:
    try:
        database = TelemetryDatabase(telemetry_database_path())
        database.initialize()
        database.record_escalation(event)
    except (OSError, sqlite3.Error):
        return


def summarize_routing() -> dict[str, object]:
    database = TelemetryDatabase(telemetry_database_path())
    database.initialize()
    return database.summarize_routing()


def get_routing_plan(request_id: str) -> RoutingPlan | None:
    database = TelemetryDatabase(telemetry_database_path())
    database.initialize()
    return database.get_routing_plan(request_id)


def record_worker_attempt(attempt: ResponseAttempt, path: Path | None = None) -> None:
    if attempt.provider is None or attempt.model is None:
        return
    target = path or statistics_path()
    event = {
        "provider": attempt.provider.value,
        "model": attempt.model,
        "kind": "worker",
        "outcome": attempt.status.value,
        "duration_seconds": attempt.duration_seconds,
        "prompt_tokens": int(attempt.usage.get("prompt_tokens", 0)),
        "completion_tokens": int(attempt.usage.get("completion_tokens", 0)),
    }
    with _LOCK:
        events = _load_events(target)
        events.append(event)
        _write_events(target, events[-_MAX_EVENTS:])
    if path is None:
        _record_attempt_telemetry(attempt)


def _load_events(path: Path) -> list[dict[str, object]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return payload if isinstance(payload, list) else []


def _write_events(path: Path, events: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".model-statistics-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(events, stream, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _record_generation_telemetry(
    metadata: GenerationMetadata,
    outcome: str,
    *,
    request_id: str | None = None,
    tier: str | None = None,
    escalation_count: int = 0,
    tool_count: int = 0,
) -> None:
    usage = metadata.usage
    has_provider_usage = "prompt_tokens" in usage or "completion_tokens" in usage
    telemetry = ModelRequestTelemetry(
        request_id=request_id or uuid4().hex,
        timestamp=metadata.completed_at,
        provider=metadata.provider.value,
        model=metadata.model,
        tier=tier or ("local" if metadata.provider.value == "ollama" else "cloud"),
        usage=TokenUsage(
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            cached_input_tokens=int(usage.get("cached_tokens", 0)),
            reasoning_tokens=int(usage.get("reasoning_tokens", 0)),
            provenance=(
                UsageProvenance.EXACT if has_provider_usage else UsageProvenance.UNAVAILABLE
            ),
        ),
        latency_ms=metadata.duration_seconds * 1000,
        escalation_count=escalation_count,
        tool_count=tool_count,
        success=outcome == "completed",
        failure_type=None if outcome == "completed" else outcome,
    )
    try:
        database = TelemetryDatabase(telemetry_database_path())
        database.initialize()
        database.record_request(telemetry)
    except (OSError, sqlite3.Error):
        return


def _record_attempt_telemetry(attempt: ResponseAttempt) -> None:
    if attempt.provider is None or attempt.model is None:
        return
    usage = attempt.usage
    has_provider_usage = "prompt_tokens" in usage or "completion_tokens" in usage
    telemetry = ModelRequestTelemetry(
        request_id=uuid4().hex,
        timestamp=datetime.now(UTC).isoformat(),
        provider=attempt.provider.value,
        model=attempt.model,
        tier="local" if attempt.provider.value == "ollama" else "cloud",
        usage=TokenUsage(
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            cached_input_tokens=int(usage.get("cached_tokens", 0)),
            reasoning_tokens=int(usage.get("reasoning_tokens", 0)),
            provenance=(
                UsageProvenance.EXACT if has_provider_usage else UsageProvenance.UNAVAILABLE
            ),
        ),
        latency_ms=attempt.duration_seconds * 1000,
        retry_count=max(attempt.attempt - 1, 0),
        success=attempt.status is ResponseStatus.VALID,
        failure_type=(
            None
            if attempt.status is ResponseStatus.VALID
            else attempt.error_category or attempt.status.value
        ),
    )
    try:
        database = TelemetryDatabase(telemetry_database_path())
        database.initialize()
        database.record_request(telemetry)
    except (OSError, sqlite3.Error):
        return
