import json
import os
import tempfile
import threading
from pathlib import Path

from .models import GenerationMetadata, ResponseAttempt, ResponseStatus

_LOCK = threading.Lock()
_MAX_EVENTS = 10_000


def statistics_path() -> Path:
    configured = os.environ.get("LOCAL_CODE_WORKER_STATISTICS_PATH")
    if configured:
        return Path(configured)
    if os.environ.get("LOCAL_CODE_WORKER_CONTAINER") == "1":
        return Path("/data/model-statistics.json")
    return Path(".local-worker/model-statistics.json")


def record_model_call(
    metadata: GenerationMetadata | None,
    *,
    kind: str,
    outcome: str,
    path: Path | None = None,
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
    models = []
    for item in grouped.values():
        duration = float(item.pop("duration_seconds"))
        completion_tokens = int(item["completion_tokens"])
        item["tokens_per_second"] = round(completion_tokens / duration, 2) if duration else 0
        models.append(item)
    return {"models": sorted(models, key=lambda item: str(item["model"]))}


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
