from pathlib import Path

from local_code_worker.models import (
    GenerationMetadata,
    JsonMode,
    ProviderName,
    ResponseAttempt,
    ResponseStatus,
)
from local_code_worker.telemetry.database import TelemetryDatabase
from local_code_worker.telemetry.models import UsageProvenance
from local_code_worker.usage_statistics import (
    record_model_call,
    record_worker_attempt,
    summarize_model_calls,
    summarize_v2_statistics,
)


def test_statistics_aggregate_tokens_speed_and_code_results(tmp_path: Path) -> None:
    path = tmp_path / "statistics.json"
    metadata = GenerationMetadata(
        provider=ProviderName.OLLAMA,
        model="qwen:test",
        base_url="http://localhost",
        started_at="2026-01-01T00:00:00Z",
        completed_at="2026-01-01T00:00:02Z",
        duration_seconds=2,
        prompt_characters=0,
        output_characters=0,
        streaming=True,
        response_format_mode=JsonMode.NONE,
        usage={"prompt_tokens": 4, "completion_tokens": 10},
    )
    record_model_call(metadata, kind="chat", outcome="completed", path=path)
    record_worker_attempt(
        ResponseAttempt(
            attempt=1,
            status=ResponseStatus.VALID,
            duration_seconds=1,
            provider=ProviderName.OLLAMA,
            model="qwen:test",
            usage={"prompt_tokens": 2, "completion_tokens": 5},
        ),
        path,
    )
    item = summarize_model_calls(path)["models"][0]
    assert item["requests"] == 2
    assert item["prompt_tokens"] == 6
    assert item["completion_tokens"] == 15
    assert item["tokens_per_second"] == 5.0
    assert item["api_completed"] == 1
    assert item["api_failed"] == 0
    assert item["code_valid"] == 1
    assert item["code_invalid"] == 0


def test_statistics_separate_api_failures_from_invalid_proposals(tmp_path: Path) -> None:
    path = tmp_path / "statistics.json"
    metadata = GenerationMetadata(
        provider=ProviderName.OLLAMA,
        model="qwen:test",
        base_url="http://localhost",
        started_at="2026-01-01T00:00:00Z",
        completed_at="2026-01-01T00:00:01Z",
        duration_seconds=1,
        prompt_characters=0,
        output_characters=0,
        streaming=False,
        response_format_mode=JsonMode.NONE,
        usage={},
    )
    record_model_call(metadata, kind="response", outcome="failed", path=path)

    item = summarize_model_calls(path)["models"][0]

    assert item["requests"] == 1
    assert item["api_completed"] == 0
    assert item["api_failed"] == 1
    assert item["code_valid"] == 0
    assert item["code_invalid"] == 0


def test_runtime_model_call_is_observed_in_sqlite(tmp_path: Path, monkeypatch) -> None:
    database_path = tmp_path / "telemetry.db"
    statistics_path = tmp_path / "statistics.json"
    monkeypatch.setenv("LOCAL_CODE_WORKER_TELEMETRY_PATH", str(database_path))
    monkeypatch.setenv("LOCAL_CODE_WORKER_STATISTICS_PATH", str(statistics_path))
    metadata = GenerationMetadata(
        provider=ProviderName.OLLAMA,
        model="qwen:test",
        base_url="http://localhost",
        started_at="2026-01-01T00:00:00Z",
        completed_at="2026-01-01T00:00:02Z",
        duration_seconds=2,
        prompt_characters=10,
        output_characters=5,
        streaming=True,
        response_format_mode=JsonMode.NONE,
        usage={"prompt_tokens": 4, "completion_tokens": 10},
    )

    record_model_call(metadata, kind="chat", outcome="completed")

    with TelemetryDatabase(database_path).connect() as connection:
        request_id = connection.execute("SELECT request_id FROM model_requests").fetchone()[0]
    observed = TelemetryDatabase(database_path).get_request(request_id)
    assert observed is not None
    assert observed.usage.provenance is UsageProvenance.EXACT
    assert observed.usage.total_tokens == 14
    assert observed.latency_ms == 2000
    assert observed.success is True


def test_runtime_worker_attempt_is_observed_in_sqlite(tmp_path: Path, monkeypatch) -> None:
    database_path = tmp_path / "telemetry.db"
    monkeypatch.setenv("LOCAL_CODE_WORKER_TELEMETRY_PATH", str(database_path))
    monkeypatch.setenv("LOCAL_CODE_WORKER_STATISTICS_PATH", str(tmp_path / "statistics.json"))
    attempt = ResponseAttempt(
        attempt=2,
        status=ResponseStatus.INVALID_JSON,
        error="response omitted",
        error_category="invalid_response",
        duration_seconds=1.5,
        provider=ProviderName.OLLAMA,
        model="qwen:test",
        usage={"prompt_tokens": 3, "completion_tokens": 2},
    )

    record_worker_attempt(attempt)

    with TelemetryDatabase(database_path).connect() as connection:
        request_id = connection.execute("SELECT request_id FROM model_requests").fetchone()[0]
    observed = TelemetryDatabase(database_path).get_request(request_id)
    assert observed is not None
    assert observed.retry_count == 1
    assert observed.success is False
    assert observed.failure_type == "invalid_response"
    assert observed.usage.total_tokens == 5


def test_v2_statistics_returns_typed_empty_summary(tmp_path: Path) -> None:
    result = summarize_v2_statistics(path=tmp_path / "telemetry.db")

    assert result == {
        "version": 2,
        "requests": {
            "request_count": 0,
            "success_count": 0,
            "failure_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "latency": {
                "average_latency_ms": 0.0,
                "p50_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
            },
        },
        "token_savings": None,
    }
