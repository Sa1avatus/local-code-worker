from local_code_worker.telemetry.models import (
    ModelRequestTelemetry,
    TokenUsage,
    UsageProvenance,
)


def test_token_usage_preserves_exact_provider_counts_and_total() -> None:
    token_usage = TokenUsage(
        input_tokens=10,
        output_tokens=6,
        cached_input_tokens=3,
        reasoning_tokens=2,
        provenance="exact",
    )

    assert token_usage.provenance is UsageProvenance.EXACT
    assert token_usage.input_tokens == 10
    assert token_usage.output_tokens == 6
    assert token_usage.cached_input_tokens == 3
    assert token_usage.reasoning_tokens == 2
    assert token_usage.total_tokens == 16


def test_model_request_telemetry_preserves_safe_request_metrics() -> None:
    telemetry = ModelRequestTelemetry(
        request_id="req-123",
        session_id="session-456",
        project_id="project-789",
        timestamp="2026-08-09T04:20:00+00:00",
        provider="ollama",
        model="qwen:test",
        tier="local",
        usage=TokenUsage(
            input_tokens=20,
            output_tokens=8,
            provenance=UsageProvenance.EXACT,
        ),
        latency_ms=125.5,
        time_to_first_token_ms=42.0,
        retry_count=1,
        escalation_count=0,
        tool_count=2,
        success=True,
    )

    assert telemetry.request_id == "req-123"
    assert telemetry.usage.total_tokens == 28
    assert telemetry.time_to_first_token_ms == 42.0
    assert telemetry.retry_count == 1
    assert telemetry.tool_count == 2
    assert telemetry.success is True
    assert telemetry.failure_type is None
