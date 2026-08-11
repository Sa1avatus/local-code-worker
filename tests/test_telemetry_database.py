import sqlite3
from concurrent.futures import ThreadPoolExecutor

from local_code_worker.models import ProviderName
from local_code_worker.routing.models import (
    EscalationEvent,
    EscalationReason,
    RouteLease,
    RoutingDecision,
    RoutingMethod,
    RoutingPlan,
)
from local_code_worker.telemetry.database import SCHEMA_VERSION, TelemetryDatabase
from local_code_worker.telemetry.models import (
    ModelRequestTelemetry,
    TokenUsage,
    UsageProvenance,
)
from local_code_worker.telemetry.savings import BaselineMethod
from local_code_worker.virtual_models import ModelTier


def test_initialize_creates_versioned_telemetry_schema(tmp_path) -> None:
    database_path = tmp_path / "state" / "telemetry.db"
    database = TelemetryDatabase(database_path)

    database.initialize()

    with sqlite3.connect(database_path) as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        version = connection.execute("SELECT version FROM schema_migrations").fetchone()
        columns = {row[1] for row in connection.execute("PRAGMA table_info(model_requests)")}

    assert tables >= {"schema_migrations", "model_requests", "routing_decisions"}
    assert version == (1,)
    with sqlite3.connect(database_path) as connection:
        latest_version = connection.execute("SELECT MAX(version) FROM schema_migrations").fetchone()
    assert latest_version == (SCHEMA_VERSION,)
    assert columns >= {
        "request_id",
        "provider",
        "model",
        "input_tokens",
        "output_tokens",
        "usage_provenance",
        "latency_ms",
        "success",
    }


def test_initialize_is_idempotent(tmp_path) -> None:
    database = TelemetryDatabase(tmp_path / "telemetry.db")

    database.initialize()
    database.initialize()

    with database.connect() as connection:
        versions = connection.execute("SELECT version FROM schema_migrations").fetchall()

    assert versions == [(1,), (2,), (SCHEMA_VERSION,)]


def test_initialize_upgrades_existing_version_one_database(tmp_path) -> None:
    database = TelemetryDatabase(tmp_path / "telemetry.db")
    with database.connect() as connection:
        connection.execute(
            "CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT)"
        )
        database._apply_version_one(connection)
        connection.execute(
            "INSERT INTO schema_migrations (version, applied_at) VALUES (1, '2026-08-09')"
        )

    database.initialize()

    with database.connect() as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        versions = connection.execute("SELECT version FROM schema_migrations").fetchall()
    assert "routing_decisions" in tables
    assert versions == [(1,), (2,), (3,)]


def test_record_routing_plan_keeps_actual_and_hypothetical_separate(tmp_path) -> None:
    database = TelemetryDatabase(tmp_path / "telemetry.db")
    database.initialize()

    def decision(model: str, method: RoutingMethod) -> RoutingDecision:
        return RoutingDecision(
            tier=ModelTier.LOCAL,
            provider=ProviderName.OLLAMA,
            model=model,
            reason="Safe routing reason",
            confidence=0.8,
            method=method,
            timestamp="2026-08-09T12:00:00+00:00",
            policy_version="test-v1",
        )

    plan = RoutingPlan(
        actual=decision("legacy-model", RoutingMethod.LEGACY),
        hypothetical=decision("routed-model", RoutingMethod.DETERMINISTIC),
    )

    database.record_routing_plan("request-1", plan)

    assert database.get_routing_plan("request-1") == plan
    assert database.get_routing_plan("missing") is None


def test_route_lease_and_escalation_metrics_are_persisted(tmp_path) -> None:
    database = TelemetryDatabase(tmp_path / "telemetry.db")
    database.initialize()
    lease = RouteLease(
        lease_id="lease-1",
        root_response_id="resp-1",
        current_route=ModelTier.MID,
        current_model="mid-model",
        created_at="2026-08-11T00:00:00+00:00",
        updated_at="2026-08-11T00:00:01+00:00",
        escalation_count=1,
        escalation_reason=EscalationReason.PROVIDER_ERROR,
    )
    event = EscalationEvent(
        from_route=ModelTier.LOCAL,
        to_route=ModelTier.MID,
        from_model="local-model",
        to_model="mid-model",
        reason=EscalationReason.PROVIDER_ERROR,
        request_id="req-1",
        response_id="resp-1",
        lease_id="lease-1",
        timestamp="2026-08-11T00:00:01+00:00",
    )

    database.record_route_lease(lease)
    database.record_escalation(event)

    metrics = database.summarize_routing()
    assert metrics["route_lease_created"] == 1
    assert metrics["escalations_total"] == 1
    assert metrics["escalations_by_reason"] == {"provider_error": 1}


def test_routing_schema_excludes_request_payloads_and_credentials(tmp_path) -> None:
    database = TelemetryDatabase(tmp_path / "telemetry.db")
    database.initialize()

    with database.connect() as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(routing_decisions)")}

    assert columns.isdisjoint(
        {"prompt", "content", "path", "api_key", "token", "cookie", "response_body"}
    )


def test_record_request_round_trips_typed_telemetry(tmp_path) -> None:
    database = TelemetryDatabase(tmp_path / "telemetry.db")
    database.initialize()
    telemetry = ModelRequestTelemetry(
        request_id="request-1",
        session_id="session-1",
        project_id="project-1",
        timestamp="2026-08-09T12:00:00Z",
        provider="ollama",
        model="qwen",
        tier="local",
        usage=TokenUsage(
            input_tokens=120,
            output_tokens=30,
            cached_input_tokens=10,
            reasoning_tokens=5,
            provenance=UsageProvenance.EXACT,
        ),
        latency_ms=42.5,
        time_to_first_token_ms=8.5,
        retry_count=1,
        escalation_count=0,
        tool_count=2,
        success=True,
    )

    database.record_request(telemetry)

    assert database.get_request("request-1") == telemetry
    assert database.get_request("missing") is None


def test_concurrent_request_writes_are_not_lost(tmp_path) -> None:
    database = TelemetryDatabase(tmp_path / "telemetry.db")
    database.initialize()

    def record(index: int) -> None:
        database.record_request(
            ModelRequestTelemetry(
                request_id=f"request-{index}",
                timestamp="2026-08-09T12:00:00Z",
                provider="ollama",
                model="qwen",
                tier="local",
                latency_ms=float(index),
                success=True,
            )
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(record, range(32)))

    with database.connect() as connection:
        count = connection.execute("SELECT COUNT(*) FROM model_requests").fetchone()

    assert count == (32,)


def test_retention_deletes_only_requests_before_cutoff(tmp_path) -> None:
    database = TelemetryDatabase(tmp_path / "telemetry.db")
    database.initialize()
    for request_id, timestamp in (
        ("old", "2026-08-01T00:00:00Z"),
        ("cutoff", "2026-08-05T00:00:00Z"),
        ("new", "2026-08-09T00:00:00Z"),
    ):
        database.record_request(
            ModelRequestTelemetry(
                request_id=request_id,
                timestamp=timestamp,
                provider="ollama",
                model="qwen",
                tier="local",
                latency_ms=1,
                success=True,
            )
        )

    deleted = database.delete_requests_before("2026-08-05T00:00:00Z")

    assert deleted == 1
    assert database.get_request("old") is None
    assert database.get_request("cutoff") is not None
    assert database.get_request("new") is not None


def test_model_requests_schema_excludes_sensitive_payload_fields(tmp_path) -> None:
    database = TelemetryDatabase(tmp_path / "telemetry.db")
    database.initialize()

    with database.connect() as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(model_requests)")}

    assert columns.isdisjoint(
        {"prompt", "content", "path", "api_key", "token", "cookie", "response_body"}
    )


def test_summarize_requests_aggregates_and_filters_by_provider(tmp_path) -> None:
    database = TelemetryDatabase(tmp_path / "telemetry.db")
    database.initialize()
    for request_id, provider, success, latency, input_tokens, output_tokens in (
        ("local-ok", "ollama", True, 10, 4, 2),
        ("local-failed", "ollama", False, 30, 6, 3),
        ("cloud-ok", "openai-compatible", True, 100, 8, 4),
    ):
        database.record_request(
            ModelRequestTelemetry(
                request_id=request_id,
                timestamp="2026-08-09T00:00:00Z",
                provider=provider,
                model="test-model",
                tier="local" if provider == "ollama" else "cloud",
                usage=TokenUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    provenance=UsageProvenance.EXACT,
                ),
                latency_ms=latency,
                success=success,
            )
        )

    summary = database.summarize_requests(provider="ollama")

    assert summary.request_count == 2
    assert summary.success_count == 1
    assert summary.failure_count == 1
    assert summary.input_tokens == 10
    assert summary.output_tokens == 5
    assert summary.latency.average_latency_ms == 20
    assert summary.latency.p50_latency_ms == 10
    assert summary.latency.p95_latency_ms == 30

    savings = database.estimate_cloud_token_savings(baseline_cloud_tokens=20)
    assert savings.baseline_method is BaselineMethod.EXPLICIT_CLOUD_TOKEN_BUDGET
    assert savings.actual_cloud_tokens == 12
    assert savings.cloud_tokens_saved == 8
    assert savings.estimated is True
