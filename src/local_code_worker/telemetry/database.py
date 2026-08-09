import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from ..routing.models import RoutingDecision, RoutingPlan
from .metrics import RequestMetrics, summarize_latencies
from .models import ModelRequestTelemetry, TokenUsage, UsageProvenance
from .savings import TokenSavings

SCHEMA_VERSION = 2


class TelemetryDatabase:
    def __init__(self, path: Path) -> None:
        self.path = path

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            applied_versions = {
                row[0] for row in connection.execute("SELECT version FROM schema_migrations")
            }
            if 1 not in applied_versions:
                self._apply_version_one(connection)
                connection.execute(
                    "INSERT INTO schema_migrations (version) VALUES (?)",
                    (1,),
                )
            if 2 not in applied_versions:
                self._apply_version_two(connection)
                connection.execute("INSERT INTO schema_migrations (version) VALUES (?)", (2,))

    def record_request(self, telemetry: ModelRequestTelemetry) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO model_requests (
                    request_id, session_id, project_id, timestamp, provider, model, tier,
                    input_tokens, output_tokens, cached_input_tokens, reasoning_tokens,
                    usage_provenance, latency_ms, time_to_first_token_ms, retry_count,
                    escalation_count, tool_count, success, failure_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    telemetry.request_id,
                    telemetry.session_id,
                    telemetry.project_id,
                    telemetry.timestamp,
                    telemetry.provider,
                    telemetry.model,
                    telemetry.tier,
                    telemetry.usage.input_tokens,
                    telemetry.usage.output_tokens,
                    telemetry.usage.cached_input_tokens,
                    telemetry.usage.reasoning_tokens,
                    telemetry.usage.provenance.value,
                    telemetry.latency_ms,
                    telemetry.time_to_first_token_ms,
                    telemetry.retry_count,
                    telemetry.escalation_count,
                    telemetry.tool_count,
                    int(telemetry.success),
                    telemetry.failure_type,
                ),
            )

    def get_request(self, request_id: str) -> ModelRequestTelemetry | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM model_requests WHERE request_id = ?",
                (request_id,),
            ).fetchone()
        if row is None:
            return None
        return ModelRequestTelemetry(
            request_id=row[0],
            session_id=row[1],
            project_id=row[2],
            timestamp=row[3],
            provider=row[4],
            model=row[5],
            tier=row[6],
            usage=TokenUsage(
                input_tokens=row[7],
                output_tokens=row[8],
                cached_input_tokens=row[9],
                reasoning_tokens=row[10],
                provenance=UsageProvenance(row[11]),
            ),
            latency_ms=row[12],
            time_to_first_token_ms=row[13],
            retry_count=row[14],
            escalation_count=row[15],
            tool_count=row[16],
            success=bool(row[17]),
            failure_type=row[18],
        )

    def record_routing_plan(self, request_id: str, plan: RoutingPlan) -> None:
        decisions = [("actual", plan.actual)]
        if plan.hypothetical is not None:
            decisions.append(("hypothetical", plan.hypothetical))
        with self.connect() as connection:
            connection.executemany(
                """
                INSERT INTO routing_decisions (
                    request_id, decision_kind, timestamp, tier, provider, model,
                    reason, confidence, method, rule_id, routellm_score,
                    routing_backend_failure, policy_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        request_id,
                        kind,
                        decision.timestamp,
                        decision.tier.value,
                        decision.provider.value,
                        decision.model,
                        decision.reason,
                        decision.confidence,
                        decision.method.value,
                        decision.rule_id,
                        decision.routellm_score,
                        int(decision.routing_backend_failure),
                        decision.policy_version,
                    )
                    for kind, decision in decisions
                ],
            )

    def get_routing_plan(self, request_id: str) -> RoutingPlan | None:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM routing_decisions WHERE request_id = ? ORDER BY decision_kind",
                (request_id,),
            ).fetchall()
        if not rows:
            return None
        decisions = {row[1]: self._routing_decision_from_row(row) for row in rows}
        return RoutingPlan(
            actual=decisions["actual"],
            hypothetical=decisions.get("hypothetical"),
        )

    def delete_requests_before(self, timestamp: str) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                "DELETE FROM model_requests WHERE timestamp < ?",
                (timestamp,),
            )
            return cursor.rowcount

    def summarize_requests(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        tier: str | None = None,
    ) -> RequestMetrics:
        clauses: list[str] = []
        parameters: list[str] = []
        if provider is not None:
            clauses.append("provider = ?")
            parameters.append(provider)
        if model is not None:
            clauses.append("model = ?")
            parameters.append(model)
        if tier is not None:
            clauses.append("tier = ?")
            parameters.append(tier)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT success, input_tokens, output_tokens, latency_ms
                FROM model_requests
                """
                + where,
                parameters,
            ).fetchall()
        success_count = sum(int(row[0]) for row in rows)
        return RequestMetrics(
            request_count=len(rows),
            success_count=success_count,
            failure_count=len(rows) - success_count,
            input_tokens=sum(int(row[1]) for row in rows),
            output_tokens=sum(int(row[2]) for row in rows),
            latency=summarize_latencies([float(row[3]) for row in rows]),
        )

    def estimate_cloud_token_savings(self, baseline_cloud_tokens: int) -> TokenSavings:
        cloud = self.summarize_requests(tier="cloud")
        return TokenSavings(
            baseline_cloud_tokens=baseline_cloud_tokens,
            actual_cloud_tokens=cloud.input_tokens + cloud.output_tokens,
        )

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        try:
            connection.execute("PRAGMA busy_timeout = 5000")
            connection.execute("PRAGMA journal_mode = WAL")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _apply_version_one(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE model_requests (
                request_id TEXT PRIMARY KEY,
                session_id TEXT,
                project_id TEXT,
                timestamp TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                tier TEXT NOT NULL,
                input_tokens INTEGER NOT NULL CHECK (input_tokens >= 0),
                output_tokens INTEGER NOT NULL CHECK (output_tokens >= 0),
                cached_input_tokens INTEGER NOT NULL CHECK (cached_input_tokens >= 0),
                reasoning_tokens INTEGER NOT NULL CHECK (reasoning_tokens >= 0),
                usage_provenance TEXT NOT NULL,
                latency_ms REAL NOT NULL CHECK (latency_ms >= 0),
                time_to_first_token_ms REAL CHECK (time_to_first_token_ms >= 0),
                retry_count INTEGER NOT NULL CHECK (retry_count >= 0),
                escalation_count INTEGER NOT NULL CHECK (escalation_count >= 0),
                tool_count INTEGER NOT NULL CHECK (tool_count >= 0),
                success INTEGER NOT NULL CHECK (success IN (0, 1)),
                failure_type TEXT
            )
            """
        )
        connection.execute(
            "CREATE INDEX model_requests_timestamp_idx ON model_requests (timestamp)"
        )

    @staticmethod
    def _apply_version_two(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE routing_decisions (
                request_id TEXT NOT NULL,
                decision_kind TEXT NOT NULL CHECK (decision_kind IN ('actual', 'hypothetical')),
                timestamp TEXT NOT NULL,
                tier TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                reason TEXT NOT NULL,
                confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
                method TEXT NOT NULL,
                rule_id TEXT,
                routellm_score REAL CHECK (routellm_score >= 0 AND routellm_score <= 1),
                routing_backend_failure INTEGER NOT NULL CHECK (
                    routing_backend_failure IN (0, 1)
                ),
                policy_version TEXT NOT NULL,
                PRIMARY KEY (request_id, decision_kind)
            )
            """
        )
        connection.execute(
            "CREATE INDEX routing_decisions_timestamp_idx ON routing_decisions (timestamp)"
        )

    @staticmethod
    def _routing_decision_from_row(row: sqlite3.Row | tuple[object, ...]) -> RoutingDecision:
        return RoutingDecision(
            timestamp=row[2],
            tier=row[3],
            provider=row[4],
            model=row[5],
            reason=row[6],
            confidence=row[7],
            method=row[8],
            rule_id=row[9],
            routellm_score=row[10],
            routing_backend_failure=bool(row[11]),
            policy_version=row[12],
        )
