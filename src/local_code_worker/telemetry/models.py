from enum import StrEnum

from pydantic import Field

from ..models import StrictModel


class UsageProvenance(StrEnum):
    EXACT = "exact"
    ESTIMATED = "estimated"
    UNAVAILABLE = "unavailable"


class TokenUsage(StrictModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    provenance: UsageProvenance = UsageProvenance.UNAVAILABLE

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ModelRequestTelemetry(StrictModel):
    request_id: str = Field(min_length=1)
    session_id: str | None = None
    project_id: str | None = None
    timestamp: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    tier: str = Field(min_length=1)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: float = Field(ge=0)
    time_to_first_token_ms: float | None = Field(default=None, ge=0)
    retry_count: int = Field(default=0, ge=0)
    escalation_count: int = Field(default=0, ge=0)
    tool_count: int = Field(default=0, ge=0)
    success: bool
    failure_type: str | None = None
