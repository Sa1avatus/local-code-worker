from enum import StrEnum

from pydantic import Field, HttpUrl, model_validator

from ..models import ProviderName, StrictModel
from ..virtual_models import ModelTier


class RoutingMode(StrEnum):
    LEGACY = "legacy"
    OBSERVE_ONLY = "observe_only"
    ROUTER = "router"
    SHADOW = "shadow"
    CANARY = "canary"
    ROUTE_LLM = "route_llm"


class RoutingMethod(StrEnum):
    LEGACY = "legacy"
    FORCED = "forced"
    DETERMINISTIC = "deterministic"
    ROUTELLM = "routellm"
    FALLBACK = "fallback"
    LEASE = "lease"


class EscalationReason(StrEnum):
    CAPABILITY_MISMATCH = "capability_mismatch"
    CONTEXT_OVERFLOW = "context_overflow"
    PROVIDER_ERROR = "provider_error"
    TIMEOUT = "timeout"
    INVALID_OUTPUT = "invalid_output"
    TOOL_CALL_FAILURE = "tool_call_failure"
    STRUCTURED_OUTPUT_FAILURE = "structured_output_failure"
    QUALITY_GUARD_FAILURE = "quality_guard_failure"
    REPEATED_FAILURE = "repeated_failure"
    MANUAL_POLICY = "manual_policy"


class TaskCategory(StrEnum):
    SIMPLE_EDIT = "simple_edit"
    TESTS = "tests"
    DOCUMENTATION = "documentation"
    REFACTOR = "refactor"
    DEBUGGING = "debugging"
    MIGRATION = "migration"
    SECURITY = "security"
    ARCHITECTURE = "architecture"
    GENERAL = "general"


class TaskFeatures(StrictModel):
    category: TaskCategory
    context_characters: int = Field(ge=0)
    message_count: int = Field(ge=0)
    tool_count: int = Field(ge=0)
    has_reasoning: bool
    has_previous_failures: bool = False
    multi_service: bool = False
    estimated_scope: int = Field(ge=1)


class TierConfig(StrictModel):
    provider: ProviderName
    model: str = Field(min_length=1)
    enabled: bool = True
    base_url: HttpUrl | None = None
    context_length: int | None = Field(default=None, ge=512, le=131_072)
    # Parallel request slots for this tier's Ollama runner. Ollama sizes the
    # runner context as num_ctx * num_parallel, so large models need 1 while
    # small matching models can use several slots. Default 1.
    num_parallel: int = Field(default=1, ge=1, le=64)
    # Reasoning "think" toggle for Ollama thinking models (qwen3.x). None = do
    # not send the parameter, leaving the model's default; False disables
    # thinking to save the token budget, True forces it.
    think: bool | None = None
    # Whether to surface the model's reasoning trace to the client. None or True
    # forward it, False hides it (the model still thinks internally).
    show_reasoning: bool | None = None
    # Optional output budget for the routing capability check; defaults to a
    # generous ceiling so client budgets (e.g. matching extraction, 16k tokens)
    # are not rejected because the tier hardcoded a 4096-token capability.
    max_output_tokens: int | None = Field(default=None, ge=1, le=131_072)
    api_key_env: str | None = None
    capabilities: "ModelCapabilities | None" = None


class ModelCapabilities(StrictModel):
    model_id: str = Field(min_length=1)
    tier: ModelTier
    context_window: int = Field(ge=512)
    supports_tools: bool
    supports_structured_output: bool
    supports_json_schema: bool
    supports_streaming: bool
    max_output_tokens: int = Field(gt=0)


class RequestCapabilities(StrictModel):
    requires_tools: bool
    requires_function_calling: bool
    requires_structured_output: bool
    requires_json_schema: bool
    context_size: int = Field(ge=0)
    estimated_input_tokens: int = Field(ge=0)
    estimated_output_tokens: int = Field(ge=0)
    multi_file_task: bool
    large_repository_context: bool
    reasoning_complexity: int = Field(ge=0, le=3)
    diff_complexity: int = Field(ge=0, le=3)


class RouteLease(StrictModel):
    lease_id: str = Field(min_length=1)
    root_response_id: str = Field(min_length=1)
    current_route: ModelTier
    current_model: str = Field(min_length=1)
    created_at: str = Field(min_length=1)
    updated_at: str = Field(min_length=1)
    escalation_count: int = Field(default=0, ge=0)
    escalation_reason: EscalationReason | None = None


class EscalationEvent(StrictModel):
    from_route: ModelTier
    to_route: ModelTier
    from_model: str = Field(min_length=1)
    to_model: str = Field(min_length=1)
    reason: EscalationReason
    request_id: str = Field(min_length=1)
    response_id: str = Field(min_length=1)
    lease_id: str = Field(min_length=1)
    timestamp: str = Field(min_length=1)


class GatewayRoutingSettings(StrictModel):
    mode: RoutingMode = RoutingMode.LEGACY
    tiers: dict[ModelTier, TierConfig] = Field(default_factory=dict)
    policy_version: str = Field(default="1", min_length=1)
    routellm_enabled: bool = False
    routellm_threshold: float = Field(default=0.5, ge=0, le=1)
    routellm_ambiguity_confidence: float = Field(default=0.65, ge=0, le=1)
    routellm_checkpoint_path: str | None = None
    local_threshold: float = Field(default=0.3, ge=0, le=1)
    strong_threshold: float = Field(default=0.7, ge=0, le=1)
    canary_percent: int = Field(default=10, ge=0, le=100)
    max_escalations_per_lease: int = Field(default=2, ge=0, le=10)

    @model_validator(mode="after")
    def validate_threshold_order(self) -> "GatewayRoutingSettings":
        if self.local_threshold > self.strong_threshold:
            raise ValueError("local_threshold must not exceed strong_threshold")
        return self


class RoutingDecision(StrictModel):
    tier: ModelTier
    provider: ProviderName
    model: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    method: RoutingMethod
    rule_id: str | None = None
    routellm_score: float | None = Field(default=None, ge=0, le=1)
    routing_backend_failure: bool = False
    timestamp: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    lease_id: str | None = None
    capability_constraints: tuple[str, ...] = ()
    excluded_models: tuple[str, ...] = ()


class RuleMatch(StrictModel):
    tier: ModelTier
    reason: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    rule_id: str = Field(min_length=1)


class RoutingPlan(StrictModel):
    actual: RoutingDecision
    hypothetical: RoutingDecision | None = None
