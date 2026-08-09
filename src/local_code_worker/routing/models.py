from enum import StrEnum

from pydantic import Field, HttpUrl

from ..models import ProviderName, StrictModel
from ..virtual_models import ModelTier


class RoutingMode(StrEnum):
    LEGACY = "legacy"
    OBSERVE_ONLY = "observe_only"
    ROUTER = "router"


class RoutingMethod(StrEnum):
    LEGACY = "legacy"
    FORCED = "forced"
    DETERMINISTIC = "deterministic"
    ROUTELLM = "routellm"
    FALLBACK = "fallback"


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
    api_key_env: str | None = None


class GatewayRoutingSettings(StrictModel):
    mode: RoutingMode = RoutingMode.LEGACY
    tiers: dict[ModelTier, TierConfig] = Field(default_factory=dict)
    policy_version: str = Field(default="1", min_length=1)
    routellm_enabled: bool = False
    routellm_threshold: float = Field(default=0.5, ge=0, le=1)
    routellm_ambiguity_confidence: float = Field(default=0.65, ge=0, le=1)
    routellm_checkpoint_path: str | None = None


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


class RuleMatch(StrictModel):
    tier: ModelTier
    reason: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    rule_id: str = Field(min_length=1)


class RoutingPlan(StrictModel):
    actual: RoutingDecision
    hypothetical: RoutingDecision | None = None
