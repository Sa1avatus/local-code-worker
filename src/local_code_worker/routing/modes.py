from datetime import UTC, datetime
from hashlib import sha256

from ..providers.base import ProviderRequest
from ..virtual_models import ModelTier
from .engine import Clock, route_request
from .models import (
    GatewayRoutingSettings,
    RoutingDecision,
    RoutingMethod,
    RoutingMode,
    RoutingPlan,
    TierConfig,
)
from .routellm_adapter import RouteLlmBackend


def _legacy_decision(
    config: TierConfig,
    tier: ModelTier,
    policy_version: str,
    clock: Clock,
) -> RoutingDecision:
    return RoutingDecision(
        tier=tier,
        provider=config.provider,
        model=config.model,
        reason="Legacy mode preserves the configured provider and model.",
        confidence=1.0,
        method=RoutingMethod.LEGACY,
        rule_id="legacy-provider",
        timestamp=clock().astimezone(UTC).isoformat(),
        policy_version=policy_version,
    )


def plan_routing(
    request: ProviderRequest,
    virtual_model_id: str,
    settings: GatewayRoutingSettings,
    legacy_config: TierConfig,
    *,
    legacy_tier: ModelTier = ModelTier.LOCAL,
    has_previous_failures: bool = False,
    routellm_backend: RouteLlmBackend | None = None,
    assignment_key: str | None = None,
    clock: Clock = lambda: datetime.now(UTC),
) -> RoutingPlan:
    legacy = _legacy_decision(legacy_config, legacy_tier, settings.policy_version, clock)
    if settings.mode is RoutingMode.LEGACY:
        return RoutingPlan(actual=legacy)

    routed = route_request(
        request,
        virtual_model_id,
        settings,
        has_previous_failures=has_previous_failures,
        routellm_backend=routellm_backend,
        clock=clock,
    )
    if settings.mode in {RoutingMode.OBSERVE_ONLY, RoutingMode.SHADOW}:
        return RoutingPlan(actual=legacy, hypothetical=routed)
    if settings.mode is RoutingMode.CANARY:
        key = assignment_key or "unassigned"
        bucket = int.from_bytes(sha256(key.encode()).digest()[:4], "big") % 100
        if bucket >= settings.canary_percent:
            return RoutingPlan(actual=legacy, hypothetical=routed)
    return RoutingPlan(actual=routed)
