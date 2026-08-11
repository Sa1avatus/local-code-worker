from collections.abc import Callable
from datetime import UTC, datetime

from ..providers.base import ProviderRequest
from ..virtual_models import VIRTUAL_MODEL_REGISTRY, ModelTier
from .analyzer import analyze_request
from .models import (
    GatewayRoutingSettings,
    RoutingDecision,
    RoutingMethod,
    TierConfig,
)
from .policy import match_deterministic_rule
from .routellm_adapter import RouteLlmBackend, validated_score

Clock = Callable[[], datetime]

_FALLBACK_ORDER: dict[ModelTier, tuple[ModelTier, ...]] = {
    ModelTier.LOCAL: (ModelTier.LOCAL, ModelTier.MID, ModelTier.STRONG),
    ModelTier.MID: (ModelTier.MID, ModelTier.LOCAL, ModelTier.STRONG),
    ModelTier.STRONG: (ModelTier.STRONG, ModelTier.MID, ModelTier.LOCAL),
}


def _resolve_available_tier(
    requested_tier: ModelTier,
    settings: GatewayRoutingSettings,
) -> tuple[ModelTier, TierConfig]:
    for tier in _FALLBACK_ORDER[requested_tier]:
        config = settings.tiers.get(tier)
        if config is not None and config.enabled:
            return tier, config
    raise ValueError("routing requires at least one enabled tier")


def route_request(
    request: ProviderRequest,
    virtual_model_id: str,
    settings: GatewayRoutingSettings,
    *,
    has_previous_failures: bool = False,
    routellm_backend: RouteLlmBackend | None = None,
    clock: Clock = lambda: datetime.now(UTC),
) -> RoutingDecision:
    virtual_model = VIRTUAL_MODEL_REGISTRY.resolve(virtual_model_id)
    if virtual_model.forced_tier is not None:
        requested_tier = virtual_model.forced_tier
        method = RoutingMethod.FORCED
        confidence = 1.0
        rule_id = "forced-tier"
        reason = f"Virtual model explicitly requested the {requested_tier.value} tier."
    else:
        match = match_deterministic_rule(
            analyze_request(request, has_previous_failures=has_previous_failures)
        )
        requested_tier = match.tier
        method = RoutingMethod.DETERMINISTIC
        confidence = match.confidence
        rule_id = match.rule_id
        reason = match.reason
        routellm_score = None
        routing_backend_failure = False
        if (
            settings.routellm_enabled
            and routellm_backend is not None
            and match.confidence <= settings.routellm_ambiguity_confidence
        ):
            try:
                routellm_score = validated_score(routellm_backend, request)
            except Exception:
                routing_backend_failure = True
            else:
                requested_tier = (
                    ModelTier.STRONG
                    if routellm_score >= settings.routellm_threshold
                    else ModelTier.LOCAL
                )
                method = RoutingMethod.ROUTELLM
                confidence = min(
                    1.0,
                    abs(routellm_score - settings.routellm_threshold) * 2,
                )
                rule_id = "routellm-weak-strong"
                reason = "RouteLLM resolved an ambiguous request using weak/strong routing."
    if virtual_model.forced_tier is not None:
        routellm_score = None
        routing_backend_failure = False

    strong = settings.tiers.get(ModelTier.STRONG)
    if (
        virtual_model.forced_tier is None
        and requested_tier is ModelTier.STRONG
        and strong is not None
        and strong.provider.value != "ollama"
    ):
        for local_tier in (ModelTier.MID, ModelTier.LOCAL):
            local_config = settings.tiers.get(local_tier)
            if (
                local_config is not None
                and local_config.enabled
                and local_config.provider.value == "ollama"
            ):
                requested_tier = local_tier
                reason = f"{reason} Cloud STRONG is reserved for fallback after local failure."
                break

    selected_tier, config = _resolve_available_tier(requested_tier, settings)
    if selected_tier is not requested_tier:
        method = RoutingMethod.FALLBACK
        reason = (
            f"{reason} The {requested_tier.value} tier is unavailable; using {selected_tier.value}."
        )

    return RoutingDecision(
        tier=selected_tier,
        provider=config.provider,
        model=config.model,
        reason=reason,
        confidence=confidence,
        method=method,
        rule_id=rule_id,
        routellm_score=routellm_score,
        routing_backend_failure=routing_backend_failure,
        timestamp=clock().astimezone(UTC).isoformat(),
        policy_version=settings.policy_version,
    )
