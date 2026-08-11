from datetime import UTC, datetime

from ..config import WorkerSettings
from ..providers.base import ProviderRequest
from ..virtual_models import ModelTier
from .models import (
    GatewayRoutingSettings,
    RoutingDecision,
    RoutingMethod,
    RoutingPlan,
    TierConfig,
)
from .modes import plan_routing
from .routellm_adapter import RouteLlmBackend


def _apply_decision(
    worker_settings: WorkerSettings,
    routing_settings: GatewayRoutingSettings,
    decision: RoutingDecision,
) -> WorkerSettings:
    tier_config = routing_settings.tiers.get(decision.tier)
    if decision.provider is not worker_settings.llm_provider and (
        tier_config is None or tier_config.base_url is None
    ):
        raise ValueError(
            "The selected tier uses a different provider, but provider instance configuration "
            "is not available yet"
        )
    updates: dict[str, object] = {
        "llm_provider": decision.provider,
        "llm_model": decision.model,
    }
    if tier_config is not None:
        if tier_config.base_url is not None:
            updates["llm_base_url"] = tier_config.base_url
        if tier_config.context_length is not None:
            updates["llm_num_ctx"] = tier_config.context_length
        updates["llm_api_key"] = None
        updates["llm_api_key_env"] = tier_config.api_key_env
    return worker_settings.model_copy(update=updates)


def resolve_gateway_route(
    request: ProviderRequest,
    virtual_model_id: str,
    worker_settings: WorkerSettings,
    routing_settings: GatewayRoutingSettings,
    routellm_backend: RouteLlmBackend | None = None,
) -> tuple[WorkerSettings, RoutingPlan]:
    plan = plan_routing(
        request,
        virtual_model_id,
        routing_settings,
        TierConfig(
            provider=worker_settings.llm_provider,
            model=worker_settings.llm_model,
        ),
        legacy_tier=ModelTier.LOCAL,
        routellm_backend=routellm_backend,
    )
    return _apply_decision(worker_settings, routing_settings, plan.actual), plan


def resolve_gateway_fallback(
    worker_settings: WorkerSettings,
    routing_settings: GatewayRoutingSettings,
    failed_tier: ModelTier,
) -> tuple[WorkerSettings, RoutingPlan] | None:
    order = {
        ModelTier.LOCAL: (ModelTier.MID, ModelTier.STRONG),
        ModelTier.MID: (ModelTier.STRONG,),
        ModelTier.STRONG: (),
    }
    for tier in order[failed_tier]:
        config = routing_settings.tiers.get(tier)
        if config is None or not config.enabled:
            continue
        decision = RoutingDecision(
            tier=tier,
            provider=config.provider,
            model=config.model,
            reason=f"The {failed_tier.value} tier failed at runtime; trying {tier.value}.",
            confidence=1.0,
            method=RoutingMethod.FALLBACK,
            rule_id="runtime-fallback",
            timestamp=datetime.now(UTC).isoformat(),
            policy_version=routing_settings.policy_version,
        )
        return (
            _apply_decision(worker_settings, routing_settings, decision),
            RoutingPlan(actual=decision),
        )
    return None
