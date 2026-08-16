from datetime import UTC, datetime

from ..config import WorkerSettings
from ..providers.base import ProviderRequest
from ..virtual_models import ModelTier
from .capabilities import capable_tiers
from .leases import apply_route_lease
from .models import (
    GatewayRoutingSettings,
    RouteLease,
    RoutingDecision,
    RoutingMethod,
    RoutingMode,
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
        if tier_config.think is not None:
            updates["llm_think"] = tier_config.think
        updates["llm_num_parallel"] = tier_config.num_parallel
        updates["llm_api_key"] = None
        updates["llm_api_key_env"] = tier_config.api_key_env
    return worker_settings.model_copy(update=updates)


def resolve_gateway_route(
    request: ProviderRequest,
    virtual_model_id: str,
    worker_settings: WorkerSettings,
    routing_settings: GatewayRoutingSettings,
    routellm_backend: RouteLlmBackend | None = None,
    route_lease: RouteLease | None = None,
    assignment_key: str | None = None,
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
        assignment_key=route_lease.root_response_id if route_lease is not None else assignment_key,
    )
    if route_lease is not None and routing_settings.mode is not RoutingMode.LEGACY:
        actual = apply_route_lease(route_lease, plan.actual, routing_settings)
        plan = plan.model_copy(update={"actual": actual})
    return _apply_decision(worker_settings, routing_settings, plan.actual), plan


def resolve_gateway_fallback(
    worker_settings: WorkerSettings,
    routing_settings: GatewayRoutingSettings,
    request: ProviderRequest,
    failed_tier: ModelTier,
) -> tuple[WorkerSettings, RoutingPlan] | None:
    order = {
        ModelTier.LOCAL: (ModelTier.MID, ModelTier.STRONG),
        ModelTier.MID: (ModelTier.STRONG,),
        ModelTier.STRONG: (),
    }
    eligible_tiers, constraints, excluded_models = capable_tiers(request, routing_settings)
    for tier in order[failed_tier]:
        config = routing_settings.tiers.get(tier)
        if config is None or not config.enabled or tier not in eligible_tiers:
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
            capability_constraints=constraints,
            excluded_models=excluded_models,
        )
        return (
            _apply_decision(worker_settings, routing_settings, decision),
            RoutingPlan(actual=decision),
        )
    return None
