from ..config import WorkerSettings
from ..providers.base import ProviderRequest
from ..virtual_models import ModelTier
from .models import GatewayRoutingSettings, RoutingPlan, TierConfig
from .modes import plan_routing
from .routellm_adapter import RouteLlmBackend


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
    tier_config = routing_settings.tiers.get(plan.actual.tier)
    if plan.actual.provider is not worker_settings.llm_provider and (
        tier_config is None or tier_config.base_url is None
    ):
        raise ValueError(
            "The selected tier uses a different provider, but provider instance configuration "
            "is not available yet"
        )
    updates: dict[str, object] = {
        "llm_provider": plan.actual.provider,
        "llm_model": plan.actual.model,
    }
    if tier_config is not None:
        if tier_config.base_url is not None:
            updates["llm_base_url"] = tier_config.base_url
        if tier_config.context_length is not None:
            updates["llm_num_ctx"] = tier_config.context_length
        updates["llm_api_key"] = None
        updates["llm_api_key_env"] = tier_config.api_key_env
    return (
        worker_settings.model_copy(update=updates),
        plan,
    )
