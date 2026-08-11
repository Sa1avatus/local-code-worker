import pytest

from local_code_worker.config import WorkerSettings
from local_code_worker.models import ProviderName
from local_code_worker.providers.base import ProviderMessage, ProviderRequest
from local_code_worker.routing.gateway import resolve_gateway_fallback, resolve_gateway_route
from local_code_worker.routing.models import (
    GatewayRoutingSettings,
    RoutingMethod,
    RoutingMode,
    TierConfig,
)
from local_code_worker.virtual_models import ModelTier


class FailingRouteLlmBackend:
    def score(self, request: ProviderRequest) -> float:
        raise RuntimeError("unavailable")


def request(content: str = "Security review") -> ProviderRequest:
    return ProviderRequest(
        messages=[ProviderMessage(role="user", content=content)],
        max_output_characters=100,
    )


def worker_settings() -> WorkerSettings:
    return WorkerSettings(
        llm_provider=ProviderName.OLLAMA,
        llm_base_url="http://localhost:11434",
        llm_model="legacy-model",
    )


def test_observe_only_keeps_legacy_execution_settings() -> None:
    worker = worker_settings()
    routing = GatewayRoutingSettings(
        mode=RoutingMode.OBSERVE_ONLY,
        tiers={
            ModelTier.STRONG: TierConfig(
                provider=ProviderName.OLLAMA,
                model="strong-model",
            )
        },
    )

    selected, plan = resolve_gateway_route(
        request(),
        "local-code-worker/auto",
        worker,
        routing,
    )

    assert selected.llm_model == "legacy-model"
    assert plan.actual.model == "legacy-model"
    assert plan.hypothetical is not None
    assert plan.hypothetical.model == "strong-model"


def test_router_applies_selected_model_for_configured_provider() -> None:
    worker = worker_settings()
    routing = GatewayRoutingSettings(
        mode=RoutingMode.ROUTER,
        tiers={
            ModelTier.STRONG: TierConfig(
                provider=ProviderName.OLLAMA,
                model="strong-model",
            )
        },
    )

    selected, plan = resolve_gateway_route(
        request(),
        "local-code-worker/auto",
        worker,
        routing,
    )

    assert selected.llm_model == "strong-model"
    assert plan.actual.model == "strong-model"


def test_router_rejects_cross_provider_route_without_instance_configuration() -> None:
    worker = worker_settings()
    routing = GatewayRoutingSettings(
        mode=RoutingMode.ROUTER,
        tiers={
            ModelTier.STRONG: TierConfig(
                provider=ProviderName.OPENAI_COMPATIBLE,
                model="strong-model",
            )
        },
    )

    with pytest.raises(ValueError, match="provider instance configuration"):
        resolve_gateway_route(
            request(),
            "local-code-worker/auto",
            worker,
            routing,
        )


def test_router_applies_complete_cross_provider_tier_configuration() -> None:
    worker = worker_settings()
    routing = GatewayRoutingSettings(
        mode=RoutingMode.ROUTER,
        tiers={
            ModelTier.STRONG: TierConfig(
                provider=ProviderName.OPENAI_COMPATIBLE,
                base_url="https://cloud.example/v1",
                model="strong-model",
                context_length=32_768,
                api_key_env="STRONG_API_KEY",
            )
        },
    )

    selected, _ = resolve_gateway_route(
        request(),
        "local-code-worker/auto",
        worker,
        routing,
    )

    assert selected.llm_provider is ProviderName.OPENAI_COMPATIBLE
    assert str(selected.llm_base_url) == "https://cloud.example/v1"
    assert selected.llm_model == "strong-model"
    assert selected.llm_num_ctx == 32_768
    assert selected.llm_api_key_env == "STRONG_API_KEY"


def test_gateway_uses_deterministic_route_when_routellm_backend_fails() -> None:
    worker = worker_settings()
    routing = GatewayRoutingSettings(
        mode=RoutingMode.ROUTER,
        routellm_enabled=True,
        tiers={
            ModelTier.LOCAL: TierConfig(
                provider=ProviderName.OLLAMA,
                model="local-model",
            )
        },
    )

    selected, plan = resolve_gateway_route(
        request("Implement the requested change"),
        "local-code-worker/auto",
        worker,
        routing,
        FailingRouteLlmBackend(),
    )

    assert selected.llm_model == "local-model"
    assert plan.actual.routing_backend_failure is True


def test_auto_defers_cloud_strong_until_local_model_fails() -> None:
    routing = GatewayRoutingSettings(
        mode=RoutingMode.ROUTER,
        tiers={
            ModelTier.MID: TierConfig(provider=ProviderName.OLLAMA, model="local-reasoner"),
            ModelTier.STRONG: TierConfig(
                provider=ProviderName.OPENAI_COMPATIBLE,
                base_url="https://cloud.example/v1",
                model="cloud-strong",
            ),
        },
    )

    selected, plan = resolve_gateway_route(
        request("Perform a security architecture review across multiple services"),
        "local-code-worker/auto",
        worker_settings(),
        routing,
    )

    assert selected.llm_model == "local-reasoner"
    assert plan.actual.tier is ModelTier.MID
    assert "reserved for fallback" in plan.actual.reason


def test_runtime_failure_advances_from_local_to_mid_then_strong() -> None:
    routing = GatewayRoutingSettings(
        mode=RoutingMode.ROUTER,
        tiers={
            ModelTier.LOCAL: TierConfig(provider=ProviderName.OLLAMA, model="local"),
            ModelTier.MID: TierConfig(provider=ProviderName.OLLAMA, model="mid"),
            ModelTier.STRONG: TierConfig(
                provider=ProviderName.OPENAI_COMPATIBLE,
                base_url="https://cloud.example/v1",
                model="strong",
            ),
        },
    )

    mid = resolve_gateway_fallback(worker_settings(), routing, request(), ModelTier.LOCAL)
    assert mid is not None
    mid_settings, mid_plan = mid
    strong = resolve_gateway_fallback(worker_settings(), routing, request(), ModelTier.MID)
    assert strong is not None
    strong_settings, strong_plan = strong

    assert mid_settings.llm_model == "mid"
    assert mid_plan.actual.method is RoutingMethod.FALLBACK
    assert strong_settings.llm_model == "strong"
    assert strong_plan.actual.provider is ProviderName.OPENAI_COMPATIBLE
    assert resolve_gateway_fallback(worker_settings(), routing, request(), ModelTier.STRONG) is None
