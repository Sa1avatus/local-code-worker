from datetime import UTC, datetime

from local_code_worker.models import ProviderName
from local_code_worker.providers.base import ProviderMessage, ProviderRequest
from local_code_worker.routing.models import (
    GatewayRoutingSettings,
    RoutingMethod,
    RoutingMode,
    TierConfig,
)
from local_code_worker.routing.modes import plan_routing
from local_code_worker.virtual_models import ModelTier

NOW = datetime(2026, 8, 9, 12, 0, tzinfo=UTC)
LEGACY = TierConfig(provider=ProviderName.OLLAMA, model="legacy-model")


def request() -> ProviderRequest:
    return ProviderRequest(
        messages=[ProviderMessage(role="user", content="Security review")],
        max_output_characters=100,
    )


def settings(mode: RoutingMode) -> GatewayRoutingSettings:
    return GatewayRoutingSettings(
        mode=mode,
        tiers={
            ModelTier.STRONG: TierConfig(
                provider=ProviderName.OPENAI_COMPATIBLE,
                model="strong-model",
            )
        },
    )


def test_legacy_mode_does_not_compute_or_apply_routed_provider() -> None:
    plan = plan_routing(
        request(),
        "local-code-worker/auto",
        settings(RoutingMode.LEGACY),
        LEGACY,
        clock=lambda: NOW,
    )

    assert plan.actual.model == "legacy-model"
    assert plan.actual.method is RoutingMethod.LEGACY
    assert plan.hypothetical is None


def test_observe_only_preserves_actual_and_exposes_hypothetical_decision() -> None:
    plan = plan_routing(
        request(),
        "local-code-worker/auto",
        settings(RoutingMode.OBSERVE_ONLY),
        LEGACY,
        clock=lambda: NOW,
    )

    assert plan.actual.model == "legacy-model"
    assert plan.hypothetical is not None
    assert plan.hypothetical.model == "strong-model"
    assert plan.hypothetical.method is RoutingMethod.DETERMINISTIC


def test_router_mode_applies_routed_provider() -> None:
    plan = plan_routing(
        request(),
        "local-code-worker/auto",
        settings(RoutingMode.ROUTER),
        LEGACY,
        clock=lambda: NOW,
    )

    assert plan.actual.model == "strong-model"
    assert plan.actual.method is RoutingMethod.DETERMINISTIC
    assert plan.hypothetical is None
