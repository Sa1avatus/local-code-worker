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


def test_shadow_mode_never_changes_actual_provider() -> None:
    plan = plan_routing(
        request(),
        "local-code-worker/auto",
        settings(RoutingMode.SHADOW),
        LEGACY,
        clock=lambda: NOW,
    )

    assert plan.actual.model == "legacy-model"
    assert plan.hypothetical is not None
    assert plan.hypothetical.model == "strong-model"


def test_canary_assignment_is_stable_for_assignment_key() -> None:
    canary = settings(RoutingMode.CANARY).model_copy(update={"canary_percent": 50})

    first = plan_routing(
        request(),
        "local-code-worker/auto",
        canary,
        LEGACY,
        assignment_key="lease-stable",
        clock=lambda: NOW,
    )
    second = plan_routing(
        request(),
        "local-code-worker/auto",
        canary,
        LEGACY,
        assignment_key="lease-stable",
        clock=lambda: NOW,
    )

    assert first == second


def test_canary_zero_percent_always_uses_legacy() -> None:
    canary = settings(RoutingMode.CANARY).model_copy(update={"canary_percent": 0})

    plan = plan_routing(
        request(),
        "local-code-worker/auto",
        canary,
        LEGACY,
        assignment_key="lease-1",
        clock=lambda: NOW,
    )

    assert plan.actual.model == "legacy-model"
    assert plan.hypothetical is not None
