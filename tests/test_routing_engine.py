from datetime import UTC, datetime

import pytest

from local_code_worker.models import ProviderName
from local_code_worker.providers.base import ProviderMessage, ProviderRequest
from local_code_worker.routing.engine import route_request
from local_code_worker.routing.models import (
    GatewayRoutingSettings,
    RoutingMethod,
    TierConfig,
)
from local_code_worker.virtual_models import ModelTier

NOW = datetime(2026, 8, 9, 12, 0, tzinfo=UTC)


def request(content: str) -> ProviderRequest:
    return ProviderRequest(
        messages=[ProviderMessage(role="user", content=content)],
        max_output_characters=100,
    )


def settings(*enabled_tiers: ModelTier) -> GatewayRoutingSettings:
    return GatewayRoutingSettings(
        tiers={
            tier: TierConfig(
                provider=ProviderName.OLLAMA,
                model=f"model-{tier.value}",
            )
            for tier in enabled_tiers
        },
        policy_version="test-v1",
    )


def test_auto_model_builds_explainable_deterministic_decision() -> None:
    decision = route_request(
        request("Fix the security vulnerability"),
        "local-code-worker/auto",
        settings(ModelTier.LOCAL, ModelTier.MID, ModelTier.STRONG),
        clock=lambda: NOW,
    )

    assert decision.tier is ModelTier.STRONG
    assert decision.model == "model-strong"
    assert decision.method is RoutingMethod.DETERMINISTIC
    assert decision.rule_id == "safety-critical"
    assert decision.timestamp == "2026-08-09T12:00:00+00:00"
    assert decision.policy_version == "test-v1"


def test_forced_virtual_model_bypasses_analysis() -> None:
    decision = route_request(
        request("Security architecture migration"),
        "local-code-worker/local",
        settings(ModelTier.LOCAL, ModelTier.STRONG),
        clock=lambda: NOW,
    )

    assert decision.tier is ModelTier.LOCAL
    assert decision.method is RoutingMethod.FORCED
    assert decision.confidence == 1.0


def test_unavailable_tier_uses_ordered_fallback() -> None:
    decision = route_request(
        request("Security review"),
        "local-code-worker/auto",
        settings(ModelTier.LOCAL, ModelTier.MID),
        clock=lambda: NOW,
    )

    assert decision.tier is ModelTier.MID
    assert decision.method is RoutingMethod.FALLBACK
    assert "strong tier is unavailable" in decision.reason


def test_router_rejects_configuration_without_enabled_tier() -> None:
    with pytest.raises(ValueError, match="at least one enabled tier"):
        route_request(
            request("Small edit"),
            "local-code-worker/auto",
            settings(),
            clock=lambda: NOW,
        )
