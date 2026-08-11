from datetime import UTC, datetime

import pytest

from local_code_worker.models import ProviderName
from local_code_worker.routing.leases import (
    apply_route_lease,
    create_route_lease,
    escalate_route_lease,
)
from local_code_worker.routing.models import (
    EscalationReason,
    GatewayRoutingSettings,
    RoutingDecision,
    RoutingMethod,
    TierConfig,
)
from local_code_worker.virtual_models import ModelTier

NOW = datetime(2026, 8, 11, tzinfo=UTC)


def decision(tier: ModelTier, model: str) -> RoutingDecision:
    return RoutingDecision(
        tier=tier,
        provider=ProviderName.OLLAMA,
        model=model,
        reason="test",
        confidence=1,
        method=RoutingMethod.DETERMINISTIC,
        timestamp=NOW.isoformat(),
        policy_version="1",
    )


def settings() -> GatewayRoutingSettings:
    return GatewayRoutingSettings(
        tiers={
            ModelTier.LOCAL: TierConfig(provider=ProviderName.OLLAMA, model="local"),
            ModelTier.MID: TierConfig(provider=ProviderName.OLLAMA, model="mid"),
            ModelTier.STRONG: TierConfig(provider=ProviderName.OLLAMA, model="strong"),
        }
    )


def test_active_lease_prevents_automatic_downgrade() -> None:
    lease = create_route_lease("resp-1", decision(ModelTier.STRONG, "strong"))

    pinned = apply_route_lease(lease, decision(ModelTier.LOCAL, "local"), settings())

    assert pinned.tier is ModelTier.STRONG
    assert pinned.model == "strong"
    assert pinned.method is RoutingMethod.LEASE
    assert pinned.lease_id == lease.lease_id


def test_escalation_is_monotonic_and_bounded() -> None:
    lease = create_route_lease("resp-1", decision(ModelTier.LOCAL, "local"))
    lease, event = escalate_route_lease(
        lease,
        decision(ModelTier.MID, "mid"),
        EscalationReason.PROVIDER_ERROR,
        request_id="req-2",
        response_id="resp-2",
        max_escalations=1,
        clock=lambda: NOW,
    )

    assert lease.current_route is ModelTier.MID
    assert lease.escalation_count == 1
    assert event.from_route is ModelTier.LOCAL
    assert event.to_route is ModelTier.MID

    with pytest.raises(ValueError, match="limit reached"):
        escalate_route_lease(
            lease,
            decision(ModelTier.STRONG, "strong"),
            EscalationReason.REPEATED_FAILURE,
            request_id="req-3",
            response_id="resp-3",
            max_escalations=1,
        )

    with pytest.raises(ValueError, match="higher tier"):
        escalate_route_lease(
            lease,
            decision(ModelTier.LOCAL, "local"),
            EscalationReason.MANUAL_POLICY,
            request_id="req-3",
            response_id="resp-3",
            max_escalations=2,
        )
