from datetime import UTC, datetime
from uuid import uuid4

from ..exceptions import ProviderError
from ..virtual_models import ModelTier
from .models import (
    EscalationEvent,
    EscalationReason,
    GatewayRoutingSettings,
    RouteLease,
    RoutingDecision,
    RoutingMethod,
)

_TIER_RANK = {ModelTier.LOCAL: 0, ModelTier.MID: 1, ModelTier.STRONG: 2}


def escalation_reason_for(error: Exception) -> EscalationReason:
    if isinstance(error, TimeoutError):
        return EscalationReason.TIMEOUT
    if isinstance(error, ProviderError):
        categories = {
            "capability_mismatch": EscalationReason.CAPABILITY_MISMATCH,
            "timeout": EscalationReason.TIMEOUT,
            "context_overflow": EscalationReason.CONTEXT_OVERFLOW,
            "invalid_output": EscalationReason.INVALID_OUTPUT,
            "tool_call_failure": EscalationReason.TOOL_CALL_FAILURE,
            "structured_output_failure": EscalationReason.STRUCTURED_OUTPUT_FAILURE,
        }
        return categories.get(error.category, EscalationReason.PROVIDER_ERROR)
    return EscalationReason.PROVIDER_ERROR


def create_route_lease(response_id: str, decision: RoutingDecision) -> RouteLease:
    return RouteLease(
        lease_id=f"lease_{uuid4().hex}",
        root_response_id=response_id,
        current_route=decision.tier,
        current_model=decision.model,
        created_at=decision.timestamp,
        updated_at=decision.timestamp,
    )


def apply_route_lease(
    lease: RouteLease,
    decision: RoutingDecision,
    settings: GatewayRoutingSettings,
) -> RoutingDecision:
    config = settings.tiers.get(lease.current_route)
    if config is None or not config.enabled:
        raise ValueError("the active RouteLease tier is no longer configured")
    return decision.model_copy(
        update={
            "tier": lease.current_route,
            "provider": config.provider,
            "model": lease.current_model,
            "reason": "Active RouteLease pins this response chain to its current route.",
            "method": RoutingMethod.LEASE,
            "rule_id": "active-route-lease",
            "lease_id": lease.lease_id,
        }
    )


def escalate_route_lease(
    lease: RouteLease,
    decision: RoutingDecision,
    reason: EscalationReason,
    *,
    request_id: str,
    response_id: str,
    max_escalations: int,
    clock=lambda: datetime.now(UTC),
) -> tuple[RouteLease, EscalationEvent]:
    if _TIER_RANK[decision.tier] <= _TIER_RANK[lease.current_route]:
        raise ValueError("RouteLease escalation must move to a higher tier")
    if lease.escalation_count >= max_escalations:
        raise ValueError("RouteLease escalation limit reached")
    timestamp = clock().astimezone(UTC).isoformat()
    event = EscalationEvent(
        from_route=lease.current_route,
        to_route=decision.tier,
        from_model=lease.current_model,
        to_model=decision.model,
        reason=reason,
        request_id=request_id,
        response_id=response_id,
        lease_id=lease.lease_id,
        timestamp=timestamp,
    )
    return (
        lease.model_copy(
            update={
                "current_route": decision.tier,
                "current_model": decision.model,
                "updated_at": timestamp,
                "escalation_count": lease.escalation_count + 1,
                "escalation_reason": reason,
            }
        ),
        event,
    )
