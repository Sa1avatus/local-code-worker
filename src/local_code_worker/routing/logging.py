import json
import logging

from .models import EscalationEvent, RoutingDecision

LOGGER = logging.getLogger("local_code_worker.routing")
if not LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(_handler)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False


def log_routing_decision(
    *,
    request_id: str,
    response_id: str,
    previous_response_id: str | None,
    decision: RoutingDecision,
    latency_ms: float | None = None,
) -> None:
    LOGGER.info(
        json.dumps(
            {
                "event": "routing_decision",
                "request_id": request_id,
                "response_id": response_id,
                "previous_response_id": previous_response_id,
                "route_lease_id": decision.lease_id,
                "route": decision.tier.value,
                "route_score": decision.routellm_score,
                "model": decision.model,
                "provider": decision.provider.value,
                "routing_reason": decision.reason,
                "escalation_reason": None,
                "latency_ms": latency_ms,
                "input_tokens": None,
                "output_tokens": None,
            },
            separators=(",", ":"),
        )
    )


def log_escalation(event: EscalationEvent) -> None:
    LOGGER.info(
        json.dumps(
            {"event": "route_escalation", **event.model_dump(mode="json")},
            separators=(",", ":"),
        )
    )
