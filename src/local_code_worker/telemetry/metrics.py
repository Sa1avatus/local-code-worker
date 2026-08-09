from math import ceil
from statistics import fmean

from pydantic import Field

from ..models import StrictModel


class LatencyMetrics(StrictModel):
    average_latency_ms: float = Field(ge=0)
    p50_latency_ms: float = Field(ge=0)
    p95_latency_ms: float = Field(ge=0)


class RequestMetrics(StrictModel):
    request_count: int = Field(ge=0)
    success_count: int = Field(ge=0)
    failure_count: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency: LatencyMetrics


def summarize_latencies(latencies_ms: list[float]) -> LatencyMetrics:
    if not latencies_ms:
        return LatencyMetrics(
            average_latency_ms=0.0,
            p50_latency_ms=0.0,
            p95_latency_ms=0.0,
        )
    if any(value < 0 for value in latencies_ms):
        raise ValueError("latencies must be non-negative")

    ordered = sorted(latencies_ms)
    return LatencyMetrics(
        average_latency_ms=round(fmean(ordered), 2),
        p50_latency_ms=_nearest_rank(ordered, 0.50),
        p95_latency_ms=_nearest_rank(ordered, 0.95),
    )


def _nearest_rank(ordered: list[float], percentile: float) -> float:
    index = max(0, ceil(percentile * len(ordered)) - 1)
    return float(ordered[index])
