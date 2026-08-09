from local_code_worker.telemetry.metrics import summarize_latencies


def test_summarize_latencies_calculates_average_and_nearest_rank_percentiles() -> None:
    metrics = summarize_latencies([10.0, 20.0, 30.0, 40.0, 50.0])

    assert metrics.average_latency_ms == 30.0
    assert metrics.p50_latency_ms == 30.0
    assert metrics.p95_latency_ms == 50.0


def test_summarize_latencies_returns_zero_metrics_for_no_requests() -> None:
    metrics = summarize_latencies([])

    assert metrics.average_latency_ms == 0.0
    assert metrics.p50_latency_ms == 0.0
    assert metrics.p95_latency_ms == 0.0
