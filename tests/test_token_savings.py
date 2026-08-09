from local_code_worker.telemetry.savings import BaselineMethod, TokenSavings


def test_token_savings_calculates_estimated_cloud_reduction() -> None:
    savings = TokenSavings(
        baseline_cloud_tokens=100,
        actual_cloud_tokens=30,
    )

    assert savings.baseline_cloud_tokens == 100
    assert savings.actual_cloud_tokens == 30
    assert savings.baseline_method is BaselineMethod.EXPLICIT_CLOUD_TOKEN_BUDGET
    assert savings.estimated is True
    assert savings.cloud_tokens_saved == 70
    assert savings.cloud_tokens_saved_percent == 70.0
