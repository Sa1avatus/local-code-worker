import pytest

from local_code_worker.routing.models import TaskCategory, TaskFeatures
from local_code_worker.routing.policy import match_deterministic_rule
from local_code_worker.virtual_models import ModelTier


def features(
    category: TaskCategory = TaskCategory.GENERAL,
    *,
    estimated_scope: int = 1,
    has_previous_failures: bool = False,
) -> TaskFeatures:
    return TaskFeatures(
        category=category,
        context_characters=100,
        message_count=1,
        tool_count=0,
        has_reasoning=False,
        has_previous_failures=has_previous_failures,
        estimated_scope=estimated_scope,
    )


@pytest.mark.parametrize(
    ("task_features", "expected_tier", "expected_rule"),
    [
        (features(TaskCategory.SECURITY), ModelTier.STRONG, "safety-critical"),
        (features(has_previous_failures=True), ModelTier.STRONG, "previous-failure"),
        (features(TaskCategory.ARCHITECTURE), ModelTier.STRONG, "architecture"),
        (features(estimated_scope=3), ModelTier.STRONG, "large-scope"),
        (features(TaskCategory.DEBUGGING), ModelTier.MID, "complex-change"),
        (features(estimated_scope=2), ModelTier.MID, "moderate-scope"),
        (features(TaskCategory.TESTS), ModelTier.LOCAL, "bounded-task"),
        (features(), ModelTier.LOCAL, "default"),
    ],
)
def test_policy_selects_first_matching_rule(
    task_features: TaskFeatures,
    expected_tier: ModelTier,
    expected_rule: str,
) -> None:
    match = match_deterministic_rule(task_features)

    assert match.tier is expected_tier
    assert match.rule_id == expected_rule


def test_security_rule_wins_over_previous_failure_and_large_scope() -> None:
    match = match_deterministic_rule(
        features(
            TaskCategory.SECURITY,
            estimated_scope=3,
            has_previous_failures=True,
        )
    )

    assert match.rule_id == "safety-critical"
