from collections.abc import Callable

from ..virtual_models import ModelTier
from .models import RuleMatch, TaskCategory, TaskFeatures

RulePredicate = Callable[[TaskFeatures], bool]


def _category_is(*categories: TaskCategory) -> RulePredicate:
    return lambda features: features.category in categories


_RULES: tuple[tuple[str, ModelTier, float, str, RulePredicate], ...] = (
    (
        "safety-critical",
        ModelTier.STRONG,
        1.0,
        "Security-sensitive work requires the strongest configured tier.",
        _category_is(TaskCategory.SECURITY),
    ),
    (
        "previous-failure",
        ModelTier.STRONG,
        0.98,
        "A previous failed attempt requires stronger recovery reasoning.",
        lambda features: features.has_previous_failures,
    ),
    (
        "architecture",
        ModelTier.STRONG,
        0.95,
        "Architecture work requires broad design reasoning.",
        _category_is(TaskCategory.ARCHITECTURE),
    ),
    (
        "large-scope",
        ModelTier.STRONG,
        0.9,
        "Large or multi-service context requires the strongest tier.",
        lambda features: features.estimated_scope >= 3,
    ),
    (
        "complex-change",
        ModelTier.MID,
        0.85,
        "Debugging, migration, and refactoring require the mid tier.",
        _category_is(
            TaskCategory.DEBUGGING,
            TaskCategory.MIGRATION,
            TaskCategory.REFACTOR,
        ),
    ),
    (
        "moderate-scope",
        ModelTier.MID,
        0.75,
        "Moderate context or tool use requires the mid tier.",
        lambda features: features.estimated_scope == 2,
    ),
    (
        "bounded-task",
        ModelTier.LOCAL,
        0.9,
        "Bounded edits, tests, and documentation fit the local tier.",
        _category_is(
            TaskCategory.SIMPLE_EDIT,
            TaskCategory.TESTS,
            TaskCategory.DOCUMENTATION,
        ),
    ),
    (
        "default",
        ModelTier.LOCAL,
        0.6,
        "Small general work defaults to the local tier.",
        lambda _features: True,
    ),
)


def match_deterministic_rule(features: TaskFeatures) -> RuleMatch:
    for rule_id, tier, confidence, reason, predicate in _RULES:
        if predicate(features):
            return RuleMatch(
                tier=tier,
                reason=reason,
                confidence=confidence,
                rule_id=rule_id,
            )
    raise RuntimeError("routing policy must contain a default rule")
