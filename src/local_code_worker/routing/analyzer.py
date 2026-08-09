from ..providers.base import ProviderRequest
from .models import TaskCategory, TaskFeatures

_CATEGORY_KEYWORDS: tuple[tuple[TaskCategory, tuple[str, ...]], ...] = (
    (TaskCategory.SECURITY, ("security", "vulnerability", "credential", "auth")),
    (TaskCategory.ARCHITECTURE, ("architecture", "design system", "cross-service")),
    (TaskCategory.MIGRATION, ("migration", "migrate", "schema change", "upgrade")),
    (TaskCategory.DEBUGGING, ("debug", "traceback", "root cause", "regression")),
    (TaskCategory.REFACTOR, ("refactor", "restructure", "extract module")),
    (TaskCategory.TESTS, ("test", "pytest", "coverage")),
    (TaskCategory.DOCUMENTATION, ("documentation", "readme", "docs")),
    (TaskCategory.SIMPLE_EDIT, ("rename", "typo", "small edit", "one line")),
)


def analyze_request(
    request: ProviderRequest,
    *,
    has_previous_failures: bool = False,
) -> TaskFeatures:
    combined = "\n".join(message.content for message in request.messages)
    normalized = combined.casefold()
    category = TaskCategory.GENERAL
    for candidate, keywords in _CATEGORY_KEYWORDS:
        if any(keyword in normalized for keyword in keywords):
            category = candidate
            break
    service_markers = sum(
        marker in normalized for marker in ("frontend", "backend", "database", "api", "worker")
    )
    context_characters = len(combined)
    estimated_scope = 1
    if context_characters >= 8_000 or len(request.messages) >= 6:
        estimated_scope += 1
    if service_markers >= 2 or request.tools:
        estimated_scope += 1
    return TaskFeatures(
        category=category,
        context_characters=context_characters,
        message_count=len(request.messages),
        tool_count=len(request.tools),
        has_reasoning=request.reasoning_effort not in {None, "none"},
        has_previous_failures=has_previous_failures,
        multi_service=service_markers >= 2,
        estimated_scope=estimated_scope,
    )
