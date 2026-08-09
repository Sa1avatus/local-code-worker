from local_code_worker.providers.base import (
    ProviderFunctionTool,
    ProviderMessage,
    ProviderRequest,
)
from local_code_worker.routing.analyzer import analyze_request
from local_code_worker.routing.models import TaskCategory


def test_analyzer_extracts_safe_explainable_features() -> None:
    secret_marker = "private-source-marker"
    request = ProviderRequest(
        messages=[
            ProviderMessage(
                role="user",
                content=f"Debug the frontend and backend regression {secret_marker}",
            )
        ],
        max_output_characters=100,
        tools=[ProviderFunctionTool(name="read_file", parameters={"type": "object"})],
        reasoning_effort="medium",
    )

    features = analyze_request(request, has_previous_failures=True)

    assert features.category is TaskCategory.DEBUGGING
    assert features.tool_count == 1
    assert features.has_reasoning is True
    assert features.has_previous_failures is True
    assert features.multi_service is True
    assert features.estimated_scope == 2
    assert secret_marker not in str(features.model_dump())


def test_analyzer_uses_ordered_category_priority() -> None:
    request = ProviderRequest(
        messages=[ProviderMessage(role="user", content="Document the security migration")],
        max_output_characters=100,
    )

    assert analyze_request(request).category is TaskCategory.SECURITY
