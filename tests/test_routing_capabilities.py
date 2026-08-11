from local_code_worker.models import ProviderName
from local_code_worker.providers.base import ProviderFunctionTool, ProviderMessage, ProviderRequest
from local_code_worker.routing.engine import route_request
from local_code_worker.routing.models import (
    GatewayRoutingSettings,
    ModelCapabilities,
    RoutingMode,
    TierConfig,
)
from local_code_worker.virtual_models import ModelTier


def model(tier: ModelTier, name: str, *, tools: bool) -> TierConfig:
    return TierConfig(
        provider=ProviderName.OLLAMA,
        model=name,
        capabilities=ModelCapabilities(
            model_id=name,
            tier=tier,
            context_window=16_384,
            supports_tools=tools,
            supports_structured_output=True,
            supports_json_schema=True,
            supports_streaming=True,
            max_output_tokens=4096,
        ),
    )


def test_tool_request_excludes_local_model_without_tool_capability() -> None:
    settings = GatewayRoutingSettings(
        mode=RoutingMode.ROUTER,
        tiers={
            ModelTier.LOCAL: model(ModelTier.LOCAL, "local-no-tools", tools=False),
            ModelTier.MID: model(ModelTier.MID, "mid-tools", tools=True),
        },
    )
    request = ProviderRequest(
        messages=[ProviderMessage(role="user", content="small edit")],
        max_output_characters=1000,
        tools=[ProviderFunctionTool(name="edit", parameters={"type": "object"})],
    )

    decision = route_request(request, "local-code-worker/auto", settings)

    assert decision.tier is ModelTier.MID
    assert decision.model == "mid-tools"
    assert decision.capability_constraints == ("tools",)
    assert decision.excluded_models == ("local-no-tools",)
