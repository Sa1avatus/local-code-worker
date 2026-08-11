from math import ceil

from ..models import JsonMode
from ..providers.base import ProviderCapability, ProviderRequest
from ..providers.registry import DEFAULT_PROVIDER_REGISTRY
from ..virtual_models import ModelTier
from .models import (
    GatewayRoutingSettings,
    ModelCapabilities,
    RequestCapabilities,
    TierConfig,
)


def analyze_capabilities(request: ProviderRequest) -> RequestCapabilities:
    content = "\n".join(message.content for message in request.messages)
    normalized = content.casefold()
    path_markers = sum(token in content for token in ("/", "\\", ".py", ".ts", ".js"))
    reasoning = {None: 0, "none": 0, "low": 1, "medium": 2}.get(
        request.reasoning_effort,
        3,
    )
    return RequestCapabilities(
        requires_tools=bool(request.tools),
        requires_function_calling=bool(request.tools),
        requires_structured_output=request.response_schema is not None
        or request.json_mode is not JsonMode.NONE,
        requires_json_schema=request.response_schema is not None,
        context_size=len(content),
        estimated_input_tokens=ceil(len(content) / 4),
        estimated_output_tokens=request.max_output_tokens or 4096,
        multi_file_task=path_markers >= 2 or "multi-file" in normalized,
        large_repository_context=len(content) >= 24_000,
        reasoning_complexity=reasoning,
        diff_complexity=min(3, max(0, path_markers)),
    )


def configured_capabilities(tier: ModelTier, config: TierConfig) -> ModelCapabilities:
    if config.capabilities is not None:
        return config.capabilities
    provider = DEFAULT_PROVIDER_REGISTRY.capabilities(config.provider)
    return ModelCapabilities(
        model_id=config.model,
        tier=tier,
        context_window=config.context_length or 16_384,
        supports_tools=provider.supports(ProviderCapability.FUNCTION_TOOLS),
        supports_structured_output=provider.supports(ProviderCapability.JSON_OBJECT),
        supports_json_schema=provider.supports(ProviderCapability.JSON_SCHEMA),
        supports_streaming=provider.supports(ProviderCapability.STREAMING),
        max_output_tokens=4096,
    )


def capable_tiers(
    request: ProviderRequest,
    settings: GatewayRoutingSettings,
) -> tuple[set[ModelTier], tuple[str, ...], tuple[str, ...]]:
    required = analyze_capabilities(request)
    constraints: list[str] = []
    if required.requires_tools:
        constraints.append("tools")
    if required.requires_json_schema:
        constraints.append("json_schema")
    elif required.requires_structured_output:
        constraints.append("structured_output")
    if request.stream:
        constraints.append("streaming")
    eligible: set[ModelTier] = set()
    excluded: list[str] = []
    for tier, config in settings.tiers.items():
        if not config.enabled:
            continue
        model = configured_capabilities(tier, config)
        compatible = (
            (not required.requires_tools or model.supports_tools)
            and (not required.requires_structured_output or model.supports_structured_output)
            and (not required.requires_json_schema or model.supports_json_schema)
            and (not request.stream or model.supports_streaming)
            and required.estimated_input_tokens <= model.context_window
            and required.estimated_output_tokens <= model.max_output_tokens
        )
        if compatible:
            eligible.add(tier)
        else:
            excluded.append(model.model_id)
    return eligible, tuple(constraints), tuple(excluded)
