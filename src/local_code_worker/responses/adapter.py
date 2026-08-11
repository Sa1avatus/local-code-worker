"""Adapt Responses API requests to provider requests with tool support."""

from __future__ import annotations

from ..models import JsonMode
from ..providers.base import (
    ProviderFunctionTool,
    ProviderFunctionToolChoice,
    ProviderMessage,
    ProviderRequest,
)
from ..tools.models import NormalizedTool, ToolKind, hosted_tool_description, HOSTED_TOOL_SCHEMAS
from ..tools.normalizer import normalize_request_tools, separate_tools
from .schemas import (
    ResponseAdditionalTools,
    ResponseCreateRequest,
    ResponseFunctionTool,
    ResponseFunctionToolChoice,
    ResponseInputFunctionCall,
    ResponseInputFunctionCallOutput,
    ResponseInputMessage,
    ResponseNamespaceTool,
)


class AdaptedRequest:
    """Provider request plus metadata about hosted tools."""

    def __init__(
        self,
        request: ProviderRequest,
        hosted_tool_names: frozenset[str],
        all_normalized: list[NormalizedTool],
    ) -> None:
        self.request = request
        self.hosted_tool_names = hosted_tool_names
        self.all_normalized = all_normalized


def adapt_response_request(
    request: ResponseCreateRequest,
    *,
    max_output_characters: int,
    json_mode: JsonMode = JsonMode.NONE,
) -> AdaptedRequest:
    """Convert a Responses API request into a provider request.

    Handles:
    - function tools (passthrough to model)
    - namespace tools (expand children)
    - web_search tools (add as function tool for model, mark as hosted)
    - additional_tools in input array
    - function_call / function_call_output in input array
    """
    messages: list[ProviderMessage] = []
    if request.instructions:
        messages.append(ProviderMessage(role="developer", content=request.instructions))

    # Process input items
    additional_tool_dicts: list[dict[str, object]] = []
    if isinstance(request.input, str):
        messages.append(ProviderMessage(role="user", content=request.input))
    else:
        for item in request.input:
            if isinstance(item, ResponseAdditionalTools):
                additional_tool_dicts.extend(item.tools)
            elif isinstance(item, ResponseInputMessage):
                messages.append(
                    ProviderMessage(
                        role=item.role,
                        content=(
                            item.content
                            if isinstance(item.content, str)
                            else "\n".join(part.text for part in item.content)
                        ),
                    )
                )
            elif isinstance(item, ResponseInputFunctionCall):
                # Previous assistant function call — add as assistant message
                messages.append(
                    ProviderMessage(
                        role="assistant",
                        content=f"[Calling tool: {item.name}({item.arguments})]",
                    )
                )
            elif isinstance(item, ResponseInputFunctionCallOutput):
                # Tool result from client — add as tool/system message
                messages.append(
                    ProviderMessage(
                        role="tool",
                        content=item.output,
                    )
                )

    # Normalize all tools to identify hosted vs passthrough
    all_normalized = normalize_request_tools(request)

    # Also normalize additional_tools from input
    from ..tools.normalizer import normalize_tool_dict
    for raw_dict in additional_tool_dicts:
        if isinstance(raw_dict, dict):
            all_normalized.extend(normalize_tool_dict(raw_dict))

    hosted, passthrough = separate_tools(all_normalized)
    hosted_names = frozenset(t.name for t in hosted)

    # Build provider tool list: passthrough tools + hosted tools as function tools
    provider_tools: list[ProviderFunctionTool] = []

    # Add passthrough (function) tools
    for tool in passthrough:
        provider_tools.append(
            ProviderFunctionTool(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
                strict=tool.metadata.get("strict", True),
            )
        )

    # Add hosted tools as function tools so the model can call them
    for tool in hosted:
        provider_tools.append(
            ProviderFunctionTool(
                name=tool.name,
                description=tool.description or hosted_tool_description(tool.kind),
                parameters=HOSTED_TOOL_SCHEMAS.get(tool.kind, tool.parameters),
                strict=False,
            )
        )

    # Handle tool_choice
    tool_choice = (
        ProviderFunctionToolChoice(name=request.tool_choice.name)
        if isinstance(request.tool_choice, ResponseFunctionToolChoice)
        else request.tool_choice
    )

    provider_request = ProviderRequest(
        messages=messages,
        max_output_characters=max_output_characters,
        max_output_tokens=request.max_output_tokens,
        json_mode=json_mode,
        stream=request.stream,
        tools=provider_tools,
        tool_choice=tool_choice,
        reasoning_effort=request.reasoning.effort if request.reasoning else None,
    )

    return AdaptedRequest(
        request=provider_request,
        hosted_tool_names=hosted_names,
        all_normalized=all_normalized,
    )
