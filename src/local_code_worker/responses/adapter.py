"""Adapt Responses API requests to provider requests with tool support."""

from __future__ import annotations

from ..models import JsonMode
from ..providers.base import (
    ProviderFunctionTool,
    ProviderFunctionToolChoice,
    ProviderMessage,
    ProviderRequest,
)
from ..tools.models import HOSTED_TOOL_SCHEMAS, NormalizedTool, ToolKind, hosted_tool_description
from ..tools.normalizer import normalize_request_tools, separate_tools
from .schemas import (
    ResponseAdditionalTools,
    ResponseCreateRequest,
    ResponseFunctionToolChoice,
    ResponseInputFunctionCall,
    ResponseInputFunctionCallOutput,
    ResponseInputMessage,
)

# Core Codex tools that should always be included when present.
_CORE_TOOL_NAMES = frozenset({
    "shell_command",
    "apply_patch",
    "update_plan",
    "read_mcp_resource",
    "list_mcp_resources",
})

# Default max passthrough tools sent to the model.
# Local models (14B) struggle with 30+ tools. 8 is a reasonable limit.
DEFAULT_MAX_PASSTHROUGH_TOOLS = 8


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


def _filter_passthrough_tools(
    tools: list[NormalizedTool],
    max_tools: int,
) -> list[NormalizedTool]:
    """Filter passthrough tools to a manageable count.

    Strategy:
    1. Always include core Codex tools (shell_command, apply_patch, etc.)
    2. Fill remaining slots with other tools in original order
    3. This ensures the model sees the most important tools first
    """
    if len(tools) <= max_tools:
        return tools

    core: list[NormalizedTool] = []
    other: list[NormalizedTool] = []
    for tool in tools:
        if tool.name in _CORE_TOOL_NAMES:
            core.append(tool)
        else:
            other.append(tool)

    # Core tools always included, fill remaining with others
    remaining = max_tools - len(core)
    if remaining < 0:
        # More core tools than limit — just take the first max_tools
        return core[:max_tools]

    return core + other[:remaining]


def adapt_response_request(
    request: ResponseCreateRequest,
    *,
    max_output_characters: int,
    json_mode: JsonMode = JsonMode.NONE,
    max_passthrough_tools: int = DEFAULT_MAX_PASSTHROUGH_TOOLS,
) -> AdaptedRequest:
    """Convert a Responses API request into a provider request.

    Handles:
    - function tools (passthrough to model, filtered to limit)
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
                messages.append(
                    ProviderMessage(
                        role="assistant",
                        content=f"[Calling tool: {item.name}({item.arguments})]",
                    )
                )
            elif isinstance(item, ResponseInputFunctionCallOutput):
                messages.append(
                    ProviderMessage(
                        role="tool",
                        content=item.output,
                    )
                )

    # Normalize all tools to identify hosted vs passthrough.
    # normalize_request_tools handles both top-level tools and
    # additional_tools from the input array.
    all_normalized = normalize_request_tools(request)

    hosted, passthrough = separate_tools(all_normalized)
    hosted_names = frozenset(t.name for t in hosted)

    # Deduplicate passthrough tools by name (keep first occurrence)
    seen_names: set[str] = set()
    deduped: list[NormalizedTool] = []
    for tool in passthrough:
        if tool.name not in seen_names:
            seen_names.add(tool.name)
            deduped.append(tool)
    passthrough = deduped

    # Filter passthrough tools to keep the model's tool list manageable
    passthrough = _filter_passthrough_tools(passthrough, max_passthrough_tools)

    # Build provider tool list: hosted tools first (most important), then passthrough
    provider_tools: list[ProviderFunctionTool] = []

    # Add hosted tools FIRST — model sees these prominently
    for tool in hosted:
        provider_tools.append(
            ProviderFunctionTool(
                name=tool.name,
                description=tool.description or hosted_tool_description(tool.kind),
                parameters=HOSTED_TOOL_SCHEMAS.get(tool.kind, tool.parameters),
                strict=False,
            )
        )

    # Add passthrough (function) tools after hosted tools
    for tool in passthrough:
        provider_tools.append(
            ProviderFunctionTool(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
                strict=tool.metadata.get("strict", True),
            )
        )

    # Inject guidance when hosted web_search is available so the model
    # prefers it over passthrough search tools (e.g. _search_documentation).
    if ToolKind.WEB_SEARCH in {t.kind for t in hosted}:
        _hint = (
            "When you need to find current information, look up facts, "
            "search the web, or answer questions requiring up-to-date knowledge, "
            "use the web_search tool. Do NOT use other search tools for web searches."
        )
        messages.append(ProviderMessage(role="developer", content=_hint))

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
