"""Normalize Responses API tools into internal NormalizedTool representation."""

from __future__ import annotations

from typing import Any

from ..responses.schemas import (
    ResponseCreateRequest,
    ResponseFunctionTool,
    ResponseNamespaceTool,
    ResponseWebSearchTool,
)
from .models import (
    HOSTED_TOOL_SCHEMAS,
    NormalizedTool,
    ToolKind,
    hosted_tool_description,
)


def _kind_from_type(tool_type: str) -> ToolKind:
    """Map a raw tool type string to ToolKind."""
    mapping = {
        "function": ToolKind.FUNCTION,
        "web_search": ToolKind.WEB_SEARCH,
        "web_fetch": ToolKind.WEB_FETCH,
        "github_search": ToolKind.GITHUB_SEARCH,
        "docs_search": ToolKind.DOCS_SEARCH,
        "local_rag_search": ToolKind.LOCAL_RAG_SEARCH,
        "namespace": ToolKind.NAMESPACE,
    }
    return mapping.get(tool_type, ToolKind.UNKNOWN)


def normalize_tool_dict(raw: dict[str, Any]) -> list[NormalizedTool]:
    """Normalize a raw tool dict (from additional_tools or top-level tools).

    Returns a list because namespace tools expand to their child function tools.
    Unknown tool types are captured as ToolKind.UNKNOWN for forward compatibility.
    """
    tool_type = raw.get("type", "unknown")
    kind = _kind_from_type(tool_type)

    if kind == ToolKind.NAMESPACE:
        # Expand namespace: normalize each child tool
        results: list[NormalizedTool] = []
        for child in raw.get("tools", []):
            if isinstance(child, dict):
                results.extend(normalize_tool_dict(child))
        return results

    if kind == ToolKind.UNKNOWN:
        # Forward-compatible: capture unknown tools without failing
        return [
            NormalizedTool(
                kind=kind,
                name=raw.get("name", "unknown"),
                description=raw.get("description"),
                parameters={},
                original_payload=raw,
            )
        ]

    if kind == ToolKind.FUNCTION:
        # Check if this function tool is actually a hosted tool by name
        # (e.g. Codex CLI may send web_search as {"type":"function","name":"web_search"})
        func_name = raw.get("name", "unknown")
        _HOSTED_NAMES = {"web_search", "web_fetch", "github_search", "docs_search", "local_rag_search"}
        if func_name in _HOSTED_NAMES:
            hosted_kind = _kind_from_type(func_name)
            return [
                NormalizedTool(
                    kind=hosted_kind,
                    name=func_name,
                    description=raw.get("description") or hosted_tool_description(hosted_kind),
                    parameters=HOSTED_TOOL_SCHEMAS.get(hosted_kind, raw.get("parameters", {})),
                    original_payload=raw,
                )
            ]
        return [
            NormalizedTool(
                kind=kind,
                name=func_name,
                description=raw.get("description"),
                parameters=raw.get("parameters", {}),
                metadata={"strict": raw.get("strict", True)},
                original_payload=raw,
            )
        ]

    # Hosted tools (web_search, web_fetch, etc.)
    # These may come from Codex as bare {"type": "web_search", ...} or
    # we synthesize them from the model catalog.
    return [
        NormalizedTool(
            kind=kind,
            name=tool_type,
            description=raw.get("description") or hosted_tool_description(kind),
            parameters=HOSTED_TOOL_SCHEMAS.get(kind, {}),
            original_payload=raw,
        )
    ]


def normalize_typed_tool(tool: ResponseFunctionTool | ResponseNamespaceTool | ResponseWebSearchTool) -> list[NormalizedTool]:
    """Normalize a typed Responses schema tool object."""
    if isinstance(tool, ResponseFunctionTool):
        return [
            NormalizedTool(
                kind=ToolKind.FUNCTION,
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
                metadata={"strict": tool.strict},
            )
        ]
    if isinstance(tool, ResponseNamespaceTool):
        results: list[NormalizedTool] = []
        for child in tool.tools:
            results.extend(normalize_typed_tool(child))
        return results
    if isinstance(tool, ResponseWebSearchTool):
        return [
            NormalizedTool(
                kind=ToolKind.WEB_SEARCH,
                name="web_search",
                description=hosted_tool_description(ToolKind.WEB_SEARCH),
                parameters=HOSTED_TOOL_SCHEMAS[ToolKind.WEB_SEARCH],
                original_payload=tool.model_dump(mode="json"),
            )
        ]
    # Fallback for unknown typed tools
    return [
        NormalizedTool(
            kind=ToolKind.UNKNOWN,
            name=getattr(tool, "name", "unknown"),
            description=getattr(tool, "description", None),
            original_payload=tool.model_dump(mode="json") if hasattr(tool, "model_dump") else {},
        )
    ]


def normalize_request_tools(request: ResponseCreateRequest) -> list[NormalizedTool]:
    """Extract and normalize ALL tools from a Responses request.

    Handles:
    - Top-level tools (function, namespace, web_search)
    - additional_tools in the input array
    """
    all_normalized: list[NormalizedTool] = []

    # Top-level tools
    for tool in request.tools:
        all_normalized.extend(normalize_typed_tool(tool))

    # additional_tools from input array
    if isinstance(request.input, list):
        for item in request.input:
            if hasattr(item, "type") and item.type == "additional_tools":
                for raw_tool in getattr(item, "tools", []):
                    if isinstance(raw_tool, dict):
                        all_normalized.extend(normalize_tool_dict(raw_tool))

    return all_normalized


def separate_tools(
    tools: list[NormalizedTool],
) -> tuple[list[NormalizedTool], list[NormalizedTool]]:
    """Separate tools into (hosted, passthrough) groups.

    Hosted tools are executed by LCW internally.
    Passthrough tools are forwarded to the downstream model.
    """
    hosted: list[NormalizedTool] = []
    passthrough: list[NormalizedTool] = []
    for tool in tools:
        if tool.is_hosted:
            hosted.append(tool)
        elif tool.is_function:
            passthrough.append(tool)
        # UNKNOWN and NAMESPACE are dropped (already expanded or captured)
    return hosted, passthrough
