"""Internal tool representation for Local Code Worker."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field

from ..models import StrictModel


class ToolKind(StrEnum):
    """Classification of tool types."""

    FUNCTION = "function"
    WEB_SEARCH = "web_search"
    WEB_FETCH = "web_fetch"
    GITHUB_SEARCH = "github_search"
    DOCS_SEARCH = "docs_search"
    LOCAL_RAG_SEARCH = "local_rag_search"
    NAMESPACE = "namespace"
    UNKNOWN = "unknown"


class NormalizedTool(StrictModel):
    """Unified internal representation of any tool from the Responses API."""

    kind: ToolKind
    name: str = Field(min_length=1)
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    original_payload: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_hosted(self) -> bool:
        """Whether LCW executes this tool internally."""
        return self.kind in (
            ToolKind.WEB_SEARCH,
            ToolKind.WEB_FETCH,
            ToolKind.GITHUB_SEARCH,
            ToolKind.DOCS_SEARCH,
            ToolKind.LOCAL_RAG_SEARCH,
        )

    @property
    def is_function(self) -> bool:
        return self.kind == ToolKind.FUNCTION

    @property
    def is_passthrough(self) -> bool:
        """Whether this tool is forwarded to the downstream model as-is."""
        return self.kind == ToolKind.FUNCTION


# Canonical parameter schemas for hosted tools that models can call.
HOSTED_TOOL_SCHEMAS: dict[ToolKind, dict[str, Any]] = {
    ToolKind.WEB_SEARCH: {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query to find current web information",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (default 5)",
                "default": 5,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    ToolKind.WEB_FETCH: {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to fetch content from",
            },
            "max_chars": {
                "type": "integer",
                "description": "Maximum characters to return (default 8000)",
                "default": 8000,
            },
        },
        "required": ["url"],
        "additionalProperties": False,
    },
    ToolKind.GITHUB_SEARCH: {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "GitHub search query"},
            "limit": {
                "type": "integer",
                "description": "Max results (default 5)",
                "default": 5,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    ToolKind.DOCS_SEARCH: {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Documentation search query"},
            "doc_url": {
                "type": "string",
                "description": "Optional base URL to restrict search",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    ToolKind.LOCAL_RAG_SEARCH: {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Local RAG search query"},
            "collection": {
                "type": "string",
                "description": "Optional collection name",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}


def hosted_tool_description(kind: ToolKind) -> str:
    """Return a model-facing description for a hosted tool."""
    descriptions = {
        ToolKind.WEB_SEARCH: (
            "Search the web for current information. Returns top results with "
            "titles, URLs, and snippets. Use for factual questions about current "
            "events, documentation, or anything requiring up-to-date knowledge."
        ),
        ToolKind.WEB_FETCH: (
            "Fetch and read the content of a specific URL. Returns the page text "
            "content. Use after web_search to read full page content from a "
            "relevant result."
        ),
        ToolKind.GITHUB_SEARCH: (
            "Search GitHub repositories, issues, and code. Returns matching "
            "results with titles, URLs, and descriptions."
        ),
        ToolKind.DOCS_SEARCH: (
            "Search documentation sites. Returns relevant documentation excerpts."
        ),
        ToolKind.LOCAL_RAG_SEARCH: (
            "Search the local codebase and documents. Returns relevant excerpts "
            "from indexed files."
        ),
    }
    return descriptions.get(kind, f"Execute {kind.value}")
