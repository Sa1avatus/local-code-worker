"""Tool executor — dispatches hosted tool calls to search providers."""

from __future__ import annotations

import logging
from typing import Any

from .models import ToolKind
from .search.base import FetchResponse, SearchResponse
from .search.duckduckgo import DuckDuckGoSearch
from .search.web_fetch import WebFetch

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Execute hosted tools and return text results for the model."""

    def __init__(
        self,
        search: DuckDuckGoSearch | None = None,
        fetcher: WebFetch | None = None,
    ) -> None:
        self._search = search or DuckDuckGoSearch()
        self._fetcher = fetcher or WebFetch()

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute a hosted tool by name and return text result.

        Returns a formatted string that the model can use as context.
        """
        kind = self._resolve_kind(tool_name)

        if kind == ToolKind.WEB_SEARCH:
            return self._exec_web_search(arguments)
        if kind == ToolKind.WEB_FETCH:
            return self._exec_web_fetch(arguments)
        if kind == ToolKind.GITHUB_SEARCH:
            return self._exec_github_search(arguments)
        if kind == ToolKind.DOCS_SEARCH:
            return self._exec_docs_search(arguments)
        if kind == ToolKind.LOCAL_RAG_SEARCH:
            return self._exec_local_rag(arguments)

        return f"Error: unknown hosted tool '{tool_name}'"

    def _resolve_kind(self, tool_name: str) -> ToolKind:
        """Resolve a tool name to ToolKind."""
        name_map = {
            "web_search": ToolKind.WEB_SEARCH,
            "web_fetch": ToolKind.WEB_FETCH,
            "github_search": ToolKind.GITHUB_SEARCH,
            "docs_search": ToolKind.DOCS_SEARCH,
            "local_rag_search": ToolKind.LOCAL_RAG_SEARCH,
        }
        kind = name_map.get(tool_name)
        if kind:
            return kind
        # Try enum value match
        try:
            return ToolKind(tool_name)
        except ValueError:
            return ToolKind.UNKNOWN

    def _exec_web_search(self, args: dict[str, Any]) -> str:
        query = args.get("query", "")
        if not query:
            return "Error: 'query' parameter is required for web_search"
        max_results = args.get("max_results", 5)
        if not isinstance(max_results, int) or max_results < 1:
            max_results = 5
        response: SearchResponse = self._search.search(query, max_results=max_results)
        if not response.results:
            return f"No web search results found for: {query}"
        lines = [f"Web search results for: {query}", ""]
        for i, result in enumerate(response.results, 1):
            lines.append(f"{i}. {result.title}")
            if result.url:
                lines.append(f"   URL: {result.url}")
            if result.snippet:
                lines.append(f"   {result.snippet}")
            lines.append("")
        lines.append(f"(Found {len(response.results)} results in {response.elapsed_ms:.0f}ms)")
        return "\n".join(lines)

    def _exec_web_fetch(self, args: dict[str, Any]) -> str:
        url = args.get("url", "")
        if not url:
            return "Error: 'url' parameter is required for web_fetch"
        max_chars = args.get("max_chars", 8000)
        if not isinstance(max_chars, int) or max_chars < 1:
            max_chars = 8000
        response: FetchResponse = self._fetcher.fetch(url, max_chars=max_chars)
        if response.status_code == 0:
            return f"Error fetching {url}: {response.text}"
        parts = [
            f"Content from: {response.url}",
            f"Status: {response.status_code}",
            f"Content-Type: {response.content_type}",
            "",
            response.text,
        ]
        if response.truncated:
            parts.append(f"\n[Content truncated at {max_chars} characters]")
        return "\n".join(parts)

    def _exec_github_search(self, args: dict[str, Any]) -> str:
        """Stub — future implementation."""
        query = args.get("query", "")
        return (
            f"GitHub search is not yet implemented. "
            f"Query was: {query}"
        )

    def _exec_docs_search(self, args: dict[str, Any]) -> str:
        """Stub — future implementation."""
        query = args.get("query", "")
        return (
            f"Documentation search is not yet implemented. "
            f"Query was: {query}"
        )

    def _exec_local_rag(self, args: dict[str, Any]) -> str:
        """Stub — future implementation."""
        query = args.get("query", "")
        return (
            f"Local RAG search is not yet implemented. "
            f"Query was: {query}"
        )
