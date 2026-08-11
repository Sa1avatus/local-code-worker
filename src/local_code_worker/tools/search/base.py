"""Base protocol for web search providers."""

from __future__ import annotations

from typing import Protocol

from pydantic import Field

from ..models import StrictModel


class SearchResult(StrictModel):
    """A single search result."""

    title: str = ""
    url: str = ""
    snippet: str = ""


class SearchResponse(StrictModel):
    """Search provider response."""

    query: str = ""
    results: list[SearchResult] = Field(default_factory=list)
    provider: str = ""
    elapsed_ms: float = 0.0


class FetchResponse(StrictModel):
    """Web fetch response."""

    url: str = ""
    status_code: int = 0
    content_type: str = ""
    text: str = ""
    truncated: bool = False
    elapsed_ms: float = 0.0


class SearchProvider(Protocol):
    """Protocol for web search implementations."""

    name: str

    def search(self, query: str, max_results: int = 5) -> SearchResponse: ...


class FetchProvider(Protocol):
    """Protocol for web fetch implementations."""

    name: str

    def fetch(self, url: str, max_chars: int = 8000) -> FetchResponse: ...
