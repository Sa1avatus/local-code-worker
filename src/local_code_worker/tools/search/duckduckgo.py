"""DuckDuckGo search provider using the HTML lite endpoint."""

from __future__ import annotations

import re
import time
from html import unescape
from urllib.parse import urlencode

import httpx

from .base import SearchResponse, SearchResult

# DuckDuckGo lite HTML endpoint — no API key required.
_DDG_LITE_URL = "https://lite.duckduckgo.com/lite/"
_DEFAULT_TIMEOUT = 10.0
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)


def _strip_html(text: str) -> str:
    """Remove HTML tags and decode entities."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    return re.sub(r"\s+", " ", text).strip()


class DuckDuckGoSearch:
    """Web search via DuckDuckGo lite HTML."""

    name = "duckduckgo"

    def __init__(self, *, timeout: float = _DEFAULT_TIMEOUT) -> None:
        self._timeout = timeout

    def search(self, query: str, max_results: int = 5) -> SearchResponse:
        started = time.monotonic()
        results: list[SearchResult] = []
        try:
            with httpx.Client(
                timeout=self._timeout,
                follow_redirects=True,
                headers={"User-Agent": _USER_AGENT},
            ) as client:
                resp = client.post(
                    _DDG_LITE_URL,
                    data={"q": query, "kl": "wt-wt"},
                )
                resp.raise_for_status()
                results = _parse_lite_html(resp.text, max_results)
        except Exception:
            # Return empty results on any error — don't crash the pipeline
            pass
        elapsed = (time.monotonic() - started) * 1000
        return SearchResponse(
            query=query,
            results=results,
            provider=self.name,
            elapsed_ms=round(elapsed, 2),
        )


def _parse_lite_html(html: str, max_results: int) -> list[SearchResult]:
    """Parse DuckDuckGo lite HTML into SearchResult list."""
    results: list[SearchResult] = []

    # DuckDuckGo lite uses <a class="result-link"> for titles/URLs
    # and <td class="result-snippet"> for snippets.
    # The structure is a table with alternating title and snippet rows.

    # Find result links
    link_pattern = re.compile(
        r"<a[^>]+href=['\"]([^'\"]*)['\"].*?class=['\"]result-link['\"]['\"]?[^>]*>(.*?)</a>",
        re.DOTALL,
    )
    # Find snippets
    snippet_pattern = re.compile(
        r"<td\s+class=['\"]result-snippet['\"][^>]*>(.*?)</td>",
        re.DOTALL,
    )

    links = link_pattern.findall(html)
    snippets = snippet_pattern.findall(html)

    for i, (url, title) in enumerate(links[:max_results]):
        snippet = _strip_html(snippets[i]) if i < len(snippets) else ""
        results.append(
            SearchResult(
                title=_strip_html(title),
                url=url,
                snippet=snippet,
            )
        )

    # Fallback: try simpler pattern if no results found
    if not results:
        results = _parse_lite_fallback(html, max_results)

    return results


def _parse_lite_fallback(html: str, max_results: int) -> list[SearchResult]:
    """Fallback parser using broader patterns."""
    results: list[SearchResult] = []
    # Try finding any links with result-related classes
    link_pattern = re.compile(
        r'<a[^>]+href="(https?://[^"]*)"[^>]*>(.*?)</a>',
        re.DOTALL,
    )
    seen_urls: set[str] = set()
    for url, title in link_pattern.findall(html):
        if "duckduckgo.com" in url:
            continue
        if url in seen_urls:
            continue
        seen_urls.add(url)
        clean_title = _strip_html(title)
        if clean_title and len(clean_title) > 3:
            results.append(SearchResult(title=clean_title, url=url))
        if len(results) >= max_results:
            break
    return results
