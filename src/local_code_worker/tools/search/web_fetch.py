"""Web fetch provider — fetches URL content as text."""

from __future__ import annotations

import time

import httpx

from .base import FetchResponse

_DEFAULT_TIMEOUT = 15.0
_MAX_REDIRECTS = 5
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

# Block private/loopback IPs to prevent SSRF
_BLOCKED_HOSTS = frozenset({
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "::1",
    "metadata.google.internal",
})


def _is_private_host(host: str) -> bool:
    """Check if a host looks like a private/internal address."""
    if host in _BLOCKED_HOSTS:
        return True
    # Block common private ranges
    for prefix in ("10.", "172.16.", "172.17.", "172.18.", "172.19.",
                    "172.20.", "172.21.", "172.22.", "172.23.",
                    "172.24.", "172.25.", "172.26.", "172.27.",
                    "172.28.", "172.29.", "172.30.", "172.31.",
                    "192.168."):
        if host.startswith(prefix):
            return True
    return False


class WebFetch:
    """Fetch web page content as text."""

    name = "web_fetch"

    def __init__(self, *, timeout: float = _DEFAULT_TIMEOUT) -> None:
        self._timeout = timeout

    def fetch(self, url: str, max_chars: int = 8000) -> FetchResponse:
        started = time.monotonic()
        try:
            from urllib.parse import urlparse

            parsed = urlparse(url)
            host = parsed.hostname or ""
            if _is_private_host(host):
                return FetchResponse(
                    url=url,
                    status_code=0,
                    text=f"Error: fetching from private/internal host '{host}' is not allowed",
                    elapsed_ms=round((time.monotonic() - started) * 1000, 2),
                )

            with httpx.Client(
                timeout=self._timeout,
                follow_redirects=True,
                max_redirects=_MAX_REDIRECTS,
                headers={"User-Agent": _USER_AGENT},
            ) as client:
                resp = client.get(url)
                resp.raise_for_status()
                content_type = resp.headers.get("content-type", "")
                text = resp.text
                truncated = len(text) > max_chars
                if truncated:
                    text = text[:max_chars]
                return FetchResponse(
                    url=str(resp.url),
                    status_code=resp.status_code,
                    content_type=content_type,
                    text=text,
                    truncated=truncated,
                    elapsed_ms=round((time.monotonic() - started) * 1000, 2),
                )
        except Exception as exc:
            return FetchResponse(
                url=url,
                status_code=0,
                text=f"Error fetching URL: {exc}",
                elapsed_ms=round((time.monotonic() - started) * 1000, 2),
            )
