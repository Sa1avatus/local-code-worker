import httpx

from ..config import WorkerSettings
from .base import LlmProvider
from .registry import DEFAULT_PROVIDER_REGISTRY


def create_provider(
    settings: WorkerSettings, transport: httpx.BaseTransport | None = None
) -> LlmProvider:
    return DEFAULT_PROVIDER_REGISTRY.create(settings, transport)
