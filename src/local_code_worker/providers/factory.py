import httpx

from ..config import WorkerSettings
from ..exceptions import ProviderConfigurationError
from ..models import ProviderName
from .base import LlmProvider
from .ollama import OllamaProvider
from .openai_compatible import OpenAICompatibleProvider


def create_provider(
    settings: WorkerSettings, transport: httpx.BaseTransport | None = None
) -> LlmProvider:
    if settings.llm_provider is ProviderName.OLLAMA:
        return OllamaProvider(settings, transport=transport)
    if settings.llm_provider is ProviderName.OPENAI_COMPATIBLE:
        return OpenAICompatibleProvider(settings, transport=transport)
    raise ProviderConfigurationError(f"Unknown LLM provider: {settings.llm_provider}")
