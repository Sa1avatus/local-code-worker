from collections.abc import Callable
from dataclasses import dataclass

import httpx
from pydantic import Field

from ..config import WorkerSettings
from ..exceptions import ProviderConfigurationError
from ..models import ProviderName, StrictModel
from .base import LlmProvider, ProviderCapabilities

ProviderFactory = Callable[[WorkerSettings, httpx.BaseTransport | None], LlmProvider]


@dataclass(frozen=True)
class ProviderRegistration:
    name: ProviderName
    factory: ProviderFactory
    capabilities: ProviderCapabilities
    is_local: bool


class ProviderModelMetadata(StrictModel):
    provider: ProviderName
    model: str = Field(min_length=1)
    capabilities: ProviderCapabilities
    is_local: bool


class ProviderRegistry:
    def __init__(self) -> None:
        self._registrations: dict[ProviderName, ProviderRegistration] = {}

    def register(self, registration: ProviderRegistration) -> None:
        if registration.name in self._registrations:
            raise ValueError(f"Provider is already registered: {registration.name}")
        self._registrations[registration.name] = registration

    def create(
        self,
        settings: WorkerSettings,
        transport: httpx.BaseTransport | None = None,
    ) -> LlmProvider:
        registration = self._registrations.get(settings.llm_provider)
        if registration is None:
            raise ProviderConfigurationError(f"Unknown LLM provider: {settings.llm_provider}")
        return registration.factory(settings, transport)

    def configured_model(self, settings: WorkerSettings) -> ProviderModelMetadata:
        registration = self._registrations.get(settings.llm_provider)
        if registration is None:
            raise ProviderConfigurationError(f"Unknown LLM provider: {settings.llm_provider}")
        return ProviderModelMetadata(
            provider=registration.name,
            model=settings.llm_model,
            capabilities=registration.capabilities,
            is_local=registration.is_local,
        )


def create_default_registry() -> ProviderRegistry:
    from .ollama import OllamaProvider
    from .openai_compatible import OpenAICompatibleProvider

    registry = ProviderRegistry()
    registry.register(
        ProviderRegistration(
            name=ProviderName.OLLAMA,
            factory=lambda settings, transport: OllamaProvider(settings, transport),
            capabilities=OllamaProvider.capabilities,
            is_local=True,
        )
    )
    registry.register(
        ProviderRegistration(
            name=ProviderName.OPENAI_COMPATIBLE,
            factory=lambda settings, transport: OpenAICompatibleProvider(settings, transport),
            capabilities=OpenAICompatibleProvider.capabilities,
            is_local=False,
        )
    )
    return registry


DEFAULT_PROVIDER_REGISTRY = create_default_registry()
