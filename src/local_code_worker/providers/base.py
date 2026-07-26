from typing import Protocol

from ..config import WorkerSettings
from ..models import GenerationMetadata, ProviderHealth


class LlmProvider(Protocol):
    settings: WorkerSettings
    last_generation_metadata: GenerationMetadata | None

    def check_connection(self) -> ProviderHealth: ...

    def list_models(self) -> list[str]: ...

    def chat(
        self,
        messages: list[dict[str, str]],
        response_schema: dict[str, object] | None,
        max_output_characters: int,
    ) -> str: ...
