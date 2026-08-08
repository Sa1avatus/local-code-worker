import httpx

from .config import WorkerSettings
from .exceptions import OllamaError, ProviderError
from .providers.ollama import OllamaProvider


class OllamaClient(OllamaProvider):
    """Backward-compatible facade for the pre-provider API."""

    def __init__(self, settings: WorkerSettings, transport: httpx.BaseTransport | None = None):
        super().__init__(settings, transport=transport)

    def check_connection(self) -> list[str]:  # type: ignore[override]
        try:
            return self.list_models()
        except ProviderError as error:
            raise OllamaError(str(error)) from error

    def chat(
        self,
        messages: list[dict[str, str]],
        response_schema: dict[str, object] | None,
        max_output_characters: int,
        max_output_tokens: int | None = None,
    ) -> str:
        try:
            return super().chat(
                messages,
                response_schema,
                max_output_characters,
                max_output_tokens,
            )
        except ProviderError as error:
            raise OllamaError(str(error)) from error


def load_system_prompt(worker_root):
    from .prompt_loader import load_system_prompt as load_prompt

    return load_prompt(worker_root)
