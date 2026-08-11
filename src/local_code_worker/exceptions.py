class WorkerError(Exception):
    """Base error with a user-facing message."""


class TaskValidationError(WorkerError):
    """The task is malformed or unsafe."""


class RepositoryError(WorkerError):
    """Repository state prevents safe operation."""


class ContextError(WorkerError):
    """The requested context cannot be built safely."""


class OllamaError(WorkerError):
    """Ollama could not return a usable implementation."""


class ProviderError(WorkerError):
    """An LLM provider request failed without exposing credentials."""

    def __init__(self, message: str, *, category: str = "provider_error"):
        super().__init__(message)
        self.category = category


class ProviderConfigurationError(WorkerError):
    """Provider configuration is missing or invalid."""


class ProviderTimeout(ProviderError):
    """A provider exceeded its configured timeout."""


class ProviderUnavailable(ProviderError):
    """A configured provider or model is unavailable."""


class ContextOverflowError(ProviderError):
    """The request exceeds the selected model context window."""


class CapabilityError(ProviderError):
    """No configured model satisfies hard request capabilities."""


class InvalidModelOutput(ProviderError):
    """A provider returned output that violates the requested contract."""


class ResponseError(WorkerError):
    """The model response is malformed."""


class PatchValidationError(WorkerError):
    """Generated changes violate task boundaries."""


class SemanticValidationError(PatchValidationError):
    """Generated file content is incomplete, placeholder, or invalid."""

    def __init__(self, message: str, *, category: str = "semantic_validation_failed"):
        super().__init__(message)
        self.category = category


class FileWriteError(WorkerError):
    """Generated files could not be applied atomically."""


class CommandValidationError(WorkerError):
    """A validation command is not permitted."""


class ProposalError(WorkerError):
    """A saved proposal is missing, changed, stale, or unsafe to apply."""
