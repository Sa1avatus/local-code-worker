from ..exceptions import ProviderConfigurationError, ProviderError
from ..telemetry.models import TokenUsage, UsageProvenance
from .base import (
    LlmProvider,
    ProviderCapability,
    ProviderFunctionCall,
    ProviderRequest,
    ProviderResult,
)


class CanonicalProviderAdapter:
    def __init__(self, provider: LlmProvider) -> None:
        self.provider = provider

    @property
    def capabilities(self):
        return self.provider.capabilities

    def complete(self, request: ProviderRequest) -> ProviderResult:
        settings = self.provider.settings
        if request.stream != settings.llm_stream:
            raise ProviderConfigurationError(
                "Canonical request stream must match configured provider stream mode"
            )
        if request.json_mode is not settings.llm_json_mode:
            raise ProviderConfigurationError(
                "Canonical request json_mode must match configured provider json mode"
            )
        if request.stream and not self.capabilities.supports(ProviderCapability.STREAMING):
            raise ProviderConfigurationError("Provider does not support streaming")
        if request.response_schema is not None and not self.capabilities.supports(
            ProviderCapability.JSON_SCHEMA
        ):
            raise ProviderConfigurationError("Provider does not support JSON Schema")

        arguments = (
            [message.model_dump() for message in request.messages],
            request.response_schema,
            request.max_output_characters,
            request.max_output_tokens or settings.llm_max_output_tokens,
        )
        content = (
            self.provider.chat(
                *arguments,
                tools=request.tools,
                tool_choice=request.tool_choice,
            )
            if request.tools
            else self.provider.chat(*arguments)
        )
        metadata = self.provider.last_generation_metadata
        if metadata is None:
            raise ProviderError(
                "Provider completed without generation metadata",
                category="missing_metadata",
            )
        usage = metadata.usage
        has_provider_usage = "prompt_tokens" in usage or "completion_tokens" in usage
        return ProviderResult(
            provider=metadata.provider,
            model=metadata.model,
            content=content,
            finish_reason=metadata.finish_reason,
            function_calls=[
                ProviderFunctionCall(
                    call_id=function_call.call_id,
                    name=function_call.name,
                    arguments=function_call.arguments,
                )
                for function_call in metadata.function_calls
            ],
            usage=TokenUsage(
                input_tokens=int(usage.get("prompt_tokens", 0)),
                output_tokens=int(usage.get("completion_tokens", 0)),
                cached_input_tokens=int(usage.get("cached_tokens", 0)),
                reasoning_tokens=int(usage.get("reasoning_tokens", 0)),
                provenance=(
                    UsageProvenance.EXACT if has_provider_usage else UsageProvenance.UNAVAILABLE
                ),
            ),
            latency_ms=metadata.duration_seconds * 1000,
        )
