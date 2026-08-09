import pytest
from pydantic import ValidationError

from local_code_worker.config import WorkerSettings
from local_code_worker.exceptions import ProviderConfigurationError
from local_code_worker.models import (
    FunctionCallMetadata,
    GenerationMetadata,
    JsonMode,
    ProviderName,
)
from local_code_worker.providers.adapter import CanonicalProviderAdapter
from local_code_worker.providers.base import (
    ProviderCapabilities,
    ProviderCapability,
    ProviderCompletedEvent,
    ProviderMessage,
    ProviderRequest,
    ProviderResult,
    ProviderStartedEvent,
    ProviderTextDeltaEvent,
    ProviderUsageEvent,
    validate_event_sequence,
)
from local_code_worker.providers.registry import DEFAULT_PROVIDER_REGISTRY
from local_code_worker.telemetry.models import TokenUsage, UsageProvenance


class FakeLegacyProvider:
    capabilities = ProviderCapabilities(
        supported=frozenset(
            {
                ProviderCapability.STREAMING,
                ProviderCapability.JSON_SCHEMA,
                ProviderCapability.USAGE,
            }
        )
    )

    def __init__(self) -> None:
        self.settings = WorkerSettings(
            _env_file=None,
            llm_model="qwen:test",
            llm_stream=False,
            llm_json_mode=JsonMode.JSON_SCHEMA,
        )
        self.last_generation_metadata: GenerationMetadata | None = None

    def chat(
        self,
        messages,
        response_schema,
        max_output_characters,
        max_output_tokens=None,
        tools=None,
        tool_choice="auto",
    ):
        self.last_generation_metadata = GenerationMetadata(
            provider=ProviderName.OLLAMA,
            model="qwen:test",
            base_url="http://localhost:11434",
            started_at="2026-08-09T00:00:00Z",
            completed_at="2026-08-09T00:00:00Z",
            duration_seconds=0.025,
            prompt_characters=5,
            output_characters=4,
            streaming=False,
            response_format_mode=JsonMode.JSON_SCHEMA,
            finish_reason="stop",
            usage={"prompt_tokens": 4, "completion_tokens": 2},
            function_calls=(
                [
                    FunctionCallMetadata(
                        call_id="call_test",
                        name=tools[0].name,
                        arguments="{}",
                    )
                ]
                if tools
                else []
            ),
        )
        return "" if tools else "done"

    def check_connection(self):
        raise NotImplementedError

    def list_models(self):
        return ["qwen:test"]


def test_provider_request_preserves_generation_constraints() -> None:
    request = ProviderRequest(
        messages=[ProviderMessage(role="user", content="hello")],
        response_schema={"type": "object"},
        max_output_characters=100,
        max_output_tokens=20,
        json_mode=JsonMode.JSON_SCHEMA,
        stream=True,
    )

    assert request.messages[0].content == "hello"
    assert request.response_schema == {"type": "object"}
    assert request.max_output_tokens == 20
    assert request.stream is True


def test_provider_request_rejects_non_positive_output_limits() -> None:
    with pytest.raises(ValidationError):
        ProviderRequest(messages=[], max_output_characters=0)


def test_provider_result_carries_typed_usage_and_latency() -> None:
    result = ProviderResult(
        provider=ProviderName.OLLAMA,
        model="qwen:test",
        content="done",
        finish_reason="stop",
        usage=TokenUsage(
            input_tokens=4,
            output_tokens=2,
            provenance=UsageProvenance.EXACT,
        ),
        latency_ms=25,
        time_to_first_token_ms=5,
    )

    assert result.usage.total_tokens == 6
    assert result.time_to_first_token_ms == 5


def test_provider_capabilities_are_explicit() -> None:
    capabilities = ProviderCapabilities(
        supported=frozenset(
            {
                ProviderCapability.STREAMING,
                ProviderCapability.USAGE,
            }
        )
    )

    assert capabilities.supports(ProviderCapability.STREAMING) is True
    assert capabilities.supports(ProviderCapability.JSON_SCHEMA) is False


def test_canonical_adapter_returns_typed_legacy_result() -> None:
    adapter = CanonicalProviderAdapter(FakeLegacyProvider())
    request = ProviderRequest(
        messages=[ProviderMessage(role="user", content="hello")],
        response_schema={"type": "object"},
        max_output_characters=100,
        max_output_tokens=20,
        json_mode=JsonMode.JSON_SCHEMA,
        stream=False,
    )

    result = adapter.complete(request)

    assert result.content == "done"
    assert result.finish_reason == "stop"
    assert result.usage.total_tokens == 6
    assert result.usage.provenance is UsageProvenance.EXACT
    assert result.latency_ms == 25


def test_canonical_adapter_rejects_unapplied_request_modes() -> None:
    adapter = CanonicalProviderAdapter(FakeLegacyProvider())
    request = ProviderRequest(
        messages=[],
        max_output_characters=100,
        json_mode=JsonMode.NONE,
        stream=False,
    )

    with pytest.raises(ProviderConfigurationError, match="json_mode"):
        adapter.complete(request)


def test_provider_event_sequence_accepts_ordered_stream() -> None:
    events = [
        ProviderStartedEvent(
            sequence=0,
            provider=ProviderName.OLLAMA,
            model="qwen:test",
        ),
        ProviderTextDeltaEvent(sequence=1, delta="hel"),
        ProviderTextDeltaEvent(sequence=2, delta="lo"),
        ProviderUsageEvent(
            sequence=3,
            usage=TokenUsage(input_tokens=2, output_tokens=1),
        ),
        ProviderCompletedEvent(sequence=4, finish_reason="stop"),
    ]

    validate_event_sequence(events)


@pytest.mark.parametrize(
    "events",
    [
        [ProviderCompletedEvent(sequence=0)],
        [
            ProviderStartedEvent(
                sequence=0,
                provider=ProviderName.OLLAMA,
                model="qwen:test",
            ),
            ProviderTextDeltaEvent(sequence=2, delta="gap"),
            ProviderCompletedEvent(sequence=3),
        ],
        [
            ProviderStartedEvent(
                sequence=0,
                provider=ProviderName.OLLAMA,
                model="qwen:test",
            ),
            ProviderStartedEvent(
                sequence=1,
                provider=ProviderName.OLLAMA,
                model="qwen:test",
            ),
            ProviderCompletedEvent(sequence=2),
        ],
    ],
)
def test_provider_event_sequence_rejects_invalid_order(events) -> None:
    with pytest.raises(ValueError):
        validate_event_sequence(events)


def test_registry_describes_configured_model_without_model_name_table() -> None:
    local = DEFAULT_PROVIDER_REGISTRY.configured_model(
        WorkerSettings(_env_file=None, llm_model="custom-local:model")
    )
    cloud = DEFAULT_PROVIDER_REGISTRY.configured_model(
        WorkerSettings(
            _env_file=None,
            llm_provider=ProviderName.OPENAI_COMPATIBLE,
            llm_base_url="https://example.test/v1",
            llm_model="custom/cloud-model",
        )
    )

    assert local.model == "custom-local:model"
    assert local.is_local is True
    assert cloud.model == "custom/cloud-model"
    assert cloud.is_local is False
    assert cloud.capabilities.supports(ProviderCapability.STREAMING)
