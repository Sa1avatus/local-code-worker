from collections.abc import Iterator
from enum import StrEnum
from typing import Literal, Protocol, TypeAlias

from pydantic import Field

from ..config import WorkerSettings
from ..models import GenerationMetadata, JsonMode, ProviderHealth, ProviderName, StrictModel
from ..telemetry.models import TokenUsage


class ProviderCapability(StrEnum):
    STREAMING = "streaming"
    JSON_OBJECT = "json_object"
    JSON_SCHEMA = "json_schema"
    USAGE = "usage"
    FUNCTION_TOOLS = "function_tools"


class ProviderCapabilities(StrictModel):
    supported: frozenset[ProviderCapability] = Field(default_factory=frozenset)

    def supports(self, capability: ProviderCapability) -> bool:
        return capability in self.supported


class ProviderMessage(StrictModel):
    role: str = Field(min_length=1)
    content: str
    tool_calls: list[dict[str, object]] | None = Field(default=None, exclude=True)
    tool_call_id: str | None = Field(default=None, exclude=True)

    def model_dump(self, **kwargs: object) -> dict[str, object]:
        base = super().model_dump(**kwargs)
        if self.tool_calls is not None:
            base["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            base["tool_call_id"] = self.tool_call_id
        return base


class ProviderFunctionTool(StrictModel):
    name: str = Field(min_length=1)
    description: str | None = None
    parameters: dict[str, object]
    strict: bool = True


class ProviderFunctionToolChoice(StrictModel):
    name: str = Field(min_length=1)


class ProviderFunctionCall(StrictModel):
    call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: str


class ProviderRequest(StrictModel):
    messages: list[ProviderMessage]
    response_schema: dict[str, object] | None = None
    max_output_characters: int = Field(gt=0)
    max_output_tokens: int | None = Field(default=None, gt=0)
    json_mode: JsonMode = JsonMode.NONE
    stream: bool = False
    tools: list[ProviderFunctionTool] = Field(default_factory=list)
    tool_choice: Literal["none", "auto", "required"] | ProviderFunctionToolChoice = "auto"
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"] | None = None


class ProviderResult(StrictModel):
    provider: ProviderName
    model: str = Field(min_length=1)
    content: str
    finish_reason: str | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)
    function_calls: list[ProviderFunctionCall] = Field(default_factory=list)
    latency_ms: float = Field(ge=0)
    time_to_first_token_ms: float | None = Field(default=None, ge=0)


class ProviderStartedEvent(StrictModel):
    type: Literal["started"] = "started"
    sequence: int = Field(ge=0)
    provider: ProviderName
    model: str = Field(min_length=1)


class ProviderTextDeltaEvent(StrictModel):
    type: Literal["text_delta"] = "text_delta"
    sequence: int = Field(ge=0)
    delta: str = Field(min_length=1)


class ProviderUsageEvent(StrictModel):
    type: Literal["usage"] = "usage"
    sequence: int = Field(ge=0)
    usage: TokenUsage


class ProviderCompletedEvent(StrictModel):
    type: Literal["completed"] = "completed"
    sequence: int = Field(ge=0)
    finish_reason: str | None = None


class ProviderToolCallsEvent(StrictModel):
    """Emitted when the model requests tool calls during streaming."""

    type: Literal["tool_calls"] = "tool_calls"
    sequence: int = Field(ge=0)
    function_calls: list[ProviderFunctionCall] = Field(default_factory=list)


ProviderEvent: TypeAlias = (
    ProviderStartedEvent
    | ProviderTextDeltaEvent
    | ProviderToolCallsEvent
    | ProviderUsageEvent
    | ProviderCompletedEvent
)


def validate_event_sequence(events: list[ProviderEvent]) -> None:
    if not events or not isinstance(events[0], ProviderStartedEvent):
        raise ValueError("provider event sequence must start with a started event")
    if not isinstance(events[-1], ProviderCompletedEvent):
        raise ValueError("provider event sequence must end with a completed event")
    for expected_sequence, event in enumerate(events):
        if event.sequence != expected_sequence:
            raise ValueError("provider event sequence numbers must be contiguous")
    if sum(isinstance(event, ProviderStartedEvent) for event in events) != 1:
        raise ValueError("provider event sequence must contain exactly one started event")
    if sum(isinstance(event, ProviderCompletedEvent) for event in events) != 1:
        raise ValueError("provider event sequence must contain exactly one completed event")


class LlmProvider(Protocol):
    settings: WorkerSettings
    last_generation_metadata: GenerationMetadata | None
    capabilities: ProviderCapabilities

    def check_connection(self) -> ProviderHealth: ...

    def list_models(self) -> list[str]: ...

    def stream(self, request: ProviderRequest) -> Iterator[ProviderEvent]: ...

    def chat(
        self,
        messages: list[dict[str, str]],
        response_schema: dict[str, object] | None,
        max_output_characters: int,
        max_output_tokens: int | None = None,
        tools: list[ProviderFunctionTool] | None = None,
        tool_choice: Literal["none", "auto", "required"] | ProviderFunctionToolChoice = "auto",
    ) -> str: ...
