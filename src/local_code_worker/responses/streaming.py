import json
from collections.abc import Iterable, Iterator

from pydantic import Field

from ..models import ProviderName, StrictModel
from ..providers.base import (
    ProviderCompletedEvent,
    ProviderEvent,
    ProviderResult,
    ProviderTextDeltaEvent,
    ProviderUsageEvent,
)
from ..telemetry.models import TokenUsage
from .builder import build_response
from .schemas import ResponseObject, ResponseOutputMessage, ResponseOutputText


class ResponseStreamEvent(StrictModel):
    type: str = Field(min_length=1)
    sequence_number: int = Field(ge=0)
    response: ResponseObject | None = None
    output_index: int | None = Field(default=None, ge=0)
    content_index: int | None = Field(default=None, ge=0)
    item_id: str | None = None
    item: ResponseOutputMessage | None = None
    part: ResponseOutputText | None = None
    delta: str | None = None
    text: str | None = None


def map_provider_events(
    events: Iterable[ProviderEvent],
    *,
    provider: ProviderName,
    model: str,
    response_id: str,
    message_id: str,
    created_at: int,
) -> Iterator[ResponseStreamEvent]:
    sequence = 0
    text_parts: list[str] = []
    usage = TokenUsage()
    finish_reason: str | None = None
    created = ResponseObject(
        id=response_id,
        created_at=created_at,
        status="in_progress",
        model=model,
        output=[],
        output_text="",
    )
    yield ResponseStreamEvent(
        type="response.created",
        sequence_number=sequence,
        response=created,
    )
    sequence += 1
    output_item = ResponseOutputMessage(
        id=message_id,
        status="incomplete",
        content=[],
    )
    yield ResponseStreamEvent(
        type="response.output_item.added",
        sequence_number=sequence,
        output_index=0,
        item=output_item,
    )
    sequence += 1
    empty_part = ResponseOutputText(text="")
    yield ResponseStreamEvent(
        type="response.content_part.added",
        sequence_number=sequence,
        output_index=0,
        content_index=0,
        item_id=message_id,
        part=empty_part,
    )
    sequence += 1
    for event in events:
        if isinstance(event, ProviderTextDeltaEvent):
            text_parts.append(event.delta)
            yield ResponseStreamEvent(
                type="response.output_text.delta",
                sequence_number=sequence,
                output_index=0,
                content_index=0,
                item_id=message_id,
                delta=event.delta,
            )
            sequence += 1
        elif isinstance(event, ProviderUsageEvent):
            usage = event.usage
        elif isinstance(event, ProviderCompletedEvent):
            finish_reason = event.finish_reason
    text = "".join(text_parts)
    completed_part = ResponseOutputText(text=text)
    yield ResponseStreamEvent(
        type="response.output_text.done",
        sequence_number=sequence,
        output_index=0,
        content_index=0,
        item_id=message_id,
        text=text,
    )
    sequence += 1
    yield ResponseStreamEvent(
        type="response.content_part.done",
        sequence_number=sequence,
        output_index=0,
        content_index=0,
        item_id=message_id,
        part=completed_part,
    )
    sequence += 1
    completed_item = output_item.model_copy(
        update={"status": "completed", "content": [completed_part]}
    )
    yield ResponseStreamEvent(
        type="response.output_item.done",
        sequence_number=sequence,
        output_index=0,
        item=completed_item,
    )
    sequence += 1
    completed = build_response(
        ProviderResult(
            provider=provider,
            model=model,
            content=text,
            finish_reason=finish_reason,
            usage=usage,
            latency_ms=0,
        ),
        response_id=response_id,
        message_id=message_id,
        created_at=created_at,
    )
    yield ResponseStreamEvent(
        type="response.completed",
        sequence_number=sequence,
        response=completed,
    )


def encode_sse(event: ResponseStreamEvent) -> bytes:
    payload = json.dumps(event.model_dump(mode="json", exclude_none=True), separators=(",", ":"))
    return f"event: {event.type}\ndata: {payload}\n\n".encode()
