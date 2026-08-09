import json

from local_code_worker.models import ProviderName
from local_code_worker.providers.base import (
    ProviderCompletedEvent,
    ProviderStartedEvent,
    ProviderTextDeltaEvent,
    ProviderUsageEvent,
)
from local_code_worker.responses.streaming import encode_sse, map_provider_events
from local_code_worker.telemetry.models import TokenUsage, UsageProvenance


def test_provider_events_map_to_ordered_responses_text_lifecycle() -> None:
    provider_events = [
        ProviderStartedEvent(
            sequence=0,
            provider=ProviderName.OLLAMA,
            model="qwen:test",
        ),
        ProviderTextDeltaEvent(sequence=1, delta="hel"),
        ProviderTextDeltaEvent(sequence=2, delta="lo"),
        ProviderUsageEvent(
            sequence=3,
            usage=TokenUsage(
                input_tokens=2,
                output_tokens=1,
                provenance=UsageProvenance.EXACT,
            ),
        ),
        ProviderCompletedEvent(sequence=4, finish_reason="stop"),
    ]

    events = list(
        map_provider_events(
            provider_events,
            provider=ProviderName.OLLAMA,
            model="qwen:test",
            response_id="resp_test",
            message_id="msg_test",
            created_at=1_786_233_600,
        )
    )

    assert [event.sequence_number for event in events] == list(range(len(events)))
    assert [event.type for event in events] == [
        "response.created",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ]
    assert events[-1].response is not None
    assert events[-1].response.output_text == "hello"
    assert events[-1].response.usage is not None
    assert events[-1].response.usage.total_tokens == 3


def test_sse_encoding_uses_event_name_and_compact_json() -> None:
    event = next(
        map_provider_events(
            [],
            provider=ProviderName.OLLAMA,
            model="qwen:test",
            response_id="resp_test",
            message_id="msg_test",
            created_at=1,
        )
    )

    encoded = encode_sse(event).decode()
    lines = encoded.splitlines()
    assert lines[0] == "event: response.created"
    assert json.loads(lines[1].removeprefix("data: "))["sequence_number"] == 0
