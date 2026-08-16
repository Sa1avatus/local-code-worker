from local_code_worker.models import ProviderName
from local_code_worker.providers.base import ProviderFunctionCall, ProviderResult
from local_code_worker.responses.builder import build_response
from local_code_worker.telemetry.models import TokenUsage, UsageProvenance


def test_build_response_maps_provider_text_and_usage() -> None:
    result = ProviderResult(
        provider=ProviderName.OLLAMA,
        model="qwen:test",
        content="Hello",
        finish_reason="stop",
        usage=TokenUsage(
            input_tokens=10,
            output_tokens=4,
            cached_input_tokens=3,
            reasoning_tokens=2,
            provenance=UsageProvenance.EXACT,
        ),
        latency_ms=25,
    )

    response = build_response(
        result,
        response_id="resp_test",
        message_id="msg_test",
        created_at=1_786_233_600,
    )

    assert response.id == "resp_test"
    assert response.object == "response"
    assert response.status == "completed"
    assert response.output_text == "Hello"
    assert response.output[0].model_dump() == {
        "id": "msg_test",
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": "Hello",
                "annotations": [],
            }
        ],
        "reasoning": None,
    }
    assert response.usage is not None
    assert response.usage.model_dump() == {
        "input_tokens": 10,
        "input_tokens_details": {"cached_tokens": 3},
        "output_tokens": 4,
        "output_tokens_details": {"reasoning_tokens": 2},
        "total_tokens": 14,
    }


def test_build_response_forwards_reasoning() -> None:
    result = ProviderResult(
        provider=ProviderName.OLLAMA,
        model="qwen:test",
        content="four",
        reasoning="The user asked for two plus two.",
        finish_reason="stop",
        usage=TokenUsage(input_tokens=10, output_tokens=4, provenance=UsageProvenance.EXACT),
        latency_ms=25,
    )

    response = build_response(result, message_id="msg_test")

    assert response.output[0].reasoning == "The user asked for two plus two."


def test_build_response_maps_function_call_output() -> None:
    result = ProviderResult(
        provider=ProviderName.OPENAI_COMPATIBLE,
        model="tool-model",
        content="",
        function_calls=[
            ProviderFunctionCall(
                call_id="call_test",
                name="lookup",
                arguments='{"key":"value"}',
            )
        ],
        latency_ms=10,
    )

    response = build_response(
        result,
        response_id="resp_test",
        function_call_item_ids=["fc_test"],
        created_at=1,
    )

    assert response.output_text == ""
    assert len(response.output) == 1
    assert response.output[0].model_dump() == {
        "id": "fc_test",
        "type": "function_call",
        "status": "completed",
        "call_id": "call_test",
        "name": "lookup",
        "arguments": '{"key":"value"}',
    }
