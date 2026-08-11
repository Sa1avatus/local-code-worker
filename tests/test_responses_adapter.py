from test_provider_contracts import FakeLegacyProvider

from local_code_worker.config import WorkerSettings
from local_code_worker.models import JsonMode
from local_code_worker.providers.adapter import CanonicalProviderAdapter
from local_code_worker.responses.adapter import adapt_response_request
from local_code_worker.responses.schemas import ResponseCreateRequest


def test_response_adapter_preserves_instructions_messages_tools_and_reasoning() -> None:
    response_request = ResponseCreateRequest.model_validate(
        {
            "model": "local-balanced",
            "instructions": "Follow project conventions.",
            "input": [{"role": "user", "content": "Fix the test."}],
            "tools": [
                {
                    "type": "function",
                    "name": "read_file",
                    "parameters": {"type": "object"},
                }
            ],
            "tool_choice": {"type": "function", "name": "read_file"},
            "reasoning": {"effort": "high"},
            "max_output_tokens": 500,
            "stream": True,
        }
    )

    adapted = adapt_response_request(
        response_request,
        max_output_characters=4_000,
        json_mode=JsonMode.JSON_OBJECT,
    )
    provider_request = adapted.request

    assert [(message.role, message.content) for message in provider_request.messages] == [
        ("developer", "Follow project conventions."),
        ("user", "Fix the test."),
    ]
    assert provider_request.tools[0].name == "read_file"
    assert provider_request.tool_choice.name == "read_file"
    assert provider_request.reasoning_effort == "high"
    assert provider_request.stream is True


def test_canonical_adapter_returns_normalized_function_call() -> None:
    response_request = ResponseCreateRequest.model_validate(
        {
            "model": "local",
            "input": "hello",
            "tools": [
                {
                    "type": "function",
                    "name": "lookup",
                    "parameters": {"type": "object"},
                }
            ],
        }
    )
    adapted = adapt_response_request(
        response_request,
        max_output_characters=100,
        json_mode=JsonMode.JSON_SCHEMA,
    )
    provider = FakeLegacyProvider()
    provider.settings = WorkerSettings(
        _env_file=None,
        llm_stream=False,
        llm_json_mode=JsonMode.JSON_SCHEMA,
    )

    result = CanonicalProviderAdapter(provider).complete(adapted.request)

    assert result.content == ""
    assert result.function_calls[0].model_dump() == {
        "call_id": "call_test",
        "name": "lookup",
        "arguments": "{}",
    }
