import pytest
from pydantic import ValidationError

from local_code_worker.responses.schemas import ResponseCreateRequest


def test_response_request_accepts_text_and_function_subset() -> None:
    request = ResponseCreateRequest.model_validate(
        {
            "model": "local-balanced",
            "instructions": "Return a concise answer.",
            "input": [
                {"type": "message", "role": "user", "content": "Hello"},
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "lookup",
                    "description": "Look up a value.",
                    "parameters": {
                        "type": "object",
                        "properties": {"key": {"type": "string"}},
                        "required": ["key"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                }
            ],
            "tool_choice": {"type": "function", "name": "lookup"},
            "reasoning": {"effort": "medium", "summary": "auto"},
            "max_output_tokens": 200,
            "stream": True,
            "store": False,
        }
    )

    assert request.input[0].content == "Hello"
    assert request.tools[0].name == "lookup"
    assert request.max_output_tokens == 200


@pytest.mark.parametrize(
    "payload",
    [
        {"model": "local", "input": [], "temperature": 0.2},
        {
            "model": "local",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_image", "image_url": "https://example.test"}],
                }
            ],
        },
        {"model": "local", "input": "hello", "max_output_tokens": 0},
        {
            "model": "local",
            "input": "hello",
            "tools": [{"type": "web_search"}],
        },
    ],
)
def test_response_request_rejects_unsupported_or_invalid_fields(payload) -> None:
    with pytest.raises(ValidationError):
        ResponseCreateRequest.model_validate(payload)
