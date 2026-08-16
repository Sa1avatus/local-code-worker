import time
from uuid import uuid4

from ..providers.base import ProviderResult
from .schemas import (
    ResponseFunctionCall,
    ResponseInputTokensDetails,
    ResponseObject,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseOutputTokensDetails,
    ResponseUsage,
)


def build_response(
    result: ProviderResult,
    *,
    response_id: str | None = None,
    message_id: str | None = None,
    function_call_item_ids: list[str] | None = None,
    created_at: int | None = None,
    model: str | None = None,
) -> ResponseObject:
    usage = result.usage
    output: list[ResponseOutputMessage | ResponseFunctionCall] = []
    if result.content or not result.function_calls:
        output.append(
            ResponseOutputMessage(
                id=message_id or f"msg_{uuid4().hex}",
                status="completed",
                content=[ResponseOutputText(text=result.content)],
                reasoning=result.reasoning,
            )
        )
    item_ids = function_call_item_ids or []
    output.extend(
        ResponseFunctionCall(
            id=item_ids[index] if index < len(item_ids) else f"fc_{uuid4().hex}",
            status="completed",
            call_id=function_call.call_id,
            name=function_call.name,
            arguments=function_call.arguments,
        )
        for index, function_call in enumerate(result.function_calls)
    )
    return ResponseObject(
        id=response_id or f"resp_{uuid4().hex}",
        created_at=created_at if created_at is not None else int(time.time()),
        status="completed",
        model=model or result.model,
        output=output,
        output_text=result.content,
        usage=ResponseUsage(
            input_tokens=usage.input_tokens,
            input_tokens_details=ResponseInputTokensDetails(
                cached_tokens=usage.cached_input_tokens
            ),
            output_tokens=usage.output_tokens,
            output_tokens_details=ResponseOutputTokensDetails(
                reasoning_tokens=usage.reasoning_tokens
            ),
            total_tokens=usage.total_tokens,
        ),
    )
