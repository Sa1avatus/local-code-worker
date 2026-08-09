from ..models import JsonMode
from ..providers.base import (
    ProviderFunctionTool,
    ProviderFunctionToolChoice,
    ProviderMessage,
    ProviderRequest,
)
from .schemas import ResponseCreateRequest, ResponseFunctionToolChoice


def adapt_response_request(
    request: ResponseCreateRequest,
    *,
    max_output_characters: int,
    json_mode: JsonMode = JsonMode.NONE,
) -> ProviderRequest:
    messages: list[ProviderMessage] = []
    if request.instructions:
        messages.append(ProviderMessage(role="developer", content=request.instructions))
    if isinstance(request.input, str):
        messages.append(ProviderMessage(role="user", content=request.input))
    else:
        messages.extend(
            ProviderMessage(role=message.role, content=message.content) for message in request.input
        )
    tools = [
        ProviderFunctionTool(
            name=tool.name,
            description=tool.description,
            parameters=tool.parameters,
            strict=tool.strict,
        )
        for tool in request.tools
    ]
    tool_choice = (
        ProviderFunctionToolChoice(name=request.tool_choice.name)
        if isinstance(request.tool_choice, ResponseFunctionToolChoice)
        else request.tool_choice
    )
    return ProviderRequest(
        messages=messages,
        max_output_characters=max_output_characters,
        max_output_tokens=request.max_output_tokens,
        json_mode=json_mode,
        stream=request.stream,
        tools=tools,
        tool_choice=tool_choice,
        reasoning_effort=request.reasoning.effort if request.reasoning else None,
    )
