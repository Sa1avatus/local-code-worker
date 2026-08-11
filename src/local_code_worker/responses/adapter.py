from ..models import JsonMode
from ..providers.base import (
    ProviderFunctionTool,
    ProviderFunctionToolChoice,
    ProviderMessage,
    ProviderRequest,
)
from .schemas import ResponseAdditionalTools, ResponseCreateRequest, ResponseFunctionToolChoice
from .schemas import ResponseFunctionTool, ResponseNamespaceTool, ResponseInputMessage


def adapt_response_request(
    request: ResponseCreateRequest,
    *,
    max_output_characters: int,
    json_mode: JsonMode = JsonMode.NONE,
) -> ProviderRequest:
    messages: list[ProviderMessage] = []
    if request.instructions:
        messages.append(ProviderMessage(role="developer", content=request.instructions))
    additional_tool_dicts: list[dict[str, object]] = []
    if isinstance(request.input, str):
        messages.append(ProviderMessage(role="user", content=request.input))
    else:
        for item in request.input:
            if isinstance(item, ResponseAdditionalTools):
                additional_tool_dicts.extend(item.tools)
            elif isinstance(item, ResponseInputMessage):
                messages.append(
                    ProviderMessage(
                        role=item.role,
                        content=(
                            item.content
                            if isinstance(item.content, str)
                            else "\n".join(part.text for part in item.content)
                        ),
                    )
                )
    response_tools: list[ResponseFunctionTool] = []
    for tool_dict in additional_tool_dicts:
        if tool_dict.get("type") == "function" and "name" in tool_dict and "parameters" in tool_dict:
            response_tools.append(
                ResponseFunctionTool(
                    name=tool_dict["name"],
                    description=tool_dict.get("description"),
                    parameters=tool_dict["parameters"],
                    strict=tool_dict.get("strict", True),
                )
            )
        elif tool_dict.get("type") == "namespace" and "tools" in tool_dict:
            for sub_tool in tool_dict["tools"]:
                if isinstance(sub_tool, dict) and sub_tool.get("type") == "function" and "name" in sub_tool and "parameters" in sub_tool:
                    response_tools.append(
                        ResponseFunctionTool(
                            name=sub_tool["name"],
                            description=sub_tool.get("description"),
                            parameters=sub_tool["parameters"],
                            strict=sub_tool.get("strict", True),
                        )
                    )
    for tool in request.tools:
        if isinstance(tool, ResponseFunctionTool):
            response_tools.append(tool)
        elif isinstance(tool, ResponseNamespaceTool):
            response_tools.extend(tool.tools)
    tools = [
        ProviderFunctionTool(
            name=tool.name,
            description=tool.description,
            parameters=tool.parameters,
            strict=tool.strict,
        )
        for tool in response_tools
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
