from typing import Literal

from pydantic import Field

from ..models import StrictModel


class ResponseAdditionalTools(StrictModel):
    type: Literal["additional_tools"] = "additional_tools"
    tools: list[dict[str, object]] = Field(default_factory=list)
    role: str | None = None


class ResponseInputMessage(StrictModel):
    type: Literal["message"] = "message"
    id: str | None = None
    role: Literal["user", "assistant", "system", "developer"]
    content: str | list["ResponseInputText"]


class ResponseInputText(StrictModel):
    type: Literal["input_text", "output_text"] = "input_text"
    text: str

class ResponseInputFunctionCall(StrictModel):
    """A function call in the input (from previous assistant turn)."""

    type: Literal["function_call"] = "function_call"
    id: str | None = None
    call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: str


class ResponseInputFunctionCallOutput(StrictModel):
    """A function call result in the input (tool output from client)."""

    type: Literal["function_call_output"] = "function_call_output"
    call_id: str = Field(min_length=1)
    output: str


class ResponseFunctionTool(StrictModel):
    type: Literal["function"] = "function"
    name: str = Field(min_length=1)
    description: str | None = None
    parameters: dict[str, object]
    strict: bool = True


class ResponseNamespaceTool(StrictModel):
    type: Literal["namespace"] = "namespace"
    name: str = Field(min_length=1)
    description: str | None = None
    tools: list[ResponseFunctionTool]


class ResponseWebSearchTool(StrictModel):
    type: Literal["web_search"] = "web_search"
    external_web_access: bool = False


class ResponseFunctionToolChoice(StrictModel):
    type: Literal["function"] = "function"
    name: str = Field(min_length=1)


class ResponseReasoning(StrictModel):
    effort: Literal["none", "low", "medium", "high", "xhigh", "max"] | None = None
    summary: Literal["auto", "concise", "detailed"] | None = None
    context: str | None = None


class ResponseTextConfig(StrictModel):
    format: dict[str, object] | None = None
    verbosity: str | None = None


class ResponseTruncationConfig(StrictModel):
    type: Literal["auto", "disabled"] = "auto"


class ResponseCreateRequest(StrictModel):
    model: str = Field(min_length=1)
    input: (
        str
        | list[
            ResponseInputMessage
            | ResponseAdditionalTools
            | ResponseInputFunctionCall
            | ResponseInputFunctionCallOutput
        ]
    )
    instructions: str | None = None
    tools: list[ResponseFunctionTool | ResponseNamespaceTool | ResponseWebSearchTool] = Field(
        default_factory=list
    )
    tool_choice: Literal["none", "auto", "required"] | ResponseFunctionToolChoice = "auto"
    reasoning: ResponseReasoning | None = None
    max_output_tokens: int | None = Field(default=None, gt=0)
    parallel_tool_calls: bool = True
    previous_response_id: str | None = None
    stream: bool = False
    store: bool = False
    include: list[str] = Field(default_factory=list)
    prompt_cache_key: str | None = None
    client_metadata: dict[str, object] = Field(default_factory=dict)
    text: ResponseTextConfig | None = None
    temperature: float | None = None
    truncation: ResponseTruncationConfig | None = None
    metadata: dict[str, object] | None = None


class ResponseErrorDetail(StrictModel):
    message: str = Field(min_length=1)
    type: str = Field(min_length=1)
    param: str | None = None
    code: str | None = None


class ResponseError(StrictModel):
    error: ResponseErrorDetail


class ResponseOutputText(StrictModel):
    type: Literal["output_text"] = "output_text"
    text: str
    annotations: list[dict[str, object]] = Field(default_factory=list)


class ResponseOutputMessage(StrictModel):
    id: str = Field(min_length=1)
    type: Literal["message"] = "message"
    status: Literal["completed", "incomplete"]
    role: Literal["assistant"] = "assistant"
    content: list[ResponseOutputText]


class ResponseFunctionCall(StrictModel):
    id: str = Field(min_length=1)
    type: Literal["function_call"] = "function_call"
    status: Literal["completed", "incomplete"]
    call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: str


class ResponseInputTokensDetails(StrictModel):
    cached_tokens: int = Field(default=0, ge=0)


class ResponseOutputTokensDetails(StrictModel):
    reasoning_tokens: int = Field(default=0, ge=0)


class ResponseUsage(StrictModel):
    input_tokens: int = Field(ge=0)
    input_tokens_details: ResponseInputTokensDetails
    output_tokens: int = Field(ge=0)
    output_tokens_details: ResponseOutputTokensDetails
    total_tokens: int = Field(ge=0)


class ResponseObject(StrictModel):
    id: str = Field(min_length=1)
    object: Literal["response"] = "response"
    created_at: int = Field(ge=0)
    status: Literal["completed", "incomplete", "failed", "in_progress"]
    model: str = Field(min_length=1)
    output: list[ResponseOutputMessage | ResponseFunctionCall]
    output_text: str
    usage: ResponseUsage | None = None
    error: ResponseErrorDetail | None = None
    incomplete_details: dict[str, str | int | float | bool | None] | None = None

