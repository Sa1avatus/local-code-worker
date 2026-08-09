from typing import Literal

from pydantic import Field

from ..models import StrictModel


class ResponseInputMessage(StrictModel):
    type: Literal["message"] = "message"
    role: Literal["user", "assistant", "system", "developer"]
    content: str


class ResponseFunctionTool(StrictModel):
    type: Literal["function"] = "function"
    name: str = Field(min_length=1)
    description: str | None = None
    parameters: dict[str, object]
    strict: bool = True


class ResponseFunctionToolChoice(StrictModel):
    type: Literal["function"] = "function"
    name: str = Field(min_length=1)


class ResponseReasoning(StrictModel):
    effort: Literal["none", "low", "medium", "high", "xhigh", "max"] | None = None
    summary: Literal["auto", "concise", "detailed"] | None = None


class ResponseCreateRequest(StrictModel):
    model: str = Field(min_length=1)
    input: str | list[ResponseInputMessage]
    instructions: str | None = None
    tools: list[ResponseFunctionTool] = Field(default_factory=list)
    tool_choice: Literal["none", "auto", "required"] | ResponseFunctionToolChoice = "auto"
    reasoning: ResponseReasoning | None = None
    max_output_tokens: int | None = Field(default=None, gt=0)
    parallel_tool_calls: bool = True
    previous_response_id: str | None = None
    stream: bool = False
    store: bool = False


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
