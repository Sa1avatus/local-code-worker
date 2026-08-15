"""Tests for tool models, normalizer, executor, and search providers."""

from __future__ import annotations

from unittest.mock import MagicMock

from local_code_worker.responses.adapter import adapt_response_request
from local_code_worker.responses.schemas import ResponseCreateRequest
from local_code_worker.tools.executor import ToolExecutor
from local_code_worker.tools.models import (
    HOSTED_TOOL_SCHEMAS,
    NormalizedTool,
    ToolKind,
    hosted_tool_description,
)
from local_code_worker.tools.normalizer import (
    normalize_request_tools,
    normalize_tool_dict,
    separate_tools,
)
from local_code_worker.tools.search.base import FetchResponse, SearchResponse, SearchResult

# ── ToolKind ──────────────────────────────────────────────────────────────────

def test_tool_kind_values() -> None:
    assert ToolKind.FUNCTION == "function"
    assert ToolKind.WEB_SEARCH == "web_search"
    assert ToolKind.WEB_FETCH == "web_fetch"
    assert ToolKind.UNKNOWN == "unknown"


# ── NormalizedTool ────────────────────────────────────────────────────────────

def test_normalized_tool_is_hosted() -> None:
    tool = NormalizedTool(kind=ToolKind.WEB_SEARCH, name="web_search")
    assert tool.is_hosted is True
    assert tool.is_function is False
    assert tool.is_passthrough is False


def test_normalized_tool_is_function() -> None:
    tool = NormalizedTool(kind=ToolKind.FUNCTION, name="shell_command")
    assert tool.is_hosted is False
    assert tool.is_function is True
    assert tool.is_passthrough is True


def test_hosted_tool_schemas_exist() -> None:
    for kind in (ToolKind.WEB_SEARCH, ToolKind.WEB_FETCH, ToolKind.GITHUB_SEARCH):
        assert kind in HOSTED_TOOL_SCHEMAS
        schema = HOSTED_TOOL_SCHEMAS[kind]
        assert "type" in schema
        assert schema["type"] == "object"


def test_hosted_tool_descriptions() -> None:
    for kind in (ToolKind.WEB_SEARCH, ToolKind.WEB_FETCH):
        desc = hosted_tool_description(kind)
        assert len(desc) > 20


# ── Normalizer ────────────────────────────────────────────────────────────────

def test_normalize_function_tool_dict() -> None:
    raw = {"type": "function", "name": "shell_command", "parameters": {"type": "object"}}
    tools = normalize_tool_dict(raw)
    assert len(tools) == 1
    assert tools[0].kind == ToolKind.FUNCTION
    assert tools[0].name == "shell_command"


def test_normalize_web_search_tool_dict() -> None:
    raw = {"type": "web_search", "external_web_access": False}
    tools = normalize_tool_dict(raw)
    assert len(tools) == 1
    assert tools[0].kind == ToolKind.WEB_SEARCH
    assert tools[0].is_hosted is True


def test_normalize_namespace_tool_dict() -> None:
    raw = {
        "type": "namespace",
        "name": "mcp__github",
        "tools": [
            {"type": "function", "name": "search", "parameters": {"type": "object"}},
            {"type": "function", "name": "fetch_file", "parameters": {"type": "object"}},
        ],
    }
    tools = normalize_tool_dict(raw)
    assert len(tools) == 2
    assert all(t.kind == ToolKind.FUNCTION for t in tools)
    assert tools[0].name == "search"
    assert tools[1].name == "fetch_file"


def test_normalize_unknown_tool_dict() -> None:
    raw = {"type": "custom_thing", "name": "mystery"}
    tools = normalize_tool_dict(raw)
    assert len(tools) == 1
    assert tools[0].kind == ToolKind.UNKNOWN
    assert tools[0].name == "mystery"


def test_normalize_request_tools_from_top_level() -> None:
    request = ResponseCreateRequest.model_validate({
        "model": "test",
        "input": "hello",
        "tools": [
            {"type": "function", "name": "shell_command", "parameters": {"type": "object"}},
            {"type": "web_search", "external_web_access": False},
        ],
    })
    tools = normalize_request_tools(request)
    assert len(tools) == 2
    kinds = {t.kind for t in tools}
    assert ToolKind.FUNCTION in kinds
    assert ToolKind.WEB_SEARCH in kinds


def test_normalize_request_tools_from_additional_tools() -> None:
    request = ResponseCreateRequest.model_validate({
        "model": "test",
        "input": [
            {
                "type": "additional_tools",
                "tools": [
                    {"type": "function", "name": "apply_patch", "parameters": {"type": "object"}},
                    {"type": "web_search"},
                ],
            },
            {"type": "message", "role": "user", "content": "hi"},
        ],
    })
    tools = normalize_request_tools(request)
    assert len(tools) == 2
    names = {t.name for t in tools}
    assert "apply_patch" in names
    assert "web_search" in names


def test_separate_tools() -> None:
    tools = [
        NormalizedTool(kind=ToolKind.FUNCTION, name="shell_command"),
        NormalizedTool(kind=ToolKind.WEB_SEARCH, name="web_search"),
        NormalizedTool(kind=ToolKind.FUNCTION, name="apply_patch"),
    ]
    hosted, passthrough = separate_tools(tools)
    assert len(hosted) == 1
    assert hosted[0].name == "web_search"
    assert len(passthrough) == 2
    assert {t.name for t in passthrough} == {"shell_command", "apply_patch"}


# ── Adapter with hosted tools ────────────────────────────────────────────────

def test_adapter_adds_web_search_as_function_tool() -> None:
    request = ResponseCreateRequest.model_validate({
        "model": "test",
        "input": "hello",
        "tools": [
            {"type": "function", "name": "shell_command", "parameters": {"type": "object"}},
            {"type": "web_search", "external_web_access": False},
        ],
    })
    from local_code_worker.models import JsonMode
    adapted = adapt_response_request(request, max_output_characters=4000, json_mode=JsonMode.NONE)

    assert "web_search" in adapted.hosted_tool_names
    assert "shell_command" not in adapted.hosted_tool_names

    tool_names = [t.name for t in adapted.request.tools]
    assert "shell_command" in tool_names
    assert "web_search" in tool_names


def test_adapter_handles_function_call_output_input() -> None:
    request = ResponseCreateRequest.model_validate({
        "model": "test",
        "input": [
            {"type": "message", "role": "user", "content": "search for X"},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "web_search",
                "arguments": '{"query":"X"}',
            },
            {"type": "function_call_output", "call_id": "call_1", "output": "search results here"},
            {"type": "message", "role": "user", "content": "summarize"},
        ],
    })
    from local_code_worker.models import JsonMode
    adapted = adapt_response_request(request, max_output_characters=4000, json_mode=JsonMode.NONE)

    # Should have messages for: user, assistant (tool call), tool (output), user
    roles = [m.role for m in adapted.request.messages]
    assert "user" in roles
    assert "tool" in roles


# ── ToolExecutor ──────────────────────────────────────────────────────────────

def test_executor_web_search_with_mock() -> None:
    mock_search = MagicMock()
    mock_search.search.return_value = SearchResponse(
        query="test query",
        results=[
            SearchResult(title="Result 1", url="https://example.com/1", snippet="First result"),
            SearchResult(title="Result 2", url="https://example.com/2", snippet="Second result"),
        ],
        provider="mock",
        elapsed_ms=50.0,
    )
    executor = ToolExecutor(search=mock_search, fetcher=MagicMock())
    result = executor.execute("web_search", {"query": "test query"})

    assert "test query" in result
    assert "Result 1" in result
    assert "https://example.com/1" in result
    mock_search.search.assert_called_once_with("test query", max_results=5)


def test_executor_web_search_no_results() -> None:
    mock_search = MagicMock()
    mock_search.search.return_value = SearchResponse(query="nothing", results=[], provider="mock")
    executor = ToolExecutor(search=mock_search, fetcher=MagicMock())
    result = executor.execute("web_search", {"query": "nothing"})
    assert "No web search results" in result


def test_executor_web_search_missing_query() -> None:
    executor = ToolExecutor()
    result = executor.execute("web_search", {})
    assert "Error" in result


def test_executor_web_fetch_with_mock() -> None:
    mock_fetch = MagicMock()
    mock_fetch.fetch.return_value = FetchResponse(
        url="https://example.com",
        status_code=200,
        content_type="text/html",
        text="<h1>Hello</h1>",
        truncated=False,
        elapsed_ms=100.0,
    )
    executor = ToolExecutor(search=MagicMock(), fetcher=mock_fetch)
    result = executor.execute("web_fetch", {"url": "https://example.com"})

    assert "example.com" in result
    assert "200" in result
    assert "<h1>Hello</h1>" in result


def test_executor_web_fetch_missing_url() -> None:
    executor = ToolExecutor()
    result = executor.execute("web_fetch", {})
    assert "Error" in result


def test_executor_unknown_tool() -> None:
    executor = ToolExecutor()
    result = executor.execute("nonexistent_tool", {})
    assert "unknown" in result.lower() or "Error" in result


def test_executor_github_search_stub() -> None:
    executor = ToolExecutor()
    result = executor.execute("github_search", {"query": "FastAPI"})
    assert "not yet implemented" in result


def test_executor_resolves_tool_kind_from_name() -> None:
    executor = ToolExecutor()
    assert executor._resolve_kind("web_search") == ToolKind.WEB_SEARCH
    assert executor._resolve_kind("web_fetch") == ToolKind.WEB_FETCH
    assert executor._resolve_kind("github_search") == ToolKind.GITHUB_SEARCH
    assert executor._resolve_kind("nonexistent") == ToolKind.UNKNOWN


# ── Search provider integration ──────────────────────────────────────────────

def test_duckduckgo_search_parse_fallback() -> None:
    from local_code_worker.tools.search.duckduckgo import _parse_lite_fallback
    html = '<html><a href="https://example.com/page">Example Page</a></html>'
    results = _parse_lite_fallback(html, 5)
    assert len(results) == 1
    assert results[0].title == "Example Page"
    assert results[0].url == "https://example.com/page"


def test_web_fetch_blocks_private_hosts() -> None:
    from local_code_worker.tools.search.web_fetch import WebFetch
    fetcher = WebFetch()
    result = fetcher.fetch("http://127.0.0.1:8765/v1/models")
    text = result.text.lower()
    assert "not allowed" in text or "private" in text or "error" in text


def test_web_fetch_blocks_localhost() -> None:
    from local_code_worker.tools.search.web_fetch import WebFetch
    fetcher = WebFetch()
    result = fetcher.fetch("http://localhost:8765/v1/models")
    assert result.status_code == 0


# ── Tool filtering ───────────────────────────────────────────────────────────

def test_adapter_limits_passthrough_tools() -> None:
    """With many tools, passthrough should be limited to max_passthrough_tools."""
    tools = [
        {"type": "function", "name": f"tool_{i}", "parameters": {"type": "object"}}
        for i in range(20)
    ]
    tools.append({"type": "web_search", "external_web_access": False})
    request = ResponseCreateRequest.model_validate({
        "model": "test",
        "input": "hello",
        "tools": tools,
    })
    from local_code_worker.models import JsonMode
    adapted = adapt_response_request(
        request, max_output_characters=4000, json_mode=JsonMode.NONE,
        max_passthrough_tools=5,
    )
    tool_names = [t.name for t in adapted.request.tools]
    assert "web_search" in tool_names
    passthrough_in_result = [n for n in tool_names if n != "web_search"]
    assert len(passthrough_in_result) == 5


def test_adapter_prioritizes_core_tools() -> None:
    """Core tools (shell_command, apply_patch) should survive filtering."""
    tools = [
        {"type": "function", "name": "_search_documentation", "parameters": {"type": "object"}},
        {"type": "function", "name": "shell_command", "parameters": {"type": "object"}},
        {"type": "function", "name": "apply_patch", "parameters": {"type": "object"}},
        {"type": "function", "name": "_linear_create_issue", "parameters": {"type": "object"}},
        {"type": "function", "name": "_figma_use", "parameters": {"type": "object"}},
        {"type": "function", "name": "_github_search", "parameters": {"type": "object"}},
        {"type": "function", "name": "_drive_search", "parameters": {"type": "object"}},
        {"type": "function", "name": "_linkedin_search", "parameters": {"type": "object"}},
        {"type": "function", "name": "_jobicy_search", "parameters": {"type": "object"}},
        {"type": "function", "name": "_hotline_search", "parameters": {"type": "object"}},
        {"type": "web_search", "external_web_access": False},
    ]
    request = ResponseCreateRequest.model_validate({
        "model": "test",
        "input": "hello",
        "tools": tools,
    })
    from local_code_worker.models import JsonMode
    adapted = adapt_response_request(
        request, max_output_characters=4000, json_mode=JsonMode.NONE,
        max_passthrough_tools=5,
    )
    tool_names = [t.name for t in adapted.request.tools]
    assert "shell_command" in tool_names
    assert "apply_patch" in tool_names
    assert "web_search" in tool_names
    passthrough = [n for n in tool_names if n != "web_search"]
    assert len(passthrough) == 5
    assert tool_names.index("shell_command") < tool_names.index("_search_documentation")


def test_adapter_no_filtering_when_under_limit() -> None:
    """When tools are under the limit, all should be preserved."""
    tools = [
        {"type": "function", "name": "shell_command", "parameters": {"type": "object"}},
        {"type": "function", "name": "apply_patch", "parameters": {"type": "object"}},
        {"type": "web_search", "external_web_access": False},
    ]
    request = ResponseCreateRequest.model_validate({
        "model": "test",
        "input": "hello",
        "tools": tools,
    })
    from local_code_worker.models import JsonMode
    adapted = adapt_response_request(
        request, max_output_characters=4000, json_mode=JsonMode.NONE,
        max_passthrough_tools=8,
    )
    tool_names = [t.name for t in adapted.request.tools]
    assert len(tool_names) == 3


def test_adapter_hosted_tools_listed_first() -> None:
    """Hosted tools should appear before passthrough tools in the list."""
    tools = [
        {"type": "function", "name": "shell_command", "parameters": {"type": "object"}},
        {"type": "web_search", "external_web_access": False},
        {"type": "function", "name": "apply_patch", "parameters": {"type": "object"}},
    ]
    request = ResponseCreateRequest.model_validate({
        "model": "test",
        "input": "hello",
        "tools": tools,
    })
    from local_code_worker.models import JsonMode
    adapted = adapt_response_request(
        request, max_output_characters=4000, json_mode=JsonMode.NONE,
    )
    tool_names = [t.name for t in adapted.request.tools]
    assert tool_names[0] == "web_search"
