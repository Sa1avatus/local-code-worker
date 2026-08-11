import io
import json
from email.message import Message
from pathlib import Path

import pytest

from local_code_worker.request_limits import (
    DEFAULT_MAX_RESPONSES_REQUEST_BYTES,
    DEFAULT_MAX_UI_REQUEST_BYTES,
    load_request_limits,
)
from local_code_worker.web_app import RequestBodyTooLarge, WorkerWebHandler


def test_request_limit_defaults_do_not_require_env_file(tmp_path: Path) -> None:
    limits = load_request_limits(tmp_path / "missing.env", environ={})

    assert limits.max_ui_request_bytes == DEFAULT_MAX_UI_REQUEST_BYTES
    assert limits.max_responses_request_bytes == DEFAULT_MAX_RESPONSES_REQUEST_BYTES


@pytest.mark.parametrize("raw", ["0", "-1", "invalid", "1.5"])
def test_request_limits_require_positive_integers(tmp_path: Path, raw: str) -> None:
    with pytest.raises(ValueError, match="LCW_MAX_RESPONSES_REQUEST_BYTES"):
        load_request_limits(
            tmp_path / "missing.env",
            environ={"LCW_MAX_RESPONSES_REQUEST_BYTES": raw},
        )


def test_request_limit_process_environment_overrides_env_file(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("LCW_MAX_UI_REQUEST_BYTES=2048\n", encoding="utf-8")

    limits = load_request_limits(env_path, environ={"LCW_MAX_UI_REQUEST_BYTES": "4096"})

    assert limits.max_ui_request_bytes == 4096


def test_body_parser_accepts_payload_just_below_16_mib_limit() -> None:
    body = json.dumps({"padding": "x" * (16 * 1024 * 1024 - 100)}).encode()
    handler = _handler(body)

    payload = handler._read_json(max_bytes=16 * 1024 * 1024)

    assert len(payload["padding"]) == 16 * 1024 * 1024 - 100


def test_body_parser_rejects_payload_above_explicit_limit() -> None:
    handler = _handler(b"{}", declared_length=1025)

    with pytest.raises(RequestBodyTooLarge) as captured:
        handler._read_json(max_bytes=1024)

    assert captured.value.max_bytes == 1024
    assert captured.value.received_bytes == 1025


def test_body_parser_requires_content_length() -> None:
    handler = _handler(b"{}", include_length=False)

    with pytest.raises(ValueError, match="Content-Length header is required"):
        handler._read_json(max_bytes=1024)


def test_body_parser_rejects_chunked_framing_explicitly() -> None:
    handler = _handler(b"{}", transfer_encoding="chunked")

    with pytest.raises(ValueError, match="Chunked request bodies are not supported"):
        handler._read_json(max_bytes=1024)


def _handler(
    body: bytes,
    *,
    declared_length: int | None = None,
    include_length: bool = True,
    transfer_encoding: str | None = None,
) -> WorkerWebHandler:
    handler = object.__new__(WorkerWebHandler)
    headers = Message()
    if include_length:
        length = declared_length if declared_length is not None else len(body)
        headers["Content-Length"] = str(length)
    if transfer_encoding:
        headers["Transfer-Encoding"] = transfer_encoding
    handler.headers = headers
    handler.rfile = io.BytesIO(body)
    return handler
