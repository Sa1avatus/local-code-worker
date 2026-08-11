from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

DEFAULT_MAX_UI_REQUEST_BYTES = 1_048_576
DEFAULT_MAX_RESPONSES_REQUEST_BYTES = 16_777_216


@dataclass(frozen=True)
class RequestLimits:
    max_ui_request_bytes: int = DEFAULT_MAX_UI_REQUEST_BYTES
    max_responses_request_bytes: int = DEFAULT_MAX_RESPONSES_REQUEST_BYTES


def load_request_limits(env_path: Path, environ: dict[str, str] | None = None) -> RequestLimits:
    import os

    values = {
        key: str(value) for key, value in dotenv_values(env_path).items() if value is not None
    }
    values.update(dict(os.environ if environ is None else environ))
    return RequestLimits(
        max_ui_request_bytes=_positive_integer(
            "LCW_MAX_UI_REQUEST_BYTES",
            values.get("LCW_MAX_UI_REQUEST_BYTES"),
            DEFAULT_MAX_UI_REQUEST_BYTES,
        ),
        max_responses_request_bytes=_positive_integer(
            "LCW_MAX_RESPONSES_REQUEST_BYTES",
            values.get("LCW_MAX_RESPONSES_REQUEST_BYTES"),
            DEFAULT_MAX_RESPONSES_REQUEST_BYTES,
        ),
    )


def _positive_integer(name: str, raw: str | None, default: int) -> int:
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value
