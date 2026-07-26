import os
import re
from pathlib import Path

from .exceptions import PatchValidationError, SemanticValidationError
from .models import ModelImplementationResponse, WorkerTask
from .repository import normalize_relative_path, resolve_repository_file

PLACEHOLDER_PHRASES = (
    "complete final file content",
    "full file content",
    "existing file content",
    "existing code",
    "unchanged code",
    "same as before",
    "rest of file unchanged",
    "omitted for brevity",
    "insert code here",
    "your code here",
    "<file content>",
    "<complete content>",
)
PYTHON_SYMBOL_PATTERN = re.compile(
    r"^(?:async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE
)


def validate_generated_changes(
    task: WorkerTask,
    response: ModelImplementationResponse,
    task_path: Path | None = None,
) -> ModelImplementationResponse:
    if not response.files:
        raise PatchValidationError("Model proposed no changed files")

    allowed_by_key = {
        os.path.normcase(normalize_relative_path(path)): normalize_relative_path(path)
        for path in task.allowed_files
    }
    readonly_keys = {
        os.path.normcase(normalize_relative_path(path)) for path in task.readonly_files
    }
    seen: set[str] = set()
    for generated_file in response.files:
        display_path = normalize_relative_path(generated_file.path)
        key = os.path.normcase(display_path)
        if key in seen:
            raise PatchValidationError(f"Duplicate generated path: {display_path}")
        seen.add(key)
        if key in readonly_keys:
            raise PatchValidationError(f"Readonly file cannot be changed: {display_path}")
        if key not in allowed_by_key:
            raise PatchValidationError(f"Generated file is outside allowed_files: {display_path}")
        generated_target = resolve_repository_file(
            task.repository_root, generated_file.path, must_exist=False
        )
        if task_path is not None and generated_target == task_path.resolve():
            raise PatchValidationError(f"Task file cannot be changed: {display_path}")
        if generated_file.path.suffix.lower() in {".bat", ".cmd", ".ps1", ".sh"}:
            raise PatchValidationError(f"Generated shell command file is forbidden: {display_path}")
        if not generated_file.content:
            raise PatchValidationError(f"Generated file is empty: {display_path}")
        if len(generated_file.content) > task.max_output_characters:
            raise PatchValidationError(f"Generated file exceeds size limit: {display_path}")
        _validate_file_content(generated_target, generated_file.content, display_path)
    return response


def _validate_file_content(target: Path, content: str, display_path: str) -> None:
    normalized = " ".join(content.lower().split())
    if normalized == "..." or any(phrase in normalized for phrase in PLACEHOLDER_PHRASES):
        raise SemanticValidationError(
            f"Generated file contains placeholder content: {display_path}",
            category="placeholder_content",
        )
    if target.suffix.lower() == ".py":
        try:
            compile(content, display_path, "exec")
        except SyntaxError as error:
            raise SemanticValidationError(
                f"Generated Python is invalid for {display_path}: "
                f"{error.msg} at line {error.lineno}"
            ) from error
    if not target.exists():
        return
    try:
        existing = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return
    if len(existing) >= 200 and len(content) < max(40, len(existing) // 5):
        raise SemanticValidationError(
            f"Generated file is suspiciously shorter than the existing file: {display_path}"
        )
    if target.suffix.lower() == ".py":
        existing_symbols = set(PYTHON_SYMBOL_PATTERN.findall(existing))
        generated_symbols = set(PYTHON_SYMBOL_PATTERN.findall(content))
        missing_symbols = sorted(existing_symbols - generated_symbols)
        if missing_symbols:
            raise SemanticValidationError(
                f"Generated file removes existing Python symbols from {display_path}: "
                + ", ".join(missing_symbols)
            )
