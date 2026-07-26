import json
from dataclasses import dataclass
from pathlib import Path

from .exceptions import ContextError
from .models import WorkerTask
from .repository import normalize_relative_path, resolve_repository_file


@dataclass(frozen=True)
class ContextStatistics:
    file_count: int
    character_count: int
    paths: tuple[str, ...]


def _read_source_file(path: Path, display_path: str) -> str:
    try:
        raw_content = path.read_bytes()
    except OSError as error:
        raise ContextError(f"Cannot read {display_path}: {error}") from error
    if b"\x00" in raw_content:
        raise ContextError(f"Binary file is not allowed in context: {display_path}")
    try:
        return raw_content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ContextError(f"File is not UTF-8: {display_path}") from error


def build_context(task: WorkerTask) -> tuple[str, ContextStatistics]:
    sections = [
        "TASK SPECIFICATION",
        json.dumps(task.model_dump(mode="json"), ensure_ascii=False, indent=2),
    ]
    paths: list[str] = []
    for category, requested_paths in (
        ("ALLOWED FILE", task.allowed_files),
        ("READONLY FILE", task.readonly_files),
    ):
        for relative_path in requested_paths:
            display_path = normalize_relative_path(relative_path)
            resolved_path = resolve_repository_file(
                task.repository_root, relative_path, must_exist=category == "READONLY FILE"
            )
            content = (
                _read_source_file(resolved_path, display_path) if resolved_path.exists() else ""
            )
            section = f"{category}\nBEGIN FILE: {display_path}\n{content}\nEND FILE: {display_path}"
            sections.append(section)
            paths.append(display_path)

    context = "\n\n".join(sections)
    if len(context) > task.max_context_characters:
        raise ContextError(
            f"Context has {len(context)} characters; limit is {task.max_context_characters}. "
            "Reduce the explicit file lists or raise the task limit."
        )
    return context, ContextStatistics(len(paths), len(context), tuple(paths))
