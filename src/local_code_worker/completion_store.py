import hashlib
from datetime import UTC, datetime
from pathlib import Path

from .models import CompletionState, WorkerTask
from .repository import normalize_relative_path, resolve_repository_file


def task_sha256(task: WorkerTask) -> str:
    canonical = task.model_dump_json(exclude_none=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def content_sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def completion_state_path(task: WorkerTask, state_directory: Path) -> Path:
    return (task.repository_root / state_directory / f"{task.task_id}.json").resolve()


def load_completion_state(
    task: WorkerTask,
    state_directory: Path,
) -> CompletionState | None:
    path = completion_state_path(task, state_directory)
    if not path.is_file():
        return None
    try:
        return CompletionState.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def is_task_completed(task: WorkerTask, state_directory: Path) -> bool:
    state = load_completion_state(task, state_directory)
    if state is None or state.task_sha256 != task_sha256(task):
        return False
    allowed_paths = {normalize_relative_path(path) for path in task.allowed_files}
    if not state.file_sha256 or not set(state.file_sha256).issubset(allowed_paths):
        return False
    for relative_path, expected_hash in state.file_sha256.items():
        path = resolve_repository_file(
            task.repository_root,
            Path(relative_path),
            must_exist=False,
        )
        if not path.is_file():
            return False
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return False
        if content_sha256(content) != expected_hash:
            return False
    return True


def write_completion_state(
    task: WorkerTask,
    state_directory: Path,
    run_id: str,
    changed_files: list[str],
) -> Path:
    hashes: dict[str, str] = {}
    for display_path in changed_files:
        relative_path = Path(display_path)
        path = resolve_repository_file(
            task.repository_root,
            relative_path,
            must_exist=True,
        )
        hashes[display_path] = content_sha256(path.read_text(encoding="utf-8"))
    state = CompletionState(
        task_id=task.task_id,
        task_sha256=task_sha256(task),
        completed_at=datetime.now(UTC).isoformat(),
        run_id=run_id,
        file_sha256=hashes,
    )
    path = completion_state_path(task, state_directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    temporary_path.replace(path)
    return path
