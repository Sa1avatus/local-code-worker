import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from .exceptions import FileWriteError
from .models import ModelImplementationResponse, WorkerTask
from .repository import normalize_relative_path, resolve_repository_file


def timestamp_slug() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")


def apply_generated_files(
    task: WorkerTask,
    response: ModelImplementationResponse,
    backups_directory: Path,
    *,
    replace_function=os.replace,
) -> tuple[list[str], Path]:
    backup_root = (
        task.repository_root / backups_directory / task.task_id / timestamp_slug()
    ).resolve()
    backup_root.mkdir(parents=True, exist_ok=False)
    applied: list[tuple[Path, Path | None]] = []
    temporary_paths: list[Path] = []

    try:
        for generated_file in response.files:
            target = resolve_repository_file(
                task.repository_root, generated_file.path, must_exist=False
            )
            backup_path: Path | None = None
            if target.exists():
                backup_path = backup_root / generated_file.path
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, backup_path)

            target.parent.mkdir(parents=True, exist_ok=True)
            file_descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".worker-tmp", dir=target.parent
            )
            temporary_path = Path(temporary_name)
            temporary_paths.append(temporary_path)
            with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="") as stream:
                stream.write(generated_file.content)
                stream.flush()
                os.fsync(stream.fileno())
            replace_function(temporary_path, target)
            temporary_paths.remove(temporary_path)
            applied.append((target, backup_path))
    except (OSError, RuntimeError) as error:
        rollback_errors: list[str] = []
        for target, backup_path in reversed(applied):
            try:
                if backup_path is None:
                    target.unlink(missing_ok=True)
                else:
                    shutil.copy2(backup_path, target)
            except OSError as rollback_error:
                rollback_errors.append(f"{target}: {rollback_error}")
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)
        detail = f" Failed rollback: {'; '.join(rollback_errors)}" if rollback_errors else ""
        raise FileWriteError(
            f"File application failed and rollback was attempted: {error}.{detail}"
        ) from error

    changed_files = [
        normalize_relative_path(generated_file.path) for generated_file in response.files
    ]
    return changed_files, backup_root
