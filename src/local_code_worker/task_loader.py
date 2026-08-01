import json
import os
from pathlib import Path

from pydantic import ValidationError

from .exceptions import TaskValidationError
from .models import WorkerTask


def map_repository_root_for_container(task: WorkerTask, task_path: Path) -> WorkerTask:
    host_root = os.environ.get("WORKER_HOST_ROOT")
    container_root = os.environ.get("WORKER_CONTAINER_ROOT")
    if not host_root and not container_root:
        return task
    if not host_root or not container_root:
        raise TaskValidationError(
            "WORKER_HOST_ROOT and WORKER_CONTAINER_ROOT must be configured together"
        )

    resolved_container = Path(container_root).resolve(strict=True)
    try:
        task_path.resolve(strict=True).relative_to(resolved_container)
    except ValueError:
        return task

    raw_repository = str(task.repository_root).replace("\\", "/").rstrip("/")
    normalized_host = host_root.replace("\\", "/").rstrip("/")
    repository_key = raw_repository.casefold()
    host_key = normalized_host.casefold()
    if repository_key == host_key:
        suffix = ""
    elif repository_key.startswith(f"{host_key}/"):
        suffix = raw_repository[len(normalized_host) :].lstrip("/")
    else:
        raise TaskValidationError(
            f"Repository root must remain under configured host workspace {normalized_host}"
        )

    mapped_repository = (resolved_container / suffix).resolve(strict=True)
    try:
        mapped_repository.relative_to(resolved_container)
    except ValueError as error:
        raise TaskValidationError("Mapped repository root escapes container workspace") from error
    return task.model_copy(update={"repository_root": mapped_repository})


def load_task(task_path: Path) -> WorkerTask:
    try:
        raw_task = task_path.read_text(encoding="utf-8")
    except OSError as error:
        raise TaskValidationError(f"Cannot read task file {task_path}: {error}") from error
    try:
        task = WorkerTask.model_validate_json(raw_task)
        return map_repository_root_for_container(task, task_path)
    except ValidationError as error:
        raise TaskValidationError(f"Invalid task JSON: {error}") from error
    except json.JSONDecodeError as error:
        raise TaskValidationError(f"Task is not valid JSON: {error}") from error
