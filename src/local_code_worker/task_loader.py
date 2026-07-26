import json
from pathlib import Path

from pydantic import ValidationError

from .exceptions import TaskValidationError
from .models import WorkerTask


def load_task(task_path: Path) -> WorkerTask:
    try:
        raw_task = task_path.read_text(encoding="utf-8")
    except OSError as error:
        raise TaskValidationError(f"Cannot read task file {task_path}: {error}") from error
    try:
        return WorkerTask.model_validate_json(raw_task)
    except ValidationError as error:
        raise TaskValidationError(f"Invalid task JSON: {error}") from error
    except json.JSONDecodeError as error:
        raise TaskValidationError(f"Task is not valid JSON: {error}") from error
