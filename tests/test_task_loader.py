import json
from pathlib import Path

import pytest

from local_code_worker.exceptions import TaskValidationError
from local_code_worker.task_loader import load_task


def valid_task_payload(repository_root: Path) -> dict:
    return {
        "task_id": "task-001",
        "title": "Update service",
        "goal": "Implement behavior",
        "repository_root": str(repository_root),
        "allowed_files": ["src/service.py"],
        "requirements": ["Keep compatibility"],
        "acceptance_criteria": ["Tests pass"],
    }


def write_task(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_task_accepts_valid_json(tmp_path: Path) -> None:
    task = load_task(write_task(tmp_path / "task.json", valid_task_payload(tmp_path)))
    assert task.task_id == "task-001"
    assert task.allowed_files == [Path("src/service.py")]


def test_load_task_rejects_missing_required_field(tmp_path: Path) -> None:
    payload = valid_task_payload(tmp_path)
    del payload["goal"]
    with pytest.raises(TaskValidationError, match="goal"):
        load_task(write_task(tmp_path / "task.json", payload))


def test_load_task_rejects_absolute_allowed_path(tmp_path: Path) -> None:
    payload = valid_task_payload(tmp_path)
    payload["allowed_files"] = [str((tmp_path / "outside.py").resolve())]
    with pytest.raises(TaskValidationError, match="absolute"):
        load_task(write_task(tmp_path / "task.json", payload))


def test_load_task_rejects_parent_traversal(tmp_path: Path) -> None:
    payload = valid_task_payload(tmp_path)
    payload["allowed_files"] = ["../outside.py"]
    with pytest.raises(TaskValidationError, match="traversal"):
        load_task(write_task(tmp_path / "task.json", payload))
