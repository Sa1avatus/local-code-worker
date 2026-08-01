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
    path.parent.mkdir(parents=True, exist_ok=True)
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


def test_load_task_maps_configured_windows_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host_workspace = "D:/OpenAIProjects"
    container_workspace = tmp_path / "workspace"
    repository = container_workspace / "project"
    repository.mkdir(parents=True)
    payload = valid_task_payload(Path("D:/OpenAIProjects/project"))
    task_path = write_task(container_workspace / "tasks" / "task.json", payload)
    monkeypatch.setenv("WORKER_HOST_ROOT", host_workspace)
    monkeypatch.setenv("WORKER_CONTAINER_ROOT", str(container_workspace))

    task = load_task(task_path)

    assert task.repository_root == repository.resolve()


def test_load_task_rejects_repository_outside_mapped_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container_workspace = tmp_path / "workspace"
    container_workspace.mkdir()
    payload = valid_task_payload(Path("C:/Other/project"))
    task_path = write_task(container_workspace / "tasks" / "task.json", payload)
    monkeypatch.setenv("WORKER_HOST_ROOT", "D:/OpenAIProjects")
    monkeypatch.setenv("WORKER_CONTAINER_ROOT", str(container_workspace))

    with pytest.raises(TaskValidationError, match="configured host workspace"):
        load_task(task_path)


def test_load_task_requires_complete_path_mapping_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_path = write_task(tmp_path / "task.json", valid_task_payload(tmp_path))
    monkeypatch.setenv("WORKER_HOST_ROOT", "D:/OpenAIProjects")
    monkeypatch.delenv("WORKER_CONTAINER_ROOT", raising=False)

    with pytest.raises(TaskValidationError, match="configured together"):
        load_task(task_path)
