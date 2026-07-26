import subprocess
from pathlib import Path

import pytest

from local_code_worker.exceptions import RepositoryError, TaskValidationError
from local_code_worker.models import WorkerTask
from local_code_worker.repository import inspect_repository, resolve_repository_file


def run_git(root: Path, *arguments: str) -> None:
    subprocess.run(["git", *arguments], cwd=root, check=True, capture_output=True)


def create_repository(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    (root / "src").mkdir(parents=True)
    (root / "src" / "service.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "notes.txt").write_text("clean\n", encoding="utf-8")
    run_git(root, "init")
    run_git(root, "add", ".")
    run_git(
        root,
        "-c",
        "user.name=Worker Test",
        "-c",
        "user.email=worker@example.invalid",
        "commit",
        "-m",
        "initial",
    )
    return root


def create_task(root: Path) -> WorkerTask:
    return WorkerTask(
        task_id="task-001",
        title="Test",
        goal="Test repository",
        repository_root=root,
        allowed_files=[Path("src/service.py")],
        requirements=["Change value"],
        acceptance_criteria=["Works"],
    )


def test_inspect_repository_allows_unrelated_dirty_file(tmp_path: Path) -> None:
    root = create_repository(tmp_path)
    (root / "notes.txt").write_text("dirty\n", encoding="utf-8")
    state = inspect_repository(create_task(root))
    assert "notes.txt" in state.initially_changed_files


def test_inspect_repository_blocks_dirty_allowed_file(tmp_path: Path) -> None:
    root = create_repository(tmp_path)
    (root / "src" / "service.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(RepositoryError, match="uncommitted"):
        inspect_repository(create_task(root))


def test_resolve_repository_file_rejects_path_outside_root(tmp_path: Path) -> None:
    root = create_repository(tmp_path)
    with pytest.raises(TaskValidationError, match="Unsafe"):
        resolve_repository_file(root, Path("../outside.py"), must_exist=False)


def test_resolve_repository_file_rejects_protected_path(tmp_path: Path) -> None:
    root = create_repository(tmp_path)
    with pytest.raises(TaskValidationError, match="Protected"):
        resolve_repository_file(root, Path(".git/config"))
