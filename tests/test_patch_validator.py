from pathlib import Path

import pytest

from local_code_worker.command_runner import run_validation_commands, validate_command
from local_code_worker.exceptions import (
    CommandValidationError,
    FileWriteError,
    PatchValidationError,
)
from local_code_worker.file_writer import apply_generated_files
from local_code_worker.models import GeneratedFile, ModelImplementationResponse, WorkerTask
from local_code_worker.patch_validator import validate_generated_changes


def create_task(root: Path) -> WorkerTask:
    return WorkerTask(
        task_id="task-001",
        title="Test",
        goal="Change files",
        repository_root=root,
        allowed_files=[Path("src/a.py"), Path("src/b.py")],
        readonly_files=[Path("src/readonly.py")],
        requirements=["Change"],
        acceptance_criteria=["Works"],
    )


def response_for(*paths: str) -> ModelImplementationResponse:
    return ModelImplementationResponse(
        summary="Change",
        files=[
            GeneratedFile(path=Path(path), content="changed\n", reason="Task") for path in paths
        ],
    )


def prepare_files(root: Path) -> None:
    (root / "src").mkdir()
    for name in ("a.py", "b.py", "readonly.py"):
        (root / "src" / name).write_text(f"original {name}\n", encoding="utf-8")


def test_patch_validator_rejects_file_outside_allowed_files(tmp_path: Path) -> None:
    prepare_files(tmp_path)
    with pytest.raises(PatchValidationError, match="outside"):
        validate_generated_changes(create_task(tmp_path), response_for("src/other.py"))


def test_patch_validator_rejects_readonly_file(tmp_path: Path) -> None:
    prepare_files(tmp_path)
    with pytest.raises(PatchValidationError, match="Readonly"):
        validate_generated_changes(create_task(tmp_path), response_for("src/readonly.py"))


def test_patch_validator_rejects_duplicate_paths(tmp_path: Path) -> None:
    prepare_files(tmp_path)
    with pytest.raises(PatchValidationError, match="Duplicate"):
        validate_generated_changes(create_task(tmp_path), response_for("src/a.py", "src/a.py"))


def test_file_writer_rolls_back_after_write_error(tmp_path: Path) -> None:
    prepare_files(tmp_path)
    task = create_task(tmp_path)
    response = response_for("src/a.py", "src/b.py")
    call_count = 0

    def fail_second_replace(source: Path, destination: Path) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise OSError("simulated failure")
        source.replace(destination)

    with pytest.raises(FileWriteError, match="rollback"):
        apply_generated_files(
            task, response, Path(".local-worker/backups"), replace_function=fail_second_replace
        )
    assert (tmp_path / "src" / "a.py").read_text(encoding="utf-8") == "original a.py\n"
    assert (tmp_path / "src" / "b.py").read_text(encoding="utf-8") == "original b.py\n"


@pytest.mark.parametrize("command", [["powershell", "-Command", "echo"], ["docker", "ps"]])
def test_command_runner_blocks_forbidden_executable(command: list[str]) -> None:
    with pytest.raises(CommandValidationError, match="not allowed"):
        validate_command(command)


def test_command_runner_blocks_shell_operator() -> None:
    with pytest.raises(CommandValidationError, match="operator"):
        validate_command(["python", "-m", "pytest", "&&", "whoami"])


def test_command_runner_executes_only_supplied_validation_command(tmp_path: Path) -> None:
    marker = tmp_path / "model-command-marker"
    results = run_validation_commands(
        [["python", "-c", "print('validation only')"]], tmp_path, timeout_seconds=10
    )
    assert results[0].exit_code == 0
    assert "validation only" in results[0].stdout
    assert not marker.exists()
