import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .exceptions import RepositoryError, TaskValidationError
from .models import WorkerTask

FORBIDDEN_PARTS = {".git", ".env", ".venv", "node_modules"}


@dataclass(frozen=True)
class RepositoryState:
    root: Path
    commit_hash: str
    initially_changed_files: frozenset[str]
    initially_existing_allowed_files: frozenset[str]


def normalize_relative_path(path: Path) -> str:
    return path.as_posix().lstrip("./")


def resolve_repository_file(root: Path, relative_path: Path, *, must_exist: bool = True) -> Path:
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise TaskValidationError(f"Unsafe relative path: {relative_path}")
    if any(part.lower() in FORBIDDEN_PARTS for part in relative_path.parts):
        raise TaskValidationError(f"Protected path is forbidden: {relative_path}")

    resolved_root = root.resolve(strict=True)
    candidate = (resolved_root / relative_path).resolve(strict=must_exist)
    try:
        candidate.relative_to(resolved_root)
    except ValueError as error:
        raise TaskValidationError(f"Path escapes repository root: {relative_path}") from error

    if must_exist and not candidate.is_file():
        raise TaskValidationError(f"Expected a regular file: {relative_path}")
    if must_exist and os.path.normcase(str(candidate)) != os.path.normcase(
        str(resolved_root / relative_path)
    ):
        expected = resolved_root / relative_path
        if expected.exists() and expected.resolve() != candidate:
            raise TaskValidationError(f"Path resolves through an unsafe link: {relative_path}")
    return candidate


def _run_git(root: Path, arguments: list[str]) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        shell=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        raise RepositoryError(completed.stderr.strip() or f"git {' '.join(arguments)} failed")
    return completed.stdout


def _run_git_diff(root: Path, arguments: list[str]) -> str:
    """Return Git diff output; status 1 is normal when a diff exists."""
    completed = subprocess.run(
        ["git", *arguments], cwd=root, shell=False, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=30, check=False,
    )
    if completed.returncode not in {0, 1}:
        raise RepositoryError(completed.stderr.strip() or f"git {' '.join(arguments)} failed")
    return completed.stdout


def inspect_repository(task: WorkerTask) -> RepositoryState:
    root = task.repository_root.resolve(strict=True)
    if not (root / ".git").exists():
        raise RepositoryError(f"Not a Git repository: {root}")

    all_paths = [*task.allowed_files, *task.readonly_files]
    normalized = [normalize_relative_path(path) for path in all_paths]
    if len(normalized) != len(set(os.path.normcase(path) for path in normalized)):
        raise TaskValidationError("Task contains duplicate or overlapping file paths")
    initially_existing_allowed_files: set[str] = set()
    for path in task.allowed_files:
        candidate = resolve_repository_file(root, path, must_exist=False)
        if candidate.exists():
            initially_existing_allowed_files.add(normalize_relative_path(path))
    for path in task.readonly_files:
        resolve_repository_file(root, path)

    status_lines = _run_git(root, ["status", "--porcelain", "-z"]).split("\0")
    changed_files: set[str] = set()
    for entry in status_lines:
        if not entry:
            continue
        status_path = entry[3:]
        if " -> " in status_path:
            status_path = status_path.split(" -> ", 1)[1]
        changed_files.add(status_path.replace("\\", "/"))

    dirty_allowed = set(normalized) & changed_files
    if dirty_allowed:
        raise RepositoryError(
            "Allowed files already have uncommitted changes: " + ", ".join(sorted(dirty_allowed))
        )
    commit_hash = _run_git(root, ["rev-parse", "HEAD"]).strip()
    return RepositoryState(
        root, commit_hash, frozenset(changed_files), frozenset(initially_existing_allowed_files)
    )


def get_allowed_diff(state: RepositoryState, allowed_files: list[Path]) -> str:
    paths = [normalize_relative_path(path) for path in allowed_files]
    parts = [_run_git_diff(state.root, ["diff", "--", *paths])]
    for path in paths:
        if path not in state.initially_existing_allowed_files:
            target = state.root / path
            if target.is_file():
                parts.append(
                    _run_git_diff(state.root, ["diff", "--no-index", "--", os.devnull, path])
                )
    return "".join(parts)
