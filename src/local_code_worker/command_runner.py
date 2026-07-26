import subprocess
from pathlib import Path

from .exceptions import CommandValidationError
from .models import CommandResult

ALLOWED_EXECUTABLES = {"python", "python.exe", "py", "pytest", "ruff", "mypy"}
FORBIDDEN_EXECUTABLES = {
    "powershell",
    "powershell.exe",
    "pwsh",
    "pwsh.exe",
    "cmd",
    "cmd.exe",
    "bash",
    "sh",
    "wsl",
    "wsl.exe",
    "docker",
    "docker.exe",
    "git",
    "git.exe",
    "curl",
    "curl.exe",
    "wget",
    "wget.exe",
}
SHELL_OPERATORS = ("&&", "||", ";", ">", "<", "|", "`")


def validate_command(command: list[str]) -> None:
    if not command:
        raise CommandValidationError("Validation command cannot be empty")
    executable = Path(command[0]).name.lower()
    if executable in FORBIDDEN_EXECUTABLES or executable not in ALLOWED_EXECUTABLES:
        raise CommandValidationError(f"Executable is not allowed: {command[0]}")
    for argument in command:
        if any(operator in argument for operator in SHELL_OPERATORS):
            raise CommandValidationError(f"Shell operator is forbidden in argument: {argument}")


def run_validation_commands(
    commands: list[list[str]], repository_root: Path, timeout_seconds: float
) -> list[CommandResult]:
    results: list[CommandResult] = []
    for command in commands:
        validate_command(command)
        try:
            completed = subprocess.run(
                command,
                cwd=repository_root,
                shell=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
                check=False,
            )
            results.append(
                CommandResult(
                    command=command,
                    exit_code=completed.returncode,
                    stdout=completed.stdout,
                    stderr=completed.stderr,
                )
            )
        except subprocess.TimeoutExpired as error:
            results.append(
                CommandResult(
                    command=command,
                    exit_code=124,
                    stdout=(error.stdout or "") if isinstance(error.stdout, str) else "",
                    stderr=(error.stderr or "") if isinstance(error.stderr, str) else "",
                    timed_out=True,
                )
            )
    return results
