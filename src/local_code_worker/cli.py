import argparse
import hashlib
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .command_runner import run_validation_commands
from .completion_store import (
    is_task_completed,
    task_sha256,
    write_completion_state,
)
from .config import WorkerSettings
from .context_builder import build_context
from .exceptions import ProposalError, ProviderError, WorkerError
from .execution import build_request_metadata, generate_implementation
from .file_writer import apply_generated_files
from .models import (
    JsonMode,
    ModelImplementationResponse,
    ProposalMetadata,
    ProviderName,
    ResponseStatus,
    WorkerReport,
    WorkerTask,
)
from .patch_validator import validate_generated_changes
from .prompt_loader import load_system_prompt
from .providers import create_provider
from .report_writer import RunReportWriter, create_run_id, format_validation_results
from .repository import get_allowed_diff, inspect_repository
from .task_loader import load_task


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def format_duration(seconds: float) -> str:
    total_seconds = max(0, round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def add_provider_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", choices=[item.value for item in ProviderName])
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--api-key-env")
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--connect-timeout", type=float)
    parser.add_argument("--read-timeout", type=float)
    parser.add_argument("--num-ctx", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--json-mode", choices=[item.value for item in JsonMode])
    parser.add_argument("--max-output-characters", type=int)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="local-code-worker")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("provider-check", "check-connection"):
        connection_parser = subparsers.add_parser(command)
        add_provider_arguments(connection_parser)
        connection_parser.add_argument("--probe-generation", action="store_true")
    for command in ("models", "list-models"):
        models_parser = subparsers.add_parser(command)
        add_provider_arguments(models_parser)
    for command in ("validate-task", "build-context", "run"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--task", required=True, type=Path)
        if command == "run":
            add_provider_arguments(command_parser)
            approval_group = command_parser.add_mutually_exclusive_group()
            approval_group.add_argument("--yes", action="store_true")
            approval_group.add_argument(
                "--codex",
                action="store_true",
                help="Save a proposal for approval in Codex without prompting or applying",
            )
            command_parser.add_argument(
                "--max-repair-attempts", type=int, choices=(0, 1), default=1
            )
            command_parser.add_argument("--report-dir", type=Path)
            command_parser.add_argument(
                "--save-invalid-response",
                action=argparse.BooleanOptionalAction,
                default=True,
            )
    apply_parser = subparsers.add_parser("apply-proposal")
    apply_parser.add_argument("--report", required=True, type=Path)
    return parser


def settings_from_arguments(parsed: argparse.Namespace) -> WorkerSettings:
    mapping = {
        "provider": "llm_provider",
        "base_url": "llm_base_url",
        "model": "llm_model",
        "api_key_env": "llm_api_key_env",
        "timeout": "llm_timeout_seconds",
        "connect_timeout": "llm_connect_timeout_seconds",
        "read_timeout": "llm_read_timeout_seconds",
        "num_ctx": "llm_num_ctx",
        "temperature": "llm_temperature",
        "stream": "llm_stream",
        "json_mode": "llm_json_mode",
        "max_output_characters": "llm_max_output_characters",
    }
    overrides: dict[str, Any] = {}
    for argument_name, setting_name in mapping.items():
        value = getattr(parsed, argument_name, None)
        if value is not None:
            overrides[setting_name] = value
    return WorkerSettings(**overrides)


def provider_check(settings: WorkerSettings, probe_generation: bool = False) -> int:
    provider = create_provider(settings)
    health = provider.check_connection()
    print(f"Provider: {health.provider}")
    print(f"Endpoint: {health.base_url}")
    print(f"Model: {health.model}")
    print(f"Status: {health.details}")
    if probe_generation and health.model_available is None:
        provider.chat(
            [{"role": "user", "content": "Return exactly: OK"}],
            None,
            100,
        )
        print("Generation probe: successful")
        return 0
    return 0 if health.reachable and health.model_available is not False else 1


def list_models(settings: WorkerSettings) -> int:
    provider = create_provider(settings)
    models = provider.list_models()
    print(f"Models reported by {settings.llm_provider}:")
    for model in models:
        print(f"- {model}")
    return 0


def check_connection(settings: WorkerSettings) -> int:
    """Backward-compatible alias for integrations importing the old function."""
    return provider_check(settings)


def validate_task_command(task_path: Path) -> int:
    task = load_task(task_path)
    state = inspect_repository(task)
    print(f"Task valid: {task.task_id}")
    print(f"Repository: {state.root}")
    print(f"Commit: {state.commit_hash}")
    return 0


def build_context_command(task_path: Path) -> int:
    task = load_task(task_path)
    inspect_repository(task)
    _, statistics = build_context(task)
    print(f"Files: {statistics.file_count}")
    print(f"Characters: {statistics.character_count}")
    for path in statistics.paths:
        print(f"- {path}")
    return 0


def run_task_command(
    task_path: Path,
    assume_yes: bool,
    settings: WorkerSettings,
    max_repair_attempts: int,
    report_directory: Path | None,
    save_invalid_response: bool,
    invoked_by_codex: bool = False,
) -> int:
    started_at = utc_now()
    started_monotonic = time.monotonic()
    print(f"Task started: {started_at}")
    try:
        return _run_task_command(
            task_path,
            assume_yes,
            settings,
            max_repair_attempts,
            report_directory,
            save_invalid_response,
            invoked_by_codex,
            started_at,
            started_monotonic,
        )
    finally:
        completed_at = utc_now()
        duration = time.monotonic() - started_monotonic
        print(f"Task finished: {completed_at}")
        print(f"Task duration: {format_duration(duration)} ({duration:.2f} seconds)")


def _run_task_command(
    task_path: Path,
    assume_yes: bool,
    settings: WorkerSettings,
    max_repair_attempts: int,
    report_directory: Path | None,
    save_invalid_response: bool,
    invoked_by_codex: bool,
    started_at: str,
    started_monotonic: float,
) -> int:
    worker_root = Path(__file__).resolve().parents[2]
    task = load_task(task_path)
    if is_task_completed(task, settings.worker_state_directory):
        print(f"Task status: {ResponseStatus.ALREADY_COMPLETED}")
        print("The task specification and applied file hashes are unchanged.")
        return 0
    state = inspect_repository(task)
    context, context_statistics = build_context(task)
    system_prompt = load_system_prompt(worker_root)
    run_id = create_run_id(task.task_id)
    reports_directory = report_directory or settings.worker_reports_directory
    report_writer = RunReportWriter(state.root, reports_directory, run_id)
    report_writer.write_task(task.model_dump_json(indent=2))
    _, api_key_env = settings.resolve_api_key()
    provider = create_provider(settings)
    safe_base_url = getattr(provider, "base_url", str(settings.llm_base_url))
    report_writer.write_metadata(
        build_request_metadata(
            run_id,
            task,
            str(settings.llm_model),
            system_prompt,
            context,
            list(context_statistics.paths),
            started_at,
            provider=settings.llm_provider,
            base_url=safe_base_url,
            api_key_env=api_key_env,
            json_mode=settings.llm_json_mode,
            stream=settings.llm_stream,
        )
    )
    generation = generate_implementation(
        task=task,
        task_path=task_path,
        client=provider,
        system_prompt=system_prompt,
        context=context,
        report_writer=report_writer,
        max_repair_attempts=max_repair_attempts,
        save_invalid_response=save_invalid_response,
    )
    response = generation.response
    report = WorkerReport(
        run_id=run_id,
        task_id=task.task_id,
        status=generation.status,
        success=False,
        changed_files=[],
        validation_results=[],
        assumptions=response.assumptions if response else [],
        warnings=response.warnings if response else [],
        model=str(settings.llm_model),
        provider=settings.llm_provider,
        base_url=safe_base_url,
        api_key_env=api_key_env,
        json_mode=settings.llm_json_mode,
        stream=settings.llm_stream,
        started_at=started_at,
        completed_at=utc_now(),
        duration_seconds=time.monotonic() - started_monotonic,
        attempts=list(generation.attempts),
        error=generation.error,
    )
    report_writer.write_report(report)
    print(f"Response status: {generation.status}")
    print(f"Report: {report_writer.report_root}")
    if response is None:
        return 1

    proposal_json = response.model_dump_json()
    report_writer.write_proposal(response)
    report_writer.write_proposal_metadata(
        ProposalMetadata(
            run_id=run_id,
            task_sha256=task_sha256(task),
            proposal_sha256=hashlib.sha256(proposal_json.encode("utf-8")).hexdigest(),
            repository_commit=state.commit_hash,
        )
    )
    print("Generated files:")
    for generated_file in response.files:
        print(f"- {generated_file.path.as_posix()}")
    if invoked_by_codex:
        report = report.model_copy(
            update={
                "status": ResponseStatus.AWAITING_APPROVAL,
                "completed_at": utc_now(),
                "duration_seconds": time.monotonic() - started_monotonic,
            }
        )
        report_writer.write_report(report)
        print(f"Task status: {ResponseStatus.AWAITING_APPROVAL}")
        print("No files were changed. Ask the user for approval in Codex.")
        print(
            "After approval run: "
            f".venv\\Scripts\\python.exe -m local_code_worker apply-proposal "
            f'--report "{report_writer.report_root}"'
        )
        return 0
    if not assume_yes:
        answer = input("Apply generated changes? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Changes were not applied.")
            return 0

    changed_files, backup_root = apply_generated_files(
        task, response, settings.worker_backups_directory
    )
    validation_results = run_validation_commands(
        task.validation_commands,
        state.root,
        settings.worker_command_timeout_seconds,
    )
    success = all(result.exit_code == 0 and not result.timed_out for result in validation_results)
    changes_diff = get_allowed_diff(state, task.allowed_files)
    report = report.model_copy(
        update={
            "success": success,
            "changed_files": changed_files,
            "validation_results": validation_results,
            "completed_at": utc_now(),
            "duration_seconds": time.monotonic() - started_monotonic,
        }
    )
    report_writer.write_report(report)
    report_writer.write_validation(format_validation_results(report))
    report_writer.write_diff(changes_diff)
    if success:
        completion_path = write_completion_state(
            task,
            settings.worker_state_directory,
            run_id,
            changed_files,
        )
        print(f"Completion state: {completion_path}")
    print(changes_diff or "No Git diff produced.")
    print(f"Backup: {backup_root}")
    print(f"Report: {report_writer.report_root}")
    return 0 if success else 1


def apply_proposal_command(
    report_root: Path,
    settings: WorkerSettings,
) -> int:
    started = time.monotonic()
    task_path = report_root / "task.json"
    proposal_path = report_root / "proposal.json"
    metadata_path = report_root / "proposal-metadata.json"
    report_path = report_root / "report.json"
    try:
        task = WorkerTask.model_validate_json(task_path.read_text(encoding="utf-8"))
        proposal = ModelImplementationResponse.model_validate_json(
            proposal_path.read_text(encoding="utf-8")
        )
        metadata = ProposalMetadata.model_validate_json(metadata_path.read_text(encoding="utf-8"))
        report = WorkerReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ProposalError(f"Cannot load saved proposal: {error}") from error

    if is_task_completed(task, settings.worker_state_directory):
        print(f"Task status: {ResponseStatus.ALREADY_COMPLETED}")
        print("The approved proposal was already applied and validated.")
        return 0
    resolved_report_root = report_root.resolve(strict=True)
    try:
        resolved_report_root.relative_to(task.repository_root.resolve(strict=True))
    except ValueError as error:
        raise ProposalError("Proposal report is outside the task repository") from error
    if metadata.run_id != report.run_id:
        raise ProposalError("Proposal run ID does not match its report")
    if metadata.task_sha256 != task_sha256(task):
        raise ProposalError("Saved task specification has changed")
    proposal_json = proposal.model_dump_json()
    proposal_hash = hashlib.sha256(proposal_json.encode("utf-8")).hexdigest()
    if metadata.proposal_sha256 != proposal_hash:
        raise ProposalError("Saved proposal has changed since generation")
    state = inspect_repository(task)
    if state.commit_hash != metadata.repository_commit:
        raise ProposalError("Repository commit changed after proposal generation")
    proposal = validate_generated_changes(task, proposal)
    writer = RunReportWriter.open_existing(task.repository_root, resolved_report_root)
    changed_files, backup_root = apply_generated_files(
        task,
        proposal,
        settings.worker_backups_directory,
    )
    validation_results = run_validation_commands(
        task.validation_commands,
        state.root,
        settings.worker_command_timeout_seconds,
    )
    success = all(result.exit_code == 0 and not result.timed_out for result in validation_results)
    changes_diff = get_allowed_diff(state, task.allowed_files)
    report = report.model_copy(
        update={
            "status": ResponseStatus.VALID,
            "success": success,
            "changed_files": changed_files,
            "validation_results": validation_results,
            "completed_at": utc_now(),
            "duration_seconds": report.duration_seconds + (time.monotonic() - started),
        }
    )
    writer.write_report(report)
    writer.write_validation(format_validation_results(report))
    writer.write_diff(changes_diff)
    print(changes_diff or "No Git diff produced.")
    print(f"Backup: {backup_root}")
    print(f"Report: {writer.report_root}")
    if success:
        completion_path = write_completion_state(
            task,
            settings.worker_state_directory,
            report.run_id,
            changed_files,
        )
        print(f"Completion state: {completion_path}")
    return 0 if success else 1


def main(arguments: list[str] | None = None) -> int:
    parser = create_parser()
    parsed = parser.parse_args(arguments)
    try:
        if parsed.command == "validate-task":
            return validate_task_command(parsed.task)
        if parsed.command == "build-context":
            return build_context_command(parsed.task)
        settings = settings_from_arguments(parsed)
        if parsed.command == "apply-proposal":
            return apply_proposal_command(parsed.report, settings)
        if parsed.command in {"provider-check", "check-connection"}:
            return provider_check(settings, parsed.probe_generation)
        if parsed.command in {"models", "list-models"}:
            return list_models(settings)
        if parsed.command == "run":
            return run_task_command(
                parsed.task,
                parsed.yes,
                settings,
                parsed.max_repair_attempts,
                parsed.report_dir,
                parsed.save_invalid_response,
                parsed.codex,
            )
    except (ProviderError, WorkerError, OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 2
