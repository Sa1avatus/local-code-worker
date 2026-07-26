import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .models import (
    ModelImplementationResponse,
    ProposalMetadata,
    RequestMetadata,
    WorkerReport,
)


def create_run_id(task_id: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{timestamp}-{task_id}-{uuid4().hex[:8]}"


class RunReportWriter:
    def __init__(self, repository_root: Path, reports_directory: Path, run_id: str):
        resolved_repository_root = repository_root.resolve()
        self.report_root = (resolved_repository_root / reports_directory / run_id).resolve()
        try:
            self.report_root.relative_to(resolved_repository_root)
        except ValueError as error:
            raise ValueError("Report directory must remain inside the repository") from error
        self.report_root.mkdir(parents=True, exist_ok=False)

    @classmethod
    def open_existing(
        cls,
        repository_root: Path,
        report_root: Path,
    ) -> "RunReportWriter":
        resolved_repository_root = repository_root.resolve()
        resolved_report_root = report_root.resolve(strict=True)
        try:
            resolved_report_root.relative_to(resolved_repository_root)
        except ValueError as error:
            raise ValueError("Report directory must remain inside the repository") from error
        writer = cls.__new__(cls)
        writer.report_root = resolved_report_root
        return writer

    def write_task(self, task_json: str) -> None:
        (self.report_root / "task.json").write_text(task_json, encoding="utf-8")

    def write_metadata(self, metadata: RequestMetadata) -> None:
        (self.report_root / "request-metadata.json").write_text(
            metadata.model_dump_json(indent=2), encoding="utf-8"
        )

    def write_raw_response(self, attempt: int, content: str) -> None:
        (self.report_root / f"model-response-attempt-{attempt}.json").write_text(
            content, encoding="utf-8"
        )

    def write_parse_error(self, attempt: int, error: str) -> None:
        (self.report_root / f"parse-error-attempt-{attempt}.txt").write_text(
            error, encoding="utf-8"
        )

    def write_report(self, report: WorkerReport) -> None:
        (self.report_root / "report.json").write_text(
            report.model_dump_json(indent=2), encoding="utf-8"
        )

    def write_proposal(self, response: ModelImplementationResponse) -> Path:
        path = self.report_root / "proposal.json"
        path.write_text(response.model_dump_json(indent=2), encoding="utf-8")
        return path

    def write_proposal_metadata(self, metadata: ProposalMetadata) -> None:
        (self.report_root / "proposal-metadata.json").write_text(
            metadata.model_dump_json(indent=2),
            encoding="utf-8",
        )

    def write_validation(self, content: str) -> None:
        (self.report_root / "validation.txt").write_text(content, encoding="utf-8")

    def write_diff(self, content: str) -> None:
        (self.report_root / "changes.diff").write_text(content, encoding="utf-8")


def format_validation_results(report: WorkerReport) -> str:
    sections: list[str] = []
    for command_result in report.validation_results:
        sections.append(
            "\n".join(
                [
                    f"COMMAND: {json.dumps(command_result.command, ensure_ascii=False)}",
                    f"EXIT CODE: {command_result.exit_code}",
                    f"TIMED OUT: {command_result.timed_out}",
                    "STDOUT:",
                    command_result.stdout,
                    "STDERR:",
                    command_result.stderr,
                ]
            )
        )
    return "\n\n".join(sections)
