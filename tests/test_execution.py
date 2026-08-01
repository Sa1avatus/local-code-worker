import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_code_worker.cli import (
    apply_proposal_command,
    format_duration,
    run_task_command,
)
from local_code_worker.config import WorkerSettings
from local_code_worker.exceptions import OllamaError, ProposalError
from local_code_worker.execution import (
    build_repair_prompt,
    build_request_metadata,
    generate_implementation,
    implementation_response_schema,
)
from local_code_worker.models import JsonMode, ProposalFormat, ResponseStatus, WorkerTask
from local_code_worker.prompt_loader import load_system_prompt
from local_code_worker.report_writer import RunReportWriter

VALID_RESPONSE = json.dumps(
    {
        "summary": "Updated file",
        "files": [{"path": "src/a.py", "content": "VALUE = 2\n", "reason": "Task"}],
        "assumptions": [],
        "warnings": [],
    }
)


class FakeOllamaClient:
    def __init__(self, responses: list[str | Exception]):
        self.responses = list(responses)
        self.calls: list[tuple[list[dict[str, str]], dict[str, object]]] = []
        self.settings = SimpleNamespace(
            llm_json_mode=JsonMode.PROMPT_ONLY,
            llm_max_output_characters=100_000,
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        response_schema: dict[str, object],
        max_output_characters: int,
    ) -> str:
        self.calls.append((messages, response_schema))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def create_task(root: Path) -> WorkerTask:
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "src" / "a.py").write_text("VALUE = 1\n", encoding="utf-8")
    return WorkerTask(
        task_id="execution-test",
        title="Test execution",
        goal="Update one file",
        repository_root=root,
        allowed_files=[Path("src/a.py")],
        requirements=["Update value"],
        acceptance_criteria=["Value updated"],
    )


def run_generation(
    tmp_path: Path,
    responses: list[str | Exception],
    *,
    max_repair_attempts: int = 1,
    context: str = "TASK CONTEXT",
):
    task = create_task(tmp_path)
    writer = RunReportWriter(tmp_path, Path(".local-worker/reports"), "run-test")
    client = FakeOllamaClient(responses)
    result = generate_implementation(
        task=task,
        task_path=tmp_path / "tasks" / "current.json",
        client=client,  # type: ignore[arg-type]
        system_prompt="SYSTEM",
        context=context,
        report_writer=writer,
        max_repair_attempts=max_repair_attempts,
        save_invalid_response=True,
    )
    return result, client, writer


def test_valid_response_succeeds_on_first_attempt(tmp_path: Path) -> None:
    result, client, writer = run_generation(tmp_path, [VALID_RESPONSE])
    assert result.status is ResponseStatus.VALID
    assert len(client.calls) == 1
    assert (writer.report_root / "model-response-attempt-1.json").exists()


@pytest.mark.parametrize(
    ("response", "expected_status"),
    [
        ("{invalid", ResponseStatus.INVALID_JSON),
        (json.dumps({"files": []}), ResponseStatus.SCHEMA_VALIDATION_FAILED),
        (
            json.dumps(
                {
                    "summary": "invalid file",
                    "files": [{"path": "src/a.py", "reason": "missing content"}],
                }
            ),
            ResponseStatus.SCHEMA_VALIDATION_FAILED,
        ),
    ],
)
def test_invalid_response_is_classified_without_repair(
    tmp_path: Path, response: str, expected_status: ResponseStatus
) -> None:
    result, _, writer = run_generation(tmp_path, [response], max_repair_attempts=0)
    assert result.status is expected_status
    assert (writer.report_root / "parse-error-attempt-1.txt").exists()


def test_repair_successfully_fixes_invalid_json(tmp_path: Path) -> None:
    result, client, writer = run_generation(tmp_path, ["{invalid", VALID_RESPONSE])
    assert result.status is ResponseStatus.VALID
    assert len(client.calls) == 2
    assert (writer.report_root / "model-response-attempt-2.json").exists()


def test_repair_invalid_response_is_repair_failed(tmp_path: Path) -> None:
    result, _, writer = run_generation(tmp_path, ["{invalid", "still invalid"])
    assert result.status is ResponseStatus.REPAIR_FAILED
    assert (writer.report_root / "parse-error-attempt-2.txt").exists()


def test_model_refusal_stops_without_repair(tmp_path: Path) -> None:
    refusal = json.dumps({"response": "I'm sorry, but I cannot assist with that request."})
    result, client, _ = run_generation(tmp_path, [refusal])
    assert result.status is ResponseStatus.MODEL_REFUSAL
    assert len(client.calls) == 1


def test_empty_response_stops_without_repair(tmp_path: Path) -> None:
    result, client, _ = run_generation(tmp_path, [""])
    assert result.status is ResponseStatus.EMPTY_RESPONSE
    assert len(client.calls) == 1


def test_transport_error_is_ollama_error(tmp_path: Path) -> None:
    result, _, _ = run_generation(tmp_path, [OllamaError("offline")])
    assert result.status is ResponseStatus.OLLAMA_ERROR


@pytest.mark.parametrize(
    "response",
    [
        json.dumps(
            {
                "summary": "outside",
                "files": [{"path": "src/b.py", "content": "x", "reason": "Task"}],
            }
        ),
        json.dumps(
            {
                "summary": "duplicate",
                "files": [
                    {"path": "src/a.py", "content": "x", "reason": "Task"},
                    {"path": "src/a.py", "content": "y", "reason": "Task"},
                ],
            }
        ),
    ],
)
def test_unsafe_generated_paths_fail_validation(tmp_path: Path, response: str) -> None:
    result, _, _ = run_generation(tmp_path, [response], max_repair_attempts=0)
    assert result.status is ResponseStatus.SCHEMA_VALIDATION_FAILED


def test_metadata_contains_hash_but_not_prompt_or_secret(tmp_path: Path) -> None:
    task = create_task(tmp_path)
    metadata = build_request_metadata(
        "run-id",
        task,
        "model",
        "SYSTEM SECRET_VALUE",
        "CONTEXT SECRET_VALUE",
        ["src/a.py"],
        "2026-01-01T00:00:00+00:00",
    )
    serialized = metadata.model_dump_json()
    assert "SECRET_VALUE" not in serialized
    assert len(metadata.prompt_sha256) == 64


def test_repair_request_does_not_repeat_repository_context(tmp_path: Path) -> None:
    marker = "UNIQUE_REPOSITORY_CONTEXT_MARKER"
    _, client, _ = run_generation(tmp_path, ["{invalid", VALID_RESPONSE], context=marker)
    repair_messages = client.calls[1][0]
    assert marker not in json.dumps(repair_messages)
    assert "JSON SCHEMA" in repair_messages[-1]["content"]


def test_patch_repair_explains_hunk_format_and_forbids_complete_file() -> None:
    prompt = build_repair_prompt("{}", "Missing unified diff hunk header", {}, ProposalFormat.PATCH)
    assert "do not return complete files" in prompt
    assert "@@ -old,count +new,count @@" in prompt
    assert "@@ -0,0 +1,2 @@" in prompt


def test_patch_system_prompt_requires_hunks_not_file_content() -> None:
    worker_root = Path(__file__).resolve().parents[1]
    prompt = load_system_prompt(worker_root, ProposalFormat.PATCH)
    assert "patches[].diff" in prompt
    assert "Never use files[].content" in prompt
    assert "files[].content contains the complete final source text" not in prompt
    assert "@@ -0,0 +1,2 @@" in prompt


def test_prompt_only_initial_request_includes_response_schema(tmp_path: Path) -> None:
    _, client, _ = run_generation(tmp_path, [VALID_RESPONSE])
    system_message = client.calls[0][0][0]["content"]
    assert "The API will not enforce the response format" in system_message
    assert "PatchImplementationResponse" in system_message
    assert '"src/a.py"' in system_message


def test_schema_restricts_paths_and_additional_properties(tmp_path: Path) -> None:
    schema = implementation_response_schema(create_task(tmp_path))
    assert schema["additionalProperties"] is False
    generated_file = schema["$defs"]["GeneratedPatch"]
    assert generated_file["additionalProperties"] is False
    assert generated_file["properties"]["path"]["enum"] == ["src/a.py"]


def test_patch_response_is_materialized_before_file_validation(tmp_path: Path) -> None:
    patch_response = json.dumps(
        {
            "summary": "Update one line",
            "patches": [
                {
                    "path": "src/a.py",
                    "diff": "@@ -1 +1 @@\n-VALUE = 1\n+VALUE = 2\n",
                    "reason": "Task",
                }
            ],
        }
    )
    result, _, _ = run_generation(tmp_path, [patch_response], max_repair_attempts=0)

    assert result.status is ResponseStatus.VALID
    assert result.response is not None
    assert result.response.files[0].content == "VALUE = 2\n"


def test_cli_creates_report_when_model_response_is_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = tmp_path / "repository"
    task = create_task(repository)
    subprocess.run(["git", "init"], cwd=repository, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=repository, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Worker Test",
            "-c",
            "user.email=worker@example.invalid",
            "commit",
            "-m",
            "initial",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    task_path = tmp_path / "task.json"
    task_path.write_text(task.model_dump_json(), encoding="utf-8")
    fake_client = FakeOllamaClient([json.dumps({"response": "I cannot assist."})])
    monkeypatch.setattr("local_code_worker.cli.create_provider", lambda settings: fake_client)
    settings = WorkerSettings(worker_reports_directory=Path(".local-worker/reports"))

    exit_code = run_task_command(task_path, False, settings, 1, None, True)

    assert exit_code == 1
    reports = list((repository / ".local-worker" / "reports").glob("*/report.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert report["status"] == "model_refusal"
    assert report["duration_seconds"] >= 0
    output = capsys.readouterr().out
    assert "Task started:" in output
    assert "Task finished:" in output
    assert "Task duration:" in output


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0, "00:00"), (65, "01:05"), (3661, "01:01:01")],
)
def test_format_duration(seconds: float, expected: str) -> None:
    assert format_duration(seconds) == expected


def test_codex_two_phase_approval_and_completed_detection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = tmp_path / "repository"
    task = create_task(repository)
    subprocess.run(["git", "init"], cwd=repository, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=repository, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Worker Test",
            "-c",
            "user.email=worker@example.invalid",
            "commit",
            "-m",
            "initial",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    task_path = tmp_path / "task.json"
    task_path.write_text(task.model_dump_json(), encoding="utf-8")
    fake_client = FakeOllamaClient([VALID_RESPONSE])
    monkeypatch.setattr("local_code_worker.cli.create_provider", lambda settings: fake_client)
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt: pytest.fail("Codex mode must not use terminal input"),
    )
    settings = WorkerSettings(
        _env_file=None,
        worker_reports_directory=Path(".local-worker/reports"),
    )

    exit_code = run_task_command(
        task_path,
        False,
        settings,
        1,
        None,
        True,
        invoked_by_codex=True,
    )

    assert exit_code == 0
    assert (repository / "src" / "a.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    report_roots = list((repository / ".local-worker" / "reports").iterdir())
    assert len(report_roots) == 1
    report_root = report_roots[0]
    report = json.loads((report_root / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "awaiting_approval"
    assert (report_root / "proposal.json").exists()
    assert (report_root / "proposal-metadata.json").exists()
    assert "Ask the user for approval in Codex" in capsys.readouterr().out

    assert apply_proposal_command(report_root, settings) == 0
    assert (repository / "src" / "a.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    state_path = repository / ".local-worker" / "state" / "execution-test.json"
    assert state_path.exists()

    monkeypatch.setattr(
        "local_code_worker.cli.create_provider",
        lambda settings: pytest.fail("Completed task must not call the provider"),
    )
    assert (
        run_task_command(
            task_path,
            False,
            settings,
            1,
            None,
            True,
            invoked_by_codex=True,
        )
        == 0
    )
    assert "already_completed" in capsys.readouterr().out


def test_apply_proposal_rejects_modified_proposal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    task = create_task(repository)
    subprocess.run(["git", "init"], cwd=repository, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=repository, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Worker Test",
            "-c",
            "user.email=worker@example.invalid",
            "commit",
            "-m",
            "initial",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    task_path = tmp_path / "task.json"
    task_path.write_text(task.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(
        "local_code_worker.cli.create_provider",
        lambda settings: FakeOllamaClient([VALID_RESPONSE]),
    )
    settings = WorkerSettings(_env_file=None)
    assert (
        run_task_command(
            task_path,
            False,
            settings,
            1,
            None,
            True,
            invoked_by_codex=True,
        )
        == 0
    )
    report_root = next((repository / ".local-worker" / "reports").iterdir())
    proposal_path = report_root / "proposal.json"
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    proposal["files"][0]["content"] = "VALUE = 99\n"
    proposal_path.write_text(json.dumps(proposal), encoding="utf-8")

    with pytest.raises(ProposalError, match="changed since generation"):
        apply_proposal_command(report_root, settings)
