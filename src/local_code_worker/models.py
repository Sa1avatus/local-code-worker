from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class InterfaceRequirement(StrictModel):
    name: str
    signature: str


class ProposalFormat(StrEnum):
    PATCH = "patch"
    FILES = "files"


class WorkerTask(StrictModel):
    task_id: str
    title: str
    goal: str
    repository_root: Path
    allowed_files: list[Path]
    readonly_files: list[Path] = Field(default_factory=list)
    requirements: list[str]
    interfaces: list[InterfaceRequirement] = Field(default_factory=list)
    acceptance_criteria: list[str]
    validation_commands: list[list[str]] = Field(default_factory=list)
    max_context_characters: int = Field(default=50_000, gt=0)
    max_output_characters: int = Field(default=100_000, gt=0)
    proposal_format: ProposalFormat = ProposalFormat.PATCH

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, task_id: str) -> str:
        if not task_id or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for character in task_id
        ):
            raise ValueError("task_id may contain only letters, numbers, hyphens, and underscores")
        return task_id

    @field_validator("allowed_files", "readonly_files")
    @classmethod
    def validate_relative_paths(cls, paths: list[Path]) -> list[Path]:
        for path in paths:
            if path.is_absolute():
                raise ValueError(f"absolute file path is forbidden: {path}")
            if ".." in path.parts:
                raise ValueError(f"parent traversal is forbidden: {path}")
        return paths

    @field_validator("validation_commands")
    @classmethod
    def validate_command_shapes(cls, commands: list[list[str]]) -> list[list[str]]:
        for command in commands:
            if not command or not all(
                isinstance(argument, str) and argument for argument in command
            ):
                raise ValueError("each validation command must be a non-empty array of strings")
        return commands


class GeneratedFile(StrictModel):
    path: Path
    content: str
    reason: str


class ModelImplementationResponse(StrictModel):
    summary: str
    files: list[GeneratedFile] = Field(min_length=1)
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class GeneratedPatch(StrictModel):
    path: Path
    diff: str
    reason: str


class PatchImplementationResponse(StrictModel):
    summary: str
    patches: list[GeneratedPatch] = Field(min_length=1)
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class CommandResult(StrictModel):
    command: list[str]
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


class ResponseStatus(StrEnum):
    VALID = "valid"
    INVALID_JSON = "invalid_json"
    SCHEMA_VALIDATION_FAILED = "schema_validation_failed"
    MODEL_REFUSAL = "model_refusal"
    EMPTY_RESPONSE = "empty_response"
    OLLAMA_ERROR = "ollama_error"
    REPAIR_FAILED = "repair_failed"
    SEMANTIC_VALIDATION_FAILED = "semantic_validation_failed"
    PLACEHOLDER_CONTENT = "placeholder_content"
    TRUNCATED_RESPONSE = "truncated_response"
    PROVIDER_REFUSAL = "provider_refusal"
    PROVIDER_ERROR = "provider_error"
    AWAITING_APPROVAL = "awaiting_approval"
    ALREADY_COMPLETED = "already_completed"


class ProviderName(StrEnum):
    OLLAMA = "ollama"
    OPENAI_COMPATIBLE = "openai-compatible"


class JsonMode(StrEnum):
    AUTO = "auto"
    JSON_SCHEMA = "json-schema"
    JSON_OBJECT = "json-object"
    PROMPT_ONLY = "prompt-only"
    NONE = "none"


class ProviderHealth(StrictModel):
    provider: ProviderName
    base_url: str
    model: str
    reachable: bool
    model_available: bool | None = None
    details: str


class GenerationMetadata(StrictModel):
    provider: ProviderName
    model: str
    base_url: str
    started_at: str
    completed_at: str
    duration_seconds: float
    prompt_characters: int
    output_characters: int
    streaming: bool
    response_format_mode: JsonMode
    finish_reason: str | None = None
    usage: dict[str, int] = Field(default_factory=dict)


class ResponseAttempt(StrictModel):
    attempt: int
    status: ResponseStatus
    error: str | None = None
    duration_seconds: float
    provider: ProviderName | None = None
    model: str | None = None
    prompt_characters: int | None = None
    output_characters: int | None = None
    finish_reason: str | None = None
    usage: dict[str, int] = Field(default_factory=dict)
    error_category: str | None = None
    response_file_path: str | None = None


class RequestMetadata(StrictModel):
    run_id: str
    task_id: str
    model: str
    provider: ProviderName = ProviderName.OLLAMA
    base_url: str = "http://localhost:11434"
    api_key_env: str | None = None
    json_mode: JsonMode = JsonMode.AUTO
    stream: bool = True
    system_prompt_characters: int
    task_context_characters: int
    full_prompt_characters: int
    context_file_paths: list[str]
    prompt_sha256: str
    started_at: str


class WorkerReport(StrictModel):
    run_id: str
    task_id: str
    status: ResponseStatus
    success: bool
    changed_files: list[str]
    validation_results: list[CommandResult]
    assumptions: list[str]
    warnings: list[str]
    model: str
    provider: ProviderName = ProviderName.OLLAMA
    base_url: str = "http://localhost:11434"
    api_key_env: str | None = None
    json_mode: JsonMode = JsonMode.AUTO
    stream: bool = True
    started_at: str
    completed_at: str
    duration_seconds: float = 0
    attempts: list[ResponseAttempt] = Field(default_factory=list)
    error: str | None = None


class ProposalMetadata(StrictModel):
    run_id: str
    task_sha256: str
    proposal_sha256: str
    repository_commit: str


class CompletionState(StrictModel):
    task_id: str
    task_sha256: str
    completed_at: str
    run_id: str
    file_sha256: dict[str, str]
