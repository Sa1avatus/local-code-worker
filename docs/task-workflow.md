# Task and proposal workflow

Read this document when creating task JSON, changing proposal formats, reviewing reports, or using
the Codex approval path.

## Task contract

A task follows `examples/task.example.json` and contains:

- a stable `task_id`, title, goal, and Git `repository_root`;
- `allowed_files`, the only files a proposal may change;
- `readonly_files`, supporting context that cannot be changed;
- requirements, interface contracts, and mechanical acceptance criteria;
- allowlisted validation commands expressed as argument arrays;
- context/output limits and a `proposal_format`.

Paths must be repository-relative, must not contain parent traversal, and must resolve inside the
Git worktree. Context files are UTF-8 text. Protected paths and files outside the explicit lists are
rejected.

Validation commands do not use a shell. Executables are restricted to Python, pytest, Ruff, and
mypy entry points. Docker, Git mutation, PowerShell, curl, redirection, and shell operators are not
accepted from a task.

## Proposal modes

`"proposal_format": "patch"` is the default for existing files. The provider returns only
headerless unified-diff hunks; the Worker materializes complete files in memory and runs the normal
scope, syntax, semantic, and validation checks.

Use `"proposal_format": "files"` only for a genuinely complete replacement or new file. The
provider must return the entire final content and cannot omit unchanged sections.

The model never selects extra files, commands, a provider, or a model.

## Prompt input formats

`"prompt_format": "xml"` is the default. It builds an Execution Contract with separate
`<context_dependencies>`, `<task_instruction>`, `<negative_constraints>`, and `<output_format>`
blocks, while the system message supplies `<system_role>`. File text remains limited to explicit
task paths and is isolated as data. Use `"prompt_format": "json"` for the legacy JSON context
format when compatibility requires it.

Prompt format never changes the proposal response format: both modes return the same strict JSON
schema so path validation, patch materialization, and two-phase approval continue to work.

## Two-phase Codex mode

Validate locally first:

```cmd
D:\OpenAIProjects\scripts\validate-local-task.cmd D:\OpenAIProjects\tasks\current.json
```

Generation requires approval and saves a proposal without writing implementation files:

```cmd
D:\OpenAIProjects\scripts\run-local-implementation.cmd D:\OpenAIProjects\tasks\current.json
```

The expected state is `awaiting_approval`. Review `task.json`, `proposal.json`,
`proposal-metadata.json`, the generated file list, warnings, and assumptions. Do not simulate the
terminal prompt or use `--yes`.

After explicit approval in Codex chat, pass the exact absolute report directory printed by
generation as the sole argument to `D:\OpenAIProjects\scripts\apply-local-proposal.cmd`.

Application verifies task/proposal hashes, repository commit and cleanliness, allowed paths,
semantic rules, and validation commands. It creates backups, writes atomically, and records scoped
diff and validation output.

## Reports and completion

Reports live under the task repository's ignored `.local-worker/reports/` directory. Metadata may
contain provider/model names, timings, path lists, counts, and hashes, but not the full prompt,
source context, headers, or secret values.

Successful application records a JSON completion file under `.local-worker/state/`, named from the
task ID. Re-running an unchanged task whose resulting file hashes still match returns
`already_completed` before provider contact.

The task repository may contain unrelated uncommitted work, but an allowed file that was already
modified blocks generation/application. This prevents the proposal from overwriting user work.

## Failure behavior

Invalid JSON or schema output may receive one compact repair attempt with the same provider and
model. Provider refusal, transport failure, empty output, semantic failure, placeholder content, and
truncation are classified in reports. The Worker never chooses a fallback model.

If application writes fail, already replaced files are restored from the run backup. Test failures
after a valid write remain visible for review; do not hide them or perform Git reset automatically.
