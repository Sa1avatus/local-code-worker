# Development

Read this document when changing Local Code Worker code, tests, prompts, Docker packaging, or the web
UI.

## Standalone setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Do not inspect or print `.env`. Use the local UI to replace or clear keys.

Run the web service on loopback:

```powershell
.\.venv\Scripts\python.exe -m local_code_worker web --host 127.0.0.1 --port 8765
```

## Checks

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_task_loader.py -q
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe -m ruff format --check src tests
.\.venv\Scripts\python.exe -m ruff check src tests
docker compose config --quiet
git diff --check
```

No mypy or other static type checker is configured. The Docker image installs development extras so
allowlisted Ruff and pytest validation commands can run when a proposal is applied in the container.

Relevant focused suites:

- task and repository boundaries: `tests/test_task_loader.py`, `tests/test_repository.py`;
- response and patch handling: `tests/test_response_parser.py`, `tests/test_patch_validator.py`,
  `tests/test_unified_diff.py`;
- provider transports: `tests/test_ollama_client.py`, `tests/test_openai_api.py`;
- web UI and metrics: `tests/test_web_ui.py`, `tests/test_system_metrics.py`;
- two-phase execution: `tests/test_execution.py`.

## Container

Static Compose validation is safe:

```powershell
docker compose config --quiet
docker build -t local-code-worker:test .
```

Starting the service can contact the configured provider and requires local Docker resources:

```powershell
docker volume create local-code-worker-data
docker compose up --build --detach
docker compose down
```

The volume is external and persists provider configuration after `docker compose down`. Do not
delete it as part of testing.

## Prompt changes

`prompts/implementation_system.txt` is used for complete-file proposals;
`prompts/implementation_patch_system.txt` is used for patch proposals. Keep their response contract
aligned with the Pydantic models and response parser. Test prompt routing and exact patch guidance in
`tests/test_execution.py` whenever either prompt changes.
