# Local Code Worker

Local Code Worker turns a narrowly scoped JSON task into a validated code proposal. The model sees
only explicitly allowed and read-only files, receives no shell, and cannot choose commands. A saved
proposal is separate from application, and an approved application is revalidated before any file is
written.

Supported providers:

- local Ollama;
- OpenAI-compatible `/models` and `/chat/completions` APIs, including OpenRouter.

The local web UI manages provider and model settings and exposes a loopback OpenAI-compatible
gateway. Repository source sent to an external provider leaves the machine; review task file lists
and provider policy before generation.

## Workspace container quick start

From `D:\OpenAIProjects`:

```cmd
D:\OpenAIProjects\scripts\start-local-worker-container.cmd
D:\OpenAIProjects\scripts\check-local-llm.cmd
```

Open `http://127.0.0.1:8765`, select the provider and model, and save the settings. API keys are
write-only in the browser and persist in `/data/.env` inside the `local-code-worker-data` Docker
volume. The UI and API never return their values.

The current UI also aggregates local per-model request counts, token counts, generation speed, and
Worker proposal outcomes. Statistics persist in `/data/model-statistics.json`; they contain no
prompt, source text, or credential values.

The container:

- publishes port 8765 on loopback only;
- mounts `D:\OpenAIProjects` at `/workspace`;
- maps only that trusted Windows prefix for task and report paths;
- stores configuration in the persistent `/data` volume;
- joins `local-code-worker-network` for local container clients.

Current Compose requests NVIDIA GPU access for the system-metrics panel. A standalone host run is
available for development when that Docker runtime is unavailable.

## Generate and apply a proposal

Create `D:\OpenAIProjects\tasks\current.json` from
[`examples/task.example.json`](examples/task.example.json), then run:

```cmd
D:\OpenAIProjects\scripts\validate-local-task.cmd D:\OpenAIProjects\tasks\current.json
D:\OpenAIProjects\scripts\run-local-implementation.cmd D:\OpenAIProjects\tasks\current.json
```

The generation wrapper always passes `--codex`; it saves a validated proposal and exits without
modifying implementation files. Review the proposal and generated file list. After explicit
approval in Codex chat, pass the exact absolute report directory printed by generation as the sole
argument to `D:\OpenAIProjects\scripts\apply-local-proposal.cmd`.

See [`docs/task-workflow.md`](docs/task-workflow.md) for task contracts, patch format, reports,
completion state, and approval behavior.

## Standalone development

Python 3.11 or newer and Git are required:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m local_code_worker web --host 127.0.0.1 --port 8765
```

Run local checks:

```powershell
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m pytest tests -q
docker compose config --quiet
```

The legacy PowerShell helpers under `scripts/` remain available for standalone development. The
workspace Codex workflow uses only the root `.cmd` wrappers.

## Local gateway

The running web service provides:

```text
GET  http://127.0.0.1:8765/v1/models
POST http://127.0.0.1:8765/v1/chat/completions
GET  http://127.0.0.1:8765/api/statistics
```

Containers on `local-code-worker-network` use `http://local-code-worker-web:8765/v1`. Local gateway
access does not require a key; clients that require one may send the non-secret value
`local-worker`.

## Documentation

- [`docs/task-workflow.md`](docs/task-workflow.md) — tasks, proposal modes, approval, and reports;
- [`docs/development.md`](docs/development.md) — setup and verification;
- [`docs/security.md`](docs/security.md) — trust boundaries and protected data;
- [`docs/providers.md`](docs/providers.md) — provider configuration and JSON modes;
- [`docs/openrouter.md`](docs/openrouter.md) — OpenRouter-specific behavior.
