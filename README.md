# Local Code Worker

Local Code Worker turns a narrowly scoped JSON task into a validated code proposal. The model sees
only explicitly allowed and read-only files, receives no shell, and cannot choose commands. A saved
proposal is separate from application, and an approved application is revalidated before any file is
written.

Task input defaults to an XML Execution Contract that isolates role, dependencies, task,
negative constraints, and output contract. Set `"prompt_format": "json"` in a task only when
legacy JSON context is required; model proposals remain strict JSON in both modes.

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
Worker proposal outcomes. The legacy UI statistics persist in `/data/model-statistics.json`.
Additive v2 telemetry persists in `/data/local-code-worker.db` using SQLite; standalone runs use
`.local-worker/local-code-worker.db`. Set `LOCAL_CODE_WORKER_TELEMETRY_PATH` to override that path.
Neither store contains prompts, source text, response bodies, filesystem paths, or credentials.

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
POST http://127.0.0.1:8765/v1/responses
GET  http://127.0.0.1:8765/api/statistics
GET  http://127.0.0.1:8765/api/v2/statistics
GET  http://127.0.0.1:8765/api/v2/router/status
GET  http://127.0.0.1:8765/api/v2/router/metrics
GET  http://127.0.0.1:8765/health
GET  http://127.0.0.1:8765/ready
```

`/api/v2/statistics` returns request/token/latency aggregates. Add the optional non-negative query
parameter `baseline_cloud_tokens` to receive estimated token savings calculated with the explicitly
named `explicit_cloud_token_budget` method. Without that parameter, `token_savings` is `null`. The
legacy `/api/statistics` response remains unchanged.

The local Responses endpoint supports strict text input, instructions, reasoning settings,
non-stream function tools, non-stream JSON responses, ordered text SSE, and bounded process-local
`previous_response_id` state. Set `store: true` to make a response available for continuation.
Stored context expires after the configurable `GATEWAY_RESPONSE_STATE_TTL_SECONDS` (two hours by
default) and is never written to telemetry or disk. Request bodies use separate limits:
`LCW_MAX_UI_REQUEST_BYTES` defaults to 1 MiB and `LCW_MAX_RESPONSES_REQUEST_BYTES` defaults to
16 MiB. Both must be positive integers, and their defaults work without `/data/.env` changes.
Chunked request bodies are rejected explicitly. Streaming requests may declare function tools and
return text, but streamed function-call output and multimodal input remain unsupported.

`GET /v1/models` exposes only the stable aliases `local-code-worker/auto`,
`local-code-worker/local`, `local-code-worker/mid`, and `local-code-worker/strong`. Physical models
remain available through the local admin endpoint `GET /api/models`. Until routing is enabled, all
four aliases execute on the active legacy `LLM_PROVIDER`/`LLM_MODEL`; forced tier metadata is
resolved but does not silently select another provider.

Containers on `local-code-worker-network` use `http://local-code-worker-web:8765/v1`. Local gateway
access does not require a key; clients that require one may send the non-secret value
`local-worker`.

## Routing architecture

```text
Request
  -> deterministic capability filter
  -> sticky RouteLease for previous_response_id chains
  -> RouteLLM / deterministic policy
  -> LOCAL / MID / STRONG
  -> configured provider
  -> monotonic escalation on a normalized failure
```

`GATEWAY_ROUTING_MODE=legacy` is the immediate rollback and preserves the original configured
provider/model. `shadow` records a hypothetical routed decision without changing execution.
`canary` applies the new policy to a deterministic percentage of root response chains, controlled
by `GATEWAY_CANARY_PERCENT`. `route_llm` applies capability filtering and the optional RouteLLM
score using `GATEWAY_LOCAL_THRESHOLD` and `GATEWAY_STRONG_THRESHOLD`. The compatibility names
`observe_only` and `router` remain accepted.

A stored Responses chain owns one process-local RouteLease. The lease pins its current tier/model,
allows only LOCAL -> MID -> STRONG movement, and stops after
`GATEWAY_MAX_ESCALATIONS_PER_LEASE`. Prompts and response bodies are not written to routing logs or
SQLite telemetry.

Run the representative routing benchmark against a loopback Worker:

```powershell
.\.venv\Scripts\python.exe -m local_code_worker.benchmarks `
  --cases benchmarks\cases.json `
  --output .local-worker\benchmarks\run.jsonl
```

The JSONL output contains selected route/model, RouteLLM score, token counts, latency, escalation
count, success, and expected-route validation. It caps model output at 128 tokens and does not copy
benchmark prompts into results.

## Documentation

- [`docs/task-workflow.md`](docs/task-workflow.md) — tasks, proposal modes, approval, and reports;
- [`docs/development.md`](docs/development.md) — setup and verification;
- [`docs/security.md`](docs/security.md) — trust boundaries and protected data;
- [`docs/providers.md`](docs/providers.md) — provider configuration and JSON modes;
- [`docs/openrouter.md`](docs/openrouter.md) — OpenRouter-specific behavior.
