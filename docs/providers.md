# LLM providers

Read this document when configuring or changing Ollama/OpenAI-compatible transports, model
discovery, streaming, JSON response modes, or provider failure handling.

Local Code Worker supports local Ollama and OpenAI-compatible HTTP APIs. It never removes a model
suffix, selects `auto`, or falls back to another free or paid model.

## Configuration sources

The workspace container reads `/data/.env`, managed by the loopback web UI. Use:

```cmd
D:\OpenAIProjects\scripts\start-local-worker-container.cmd
D:\OpenAIProjects\scripts\check-local-llm.cmd
```

For standalone development, copy `.env.example` to ignored `.env` and run:

```powershell
.\.venv\Scripts\python.exe -m local_code_worker check-connection
.\.venv\Scripts\python.exe -m local_code_worker list-models
```

Configuration precedence is a deliberate CLI override, current `LLM_*` environment or `.env`
values, legacy `OLLAMA_*` values, then built-in Ollama defaults. Workspace project workflows do not
pass overrides; they use the provider and model selected in the UI.

API keys are never accepted as CLI values. `LLM_API_KEY_ENV` names the environment variable holding
the key. Reports store only the variable name, never its value or an Authorization header.

## Ollama

Ollama uses `/api/tags`, `/api/chat`, and `/api/ps`. Streaming consumes NDJSON until a `done` chunk,
rejects explicit error chunks without persisting their potentially sensitive body, preserves
`done_reason`, sends `num_predict`, and enforces the character limit while reading. The endpoint
is restricted to loopback or `host.docker.internal`.

An explicit standalone health check is:

```powershell
.\.venv\Scripts\python.exe -m local_code_worker provider-check --provider ollama --base-url http://localhost:11434 --model qwen2.5-coder:3b
```

Model download is available through the local UI and Ollama's native streaming pull API. Downloads
consume network, disk, and provider resources and require approval.

## OpenAI-compatible APIs

The provider uses `/models` and `/chat/completions` with Bearer authentication. Streaming uses SSE;
non-streaming remains available for compatibility. A generation probe is never sent by default. If
`/models` is unavailable, `provider-check --probe-generation` opts into a minimal billable request
and therefore requires approval.

Use the web UI to select the endpoint and a model returned by provider discovery. Then verify the
non-secret summary with `check-local-llm.cmd`. OpenRouter-specific configuration is in
`openrouter.md`.

## JSON modes

- `auto` — a conservative JSON object for Ollama and OpenAI-compatible providers; a clearly
  rejected OpenAI-compatible response format receives one retry in prompt-only mode.
- `json-schema` — request strict server-side JSON Schema support.
- `json-object` — request a JSON object without assuming schema support.
- `prompt-only` — rely on the system prompt and local validation.
- `none` — send no response-format option.

Every mode is parsed and validated locally. Invalid JSON or schema output may receive one bounded
repair attempt with the same provider and model. HTTP 402, 404, 429, 5xx, refusal, timeout, and
transport failures stop the run; they never select a different model.

## Data boundary

An external provider receives all source text explicitly listed in the task's `allowed_files` and
`readonly_files`. Review those lists and the provider's data policy. Never include secrets, `.env`,
private keys, raw telemetry, browser sessions, personal data, or unrelated source.

Endpoint user information and query parameters are removed from report metadata. Raw model
responses are stored only in ignored report directories and never include HTTP headers.
