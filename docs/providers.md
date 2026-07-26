# LLM providers

Local Code Worker supports local Ollama and OpenAI-compatible HTTP APIs. Provider and
model selection is explicit for every run; the worker never changes the selected model,
removes a `:free` suffix, or falls back to a paid model.

## Configuration

Configuration priority is CLI, current `LLM_*` environment or `.env` values, legacy
`OLLAMA_*` values, then defaults. Copy `.env.example` to `.env` for local configuration.
Do not commit `.env`.

API keys are never accepted as CLI values. Set `LLM_API_KEY_ENV` to the name of a
variable holding the key, or pass `--api-key-env NAME`. The report records only that
variable name. It does not store the value, Authorization header, or full environment.

## Ollama

```cmd
.venv\Scripts\python.exe -m local_code_worker provider-check ^
  --provider ollama ^
  --base-url http://localhost:11434 ^
  --model qwen2.5-coder:3b

.venv\Scripts\python.exe -m local_code_worker run ^
  --task D:\OpenAIProjects\tasks\current.json ^
  --provider ollama ^
  --model qwen2.5-coder:3b
```

Ollama uses `/api/tags` and `/api/chat`. Streaming reads NDJSON until a `done` chunk,
preserves `done_reason`, and enforces the character limit while reading. In `auto` or
`json-schema` mode the response schema is sent as Ollama's `format`.

## Generic OpenAI-compatible API

```cmd
set COMPATIBLE_API_KEY=...

.venv\Scripts\python.exe -m local_code_worker provider-check ^
  --provider openai-compatible ^
  --base-url https://provider.example/v1 ^
  --model provider/model-name ^
  --api-key-env COMPATIBLE_API_KEY
```

The provider uses `/models` and `/chat/completions` with Bearer authentication. Streaming
uses SSE; non-streaming is available with `--no-stream`. A generation probe is never made
by default. If `/models` is unavailable, opt into a minimal request with
`--probe-generation`.

## JSON modes

- `auto`: Ollama schema format; safe `json-object` for OpenAI-compatible APIs. If that
  API clearly rejects `response_format`, one retry uses `prompt-only`.
- `json-schema`: request strict server-side OpenAI JSON Schema support.
- `json-object`: request a JSON object without assuming JSON Schema support.
- `prompt-only`: rely on the existing strict system prompt and local validation.
- `none`: send no response-format option and add no mode-specific prompt.

Every response is still parsed and validated locally. Invalid JSON, schema failures,
placeholder content, semantic failures, and truncation may receive one compact repair
attempt using the same provider and model.

## Model listing and health

```cmd
.venv\Scripts\python.exe -m local_code_worker models --provider ollama
.venv\Scripts\python.exe -m local_code_worker models ^
  --provider openai-compatible ^
  --base-url https://provider.example/v1 ^
  --model provider/model-name ^
  --api-key-env COMPATIBLE_API_KEY
```

Rate limits and provider failures are reported with stable categories. HTTP 402, 404,
429, and 5xx responses stop the run; they never trigger a different model.

## Data and security

Ollama keeps prompts local. An external provider receives the task context, including
the source files explicitly listed in `allowed_files` and `readonly_files`. Review those
lists and the provider's data policy before running. Never put secrets or `.env` files
in task context.

Endpoint query parameters and user information are removed from report metadata.
Raw model responses are stored without HTTP headers.
