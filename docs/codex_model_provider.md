# Codex Beta model provider

Local Code Worker exposes an OpenAI Responses-compatible loopback gateway at
`http://127.0.0.1:8765/v1`. Codex remains the agent and owns filesystem, shell, approvals, and tool
execution. The Worker selects a configured model and proxies inference. It executes only hosted web
tools explicitly declared by the client; ordinary function tools remain client-side passthrough
calls and never grant the Worker filesystem or shell access.

## Compatibility profile

The gateway supports:

- `GET /v1/models` with `local-code-worker/auto`, `local`, `mid`, and `strong`;
- `POST /v1/responses` for text input and output;
- ordered text and function-call SSE streaming;
- non-streaming and streaming function tools;
- bounded hosted `web_search` and function-style `web_fetch` tool loops;
- bounded `previous_response_id` continuation;
- legacy, shadow, canary, RouteLLM, and compatibility routing modes.

Streaming passthrough calls emit output-item, function-argument, item-completion, and final-response
events in order. Hosted web calls are executed by the Worker and returned to the model before the
stream continues. Multimodal input remains unsupported and is rejected explicitly. A routed tier
may change provider only when that tier has its own endpoint, credentials, and health configuration;
the gateway never reuses credentials or a base URL for the wrong provider.

The OpenAI model documentation identifies Responses as the API for current coding models and lists
streaming and function calling as supported capabilities:
<https://developers.openai.com/api/docs/models>.

## Codex Beta profile

Back up the existing Codex configuration and verify the field names against the installed Codex
Beta build before adding this profile. Do not replace an existing default provider until the
loopback checks below pass.

```toml
model = "local-code-worker/auto"
model_provider = "local-code-worker"

[model_providers.local-code-worker]
name = "Local Code Worker"
base_url = "http://127.0.0.1:8765/v1"
wire_api = "responses"
requires_openai_auth = false
```

Codex includes instructions and tool schemas even for short messages. `/v1/responses` therefore
accepts request bodies up to 16 MiB by default through `LCW_MAX_RESPONSES_REQUEST_BYTES`. UI and
configuration JSON use the separate 1 MiB `LCW_MAX_UI_REQUEST_BYTES` limit. Oversized requests
return HTTP 413 with `request_too_large`; diagnostic logs contain only request ID, method, path,
sizes, status, and elapsed time.

The Worker binds to loopback by default. Do not expose port 8765 publicly, and do not place provider
API keys in Codex prompts or this TOML block.

## Verification

Start the existing container workflow:

```cmd
D:\OpenAIProjects\scripts\start-local-worker-container.cmd
```

Check model discovery without invoking a model:

```powershell
Invoke-RestMethod http://127.0.0.1:8765/v1/models
```

Then run one small text request, one streaming text request, one non-streaming function-tool request,
and one streaming function-tool request. Use `local-code-worker/local` first so the test cannot
select a cloud tier. Verify
`GET /api/v2/statistics` and the SQLite routing decision after each request.

Do not claim Codex Beta integration is complete until an installed Codex client has exercised the
profile. The repository test suite validates the same loopback HTTP shapes with an in-process fake
provider, so it does not prove compatibility with a particular Codex Beta build.

## Verified loopback matrix

The following checks were exercised on 2026-08-09 against the Docker Compose service and the local
`qwen2.5-coder-14b-iq4xs:latest` Ollama model:

| Scenario | Result |
| --- | --- |
| Virtual model discovery | Four stable aliases returned |
| Non-streaming text | Completed with usage |
| Text SSE | Ordered lifecycle through `response.completed` |
| Non-streaming function tool | Canonical `function_call` returned |
| Stored response continuation | Boundary value recovered through `previous_response_id` |
| Client disconnect | Stream aborted after 128 bytes; gateway remained healthy |
| 30,000-character context | Both boundary markers recovered; 3,800 input tokens reported |

The installed Codex Beta executable could not be launched from the Windows App sandbox (`Access is
denied`). Therefore the TOML profile itself remains unverified in this environment even though its
Responses HTTP contract passed direct loopback testing.

## Rollback

Remove the profile selection (or restore the backed-up Codex configuration) and leave
`GATEWAY_ROUTING_MODE=legacy`. This does not modify the legacy Worker proposal/apply workflow or
delete telemetry.

## Optional RouteLLM image

The default image does not install RouteLLM. Its `0.2.0` package pulls Torch, Transformers,
Datasets, PyArrow, LiteLLM, and a scientific Python stack, so enabling it materially increases build
time and image size.

On the verified Windows/Docker Desktop host, the default image built in about 107 seconds and was
about 104 MB. The first opt-in RouteLLM build did not finish within a 10-minute bounded check and
did not export an image. Treat build duration, disk use, and checkpoint download time as deployment
gates rather than enabling this extra in the default image.

Build the opt-in image in PowerShell:

```powershell
$env:INSTALL_ROUTELLM = "1"
docker compose build worker
docker compose up --detach
```

Then set `GATEWAY_ROUTELLM_ENABLED=true` and optionally
`GATEWAY_ROUTELLM_CHECKPOINT_PATH=<hugging-face-checkpoint>` in the Worker's `/data/.env`. The first
MF initialization may download checkpoint files. Do not put Hugging Face credentials or provider
API keys in routing telemetry or repository configuration.

Rollback does not require deleting any volume:

```powershell
$env:INSTALL_ROUTELLM = "0"
docker compose build worker
docker compose up --detach
```

Keep `GATEWAY_ROUTELLM_ENABLED=false` until the opt-in image and checkpoint have passed a smoke test.
