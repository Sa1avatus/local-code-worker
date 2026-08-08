# OpenRouter

Read this document only when configuring the generic OpenAI-compatible provider for OpenRouter.

Open the local UI at `http://127.0.0.1:8765`, select `openai-compatible`, set the base URL to
`https://openrouter.ai/api/v1`, enter the API key in the write-only key field, refresh models, select
an exact returned model ID, and save. The key remains in `/data/.env` inside the persistent Docker
volume and is never returned by the API.

Verify the non-secret selection:

```cmd
D:\OpenAIProjects\scripts\check-local-llm.cmd
```

Use `prompt-only` when the chosen model does not advertise reliable structured-output support.
Local Pydantic, patch, placeholder, and semantic validation remain active in every mode.

Local Code Worker never uses `openrouter/auto`, removes a `:free` suffix, or chooses another model.
Availability and pricing change independently of this repository; refresh the model list before a
large task. HTTP 402, 404, 429, and 5xx responses stop the run without fallback.

Optional attribution headers are configured through `OPENROUTER_HTTP_REFERER` and
`OPENROUTER_APP_TITLE`. They are not required for local operation.

OpenRouter receives every explicit source file in the task context. Do not include credentials,
private keys, `.env`, browser state, personal data, or unrelated code, and obtain approval before
each provider generation.
