# OpenRouter

OpenRouter is used through the generic OpenAI-compatible provider. A model must always
be selected explicitly in configuration or on the command line. Local Code Worker does
not use `openrouter/auto`, remove `:free`, or choose another free or paid model.

```cmd
set OPENROUTER_API_KEY=...

.venv\Scripts\python.exe -m local_code_worker provider-check ^
  --provider openai-compatible ^
  --base-url https://openrouter.ai/api/v1 ^
  --model qwen/qwen3-14b:free ^
  --api-key-env OPENROUTER_API_KEY

.venv\Scripts\python.exe -m local_code_worker run ^
  --task D:\OpenAIProjects\tasks\current.json ^
  --provider openai-compatible ^
  --base-url https://openrouter.ai/api/v1 ^
  --model qwen/qwen3-14b:free ^
  --api-key-env OPENROUTER_API_KEY ^
  --json-mode prompt-only
```

The documented conservative preset uses `prompt-only` because model capabilities vary.
Local Pydantic and semantic validation remain active, with a bounded repair attempt.

Optional attribution headers can be configured without code changes:

```text
OPENROUTER_HTTP_REFERER=https://your-application.example
OPENROUTER_APP_TITLE=Local Code Worker
```

HTTP 402 means credits or payment are required, 404 commonly means the selected model
or endpoint is unavailable, 429 is a rate limit, and 5xx is a provider failure. These
errors stop the run. Free-model availability can change, so verify it with `models` or
`provider-check` before starting a large task.

Using OpenRouter sends the explicitly selected repository context to an external
provider. Do not include credentials, private keys, `.env` files, or unrelated source
files in the task.
