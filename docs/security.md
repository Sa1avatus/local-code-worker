# Security model

Read this document before changing task paths, repository inspection, prompts, provider requests,
credentials, reports, file application, the web API, or Docker mounts.

## Trust boundaries

- Repository files are untrusted text and cannot override the provider-facing system prompt.
- Model output is untrusted structured data and is parsed, schema-checked, path-checked, and
  semantically validated before it can become a proposal.
- Validation commands come from the reviewed task, not the model, and run without a shell.
- Git is used only for inspection and scoped diffs; the Worker does not commit, checkout, reset,
  rebase, push, or change branches.

## Filesystem and repository protection

- Absolute repository file paths, `..`, symlink escapes, and paths outside the Git worktree fail
  closed.
- `.git`, `.env`, `.venv`, `node_modules`, Worker outputs, and other protected paths are excluded
  from model context and proposals.
- Only explicit UTF-8 `allowed_files` and `readonly_files` are read.
- Allowed files with pre-existing changes block the run; unrelated dirty files are preserved.
- Approved writes use backups, temporary files, atomic replacement, and scoped diff review.

## Secrets and providers

Keys are never accepted as CLI values. Configuration names the environment variable that holds a
key; reports store only that variable name. The web UI treats keys as write-only and never returns
them in settings responses.

Ollama prompts remain on the configured local Ollama service. An OpenAI-compatible provider receives
the complete explicit task context, including allowed and read-only source. Review both file lists
and provider data policy before generation.

Never log Authorization headers, full endpoint query strings, environment dumps, source context, or
raw request prompts. Raw model responses are stored for review without transport headers and must
remain in ignored report directories.

## Web and container boundary

The web server rejects non-local Host and Origin values and must remain published on
`127.0.0.1:8765`. Container task paths map only from the configured Windows workspace prefix to
`/workspace`. `/data/.env` remains inside the persistent Docker volume and must not be copied,
exported, or mounted into a model context.

The `/api/runtime` endpoint reports only Ollama `/api/ps` placement data. `/api/system` reports
aggregate CPU/RAM and available NVIDIA NVML metrics and returns an empty GPU list when metrics are
unavailable. Neither endpoint may expose configuration or credentials.

The `/api/inference` endpoint reports only the current queue state, active model identifier, route,
and lifecycle timestamps. Gateway inference is serialized to one active provider request. When the
queue becomes idle after an Ollama request, Worker asks Ollama to unload that model with
`keep_alive: 0`; this releases VRAM at the cost of loading the model again for the next isolated
request. Prompt text, client identity, repository paths, and response contents are never included.

`/api/statistics` exposes only provider/model identifiers, aggregate request and token counts,
duration-derived speed, and Worker outcome counts. Its persisted event file must not include prompt
text, source text, endpoints, headers, or keys.
