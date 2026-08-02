# Release notes

## Unreleased

### Container workflow

- Added a Docker Compose runtime that mounts the workspace at `/workspace` and preserves
  provider configuration in the existing `local-code-worker-data` volume.
- Added guarded Windows-host to container path mapping for task repository roots.
- Added Git and development validation dependencies to the image so generation,
  proposal application, Ruff, and pytest can all run inside the container.
- Replaced the browser-dependent model datalist with a native model dropdown and a
  separate manual Ollama download field, including automatic selection after download.
- Added local OpenAI-compatible `/v1/models` and `/v1/chat/completions` endpoints to the
  Worker container, plus a named Docker network for container-to-container clients.
- Added compact patch proposals: the model can return unified-diff hunks, while the Worker
  materializes and validates complete files locally before the existing approval workflow.
- Fixed the patch-proposal protocol: patch tasks now receive a hunk-only prompt and example,
  accept safe diff wrappers and new-file `@@ -0,0` hunks, and provide actionable repair guidance.
- Fixed proposal application reports for newly created files: allowed untracked files are now
  recorded as `new file` diffs without staging them.
- Fixed the model refresh control so browser `PointerEvent` objects cannot appear as bogus
  model names in the native model dropdown.
- Added a validated context-length control to the web settings page. It persists
  `LLM_NUM_CTX` and reminds users to unload the model before applying the new context.
- Added a live Ollama runtime panel in the web UI. It refreshes every 15 seconds and
  shows loaded-model VRAM allocation, estimated GPU/CPU placement, and context length.
- Expanded the panel with CPU/RAM cards and NVIDIA GPU utilization, temperature, VRAM,
  power, and graphics clock; the GPU is requested explicitly from Docker Compose.

### Validation

- `python -m ruff check src tests`
- `python -m pytest tests -q` (`106 passed` in the container)

## 1.1.0

### User-visible changes

- Added a local web interface for selecting Ollama or OpenAI-compatible providers,
  configuring endpoints and models, and saving or explicitly removing API keys.
- Added provider model discovery and streamed Ollama model downloads with progress.
- Added the `web` CLI command and a Docker image that serves the UI on port 8765 as a
  non-root user with persistent configuration in `/data`.
- Expanded the README with native Ollama installation, model-directory configuration,
  recommended coding-model downloads, API checks, and Docker connectivity guidance.

### Important fixes and security

- API-key values are write-only in the browser, excluded from API responses, and stored
  only in the ignored local environment file or Docker data volume.
- Browser API requests are restricted to local `Host` and `Origin` values.
- Ollama stream failures are validated and classified without exposing response bodies.
- Docker-hosted Worker instances can reach native Windows Ollama through
  `host.docker.internal`.

### Compatibility and migration

- Existing CLI commands and `LLM_*`/legacy `OLLAMA_*` configuration remain compatible.
- Docker users should mount `local-code-worker-data:/data` and publish port 8765 only on
  `127.0.0.1`.
- Native Windows Ollama must be running on port 11434 before local models can be listed
  or downloaded.

### Validation

- `python -m ruff check src tests`
- `python -m pytest tests -q` (`96 passed`)
- Docker image build and local HTTP smoke test on `127.0.0.1:8765`
