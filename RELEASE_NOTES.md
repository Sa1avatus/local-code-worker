# Release notes

## 1.1.0

### User-visible changes

- Added a local web interface for selecting Ollama or OpenAI-compatible providers,
  configuring endpoints and models, and saving or explicitly removing API keys.
- Added provider model discovery and streamed Ollama model downloads with progress.
- Added the `web` CLI command and a Docker image that serves the UI on port 8765 as a
  non-root user with persistent configuration in `/data`.

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
