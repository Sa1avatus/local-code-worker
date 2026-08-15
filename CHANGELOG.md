# Changelog

All notable changes to Local Code Worker are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and released versions
follow Semantic Versioning. The repository currently has no Git tags; dates and versions below are
derived from project metadata and commit history.

## [1.2.0] - 2026-08-15

### Added

- Added XML Execution Contracts as the default model-input format, separating system role,
  explicit dependencies, task instructions, negative constraints, acceptance criteria, validation
  commands, and the JSON proposal output contract. Legacy JSON prompt input remains available as an
  explicit compatibility mode.
- Added compact unified-diff proposals for existing and new files while retaining complete-file
  proposals for genuine replacements.
- Added a Docker Compose workspace runtime with guarded Windows-to-container path mapping, a
  persistent configuration volume, the shared `local-code-worker-network`, and validation tools in
  the image.
- Added OpenAI-compatible `GET /v1/models`, `POST /v1/chat/completions`, and
  `POST /v1/responses` endpoints on the loopback web service.
- Added strict Responses request and response models for text messages, instructions, reasoning
  settings, output limits, function tools, tool choice, JSON responses, and usage metadata.
- Added ordered Responses SSE streaming for text and the complete function-call lifecycle,
  including output-item, argument delta/completion, item completion, and final response events.
- Added bounded process-local continuation through `store: true` and `previous_response_id`, with
  configurable TTL and LRU eviction.
- Added hosted `web_search` through DuckDuckGo Lite and function-style `web_fetch` with bounded
  output, timeouts, redirect limits, and private-host checks. Hosted calls can run in both
  non-streaming and streaming response loops.
- Added normalization, deduplication, prioritization, and bounded forwarding of client-side tools
  for smaller local models.
- Added the stable virtual models `local-code-worker/auto`, `/local`, `/mid`, and `/strong`, while
  keeping physical model discovery on the local admin API.
- Added deterministic capability filtering, typed LOCAL/MID/STRONG routing, optional RouteLLM
  scoring, legacy/shadow/canary/routed modes, sticky route leases, and bounded monotonic escalation.
- Added privacy-safe SQLite telemetry for request counts, exact provider token usage, latency,
  routing decisions, leases, and escalation events. Actual and hypothetical routes are stored
  separately.
- Added versioned statistics, router status and metrics endpoints, an inference queue status API,
  model/system monitoring, and a configurable model unload policy.
- Added a routing benchmark runner that records route, model, usage, latency, escalation, success,
  and expected-route checks without copying benchmark prompts into its JSONL output.
- Expanded the web UI with routing configuration, model usage statistics, queue/runtime status,
  system metrics, and unload controls.

### Changed

- Changed the packaged project version from `1.1.0` to `1.2.0`.
- Changed task generation to prefer XML prompt contracts while keeping model responses as strict,
  schema-validated JSON proposals.
- Changed the gateway request limit to use separate defaults: 1 MiB for UI/configuration requests
  and 16 MiB for Codex Responses requests.
- Changed public model discovery to expose only stable virtual aliases; configured physical models
  remain available through `GET /api/models`.
- Changed provider integration to use typed canonical requests, results, capability declarations,
  token usage, and ordered stream events while preserving legacy `chat()` compatibility.
- Changed inference execution to serialize model requests and expose queue state, reducing competing
  model loads and GPU-memory pressure.
- Changed routing fallback so a tier must have its own configured endpoint, credentials, and health
  policy; configuration is never borrowed silently from another provider.

### Fixed

- Fixed generated new-file diff reporting so approved untracked files are shown without staging.
- Fixed compact patch parsing for safe diff wrappers, CRLF input, context validation, and
  `@@ -0,0` new-file hunks.
- Fixed model discovery and selection, including native dropdown population, automatic selection
  after Ollama download, and accidental browser event objects being treated as model names.
- Fixed runaway Ollama generation by sending native output limits and classifying safe stream-error
  categories without exposing response bodies.
- Fixed Codex-sized Responses requests and Codex CLI v0.146-compatible input shapes.
- Fixed bare JSON function-call parsing and Ollama/OpenAI-compatible tool-call normalization.
- Fixed Ollama tool round-trips to use the provider's expected `tool_calls` structure and dictionary
  arguments rather than a double-encoded JSON string.
- Fixed hosted-tool discovery when Codex presents hosted tools through function-style schemas.
- Fixed DuckDuckGo Lite parsing for both single-quoted and double-quoted HTML attributes.
- Fixed streamed passthrough calls so they are emitted as function-call output items with the full
  Responses SSE lifecycle.
- Fixed duplicate passthrough tools and repeated normalization that could send redundant schemas to
  local models.
- Fixed Responses input validation to accept both `input_text` and `output_text` content parts used
  in continuation flows.
- Fixed Responses requests incorrectly enabling provider JSON mode when ordinary text or tool
  output was expected.

### Security

- Kept the gateway bound to loopback by default and preserved local Host/Origin checks for browser
  configuration endpoints.
- Kept provider keys write-only in the UI and excluded credentials, prompts, source contents,
  response bodies, filesystem paths, and tool payloads from telemetry.
- Kept the model isolated from shell access, arbitrary file selection, Git mutation, validation
  command selection, and proposal application.
- Added explicit request-size validation and explicit rejection of chunked Responses bodies and
  unsupported multimodal input.

## [1.1.0] - 2026-08-01

### Added

- Added a local web interface for selecting Ollama or OpenAI-compatible providers, configuring
  endpoints and models, and saving or explicitly removing provider API keys.
- Added provider model discovery and streamed Ollama model downloads with progress reporting.
- Added the `web` CLI command and a non-root Docker image serving the UI on port 8765.
- Added persistent provider configuration through an ignored `.env` file or `/data` Docker volume.

### Changed

- Changed the packaged project version from `0.1.0` to `1.1.0`.
- Preserved existing CLI commands and both `LLM_*` and legacy `OLLAMA_*` environment variables.
- Allowed Docker-hosted Worker instances to reach native Windows Ollama through
  `host.docker.internal`.

### Fixed

- Fixed provider stream error handling so failures are classified without returning raw response
  bodies.
- Fixed browser configuration endpoints to reject non-local Host and Origin values.
- Fixed API-key handling so values are write-only, never serialized in API responses, and preserved
  when switching temporarily back to Ollama.

### Validation

- Ruff completed successfully.
- The project test suite reported 96 passing tests.
- The Docker image and loopback HTTP service were smoke-tested for this release.

## [0.1.0] - 2026-07-26

### Added

- Added the initial constrained code-proposal worker for Python 3.11 or newer.
- Added local Ollama and OpenAI-compatible provider support behind interchangeable provider
  interfaces.
- Added strict JSON task loading with explicit writable and read-only file lists, bounded context,
  protected paths, and allowlisted validation commands.
- Added schema-validated full-file proposals, one bounded repair attempt for malformed structured
  output, and local semantic validation before writing.
- Added the two-phase Codex workflow: generate and save a proposal first, then revalidate and apply
  only after explicit approval.
- Added repository cleanliness, commit, task-hash, proposal-hash, path-scope, and completion-state
  checks before application.
- Added atomic writes, scoped backups, rollback on write failure, validation reports, and Git diffs
  limited to the approved files.
- Added privacy-aware reports that record metadata and hashes without storing the full prompt in
  request metadata.

### Security

- Prevented the model from choosing files, commands, shell operations, Git mutations, or arbitrary
  output paths.
- Blocked absolute task paths, parent traversal, escaping symlinks, and protected locations such as
  `.git`, `.env`, `.venv`, and `node_modules`.
