# Local Code Worker v2.0: аудит и целевая архитектура

Статус документа: завершённый аудит PHASE 1 и план следующих небольших инкрементов.

Дата аудита: 2026-08-09. Аудит выполнен для ветки
`feat/xml-contract-statistics`, commit `da54086`.

## 1. Цель и границы

Local Code Worker v2.0 должен добавить новый режим model gateway для Codex поверх существующего
ограниченного генератора предложений. Это не замена Codex agent loop и не переписывание Worker с
нуля.

Codex по-прежнему отвечает за workspace, чтение и запись файлов, shell, инструменты, approvals,
применение изменений и тесты. Worker принимает model inference request, выбирает настроенный
provider/model, проксирует запрос и возвращает OpenAI Responses-совместимый ответ.

Существующий двухфазный workflow предложения и применения остаётся самостоятельным legacy-путём.
В репозитории нет HTTP endpoint `/delegate`; его функциональный аналог — команды `run --codex` и
`apply-proposal`, вызываемые доверенными workspace wrappers. Их нельзя связывать с новым gateway
так, чтобы model request мог автоматически изменить файлы.

Этот документ не реализует PHASE 2 и не добавляет runtime dependencies, публичные endpoints или
миграции данных. Такие изменения требуют отдельных небольших инкрементов и проверок.

## 2. Краткий вывод аудита

Репозиторий уже содержит полезную основу для v2:

- единые `WorkerSettings` и provider factory;
- отдельные Ollama и OpenAI-compatible transports;
- сбор provider usage и безопасная нормализация ошибок;
- локальный OpenAI-compatible gateway для `/v1/models` и `/v1/chat/completions`;
- loopback web UI, write-only API keys и Docker volume `/data`;
- атомарную JSON-статистику и отчёты legacy workflow;
- хорошие проверки path, prompt, proposal, Git state и atomic apply.

Но текущий gateway не является Responses API gateway:

- отсутствует `POST /v1/responses`;
- provider protocol знает только `chat(...)`, а не Responses items, tools и incremental events;
- `stream=true` на публичном gateway сначала полностью буферизует downstream ответ;
- нет virtual models, logical tiers, routing decision, RouteLLM или escalation;
- конфигурация описывает один активный provider/model;
- статистика недостаточна для baseline, routing, savings, latency percentiles и стоимости;
- нет database layer, migrations и response/session state;
- health — синхронный снимок текущего provider без истории состояния;
- UI является одной HTML/JS строкой внутри `web_app.py` и не имеет раздела Routing.

Следовательно, правильный путь — расширять существующие seams, но не помещать Responses parsing,
routing, provider transport, telemetry и UI logic обратно в монолитный `web_app.py`.

## 3. Карта репозитория

| Область | Текущие файлы | Ответственность |
| --- | --- | --- |
| Entrypoint и CLI | `src/local_code_worker/__main__.py`, `cli.py` | Команды provider check, task validation, legacy generation/apply и web server |
| Task contract | `models.py`, `task_loader.py` | Pydantic schema, контейнерное отображение workspace paths |
| Контекст и prompts | `context_builder.py`, `prompt_loader.py`, `prompts/` | Явно ограниченный source context и XML/JSON execution contracts |
| Legacy execution | `execution.py`, `response_parser.py`, `patch_validator.py` | Provider call, repair, JSON/schema/semantic validation |
| Безопасное применение | `repository.py`, `file_writer.py`, `command_runner.py` | Git state, path boundary, backup, atomic write и allowlisted validation |
| Состояние и отчёты | `completion_store.py`, `report_writer.py` | Completion hashes и reviewable run artifacts |
| Providers | `providers/base.py`, `factory.py`, `ollama.py`, `openai_compatible.py` | Текущие transports Ollama и Chat Completions-compatible APIs |
| Gateway и UI | `web_app.py`, `web_models.py`, `web_config.py` | HTTP routes, local UI, provider configuration и Chat Completions adapter |
| Наблюдаемость | `usage_statistics.py`, `system_metrics.py` | Model-call event JSON и CPU/RAM/GPU snapshot |
| Packaging | `pyproject.toml`, `Dockerfile`, `compose.yaml` | Python 3.12 container, one service, `/data` volume, workspace mount |
| Проверки | `tests/` | Unit и local HTTP tests с fake/mock providers |

FastAPI в проекте отсутствует. Web entrypoint — `run_web_server()` в `web_app.py`, использующий
`ThreadingHTTPServer` и `BaseHTTPRequestHandler`. Добавлять FastAPI только ради v2 не требуется:
это новая runtime dependency и ненужная смена server stack. Существующий сервер можно сохранить,
вынеся schema, adapters и route handlers в отдельные модули.

## 4. Текущий legacy request flow

```text
tasks/current.json
  -> cli.run_task_command()
  -> load_task() + container path mapping
  -> inspect_repository()
  -> build_context() + load_system_prompt()
  -> create_provider(settings)
  -> generate_implementation()
  -> provider.chat()
  -> parse + schema/path/syntax/semantic validation
  -> ignored report directory + awaiting_approval

explicit user approval
  -> cli.apply_proposal_command()
  -> verify task/proposal hashes + commit + cleanliness
  -> atomic file replacement + backup
  -> allowlisted validation commands
  -> scoped diff + completion state
```

Ключевые свойства, которые v2 обязан сохранить:

- модель видит только `allowed_files` и `readonly_files`;
- модель не выбирает произвольные files или shell commands;
- provider/model выбираются до generation и не меняются скрыто;
- proposal не применяется во время generation;
- approval и application разделены;
- task hash, proposal hash, commit и cleanliness повторно проверяются;
- prompts, source, headers и secrets не попадают в обычную telemetry.

Новый gateway не должен вызывать `generate_implementation()` или `apply_proposal_command()`.
Gateway возвращает только model output; Codex сам выполняет agent actions.

## 5. Текущий gateway request flow

```text
POST /v1/chat/completions
  -> WorkerWebHandler._chat_request()
  -> validate messages/model/temperature/max_tokens/response_format
  -> load one active WorkerSettings object
  -> create_provider(settings)
  -> provider.chat()
  -> record_model_call()
  -> convert buffered text to Chat Completions JSON or two SSE chunks
```

`GET /v1/models` сейчас возвращает физические models активного provider. Request может выбрать
другое имя model, но provider и endpoint остаются единственными глобально настроенными значениями.

Публичный `stream=true` не является настоящим proxy streaming. `_chat_request()` принудительно
создаёт settings с `llm_stream=False`; `_chat_completion()` ждёт полный text и затем отправляет один
content chunk, один finish chunk и `[DONE]`. Этот путь нужно оставить для backward compatibility,
но нельзя использовать как основу Responses streaming.

## 6. Текущая provider abstraction

`LlmProvider` в `providers/base.py` — структурный `Protocol` со следующими операциями:

- `check_connection()`;
- `list_models()`;
- `chat(messages, response_schema, max_output_characters, max_output_tokens)`;
- `last_generation_metadata`.

`OllamaProvider` использует `/api/tags`, `/api/ps`, `/api/pull` и `/api/chat`. Он умеет читать
downstream NDJSON потоком, но собирает text в список и возвращает одну строку. Usage берётся из
`prompt_eval_count` и `eval_count` финального Ollama chunk.

`OpenAICompatibleProvider` использует `/models` и `/chat/completions` с Bearer auth. Он читает SSE,
но также собирает content в памяти. Usage извлекается только если provider прислал `usage`; request
не просит `stream_options.include_usage`, поэтому streaming usage зависит от поведения конкретного
provider.

`OllamaClient` — backward-compatible facade старого API. Его нужно сохранить до удаления всех
внешних imports, если такие consumers существуют.

Текущий protocol нельзя просто переименовать в Responses provider: он теряет typed input items,
instructions, tool definitions, tool calls, reasoning, previous response state, cached/reasoning
tokens, time to first token и incremental event boundaries.

## 7. Конфигурация

`WorkerSettings` читает CLI overrides, `LLM_*` environment или `.env`, legacy `OLLAMA_*`, затем
Ollama defaults. Конфигурация содержит один provider/model/base URL, timeouts, context/output limits,
temperature, streaming и JSON mode.

`web_config.py` сохраняет не-secret поля и write-only key в `.env`; container использует
`/data/.env`. API возвращает только `api_key_configured` и имя environment variable, не значение.

Для v2 нельзя размножать `LLM_*` в три неструктурированных набора. Нужны:

- typed `GatewaySettings` с `mode = legacy | observe_only | router`;
- typed `TierConfig` для LOCAL, MID и STRONG;
- отдельная registry конфигурация provider instances;
- routing, fallback, escalation и health policy;
- сохранение secrets только через environment/secrets mechanism;
- атомарное сохранение non-secret config с version/schema field;
- preview и явное подтверждение preset, если он заменяет CUSTOM значения.

Legacy `LLM_*` остаётся источником фактического provider в `legacy` и начальным default для миграции.

## 8. Хранилище и migrations

Database layer и migration tool в репозитории отсутствуют. Постоянные данные сейчас состоят из:

- `/data/.env` для container configuration и credentials;
- `/data/model-statistics.json` для максимум 10 000 model-call events;
- `.local-worker/reports/` для per-run legacy artifacts;
- `.local-worker/state/` для completion hashes;
- `.local-worker/backups/` для apply backups.

JSON event list переписывается атомарно и защищён process-local lock, но плохо подходит для
concurrent queries, percentiles, retention, response state и schema evolution. Для v2 рекомендуется
stdlib `sqlite3` в `/data/local-code-worker.db`, без нового Python dependency. Первый migration
должен только создавать новые таблицы; существующий JSON нельзя удалять. Одноразовый optional import
старой статистики должен быть идемпотентным и помечать неполные поля как unknown/estimated.

Минимальные таблицы:

- `schema_migrations(version, applied_at)`;
- `model_requests(...)` для безопасной request telemetry;
- `routing_decisions(...)` с initial, hypothetical и final routes;
- `provider_health(...)` для последнего состояния и агрегатов;
- `response_state(...)` для bounded/expiring `previous_response_id` support;
- при необходимости `gateway_config_versions(...)` только для non-secret configuration history.

Rollback первой миграции — остановить запись в новые таблицы и вернуться к legacy JSON reader.
Destructive downgrade и автоматическое удаление таблиц не нужны.

## 9. Текущая telemetry и token accounting

`GenerationMetadata` уже содержит provider, model, sanitized base URL, start/end, duration,
prompt/output characters, streaming flag, response format, finish reason и provider usage.

`usage_statistics.py` сохраняет только:

- provider и model;
- kind и outcome;
- duration;
- prompt/completion tokens.

В event отсутствуют timestamp, request/session/project IDs, route, task analysis, retry/escalation,
time to first token, cached/reasoning tokens, tools, cost и success semantics. `summarize_model_calls()`
агрегирует requests, tokens, tokens/sec и valid/invalid Worker proposals только по model.

Для v2 нужно различать:

- provider-reported exact usage;
- locally estimated usage;
- unavailable usage;
- baseline estimate;
- actual local/mid/strong/cloud usage;
- savings, вычисленные из явно названного baseline method.

`estimated_cloud_tokens_saved` нельзя считать точным, а provider request success нельзя считать
успешной coding task без feedback signal от Codex/legacy workflow. В первой версии следует хранить
`inference_success` отдельно от `task_success`, причём `task_success = null`, пока signal отсутствует.

Полный prompt, source code, tool arguments, Authorization headers и credentials по умолчанию не
записываются. Debug logging должен быть отдельным opt-in режимом с redaction и коротким retention.

## 10. Web UI

UI встроен как `INDEX_HTML` в `web_app.py`. Он настраивает один provider/model, позволяет pull
Ollama model, показывает system/runtime metrics и агрегаты model usage. Отдельной сборки frontend
нет.

Для первых gateway phases UI менять не нужно. После стабилизации inference path следует вынести
HTML/JS в package resources или небольшие модули и добавить раздел Routing:

- cards LOCAL/MID/STRONG с явной меткой local/cloud;
- mode и preset с preview diff;
- RouteLLM enabled/threshold;
- bounded escalation/fallback settings;
- provider health;
- routing/savings/latency charts;
- таблицу последних requests без prompts/tool payloads.

UI не должен возвращать secrets, автоматически тестировать платный provider при открытии страницы
или применять destructive preset без предупреждения.

## 11. Health checks

Текущий `ProviderHealth` содержит provider, base URL, model, reachable, model availability и details.
`/api/health` проверяет только активный provider синхронно во время запроса. Docker Compose не имеет
service-level `healthcheck`.

V2 health registry должен хранить для каждого configured provider/model:

- `available | unavailable | degraded`;
- last checked и last successful timestamps;
- sanitized last error category/message;
- rolling average latency;
- configured provider/model/tier;
- optional cooldown/circuit-open deadline.

Router должен исключать явно unavailable target, если есть допустимый fallback. Health probe не
должен отправлять billable generation без отдельной настройки. Для Docker достаточно HTTP probe
локального gateway readiness; downstream readiness показывается отдельно и не должна постоянно
перезапускать контейнер.

## 12. Docker и runtime topology

Compose содержит один `worker` service, публикуемый только как `127.0.0.1:8765`, persistent external
volume `/data`, read-write `/workspace` mount и существующую network. Dockerfile запускает
`python -m local_code_worker` от non-root user.

Эту topology нужно сохранить. RouteLLM в первой реализации является optional library/adapter внутри
Worker, а не отдельным service. Новый service оправдан только измеренной потребностью в иной runtime,
GPU/process isolation или независимом lifecycle.

Порт 8765 следует сохранить по умолчанию для backward compatibility. Документация Codex должна
использовать `http://127.0.0.1:8765/v1`, если пользователь явно не перенастроил Compose.

## 13. Целевая модульная архитектура

Предлагаемая структура адаптирована к существующему package и не требует смены web framework:

```text
src/local_code_worker/
  responses/
    schemas.py       # strict request/response/item/event models
    adapter.py       # Responses <-> canonical provider request/result
    streaming.py     # ordered SSE event encoder, disconnect handling
    api.py           # endpoint handler, не routing rules
  providers/
    base.py          # extended capabilities + responses/stream protocol
    registry.py      # configured provider instances and model metadata
    ollama.py
    openai_compatible.py
    openai_responses.py  # только если native Responses transport полезен
  routing/
    models.py        # Tier, TaskAnalysis, RoutingDecision
    analyzer.py      # deterministic feature extraction
    rules.py         # data-driven policy rules
    engine.py        # sequence and forced virtual models
    routellm_adapter.py
    fallback.py
    escalation.py
  telemetry/
    models.py
    store.py
    metrics.py
    savings.py
  storage/
    sqlite.py
    migrations.py
  health.py
  gateway_config.py
  web_app.py         # thin route dispatch + legacy endpoints
```

Названия можно скорректировать по мере небольших инкрементов, но границы обязательны:

- Responses parsing не знает routing policy;
- router не знает HTTP детали Ollama/OpenAI;
- providers не выбирают tier и не делают silent fallback;
- telemetry получает sanitized lifecycle events;
- legacy execution использует совместимый adapter, но не зависит от Responses HTTP schemas;
- UI не является источником бизнес-правил.

## 14. Canonical provider contract

Текущий `chat()` следует сохранить как compatibility method, реализованный поверх нового
канонического contract или адаптирующийся к нему. Новый protocol концептуально должен предоставлять:

- `responses(request) -> ProviderResult`;
- `responses_stream(request) -> Iterator[ProviderEvent]`;
- `health() -> ProviderHealthSnapshot`;
- `list_models()` и model information;
- capabilities: tools, reasoning, temperature, native responses, streaming;
- context/output limits;
- usage с exact/estimated provenance;
- cost metadata без hardcoded model names в routing code.

`ProviderRequest` должен содержать normalized input items, instructions, tools, tool choice,
reasoning, temperature, output limit и metadata. `ProviderEvent` должен представлять semantic
delta/tool/usage/completion/error events, а не сырые provider bytes. Responses SSE encoder переводит
их в публичные event types.

LM Studio обслуживается generic OpenAI-compatible provider configuration, если его API действительно
совместим. Отдельный class нужен только при отличающемся capability detection или transport.

## 15. Responses API compatibility

`POST /v1/responses` добавляется отдельным route. Минимально принимаются и валидируются:

- `model`;
- `input` (string и supported input items);
- `instructions`;
- `stream`;
- `tools`;
- `tool_choice`;
- `previous_response_id`;
- `reasoning`;
- `temperature`;
- `max_output_tokens`;
- `metadata`.

Unsupported fields/capabilities должны возвращать Responses-shaped validation error, а не тихо
исчезать. `temperature` передаётся только provider/model с такой capability. Tools преобразуются в
downstream native tool definitions; function calls возвращаются как typed output items. Worker не
исполняет tools.

Для streaming нельзя вызывать существующий buffered `chat()`. Provider iterator должен передавать
delta в client сразу, одновременно измеряя time to first token. Encoder обязан сохранить порядок
created/in-progress/output-item/content-delta/completed events и корректно закрыть stream при error
или disconnect. Точные event schemas следует фиксировать contract tests на основе фактического
Codex traffic, не только упрощённых примеров.

`previous_response_id` требует `ResponseStateStore`. До появления безопасного bounded store endpoint
не следует утверждать полную поддержку multi-turn continuation. Stored state имеет TTL/size limits и
не содержит secrets; возможность не сохранять raw input должна быть предусмотрена конфигурацией.

## 16. Virtual models и routing modes

Публичные virtual models:

- `local-code-worker/auto` — полный routing sequence;
- `local-code-worker/local` — forced LOCAL;
- `local-code-worker/mid` — forced MID;
- `local-code-worker/strong` — forced STRONG.

`/v1/models` должен возвращать virtual models независимо от физического active provider. Physical
models остаются видимыми через admin API, а не смешиваются с стабильным Codex contract.

Режимы:

- `legacy`: фактический route — текущий `LLM_*` provider/model, router не вызывается;
- `observe_only`: фактический route остаётся legacy, hypothetical decision сохраняется;
- `router`: фактический route выбирается engine;
- forced virtual model: выбранный tier используется без RouteLLM, но с health/fallback policy,
  если policy явно это разрешает.

Observe-only telemetry хранит actual и hypothetical route отдельно. Она не должна приписывать
hypothetical route фактические latency, token usage, cost или success.

## 17. Routing engine, rules и RouteLLM

Routing sequence:

1. Responses request validation.
2. Forced virtual model и hard overrides.
3. Deterministic task analysis.
4. Ordered, configurable policy rules.
5. RouteLLM только для ambiguous decision.
6. Configured deterministic fallback.
7. Provider health selection.

`RoutingDecision` минимум содержит selected tier/provider/model, reason, confidence,
routing method, rule id, RouteLLM score и timestamp. Дополнительно нужны initial route,
hypothetical route, final route и policy/config version для воспроизводимости.

Hard rules хранятся как typed ordered policy records с predicate/action, а не большой `if/elif`.
Первый analyzer использует только объяснимые признаки: task category, architecture/security/migration,
debugging, multi-service, tests/docs/refactor/simple edit, context size, estimated scope и previous
failures. Он не читает secrets и не логирует source text.

RouteLLM по умолчанию выключен. Adapter рассматривает LOCAL как weak, STRONG как strong; MID
выбирается deterministic rules. Import/inference failure даёт telemetry event и deterministic
fallback, но не роняет model request. Добавление RouteLLM dependency выполняется только в PHASE 8
после отдельного approval и совместимости с Python/container.

## 18. Fallback и bounded escalation

Разрешённые переходы: LOCAL -> MID, LOCAL -> STRONG, MID -> STRONG. Нужны отдельные counters
`max_local_attempts`, `max_mid_attempts`, `max_total_attempts`; каждый provider call увеличивает
общий счётчик. Повтор одного tier/provider/model без причины запрещён.

Первая версия может эскалировать только по наблюдаемым сигналам: timeout, provider error, malformed
provider response, invalid tool call shape, context overflow, unsupported capability и explicit low
confidence. Semantic failure coding task без feedback от Codex не симулируется.

Fallback и escalation — разные решения и должны иметь разные telemetry reasons. Client disconnect
не должен запускать дорогой fallback. STRONG cloud request допустим только после routing decision и
явной конфигурации cloud provider.

## 19. Backward compatibility strategy

Обязательные compatibility gates:

- CLI names, task schema defaults, prompts и two-phase proposal/apply сохраняются;
- `OllamaClient` facade сохраняется;
- `LLM_*` и legacy `OLLAMA_*` продолжают загружаться;
- `/v1/models` и `/v1/chat/completions` не удаляются;
- `/api/settings`, `/api/health`, `/api/runtime`, `/api/system`, `/api/statistics` не меняют
  существующую форму без versioned endpoint;
- Compose service, port 8765, network, volume и workspace mount сохраняются;
- старый `model-statistics.json` не удаляется и не перезаписывается миграцией;
- router default mode при upgrade — `legacy`, затем explicit `observe_only`, затем `router`.

Новый provider contract должен иметь adapter для legacy `execution.generate_implementation()`, чтобы
не переписывать approval workflow одновременно с gateway.

## 20. Codex configuration contract

Актуальная официальная Codex config reference подтверждает:

- `model_provider` ссылается на id в `model_providers`;
- custom provider поддерживает `name`, `base_url` и authentication options;
- `wire_api = "responses"` — единственное поддерживаемое значение и default;
- direct `experimental_bearer_token` поддерживается, но не рекомендуется; предпочтителен `env_key`;
- `model_provider` и `model_providers` являются machine-local settings и игнорируются в project
  `.codex/config.toml`.

Источник: <https://developers.openai.com/codex/config-reference> и
<https://developers.openai.com/codex/config-advanced>.

После реализации endpoint рекомендуемый user-level `~/.codex/config.toml`:

```toml
model = "local-code-worker/auto"
model_provider = "local-code-worker"

[model_providers.local-code-worker]
name = "Local Code Worker"
base_url = "http://127.0.0.1:8765/v1"
wire_api = "responses"
requires_openai_auth = false
```

Статический bearer token для loopback unauthenticated gateway не нужен. Если позже вводится auth,
следует использовать `env_key`, а не помещать secret literal в config. Конкретный snippet должен
быть проверен end-to-end установленным Codex Beta после появления `/v1/responses`; наличие ключей в
reference ещё не доказывает wire compatibility реализации.

Отдельный `docs/codex_model_provider.md` создаётся в PHASE 7 после contract tests, чтобы не
публиковать неработающую инструкцию заранее.

## 21. Риски и меры

| Риск | Последствие | Мера |
| --- | --- | --- |
| Частичная Responses совместимость | Codex loop ломается на tools/stream events | Captured contract fixtures и integration test с установленным Codex |
| Buffered streaming | Высокий TTFT/memory, idle timeout | Iterator-based provider contract и immediate SSE flush |
| Смешение gateway и legacy workflow | Model request может повлиять на files | Жёсткая модульная граница; gateway не импортирует apply path |
| Silent field loss | Неверное reasoning/tool поведение | Strict schemas и explicit unsupported-capability errors |
| Неограниченная escalation | Рост cloud tokens/cost | Per-tier и total attempt limits, no retry after disconnect |
| Ошибочный savings baseline | Вводящий в заблуждение dashboard | Provenance и `estimated` flags; actual/hypothetical separation |
| Router выбирает unhealthy provider | Повторные failures | Health registry, cooldown и deterministic fallback |
| Race в JSON statistics | Потеря events | SQLite transaction store и append-only migrations |
| Leakage через telemetry/debug | Source/secrets в persistence | Allowlist fields, redaction, opt-in debug, retention |
| Config migration ломает legacy | Worker перестаёт запускаться | Default legacy mode и additive migration |
| Монолитный UI/API растёт | Трудно тестировать и расширять | Responses/routing/telemetry modules до dashboard work |
| RouteLLM dependency несовместима | Container build/runtime fail | Optional adapter, disabled default, separate approved increment |

## 22. Проверенный baseline

На момент аудита выполнены:

- `python -m ruff check src tests` — успешно;
- `docker compose config --quiet` — успешно, Docker CLI дополнительно сообщил sandbox warning о
  недоступном user config, но Compose schema прошла;
- первоначальный `pytest` без `test_openai_api.py`, `test_system_metrics.py` и `test_web_ui.py` —
  118 passed;
- после установки уже объявленных `psutil` и `nvidia-ml-py` в существующую `.venv` полный
  `pytest tests -q` — 129 passed;
- `ruff format --check src tests` — неуспешно: 21 существующий файл требует форматирования. В рамках
  архитектурного аудита массовое форматирование не выполнялось.

Development environment теперь соответствует `pyproject.toml`, а полный test baseline зелёный до
изменения поведения.

## 23. Детальный implementation plan

Каждый пункт ниже — отдельный reviewable increment. Runtime provider generation и применение
предложений подчиняются workspace approval workflow; один approval не разрешает следующие calls.

### PHASE 2 — telemetry и baseline

1. Добавить typed telemetry models и чистые calculations в новые
   `telemetry/models.py`, `telemetry/savings.py`, `telemetry/metrics.py`; тесты —
   `tests/test_telemetry_models.py`, `tests/test_token_savings.py`.
2. Добавить additive SQLite migration/store в `storage/sqlite.py`, `storage/migrations.py` и
   `telemetry/store.py`; оставить `usage_statistics.py` compatibility facade; тесты на migration,
   concurrency, retention и отсутствие sensitive fields.
3. Инструментировать существующие `web_app.py` Chat Completions и `cli.py` Worker attempts без
   изменения routing. Ввести baseline method и exact/estimated provenance.
4. Расширить `/api/statistics` additively или добавить versioned `/api/v2/statistics`; не ломать
   текущий UI contract.

### PHASE 3 — provider abstraction

1. Расширить `providers/base.py` canonical request/result/event/capability protocols.
2. Адаптировать `providers/ollama.py` и `providers/openai_compatible.py` к incremental iterator,
   сохранив `chat()`.
3. Добавить registry и per-provider model metadata без hardcoded routing model names.
4. Расширить `tests/test_providers.py` contract tests: capabilities, usage provenance, TTFT,
   cancellation и настоящий streaming.

### PHASE 4 — Responses API compatibility

1. Создать `responses/schemas.py` с strict supported subset и Responses errors.
2. Создать `responses/adapter.py` для input/instructions/tools/tool_choice/reasoning conversion.
3. Создать `responses/streaming.py` для ordered SSE events без buffering.
4. Создать `responses/api.py` и тонко подключить route в `web_app.py`.
5. Добавить `tests/test_responses_api.py`, `tests/test_responses_streaming.py` и tool-call fixtures.
6. Добавить bounded response state только вместе с tests `previous_response_id`, TTL и restart.

### PHASE 5 — virtual models

1. Добавить stable virtual model registry и изменить только public `/v1/models` contract.
2. Оставить physical discovery в `/api/models`.
3. Добавить forced tier resolution без RouteLLM и без automatic escalation.
4. Contract tests для auto/local/mid/strong и неизвестных names.

### PHASE 6 — deterministic router

1. Добавить routing models, analyzer, data-driven rules и engine.
2. Добавить typed tier/provider/model configuration и modes legacy/observe_only/router.
3. В `observe_only` сохранять hypothetical decision, фактически используя legacy provider.
4. Unit tests всех hard rules, order/priority, context/scope/failure features и fallback policy.

### PHASE 7 — Codex Beta integration

1. Проверить user-level config на фактически установленной Codex Beta версии.
2. Запустить non-streaming, streaming и tool-calling end-to-end requests через loopback gateway.
3. Зафиксировать минимальные sanitized fixtures и создать `docs/codex_model_provider.md`.
4. Проверить disconnect, retry semantics, long context и `previous_response_id`.

### PHASE 8 — RouteLLM

1. После отдельного dependency approval добавить optional `routing/routellm_adapter.py`.
2. RouteLLM вызывается только для ambiguous cases и может быть полностью отключён.
3. Failure/timeout/malformed score дают deterministic fallback и telemetry.
4. Tests не требуют реальной model/network и проверяют weak/strong mapping.

### PHASE 9 — fallback и escalation

1. Добавить health-aware selection, bounded counters и typed escalation reasons.
2. Интегрировать только observable failures; semantic feedback оставить отдельным будущим API.
3. Integration tests LOCAL failure -> MID/STRONG, unavailable provider, total-attempt limit и client
   disconnect without escalation.

### PHASE 10 — dashboard и configuration

1. Вынести UI assets из `web_app.py` без framework rewrite.
2. Добавить routing config schemas/API с safe preset preview и write-only secrets.
3. Добавить versioned metrics/recent requests API и затем charts/cards/table.
4. UI tests проверяют labels local/cloud, no secret serialization и destructive preset warning.

### PHASE 11 — A/B comparison

1. Сравнить legacy baseline, observe-only hypothetical и router actual на одинаковых task cohorts.
2. Основная метрика — cloud tokens per successful coding task; до появления feedback signal
   использовать inference metrics, явно не называя их task success.
3. Дополнительно считать cost per successful task, tasks without STRONG и escalation-adjusted cost.
4. Не включать router по умолчанию до достаточного sample, стабильных integration tests и
   документированного rollback.

## 24. Следующий безопасный шаг

Следующий инкремент — PHASE 2.1: только typed telemetry schema и pure baseline/savings calculations
с unit tests, без SQLite, endpoints, provider calls, Docker topology и UI. Это минимальная база, на
которой можно согласовать определения exact/estimated metrics до persistence migration.

## 25. Статус PHASE 2

PHASE 2 реализована additively:

- typed request/usage, latency и token-savings models;
- SQLite schema version 1 с `schema_migrations` и `model_requests`;
- WAL, bounded lock wait, idempotent initialization, retention и concurrency tests;
- observe-only запись Chat Completions и Worker attempts при сохранении legacy JSON facade;
- request/token/success/latency aggregates с provider/model/tier filters;
- baseline method `explicit_cloud_token_budget`, всегда маркированный как estimated;
- новый `GET /api/v2/statistics`; legacy `GET /api/statistics` не изменён.

Runtime path — `/data/local-code-worker.db` в контейнере и
`.local-worker/local-code-worker.db` standalone. Override:
`LOCAL_CODE_WORKER_TELEMETRY_PATH`. Схема намеренно не содержит prompt, response content,
filesystem path, API key, token или cookie fields.

## 26. Статус PHASE 3

PHASE 3 реализована с сохранением legacy `chat()`:

- canonical request/result/message/capability contracts;
- ordered `started`, `text_delta`, `usage`, `completed` events;
- incremental Ollama NDJSON и OpenAI-compatible SSE iterators;
- TTFT metadata и exact/unavailable usage provenance;
- generator cancellation закрывает underlying HTTP stream без completed metadata;
- registry-based provider factory и metadata произвольной configured model без routing-name table;
- canonical adapter поверх legacy provider interface с явной проверкой stream/JSON modes.

## 27. Статус PHASE 4

PHASE 4 реализована как strict, additive Responses subset:

- text/message input, instructions, reasoning, tools/tool choice и output limits schemas;
- canonical request adapter и OpenAI-shaped response/output/usage models;
- ordered text SSE lifecycle с `response.completed` и `response.failed`;
- additive `POST /v1/responses` без изменения Chat Completions;
- bounded process-local `previous_response_id` state с TTL/LRU и opt-in `store: true`;
- non-stream function calls для Ollama и OpenAI-compatible provider formats;
- явный отказ для multimodal input и streaming function tools вместо silent fallback.

## 28. Статус PHASE 5

PHASE 5 реализована без преждевременного включения router:

- stable aliases `local-code-worker/auto`, `/local`, `/mid`, `/strong`;
- typed LOCAL/MID/STRONG forced-tier resolution;
- public `/v1/models` содержит только virtual catalog;
- admin `/api/models` сохраняет physical provider discovery;
- Chat Completions и Responses отклоняют неизвестные aliases и возвращают public alias;
- до PHASE 6 все aliases фактически используют active legacy provider/model.
# Production routing lifecycle

The Responses gateway analyzes hard request capabilities before scoring route quality. Unsupported
tiers are excluded before provider invocation. New stored chains create a bounded in-memory
`RouteLease` inside the existing response state; continuations reuse it through
`previous_response_id`. Automatic movement is monotonic (`LOCAL -> MID -> STRONG`) and each change
records a normalized escalation event in SQLite.

Modes are selected with `GATEWAY_ROUTING_MODE`: `legacy`, `shadow`, `canary`, or `route_llm`.
Compatibility values `observe_only` and `router` remain supported. Shadow always executes legacy.
Canary assignment hashes the root response ID, so every continuation in a lease receives the same
assignment. Legacy is the rollback path and does not require source changes.

The admin surface exposes `/api/v2/router/status`, `/api/v2/router/metrics`, and a response-scoped
decision lookup. `/health` is a liveness check; `/ready` validates and reports the loaded routing
configuration without loading a model. Structured logs and telemetry contain identifiers,
routes/models, scores, latency/token metadata, and escalation reasons, but never prompts, source,
tool payloads, response bodies, paths, or credentials.
