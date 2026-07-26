# Local Code Worker

> Provider update: Local Code Worker supports both local Ollama and OpenAI-compatible
> APIs. See [provider configuration](docs/providers.md) and the
> [OpenRouter guide](docs/openrouter.md). Existing Ollama commands and `OLLAMA_*`
> variables remain supported.

Every `run` prints its UTC start time immediately and always prints the UTC finish time
and total wall-clock duration. The same duration in seconds is stored in `report.json`.

## Codex approval workflow

When Codex launches the Worker, use the two-phase mode:

```cmd
.venv\Scripts\python.exe -m local_code_worker run --codex ^
  --task D:\OpenAIProjects\tasks\current.json
```

This generates and validates the response, stores `proposal.json` and
`proposal-metadata.json` in the run report, prints the proposed file list, and exits with
`awaiting_approval`. It never opens the terminal confirmation prompt and does not change
implementation files.

After the user approves the proposal in Codex, apply that exact saved proposal:

```cmd
.venv\Scripts\python.exe -m local_code_worker apply-proposal ^
  --report D:\path\to\repository\.local-worker\reports\<run-id>
```

Before writing, the Worker verifies the task hash, proposal hash, repository commit,
repository cleanliness, allowed paths, and semantic validation. Successful validations
create `.local-worker/state/<task-id>.json`. A later run with the same task specification
and unchanged resulting file hashes returns `already_completed` before contacting the
provider.

Local Code Worker — ограниченный исполнитель технических заданий. Он читает только явно перечисленные файлы Git-репозитория, получает структурированное предложение изменений от локальной Ollama, повторно проверяет границы, запрашивает подтверждение и атомарно записывает только разрешённые файлы.

Worker не является автономным агентом: модель не получает shell, не выбирает файлы и команды, не видит `.git`, `.env` или остальной репозиторий. OpenAI API, Codex и n8n не используются.

## Архитектура

```text
task JSON → Git/path validation → explicit file context → Ollama /api/chat
          → JSON validation → confirmation → backup/atomic write
          → allowlisted validations → report + scoped git diff
```

Git-команды Worker фиксированы: `status --porcelain`, `rev-parse HEAD` и `diff --` для разрешённых файлов. Worker не выполняет commit, checkout, reset, push и не меняет ветку.

## Установка и настройка

Требуются Python 3.11+, Git, PowerShell и запущенная Ollama с `qwen2.5-coder:3b` по умолчанию.

```powershell
cd local-code-worker
Copy-Item .env.example .env
.\scripts\install.ps1
.\scripts\test-connection.ps1
```

Настройки `.env` задают URL и модель Ollama, тайм-ауты, каталоги отчётов и backup. `.env`, `.venv` и `.local-worker` исключены из Git.

Модель выбирается через `OLLAMA_MODEL` в `.env`. Для разового запуска можно не менять файл:

```powershell
python -m local_code_worker check-connection --model qwen2.5-coder:7b
python -m local_code_worker run --task ..\tasks\current.json --model qwen2.5-coder:7b
```

Параметр `--model` имеет приоритет над `.env`; без него используется `OLLAMA_MODEL`.

Список установленных моделей через безопасный интерфейс Worker:

```powershell
python -m local_code_worker list-models
```

## Формат задания

Скопируйте [examples/task.example.json](examples/task.example.json) за пределы Worker и замените значения. `repository_root` должен указывать на существующий Git-репозиторий. `allowed_files` — единственные файлы, которые модель может предложить изменить; `readonly_files` добавляются только как контекст.

Команды проверки задаются массивами аргументов, не строками:

```json
[
  ["python", "-m", "ruff", "check", "src", "tests"],
  ["python", "-m", "pytest", "tests/unit/test_service.py", "-q"]
]
```

Разрешены только `python`, `python.exe`, `py`, `pytest`, `ruff` и `mypy`. Shell, Docker, Git, PowerShell, WSL, curl и shell-операторы блокируются.

## Создание задания с помощью Codex

Codex можно использовать только для подготовки JSON вручную: передайте ему бизнес-требования и самостоятельно проверьте `repository_root`, списки файлов и команды. Worker не запускает Codex и не принимает от него команды автоматически.

Проверка задания без модели и без записи:

```powershell
python -m local_code_worker validate-task --task ..\tasks\current.json
python -m local_code_worker build-context --task ..\tasks\current.json
```

`build-context` выводит только число файлов, символов и пути. Исходный код в консоль или отчёт не выводится.

## Пробный и рабочий запуск

```powershell
.\scripts\run-task.ps1 -TaskPath ..\tasks\current.json
```

После ответа модели Worker показывает пути и спрашивает:

```text
Apply generated changes? [y/N]
```

Любой ответ кроме `y` или `yes` отменяет запись. Флаг `--yes` существует для будущей контролируемой автоматизации, но скрипт его не использует.

Worker сохраняет каждую попытку ответа до парсинга. Для невалидного JSON или ошибки schema выполняется не более одной компактной repair-попытки; отказ модели, пустой ответ и transport error repair не запускают.

```cmd
local-code-worker\.venv\Scripts\python.exe -m local_code_worker run ^
  --task tasks\current.json ^
  --model qwen2.5-coder:7b ^
  --max-repair-attempts 1 ^
  --report-dir .local-worker\reports ^
  --save-invalid-response
```

`--report-dir` должен оставаться внутри репозитория задачи. `--save-invalid-response` включён по умолчанию. Полный prompt и содержимое исходных файлов в metadata не сохраняются.

## Отчёты и восстановление

Каждый запуск создаёт уникальный каталог в репозитории задачи, включая неудачные ответы:

```text
.local-worker/reports/<run-id>/
├── task.json
├── request-metadata.json
├── model-response-attempt-1.json
├── parse-error-attempt-1.txt
├── model-response-attempt-2.json
├── parse-error-attempt-2.txt
├── report.json
├── validation.txt
└── changes.diff
```

Файлы второй попытки появляются только при repair; `validation.txt` и `changes.diff` — после применения. `request-metadata.json` содержит только идентификаторы, модель, размеры, список путей и SHA-256 prompt, но не сам prompt.

Исходные версии находятся в `.local-worker/backups/<task_id>/<timestamp>/` с сохранённой структурой путей. При ошибке записи Worker автоматически откатывает уже заменённые файлы. Для ручного отката остановите работу, сравните backup с рабочим файлом и скопируйте нужную исходную версию обратно; Worker намеренно не выполняет `git checkout` или `reset`.

## Ограничения безопасности

- Абсолютные пути, `..`, выход из корня и symlink наружу блокируются.
- `.git`, `.env`, `.venv` и `node_modules` защищены.
- Уже изменённый allowed-файл блокирует запуск; посторонние незакоммиченные изменения допускаются.
- Читаются только UTF-8 текстовые файлы из двух явных списков.
- Превышенный контекст не обрезается, а отклоняется.
- Модель возвращает только полное содержимое файлов в JSON и не определяет команды.
- Запись выполняется через временный файл и атомарную замену.

## Типичные ошибки

- **Not a Git repository:** исправьте `repository_root`.
- **Allowed files already have uncommitted changes:** сохраните или откатите собственные изменения перед запуском.
- **Context exceeds limit:** сократите явные списки или осознанно увеличьте лимит задания.
- **Ollama/model unavailable:** запустите `local-ollama`, проверьте `/api/tags` и установленную модель.
- **Ollama timeout:** модель 7B с частичной CPU-выгрузкой может отвечать медленно; увеличьте `OLLAMA_TIMEOUT_SECONDS`.
- **Validation failed:** изучите `validation.txt` и `changes.diff`; Worker не откатывает корректно записанные файлы только из-за падения тестов.

## Полный рабочий процесс

Основной запуск не зависит от PowerShell Execution Policy и использует Python из `.venv` напрямую:

```cmd
local-code-worker\.venv\Scripts\python.exe -m local_code_worker check-connection
local-code-worker\.venv\Scripts\python.exe -m local_code_worker validate-task --task tasks\current.json
local-code-worker\.venv\Scripts\python.exe -m local_code_worker run --task tasks\current.json
```

Эквивалентные CMD-обёртки из корня workspace:

```cmd
scripts\check-local-llm.cmd
scripts\validate-local-task.cmd tasks\current.json
scripts\run-local-implementation.cmd tasks\current.json
```

CMD-обёртки не используют глобальный Python, PowerShell или `--yes`. Если отдельный `.ps1` всё же необходим, запускайте его только с временным process-level `powershell.exe -NoProfile -ExecutionPolicy Bypass -File <script-path>`; системную и пользовательскую Execution Policy менять не нужно.
