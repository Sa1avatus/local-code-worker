$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

try {
    & ".\.venv\Scripts\Activate.ps1"
    python -m local_code_worker check-connection
    if ($LASTEXITCODE -ne 0) { throw "Проверка Ollama завершилась с ошибкой." }
} catch {
    Write-Error $_
    exit 1
}
