$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

try {
    if (-not (Test-Path -LiteralPath ".venv")) {
        python -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw "Не удалось создать .venv." }
    }
    & ".\.venv\Scripts\Activate.ps1"
    python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "Не удалось обновить pip." }
    python -m pip install -e ".[dev]"
    if ($LASTEXITCODE -ne 0) { throw "Не удалось установить Code Worker." }
} catch {
    Write-Error $_
    exit 1
}
