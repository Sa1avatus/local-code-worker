param(
    [Parameter(Mandatory = $true)]
    [string]$TaskPath
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

try {
    & ".\.venv\Scripts\Activate.ps1"
    python -m local_code_worker validate-task --task $TaskPath
    if ($LASTEXITCODE -ne 0) { throw "Задание не прошло проверку." }
    python -m local_code_worker run --task $TaskPath
    if ($LASTEXITCODE -ne 0) { throw "Code Worker завершился с ошибкой." }
} catch {
    Write-Error $_
    exit 1
}
