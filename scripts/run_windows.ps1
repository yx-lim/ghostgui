$ErrorActionPreference = "Stop"

Set-Location (Join-Path $PSScriptRoot "..")

if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    Write-Error ".venv was not found. Run scripts/install_windows.ps1 first."
}

. ".\.venv\Scripts\Activate.ps1"
ghostgui @args
