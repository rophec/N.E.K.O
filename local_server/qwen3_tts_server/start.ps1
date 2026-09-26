$ErrorActionPreference = "Stop"

Push-Location $PSScriptRoot
try {
    $env:KMP_DUPLICATE_LIB_OK = "TRUE"
    uv run python -m server.main
}
finally {
    Pop-Location
}
