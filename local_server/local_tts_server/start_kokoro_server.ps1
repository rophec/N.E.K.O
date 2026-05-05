[CmdletBinding()]
param(
    [switch]$ServerOnly
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent (Split-Path -Parent $scriptDir)
Set-Location $repoRoot

$uvExe = (Get-Command uv -ErrorAction Stop).Source

if (-not $env:UV_CACHE_DIR) {
    $env:UV_CACHE_DIR = Join-Path $repoRoot ".uv-cache-local"
}

if (-not (Test-Path $env:UV_CACHE_DIR)) {
    New-Item -ItemType Directory -Force -Path $env:UV_CACHE_DIR | Out-Null
}

if (-not $env:LOCAL_TTS_DEFAULT_MODEL) {
    $env:LOCAL_TTS_DEFAULT_MODEL = "kokoro"
}

if (-not $env:LOCAL_TTS_KOKORO_DEFAULT_VOICE) {
    $env:LOCAL_TTS_KOKORO_DEFAULT_VOICE = "zf_xiaobei"
}

if (-not $env:LOCAL_TTS_DEFAULT_VOICE) {
    $env:LOCAL_TTS_DEFAULT_VOICE = "kokoro:zf_xiaobei"
}

if (-not $env:LOCAL_TTS_HOST) {
    $env:LOCAL_TTS_HOST = "127.0.0.1"
}

if (-not $env:LOCAL_TTS_PORT) {
    $env:LOCAL_TTS_PORT = "50000"
}

$serverScript = Join-Path $scriptDir "server.py"
$launcherScript = Join-Path $repoRoot "launcher.py"
$wsUrl = "ws://{0}:{1}" -f $env:LOCAL_TTS_HOST, $env:LOCAL_TTS_PORT
$healthUrl = "http://{0}:{1}/health" -f $env:LOCAL_TTS_HOST, $env:LOCAL_TTS_PORT

Write-Host ""
Write-Host "=== NEKO Kokoro Local TTS ===" -ForegroundColor Cyan
Write-Host "Repo Root : $repoRoot"
Write-Host "UV Cache  : $env:UV_CACHE_DIR"
Write-Host "Voice     : $env:LOCAL_TTS_DEFAULT_VOICE"
Write-Host "WS URL    : $wsUrl"
Write-Host "Health    : $healthUrl"
Write-Host ""
Write-Host "NEKO 自定义 API TTS 请填写：" -ForegroundColor Yellow
Write-Host "  $wsUrl"
Write-Host ""

if ($ServerOnly) {
    & $uvExe run --no-project --with fastapi --with uvicorn --with kokoro-onnx python $serverScript --host $env:LOCAL_TTS_HOST --port $env:LOCAL_TTS_PORT
    exit $LASTEXITCODE
}

$serverArgs = @(
    "run",
    "--no-project",
    "--with", "fastapi",
    "--with", "uvicorn",
    "--with", "kokoro-onnx",
    "python",
    $serverScript,
    "--host", $env:LOCAL_TTS_HOST,
    "--port", $env:LOCAL_TTS_PORT
)

$serverProcess = Start-Process -FilePath $uvExe -ArgumentList $serverArgs -WorkingDirectory $repoRoot -PassThru

try {
    $ready = $false
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        Start-Sleep -Milliseconds 500
        try {
            $health = Invoke-WebRequest $healthUrl -UseBasicParsing -TimeoutSec 2
            if ($health.StatusCode -eq 200) {
                $ready = $true
                break
            }
        } catch {
        }
    }

    if (-not $ready) {
        throw "Kokoro local TTS server failed to become ready: $healthUrl"
    }

    Write-Host "Kokoro local TTS server ready, launching NEKO launcher..." -ForegroundColor Green
    & python $launcherScript
    exit $LASTEXITCODE
}
finally {
    if ($serverProcess -and -not $serverProcess.HasExited) {
        Stop-Process -Id $serverProcess.Id -Force -ErrorAction SilentlyContinue
    }
}
