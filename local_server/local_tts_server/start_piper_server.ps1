[CmdletBinding()]
param(
    [switch]$ServerOnly
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent (Split-Path -Parent $scriptDir)
$modelDir = Join-Path $scriptDir "piper_models"
Set-Location $repoRoot

$uvExe = (Get-Command uv -ErrorAction Stop).Source

if (-not $env:UV_CACHE_DIR) {
    $env:UV_CACHE_DIR = Join-Path $repoRoot ".uv-cache-local"
}

if (-not (Test-Path $env:UV_CACHE_DIR)) {
    New-Item -ItemType Directory -Force -Path $env:UV_CACHE_DIR | Out-Null
}

if (-not (Test-Path $modelDir)) {
    New-Item -ItemType Directory -Force -Path $modelDir | Out-Null
}

if (-not $env:LOCAL_TTS_DEFAULT_MODEL) {
    $env:LOCAL_TTS_DEFAULT_MODEL = "piper"
}

if (-not $env:LOCAL_TTS_PIPER_DEFAULT_VOICE) {
    $env:LOCAL_TTS_PIPER_DEFAULT_VOICE = "default"
}

if (-not $env:LOCAL_TTS_DEFAULT_VOICE) {
    $env:LOCAL_TTS_DEFAULT_VOICE = "piper:$env:LOCAL_TTS_PIPER_DEFAULT_VOICE"
}

if (-not $env:LOCAL_TTS_PIPER_VOICE_DIR) {
    $env:LOCAL_TTS_PIPER_VOICE_DIR = $modelDir
}

if (-not $env:LOCAL_TTS_HOST) {
    $env:LOCAL_TTS_HOST = "127.0.0.1"
}

if (-not $env:LOCAL_TTS_PORT) {
    $env:LOCAL_TTS_PORT = "50000"
}

$serverScript = Join-Path $scriptDir "piper_server.py"
$launcherScript = Join-Path $repoRoot "launcher.py"
$wsUrl = "ws://{0}:{1}" -f $env:LOCAL_TTS_HOST, $env:LOCAL_TTS_PORT
$healthUrl = "http://{0}:{1}/health" -f $env:LOCAL_TTS_HOST, $env:LOCAL_TTS_PORT
$firstModel = Get-ChildItem -Path $env:LOCAL_TTS_PIPER_VOICE_DIR -Filter "*.onnx" -File -ErrorAction SilentlyContinue | Select-Object -First 1

Write-Host ""
Write-Host "=== NEKO Piper Local TTS ===" -ForegroundColor Cyan
Write-Host "Repo Root : $repoRoot"
Write-Host "UV Cache  : $env:UV_CACHE_DIR"
Write-Host "Model Dir : $env:LOCAL_TTS_PIPER_VOICE_DIR"
Write-Host "Voice     : $env:LOCAL_TTS_DEFAULT_VOICE"
Write-Host "WS URL    : $wsUrl"
Write-Host "Health    : $healthUrl"
Write-Host ""
Write-Host "Fill this in NEKO custom API TTS:" -ForegroundColor Yellow
Write-Host "  $wsUrl"
Write-Host ""

if (-not $firstModel -and -not $env:LOCAL_TTS_PIPER_MODEL) {
    Write-Host "No Piper .onnx voice was found yet." -ForegroundColor Yellow
    Write-Host "Put a Piper voice pair here, then run this script again:"
    Write-Host "  $modelDir"
    Write-Host "Example files:"
    Write-Host "  voice-name.onnx"
    Write-Host "  voice-name.onnx.json"
    Write-Host ""
}

if ($ServerOnly) {
    & $uvExe run --no-project --with fastapi --with uvicorn --with numpy --with piper-tts python $serverScript --host $env:LOCAL_TTS_HOST --port $env:LOCAL_TTS_PORT
    exit $LASTEXITCODE
}

$serverArgs = @(
    "run",
    "--no-project",
    "--with", "fastapi",
    "--with", "uvicorn",
    "--with", "numpy",
    "--with", "piper-tts",
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
        throw "Piper local TTS server failed to become ready: $healthUrl"
    }

    Write-Host "Piper local TTS server ready, launching NEKO launcher..." -ForegroundColor Green
    & python $launcherScript
    exit $LASTEXITCODE
}
finally {
    if ($serverProcess -and -not $serverProcess.HasExited) {
        Stop-Process -Id $serverProcess.Id -Force -ErrorAction SilentlyContinue
    }
}
