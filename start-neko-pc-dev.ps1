param(
    [string]$NekoRoot = $PSScriptRoot,
    [string]$PcRoot = "",
    [int]$MainPort = 48911,
    [int]$MemoryPort = 48912,
    [int]$ToolPort = 48915,
    [switch]$SkipBackend,
    [switch]$InstallNpm
)

$ErrorActionPreference = "Stop"

function Resolve-ExistingDirectory {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Label
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "$Label does not exist: $Path"
    }
    return (Resolve-Path -LiteralPath $Path).Path
}

function Find-PcRoot {
    param([string]$Root)

    $parent = Split-Path -Parent $Root
    $candidates = @(
        (Join-Path $Root ".codexworktree\N.E.K.O.-PC"),
        (Join-Path $Root "N.E.K.O.-PC"),
        (Join-Path $parent "N.E.K.O.-PC"),
        (Join-Path $parent "neko-pc"),
        (Join-Path $parent "NEKO-PC")
    )

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath (Join-Path $candidate "package.json") -PathType Leaf) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }

    throw @"
Could not find N.E.K.O.-PC.

Pass it explicitly from PyCharm script parameters, for example:
  -PcRoot "F:\NEKO_bugfix\NEKO_latest_clean\.codexworktree\N.E.K.O.-PC"
"@
}

function Test-NekoHealth {
    param(
        [int]$Port,
        [string]$ServiceName
    )

    try {
        $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2
        return ($resp.app -eq "N.E.K.O")
    } catch {
        return $false
    }
}

function Test-AllNekoServices {
    return (
        (Test-NekoHealth -Port $MainPort -ServiceName "main") -and
        (Test-NekoHealth -Port $MemoryPort -ServiceName "memory") -and
        (Test-NekoHealth -Port $ToolPort -ServiceName "tool")
    )
}

function Drain-JobOutput {
    param($Job)

    if ($null -eq $Job) {
        return
    }

    Receive-Job -Job $Job -Keep -ErrorAction SilentlyContinue | ForEach-Object {
        Write-Host $_
    }
}

$NekoRoot = Resolve-ExistingDirectory -Path $NekoRoot -Label "N.E.K.O root"
if ([string]::IsNullOrWhiteSpace($PcRoot)) {
    $PcRoot = Find-PcRoot -Root $NekoRoot
} else {
    $PcRoot = Resolve-ExistingDirectory -Path $PcRoot -Label "N.E.K.O.-PC root"
}

Write-Host "N.E.K.O root:    $NekoRoot"
Write-Host "N.E.K.O.-PC root: $PcRoot"

$startedBackendJob = $null

try {
    if (-not $SkipBackend) {
        if (Test-AllNekoServices) {
            Write-Host "Existing N.E.K.O backend detected. Reusing it."
        } else {
            Write-Host "Starting N.E.K.O backend with: uv run launcher.py"
            $startedBackendJob = Start-Job -Name "neko-backend-dev" -ScriptBlock {
                param($Root)
                Set-Location -LiteralPath $Root
                & uv run launcher.py 2>&1 | ForEach-Object { "[backend] $_" }
            } -ArgumentList $NekoRoot

            Write-Host "Waiting for backend health checks..."
            $ready = $false
            for ($i = 1; $i -le 120; $i++) {
                Drain-JobOutput -Job $startedBackendJob

                if ($startedBackendJob.State -in @("Failed", "Stopped", "Completed")) {
                    Drain-JobOutput -Job $startedBackendJob
                    throw "Backend job exited before all health checks became ready. Job state: $($startedBackendJob.State)"
                }

                if (Test-AllNekoServices) {
                    $ready = $true
                    break
                }

                Start-Sleep -Seconds 1
            }

            Drain-JobOutput -Job $startedBackendJob

            if (-not $ready) {
                throw "N.E.K.O backend did not become ready on ports $MainPort/$MemoryPort/$ToolPort."
            }

            Write-Host "N.E.K.O backend is ready."
        }
    } else {
        Write-Host "SkipBackend is set. Not starting launcher.py."
    }

    Set-Location -LiteralPath $PcRoot

    if ($InstallNpm -or -not (Test-Path -LiteralPath (Join-Path $PcRoot "node_modules") -PathType Container)) {
        Write-Host "Installing N.E.K.O.-PC npm dependencies..."
        & npm install
        if ($LASTEXITCODE -ne 0) {
            throw "npm install failed with exit code $LASTEXITCODE"
        }
    }

    Write-Host "Starting N.E.K.O.-PC with: npm start"
    & npm start
    if ($LASTEXITCODE -ne 0) {
        throw "npm start failed with exit code $LASTEXITCODE"
    }
} finally {
    if ($null -ne $startedBackendJob) {
        Write-Host "Stopping backend job started by this script..."
        Stop-Job -Job $startedBackendJob -ErrorAction SilentlyContinue
        Remove-Job -Job $startedBackendJob -Force -ErrorAction SilentlyContinue
    }
}
