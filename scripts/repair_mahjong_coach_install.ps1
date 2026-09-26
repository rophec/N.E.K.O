[CmdletBinding()]
param(
    [switch]$Apply,
    [string]$NekoDataRoot = ""
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host "[Mahjong Coach] $Message"
}

function Resolve-SafeChildPath {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$Child
    )
    $resolvedRoot = [System.IO.Path]::GetFullPath($Root).TrimEnd('\')
    $resolvedChild = [System.IO.Path]::GetFullPath($Child)
    if (-not $resolvedChild.StartsWith($resolvedRoot + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to operate outside the N.E.K.O data directory: $resolvedChild"
    }
    return $resolvedChild
}

if ([string]::IsNullOrWhiteSpace($NekoDataRoot)) {
    if ([string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) {
        throw "LOCALAPPDATA is unavailable; pass -NekoDataRoot explicitly."
    }
    $NekoDataRoot = Join-Path $env:LOCALAPPDATA "N.E.K.O"
}

$dataRoot = [System.IO.Path]::GetFullPath($NekoDataRoot)
if (-not (Test-Path -LiteralPath $dataRoot -PathType Container)) {
    throw "N.E.K.O data directory was not found: $dataRoot"
}

$pluginsRoot = Resolve-SafeChildPath -Root $dataRoot -Child (Join-Path $dataRoot "plugins")
$profilesRoot = Resolve-SafeChildPath -Root $dataRoot -Child (Join-Path $dataRoot ".neko-package-profiles")

Write-Step "Data directory: $dataRoot"
Write-Step "This tool never deletes plugin files or user profiles."

$runningNeko = @(
    Get-Process -ErrorAction SilentlyContinue |
        Where-Object { $_.ProcessName -match '^(N\.E\.K\.O|NEKO)$' }
)
if ($Apply -and $runningNeko.Count -gt 0) {
    Write-Error "N.E.K.O is still running. Close it completely, then run this tool again."
    exit 3
}

$candidates = @()
if (Test-Path -LiteralPath $pluginsRoot -PathType Container) {
    foreach ($directory in Get-ChildItem -LiteralPath $pluginsRoot -Directory -Force) {
        $nameMatches = $directory.Name -match '^mahjong_coach(?:_\d+)?$'
        $manifestMatches = $false
        $manifestPath = Join-Path $directory.FullName "plugin.toml"
        if (Test-Path -LiteralPath $manifestPath -PathType Leaf) {
            try {
                $manifestText = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8
                $manifestMatches = $manifestText -match '(?m)^\s*id\s*=\s*"mahjong_coach"\s*$'
            }
            catch {
                # An unreadable manifest in the canonical directory is itself
                # an identity-less remainder and should still be recoverable.
                $manifestMatches = $false
            }
        }
        if ($nameMatches -or $manifestMatches) {
            $candidates += $directory
        }
    }
}

if ($candidates.Count -eq 0) {
    Write-Step "No conflicting Mahjong Coach plugin directory was found."
    if (Test-Path -LiteralPath (Join-Path $profilesRoot "mahjong_coach")) {
        Write-Step "The existing mahjong_coach profile was preserved; current packages do not overwrite it."
    }
    Write-Step "If import still fails, collect the install-plan reason from a newer N.E.K.O build."
    exit 0
}

Write-Step "Conflicting or stale plugin directories:"
foreach ($candidate in $candidates) {
    Write-Host "  - $($candidate.FullName)"
}

if (-not $Apply) {
    Write-Step "Audit only. Re-run with -Apply to move these directories into a recoverable backup."
    exit 2
}

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$recoveryRoot = Resolve-SafeChildPath -Root $dataRoot -Child (
    Join-Path $dataRoot ".mahjong-coach-install-recovery\$stamp"
)
New-Item -ItemType Directory -Path $recoveryRoot -Force | Out-Null

$moved = 0
foreach ($candidate in $candidates) {
    $source = Resolve-SafeChildPath -Root $pluginsRoot -Child $candidate.FullName
    $destination = Join-Path $recoveryRoot $candidate.Name
    if (Test-Path -LiteralPath $destination) {
        $destination = Join-Path $recoveryRoot ("{0}-{1}" -f $candidate.Name, $moved)
    }
    Move-Item -LiteralPath $source -Destination $destination
    $moved += 1
    Write-Step "Backed up $source -> $destination"
}

if (Test-Path -LiteralPath (Join-Path $pluginsRoot "mahjong_coach")) {
    throw "The canonical target still exists after recovery; installation remains unsafe."
}

Write-Step "Recovery complete. $moved plugin director$(if ($moved -eq 1) { 'y' } else { 'ies' }) moved."
Write-Step "User profiles were not changed. Start N.E.K.O and import the latest Mahjong Coach package again."
Write-Step "Backup directory: $recoveryRoot"

