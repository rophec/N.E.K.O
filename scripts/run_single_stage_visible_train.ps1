param(
    [string]$Data = "datasets/mahjong_yolo_active/dataset_registry/02_yolo_hbb_safe_aug_221/hbb_v1_safe_aug/data.yaml",
    [string]$TestData = "datasets/mahjong_yolo_active/dataset_registry/05_yolo_hbb_fixed_test_21/data.yaml",
    [string]$Model = "yolo26n.pt",
    [int[]]$Epochs = @(150),
    [int]$ImgSize = 800,
    [int]$Batch = 8,
    [string]$Device = "0",
    [string]$Cache = "False",
    [double]$HsvH = 0.015,
    [double]$HsvS = 0.7,
    [double]$HsvV = 0.4,
    [double]$Scale = 0.2,
    [double]$Translate = 0.05,
    [double]$Erasing = 0.4,
    [string]$RunName = ""
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

if ([string]::IsNullOrWhiteSpace($RunName)) {
    $RunName = "single_stage_" + (Get-Date -Format "yyyyMMdd-HHmmss")
}

$outputRoot = Join-Path $repoRoot "runs/mahjong_yolo26_hbb/single_stage_improve/$RunName"
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null
$log = Join-Path $outputRoot "visible_train.log"
New-Item -ItemType File -Force -Path $log | Out-Null

$epochArgs = ($Epochs | ForEach-Object { [string]$_ }) -join " "
$childScript = Join-Path $outputRoot "run_visible_training.ps1"
$command = @"
`$ErrorActionPreference = 'Continue'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
`$Host.UI.RawUI.WindowTitle = 'Mahjong YOLO single-stage training: $RunName'
Set-Location '$repoRoot'
`$env:UV_PROJECT_ENVIRONMENT = '.venv-yolo-train'
Write-Host 'Output root: $outputRoot'
Write-Host 'Log: $log'
Write-Host 'Data: $Data'
Write-Host 'Model: $Model'
uv run --no-sync python tools/mahjong_yolo/run_epoch_comparison.py --data '$Data' --test-data '$TestData' --model '$Model' --output-root '$outputRoot' --epochs $epochArgs --imgsz $ImgSize --batch $Batch --device '$Device' --workers 0 --cache '$Cache' --optimizer AdamW --lr0 0.002 --hsv-h $HsvH --hsv-s $HsvS --hsv-v $HsvV --scale $Scale --translate $Translate --erasing $Erasing 2>&1 | Tee-Object -FilePath '$log'
Write-Host ''
Write-Host 'Training command finished or failed. Press Enter to close this window.'
Read-Host
"@

Set-Content -LiteralPath $childScript -Value $command -Encoding UTF8
Start-Process -FilePath powershell -ArgumentList @("-NoExit", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $childScript) -WorkingDirectory $repoRoot

[PSCustomObject]@{
    run_name = $RunName
    output_root = $outputRoot
    log = $log
    child_script = $childScript
    data = $Data
    test_data = $TestData
    model = $Model
    epochs = $Epochs -join ","
} | ConvertTo-Json
