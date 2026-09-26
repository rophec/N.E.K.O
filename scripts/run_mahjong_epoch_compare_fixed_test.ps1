$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

# Keep Ultralytics progress output readable in Windows Terminal.
# 保证 Windows 终端里 Ultralytics 的进度条和指标不会被 GBK 解码成乱码。
chcp 65001 | Out-Null
$Utf8NoBom = New-Object System.Text.UTF8Encoding $false
[Console]::OutputEncoding = $Utf8NoBom
$OutputEncoding = $Utf8NoBom
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:LANG = "C.UTF-8"
$env:LC_ALL = "C.UTF-8"

$env:UV_PROJECT_ENVIRONMENT = ".venv-yolo-train"
$OutputRoot = Join-Path $RepoRoot "runs\mahjong_yolo26_hbb\epoch_compare_fixed_test"
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$LogPath = Join-Path $OutputRoot "epoch_compare_fixed_test_launcher_$Timestamp.log"

Write-Host "Mahjong YOLO26 HBB epoch comparison with fixed test set"
Write-Host "Repo: $RepoRoot"
Write-Host "Log:  $LogPath"
Write-Host "Epochs: 80, 150, 200"
Write-Host "Metrics: validation + fixed_test_21 precision/recall/mAP50/mAP50-95"
Write-Host ""

uv run --no-sync python tools\mahjong_yolo\run_epoch_comparison.py `
  --epochs 80 150 200 `
  --imgsz 800 `
  --batch 8 `
  --device 0 `
  --workers 0 `
  --conf 0.05 `
  --output-root "$OutputRoot" `
  --data "F:\NEKO_bugfix\A_NEKO_Base\N.E.K.O\.codexworktree\automajhong\datasets\mahjong_yolo_active\dataset_registry\02_yolo_hbb_safe_aug_221\hbb_v1_safe_aug\data.yaml" `
  --test-data "F:\NEKO_bugfix\A_NEKO_Base\N.E.K.O\.codexworktree\automajhong\datasets\mahjong_yolo_active\dataset_registry\05_yolo_hbb_fixed_test_21\data.yaml" 2>&1 | Tee-Object -FilePath $LogPath

Write-Host ""
Write-Host "Done. Press Enter to close this terminal."
Read-Host | Out-Null
