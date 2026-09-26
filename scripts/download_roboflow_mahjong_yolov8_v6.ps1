$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if (-not $env:ROBOFLOW_API_KEY) {
  throw "Set ROBOFLOW_API_KEY first, or download the Roboflow zip manually from the browser."
}

$env:UV_PROJECT_ENVIRONMENT = ".venv-yolo-train"
$DownloadDir = Join-Path $RepoRoot "datasets\mahjong_yolo_active\dataset_registry\06_external_roboflow_mahjong_yolov8_v6\raw_download"
New-Item -ItemType Directory -Force -Path $DownloadDir | Out-Null

uv run --no-sync --with roboflow python -c @"
from pathlib import Path
from roboflow import Roboflow

api_key = r"$env:ROBOFLOW_API_KEY"
download_dir = Path(r"$DownloadDir")
rf = Roboflow(api_key=api_key)
project = rf.workspace("mahjongdetection-klhs0").project("mahjong-yolov8")
version = project.version(6)

# EN: Roboflow's Python SDK accepts export format names documented by Roboflow.
# ZH: Roboflow Python SDK 使用其官方文档中的导出格式名。
dataset = version.download("yolov8", location=str(download_dir), overwrite=True)
print(dataset.location)
"@

uv run --no-sync python tools\mahjong_yolo\import_roboflow_mahjong_yolov8.py --overwrite
