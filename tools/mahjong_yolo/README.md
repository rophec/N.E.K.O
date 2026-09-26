# Mahjong YOLO26 HBB Training Draft

This folder contains reviewable training utilities for the manually labeled
Mahjong HBB dataset.

## Data locations

- Raw HBB export:
  `datasets/mahjong_yolo_active/manual_labeling_round2/exports/yolo_hbb_export`
- Prepared train/val dataset:
  `datasets/mahjong_yolo_active/train_ready/hbb_v1`
- Local YOLO26 HBB seed model:
  `yolo26n.pt`

## Prepare data

```powershell
uv run python tools/mahjong_yolo/prepare_hbb_dataset.py --clean
```

The prepare step validates YOLO HBB labels, makes a deterministic 80/20
train/val split, writes `data.yaml`, and writes `dataset_manifest.json`.

## Review training command

```powershell
uv run python tools/mahjong_yolo/train_yolo26_hbb.py --dry-run
```

## Start training after review

```powershell
uv run python tools/mahjong_yolo/train_yolo26_hbb.py
```

The training script imports `ultralytics` only when training actually starts.
This keeps plugin runtime separate from the training environment.

The default training arguments intentionally disable horizontal and vertical
flips. Mahjong tile faces contain non-symmetric glyphs, so mirrored tiles become
invalid training samples and can damage class recognition.

## Validate raw screenshots

Put raw validation screenshots here:

```text
datasets/mahjong_yolo_active/manual_validation/images
```

Then run the full raw-image pipeline with the training virtual environment:

```powershell
uv run --python .venv-yolo-train\Scripts\python.exe --no-project python tools/mahjong_yolo/run_raw_validation_pipeline.py
```

The script creates a timestamped folder under:

```text
runs/mahjong_yolo26_hbb/manual_validation
```

Each image gets its own subfolder with numbered outputs:

```text
01-source.png
02-mask-before.png
03-mask-after.png
04-largest-component.png
05-external-contour.png
06-selected-lines.png
07-lines-on-source.png
08-extended-lines.png
09-warp-square-800.png
10-yolo-prediction.png
10-yolo-prediction.json
summary.json
```
