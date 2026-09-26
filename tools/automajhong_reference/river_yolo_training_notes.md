# River YOLO Training Notes

This note records the current practical path for Mahjong Soul river recognition.

## Current boundary

- Local hand recognition stays on the original screenshot.
- River / exposed meld / center-table visible tiles use the tablecloth-warped
  `800x800` image.
- The current script generates a YOLO detection dataset for river tile boxes.
- It does **not** automatically assign 34 tile names, because the template
  classifier is not reliable enough and would poison the training data.

中文：

- 自家手牌在原图坐标识别。
- 牌河、副露、桌面中部可见牌在桌布透视变换后的 `800x800` 图上识别。
- 当前脚本先生成 YOLO 检测框数据集。
- 暂时不自动写 34 类牌名，因为模板分类会错，不能拿错误标签训练模型。

## Dataset draft command

For raw screenshots, run tablecloth warp first:

```powershell
uv run --with opencv-python-headless --with numpy python tools/automajhong_reference/run_tablecloth_warp_batch.py `
  --input-root "C:\Users\19079\Desktop\识别图" `
  --output-root "C:\Users\19079\Desktop\识别图\prelabel_20260626_v3\table_warp"
```

Then create river pre-labels from the successful warped images:

```powershell
uv run --with opencv-python-headless --with numpy python tools/automajhong_reference/prepare_river_yolo_dataset.py `
  --input-root "C:\Users\19079\Desktop\识别图\prelabel_20260626_v3\table_warp" `
  --output-root "C:\Users\19079\Desktop\识别图\prelabel_20260626_v3\river_detection_draft" `
  --split train
```

Current raw pre-label result:

```text
Input:  C:\Users\19079\Desktop\识别图
Warp:   C:\Users\19079\Desktop\识别图\prelabel_20260626_v3\table_warp
Labels: C:\Users\19079\Desktop\识别图\prelabel_20260626_v3\river_detection_draft
Warp success: 18 images
Warp failed:  1 image, 牌.png
River slots:  218 pre-label boxes
Review sheet:
  C:\Users\19079\Desktop\识别图\prelabel_20260626_v3\river_detection_draft\review\debug\00_slots_contact_sheet.png
Review CSV:
  C:\Users\19079\Desktop\识别图\prelabel_20260626_v3\river_detection_draft\review\river_review.csv
```

Important: these are pre-labels. Human review is required before training a
trusted detector or a 34-class tile recognizer.

中文：这批是预标注，不是最终真值。训练前必须检查框是否包住单张牌；坏框填
`review_keep=0`，牌名要人工填入 `review_tile_label`。

Legacy command for already-warped test folders:

```powershell
uv run --with opencv-python-headless --with numpy python tools/automajhong_reference/prepare_river_yolo_dataset.py `
  --input-root "C:\Users\19079\Desktop\识别图\normal_tablecloth_flow_11_12_13" `
  --output-root "C:\Users\19079\Desktop\识别图\yolo_datasets\river_detection_draft" `
  --split train

uv run --with opencv-python-headless --with numpy python tools/automajhong_reference/prepare_river_yolo_dataset.py `
  --input-root "C:\Users\19079\Desktop\识别图\normal_tablecloth_flow_11_12_13" `
  --output-root "C:\Users\19079\Desktop\识别图\yolo_datasets\river_detection_draft" `
  --split val
```

## Output

```text
C:\Users\19079\Desktop\识别图\yolo_datasets\river_detection_draft
├─ data.yaml
├─ images
│  ├─ train
│  └─ val
├─ labels
│  ├─ train
│  └─ val
├─ review
│  ├─ crops
│  ├─ debug
│  └─ river_review.csv
└─ summary.json
```

## Training command, detection-only sanity check

Use Ultralytics only during training/export. Do not add Ultralytics, PyTorch, or
TensorFlow to the plugin runtime dependency path.

The exact YOLO26 model filename can change with Ultralytics releases. Prefer the
official YOLO26 detect model when available. If the local package does not expose
YOLO26 yet, use the closest current small detect model only as a pipeline sanity
check and record that fallback.

Example:

```powershell
uv run --with ultralytics yolo detect train `
  model=yolo26n.pt `
  data="C:\Users\19079\Desktop\识别图\yolo_datasets\river_detection_draft\data.yaml" `
  epochs=3 `
  imgsz=800 `
  batch=2 `
  project="C:\Users\19079\Desktop\识别图\yolo_runs" `
  name=river_detection_draft_yolo26
```

Verified local sanity run:

```powershell
uv run --with ultralytics yolo detect train `
  model=yolo26n.pt `
  data="C:\Users\19079\Desktop\识别图\yolo_datasets\river_detection_draft\data.yaml" `
  epochs=1 `
  imgsz=800 `
  batch=2 `
  project="C:\Users\19079\Desktop\识别图\yolo_runs" `
  name=river_detection_draft_yolo26_sanity `
  exist_ok=True
```

Result:

```text
C:\Users\19079\Desktop\识别图\yolo_runs\river_detection_draft_yolo26_sanity\weights\best.pt
```

Important: this run only proves the YOLO26 training pipe works. It is not a
usable recognizer because the dataset has only 3 images and one detection class
named `tile`.

## Export sanity check

Verified command:

```powershell
uv run --with ultralytics --with onnx --with onnxslim yolo export `
  model="C:\Users\19079\Desktop\识别图\yolo_runs\river_detection_draft_yolo26_sanity\weights\best.pt" `
  format=onnx `
  imgsz=800 `
  simplify=True
```

Result:

```text
C:\Users\19079\Desktop\识别图\yolo_runs\river_detection_draft_yolo26_sanity\weights\best.onnx
```

Verified ONNX inference command:

```powershell
uv run --with ultralytics --with onnxruntime yolo predict `
  task=detect `
  model="C:\Users\19079\Desktop\识别图\yolo_runs\river_detection_draft_yolo26_sanity\weights\best.onnx" `
  source="C:\Users\19079\Desktop\识别图\yolo_datasets\river_detection_draft\images\val\11_river.png" `
  imgsz=800 `
  conf=0.001 `
  project="C:\Users\19079\Desktop\识别图\yolo_runs" `
  name=river_detection_draft_onnx_predict `
  exist_ok=True `
  save=True
```

Result:

```text
C:\Users\19079\Desktop\识别图\yolo_runs\river_detection_draft_onnx_predict
```

The ONNX file loads and runs with ONNX Runtime. The prediction quality is not
meaningful yet because the model was trained for only one epoch on draft boxes.

## From box detector to real tile recognition

1. Open `review/river_review.csv`.
2. For each crop in `review/crops`, fill:
   - `review_tile_label`: one of `1m..9m`, `1p..9p`, `1s..9s`, `1z..7z`.
   - `review_keep`: `1` for valid tile, `0` for bad split/crop.
3. Convert reviewed rows into a 34-class YOLO dataset.
4. Train the 34-class model.
5. Export to a lightweight runtime artifact, such as ONNX, OpenVINO, NCNN, or
   another Ultralytics-supported export that does not require the full training
   framework in the plugin.

中文后续：

1. 打开 `review/river_review.csv`。
2. 对每张 `review/crops` 里的裁图填写真实牌名。
3. 坏的拆槽填 `review_keep=0`。
4. 审核后再生成 34 类 YOLO 数据集。
5. 训练并导出轻量运行产物，插件运行时只加载导出模型，不塞 PyTorch /
   TensorFlow / 完整 Ultralytics。

Conversion command after review:

```powershell
uv run python tools/automajhong_reference/build_reviewed_river_yolo34_dataset.py `
  --draft-root "C:\Users\19079\Desktop\识别图\yolo_datasets\river_detection_draft" `
  --output-root "C:\Users\19079\Desktop\识别图\yolo_datasets\river_tile34_reviewed" `
  --split train
```
