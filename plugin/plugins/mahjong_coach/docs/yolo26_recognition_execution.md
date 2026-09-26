# YOLO26 Recognition Execution Notes

Mahjong Coach has two tile recognition modes.

- `legacy`: fixed screenshot regions, ROI occupancy checks, and the existing template/ONNX tile classifier.
- `yolo26`: a unified YOLO26 visible-tile path for hand tiles, self melds, and discard rivers.

The runtime plugin must remain lightweight. Training may use Ultralytics YOLO26, but the plugin runtime must not require `ultralytics`, `torch`, `tensorflow`, or a training environment. The final deployment artifact may be ONNX, OpenVINO, TensorRT, DirectML, or another lightweight YOLO export, as long as it runs out of the box inside the plugin.

The table-surface normalization path intentionally depends on `opencv-python-headless`. This mirrors the useful part of AutoMajsoul's `Android/ingame_recognizer.py`: sample the table cloth color in HSV, find the largest table contour, and warp the table quad into a stable top-down image. The dependency is the headless `cv2` wheel, not a full GUI OpenCV install.

## Legacy flow

1. Capture the Mahjong Soul window.
2. Scale fixed layouts from the calibration profile.
3. Scan hand, self meld, and river regions separately.
4. Use ROI brightness metrics to decide whether a slot is occupied.
5. Classify occupied crops with the existing tile classifier.
6. Convert results to `hand_tiles`, `melds`, `discard_piles`, and `visible_tiles`.

## YOLO26 flow

1. Capture the Mahjong Soul window.
2. Resolve or estimate the Mahjong Soul content area.
3. Detect the table surface from color samples and edge contours, then save the table quad and perspective-warp diagnostics when debugging is enabled.
4. Run the lightweight YOLO26 deployment artifact.
5. Postprocess detections:
   - drop empty/low-confidence classes,
   - suppress overlapping boxes,
   - split detections into `hand`, `self_meld`, and `river`,
   - assign river detections to `self`, `left_opponent`, `top_opponent`, or `right_opponent`,
   - sort hand tiles from left to right,
   - group self meld tiles for display and strategy.
6. Convert the unified result to existing engine result objects.
7. If YOLO26 is unavailable, fall back to `legacy` and show `fallback_reason`.

The table-surface detector uses OpenCV for the production/reference path. Its HSV mask, contour filtering, polygon approximation, and perspective transform are aligned with AutoMajsoul so different Mahjong Soul window sizes are normalized before YOLO work. A pure `numpy`/`Pillow` fallback remains for degraded diagnostics when `cv2` cannot be imported, but training data generation and serious recognition debugging should use the OpenCV path.

Runtime dependency boundary:

- Required for table normalization: `opencv-python-headless`.
- Still forbidden at plugin runtime: `ultralytics`, `torch`, `tensorflow`, training notebooks, and raw training environments.
- Allowed at plugin runtime: lightweight exported YOLO artifacts plus their small inference backend, such as ONNXRuntime if ONNX is chosen.

## Runtime artifact layout

Place the deployment artifact here:

```text
plugin/plugins/mahjong_coach/data/models/yolo26_mahjong/
```

Required files:

- `labels.json`: tile labels or names.
- `metadata.json`: runtime format and model filename.
- The exported model file, for example `model.onnx`.

Example metadata:

```json
{
  "runtime": "onnxruntime",
  "model_file": "model.onnx",
  "model_family": "yolo26",
  "task": "obb",
  "dataset_version": "todo",
  "exported_at": "todo"
}
```

## Training TODO

Model training is required before the YOLO26 mode can outperform legacy recognition.

1. Collect screenshots across resolutions, title-bar states, game themes, and open/closed hand states.
2. Annotate all visible tile faces in the hand, self melds, and all four rivers.
3. Prefer YOLO26-OBB labels because Mahjong Soul river tiles can be rotated or skewed.
4. Train the model with Ultralytics YOLO26 outside the plugin runtime.
5. Validate by zone: hand, self meld, self river, left river, top river, right river.
6. Export a lightweight artifact and place it under `data/models/yolo26_mahjong/`.
7. Record the dataset version, training command, export command, validation metrics, and checksum.

## Debug proof requirements

For every real screenshot debug run, save or expose:

- source screenshot,
- content/zone overlay,
- table quad overlay and perspective-warp image,
- YOLO detection overlay,
- accepted/rejected detections with class and confidence,
- final hand, meld, and river outputs.

Generated debug images should stay out of commits unless explicitly requested.
