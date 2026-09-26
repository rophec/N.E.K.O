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
4. Run the lightweight YOLO26 deployment artifact on the warped 800 x 800 table image.
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

## Current deployment candidate

The current runnable candidate is a single-stage YOLO26m HBB detector exported to ONNX. It recognizes all 34 tile identities and locations in one pass.

- Training run: `yolo26m_clean_rebuild_b8_e50_20260714`
- Runtime: ONNXRuntime, opset 18
- Input: RGB letterboxed `800 x 800` warped table image
- Output: end-to-end rows `[x1, y1, x2, y2, confidence, class_id]`
- SHA256: `fca49295f8d1e9134803f6d069406f7a2e8ef9c7223d9d1fe1ed4cacdf143521`
- Runtime imports: `onnxruntime`, `numpy`, `Pillow`, and the existing OpenCV table normalizer; no Ultralytics, PyTorch, or TensorFlow import is required.

The current runtime also separates an exposed self meld from the concealed hand, assigns detections only inside four explicit central river polygons, and keeps outer opponent melds or animation tiles out of `discard_piles`. In live mode, full YOLO snapshots are matched to history by player and bounding-box IoU: a changed class or a normal new discard needs two consecutive frames before it replaces or appends state, while a call window can append immediately. Opponent riichi is inferred from the orientation anomaly in that player's river and confirmed across two frames before defense mode is activated.

The HBB model is a deployment candidate, not the end of model work. OBB training/export and broader screenshot acceptance remain listed in the TODO.

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

The plugin panel exposes the first two coordinate spaces as separate previews:

- `原始截图`: the untouched captured Mahjong Soul window from `last_frame_path`.
- `变换后牌桌分区`: the `800 x 800` `warped_table` image used as the YOLO input, with the four river polygons, hand/meld zones, and current detections drawn in warped-table coordinates.

The transformed preview is produced in memory by `mahjong_coach_table_region_preview`. If table-surface detection fails, the panel reports the failure and leaves this preview empty; it must never substitute the raw frame under the transformed-preview label.
