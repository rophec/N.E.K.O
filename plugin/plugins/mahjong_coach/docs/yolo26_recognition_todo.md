# YOLO26 Recognition TODO

This file tracks unfinished work for the Mahjong Coach YOLO26 recognition mode.

## Runtime integration

- [x] Add `perception.tile_recognition_mode = "legacy" | "yolo26"`.
- [x] Keep `legacy` as the default mode.
- [x] Add UI controls for switching between legacy and YOLO26.
- [x] Add a lightweight YOLO26 backend interface that avoids importing `ultralytics`, `torch`, or `tensorflow` at plugin runtime.
- [x] Add optional table-surface detection and perspective-warp diagnostics inspired by AutoMajsoul.
- [x] Provide `opencv-python-headless` as the runtime dependency for the AutoMajsoul-style HSV/contour/warp path.
- [x] Add fallback metadata when the YOLO26 runtime artifact is missing or unavailable.
- [x] Show tile recognition mode and fallback reason in the plugin timing log.
- [x] Display self melds separately from the hand while keeping them on the same YOLO visible-tile path.
- [x] Use the detected and perspective-warped table image as the deployed HBB model input.
- [ ] Treat the pure `numpy`/`Pillow` table-surface fallback as degraded diagnostics only; use OpenCV output for training-quality proof images.

## Model training pipeline

- [ ] Collect Mahjong Soul screenshots across resolutions, title-bar states, window scales, and table themes.
- [ ] Include closed hands, open hands, self melds, all four discard rivers, late-round dense rivers, and action-window frames.
- [ ] Define the annotation schema for YOLO26-OBB: tile class, rotated box, area tag, and optional owner tag.
- [ ] Build or choose an annotation workflow that can export Ultralytics-compatible YOLO OBB labels.
- [ ] Train the first YOLO26-OBB model for 34 tile classes: `1m-9m`, `1p-9p`, `1s-9s`, `1z-7z`.
- [ ] Validate per-zone accuracy separately for hand, self meld, self river, left river, top river, and right river.
- [x] Export the current YOLO26m HBB candidate as a lightweight ONNX artifact that does not require PyTorch, TensorFlow, or the full Ultralytics package at runtime.
- [x] Add `labels.json` and `metadata.json` beside the exported model artifact.
- [x] Document the deployed training run, export format, and model checksum.
- [ ] Add a model replacement checklist so newer YOLO26 models can be swapped without changing plugin code.

## YOLO backend completion

- [x] Implement ONNXRuntime preprocessing and YOLO26 end-to-end detect output decoding.
- [ ] Add YOLO26-OBB rotated-output decoding; the current deployment candidate is HBB detect.
- [ ] Tune confidence thresholds and overlap suppression for small Mahjong tiles.
- [ ] Save diagnostic overlays for successful YOLO frames, not only for postprocessed detections.
- [x] Use the table-surface perspective warp as the default input for the deployed HBB model.
- [ ] Add a CLI/debug entry point that runs one screenshot through the YOLO path and writes proof images.

## Tests and acceptance

- [x] Add postprocessing tests using mock YOLO detections.
- [x] Add missing-model fallback tests.
- [x] Run real-model smoke tests on closed-hand and open-hand fixed-test screenshots.
- [ ] Add screenshot fixture tests for at least three resolutions.
- [ ] Compare legacy and YOLO26 outputs on the same screenshot set.
- [ ] Require UI screenshots before marking the YOLO26 path ready for user testing.
