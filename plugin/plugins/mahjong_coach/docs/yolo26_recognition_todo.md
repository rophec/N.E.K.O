# YOLO26 Recognition TODO

This file tracks unfinished work for the Mahjong Coach YOLO26 recognition mode.

## Runtime integration

- [x] Add `perception.tile_recognition_mode = "legacy" | "yolo26"`.
- [x] Keep `legacy` as the default mode.
- [x] Add UI controls for switching between legacy and YOLO26.
- [x] Add a lightweight YOLO26 backend interface that avoids importing `ultralytics`, `torch`, or `tensorflow` at plugin runtime.
- [x] Add optional table-surface detection and perspective-warp diagnostics inspired by AutoMajsoul.
- [x] Provide `opencv-python-headless` as the runtime dependency for the AutoMajsoul-style HSV/contour/warp path.
- [x] Replace the production single-sample quad approximation with the validated multi-sample cloth-component and Hough support-line flow.
- [x] Reject implausible support-line quads by coverage, convexity, span, direction, and extrapolation while retaining an explicit `legacy` detector mode.
- [x] Replace the old outermost-line rule with near-collinear clustering, merged span coverage, and weighted-median line fitting so avatar/UI slashes cannot dominate by extrapolation.
- [x] Visually and numerically verify stable support quads on all 18 available real screenshots without treating historical prepared warps as ground truth.
- [x] Add fallback metadata when the YOLO26 runtime artifact is missing or unavailable.
- [x] Show tile recognition mode and fallback reason in the plugin timing log.
- [x] Display self melds separately from the hand while keeping them on the same YOLO visible-tile path.
- [x] Use the detected and perspective-warped table image as the deployed HBB model input.
- [x] Add geometry-gated low-confidence recovery for the original bottom hand/self-meld row while keeping the warped river threshold unchanged and marking recovered meld identities unreliable.
- [x] Use tile orientation only as a narrow diagonal river-owner tie-breaker to prevent corner leakage without overriding genuine in-sector riichi rotations.
- [x] Recognize opponent meld shelves on the warped table, assign them to left/top/right opponents, and partition them into coherent three/four-tile groups.
- [x] Classify opponent groups as chi/pon/kan and gate called-tile identity recovery behind a unique turned tile plus a legal triplet/quad invariant.
- [x] Persist structured opponent melds across temporary missed frames and feed reliable exposed tiles into remaining-copy/wall calculations without treating them as genbutsu.
- [x] Display opponent meld groups by seat and kind in both the live panel and round archive.
- [x] Retain called discards as genbutsu history, mark them `claimed_into_meld`, and exclude that historical physical instance from remaining-copy/wall counts while counting the meld once.
- [x] Limit opponent claim linking to newly observed melds and a three-scan synchronization window so a long-standing meld cannot consume a later detector dropout; restrict chi to its legal source seat.
- [x] Treat the pure `numpy`/`Pillow` table-surface fallback as degraded legacy diagnostics only; use OpenCV support-line output for training-quality proof images.

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
- [x] Bundle the current `model.onnx` in the plugin's default model directory so normal YOLO runs do not depend on another worktree or a local override.
- [x] Build and inspect a `.neko-plugin` package to verify the model entry, uncompressed size, and SHA256 survive distribution packaging.
- [ ] Add a model replacement checklist so newer YOLO26 models can be swapped without changing plugin code.

## YOLO backend completion

- [x] Implement ONNXRuntime preprocessing and YOLO26 end-to-end detect output decoding.
- [ ] Add YOLO26-OBB rotated-output decoding; the current deployment candidate is HBB detect.
- [ ] Tune confidence thresholds and overlap suppression for small Mahjong tiles.
- [x] Save successful original-frame and warped-table diagnostic overlays, including opponent-meld group outlines.
- [x] Use the table-surface perspective warp as the default input for the deployed HBB model.
- [ ] Add a CLI/debug entry point that runs one screenshot through the YOLO path and writes proof images.

## Tests and acceptance

- [x] Add postprocessing tests using mock YOLO detections.
- [x] Add missing-model fallback tests.
- [x] Run real-model smoke tests on closed-hand and open-hand fixed-test screenshots.
- [x] Validate opponent-meld ownership/grouping on all 18 available real screenshots: six positive images and twelve zero-group images.
- [x] Re-run the three special table-edge screenshots through the full surface-warp and meld postprocess path with zero false opponent melds.
- [x] Verify the bundled default model artifact against its metadata SHA256 and load it without a path override in automated tests.
- [x] Cover called-discard deduplication, genbutsu retention, delayed river/meld synchronization, old-meld dropout rejection, legal chi source, and self-call linking in engine tests.
- [ ] Add screenshot fixture tests for at least three resolutions.
- [ ] Compare legacy and YOLO26 outputs on the same screenshot set.
- [ ] Require UI screenshots before marking the YOLO26 path ready for user testing.
