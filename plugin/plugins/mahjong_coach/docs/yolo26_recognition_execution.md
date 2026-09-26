# YOLO26 Recognition Execution Notes

Mahjong Coach has two tile recognition modes.

- `legacy`: fixed screenshot regions, ROI occupancy checks, and the existing template/ONNX tile classifier.
- `yolo26`: a shared YOLO26 backend with separate original-frame hand/self-meld and warped-table river/opponent-meld passes.

The runtime plugin must remain lightweight. Training may use Ultralytics YOLO26, but the plugin runtime must not require `ultralytics`, `torch`, `tensorflow`, or a training environment. The final deployment artifact may be ONNX, OpenVINO, TensorRT, DirectML, or another lightweight YOLO export, as long as it runs out of the box inside the plugin.

The table-surface normalization path intentionally depends on `opencv-python-headless`. It samples several likely cloth areas in HSV, keeps the largest connected cloth component, extracts only its external contour, groups near-collinear Hough segments, and fits four directional support lines from their distributed coverage. The dependency is the headless `cv2` wheel, not a full GUI OpenCV install.

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
3. Detect the table surface from multiple color samples, the largest cloth component, its external contour, and four support-line clusters. Rank a cluster primarily by merged edge coverage and only secondarily by how far outside it lies, so a short avatar/UI slash cannot win merely by looking outermost after extrapolation. Reject geometrically implausible or weakly supported quads before perspective warping, then save the complete diagnostic chain when debugging is enabled.
4. Run the same lightweight YOLO26 deployment artifact twice:
   - on the original screenshot for the near-facing concealed hand and self melds,
   - on the perspective-warped `800 x 800` table for discard rivers and opponent meld shelves.
5. Postprocess the two coordinate spaces independently:
   - drop empty/low-confidence classes,
   - suppress overlapping boxes,
   - find the original-frame bottom tile row from the detected baselines and split a separated right-side self meld,
   - keep the normal `0.25` acceptance threshold, but decode an original-bottom-only recovery pool down to `0.01`; accept an interior hand tile only above `0.15` when two accepted neighbors bracket it, and accept low-confidence meld boxes only when they form complete aligned groups of three or four and the effective hand count is exactly 13 or 14,
   - mark geometrically recovered meld identities as unreliable so the strategy layer uses the recovered open-meld count without trusting low-confidence tile names,
   - estimate the river center from the current warped-frame detection distribution,
   - assign warped river detections to `self`, `left_opponent`, `top_opponent`, or `right_opponent` by directional sector; only inside a narrow diagonal ambiguity band, use the detected tile orientation to break the owner tie so a right/left river corner tile cannot leak into the top/bottom river and create a false riichi declaration,
   - reject farther warped-table tiles instead of treating them as discards,
   - promote only far-out tiles on the correct seat's meld-side corner into opponent-meld candidates; cluster them along that seat's table edge and partition each shelf into coherent groups of three or four,
   - require exactly one called-tile orientation and classify exact triplets, sequences, and quads as `pon`, `chi`, and `kan`; this rejects aligned dora indicators, and when the turned called tile conflicts with an otherwise exact triplet/quad, correct only that tile and preserve `observed_tiles` plus a structured correction reason,
   - keep isolated outer detections and incomplete groups as `excluded_table_tile`, so UI/animation tiles cannot become a meld merely by being outside the river,
   - sort hand tiles from left to right,
   - group self meld tiles for display and strategy.
6. Convert the unified result to existing engine result objects, including structured `opponent_melds` and reliable `opponent_meld_tiles`.
7. If either coordinate-space pass is unavailable, fall back to `legacy` for that component and show `fallback_reason`. A table-warp failure no longer discards a valid original-frame hand result.

The default `support_lines` table-surface mode uses OpenCV for the production path. Its seven HSV samples, connected-component selection, external-contour filtering, directional line clustering, weighted-median fitting, and perspective transform are independent of any historical prepared-image geometry. It records per-side support coverage, component/quad area ratios, corner extrapolation, line angles, and a quality score. If this path cannot form a trustworthy quad, the YOLO river component is reported unavailable so the coach can use its existing component-level legacy recognition; it does not silently stretch an inner river contour.

The previous single-sample contour approximation and whole-frame Canny paths remain available through `detect_table_surface(..., mode="legacy")`. Callers may also opt into `legacy_fallback=True` for compatibility diagnostics. Neither behavior is the production default.

Runtime dependency boundary:

- Required for table normalization: `opencv-python-headless`.
- Still forbidden at plugin runtime: `ultralytics`, `torch`, `tensorflow`, training notebooks, and raw training environments.
- Allowed at plugin runtime: lightweight exported YOLO artifacts plus their small inference backend, such as ONNXRuntime if ONNX is chosen.

## Runtime artifact layout

The current deployment artifact is bundled at the default runtime path:

```text
plugin/plugins/mahjong_coach/data/models/yolo26_mahjong/
```

Required files:

- `labels.json`: tile labels or names.
- `metadata.json`: runtime format and model filename.
- The exported model file, for example `model.onnx`.

The loader resolves this directory relative to the plugin, so a normal `yolo26`
run needs no path override. If any required file is absent or the runtime cannot
load it, only that recognition component reports a `fallback_reason` and uses
the retained legacy path. The N.E.K.O plugin builder recursively includes this
directory: the verified `.neko-plugin` contained the ONNX at
`payload/plugins/mahjong_coach/data/models/yolo26_mahjong/model.onnx` with the
same size and SHA256 as the source artifact.

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
- Inputs: native-size RGB original screenshot for hand/meld plus RGB letterboxed `800 x 800` warped table for rivers
- Output: end-to-end rows `[x1, y1, x2, y2, confidence, class_id]`
- SHA256: `fca49295f8d1e9134803f6d069406f7a2e8ef9c7223d9d1fe1ed4cacdf143521`
- Bundled model: `model.onnx`, 81,915,072 bytes, tracked as a binary artifact.
- Runtime imports: `onnxruntime`, `numpy`, `Pillow`, and the existing OpenCV table normalizer; no Ultralytics, PyTorch, or TensorFlow import is required.

The current runtime separates an exposed self meld from the concealed hand in original-frame coordinates. A low-confidence recovery pool exists only for that original bottom row; the warped-table river/opponent-meld pass still uses the configured acceptance threshold without recovery. Recovery is rejected unless the geometry and Mahjong tile-count invariant close together, and recovered self-meld names are explicitly flagged unreliable. The runtime no longer uses four fixed river polygons: the warped-table pass estimates the current river center, assigns river ownership by directional sector, and then independently recognizes complete opponent meld shelves in the outer seat corners. Structured opponent melds are retained in round state across a temporary missed frame. Their reliable tile identities participate in visible-tile, wall, and remaining-copy calculations, but never in player-specific genbutsu. Every raw detection records `coordinate_space` so original and warped boxes cannot be overlaid or reconciled accidentally. In live mode, full YOLO river snapshots are matched to history by player and bounding-box IoU: a changed class or a normal new discard needs two consecutive frames before it replaces or appends state, while a call window can append immediately. Opponent riichi is inferred from the orientation anomaly in that player's river and confirmed across two frames before defense mode is activated.

When a called discard disappears from a river and a corresponding new meld is
observed, the historical discard record is retained and marked
`claimed_into_meld`. Player-specific genbutsu therefore still sees the original
discard, while remaining-copy, wall, and fully-visible calculations exclude that
historical physical instance and count the meld tiles once. Opponent linking is
limited to a three-scan window on a newly observed meld; an old unlinked meld
cannot consume a later same-tile detector dropout. Chi linking additionally
restricts the source to the only legal previous player. The self-call path stores
the unique call-window river delta and links it only after a new self meld is
observed. The panel renders claimed historical tiles with a dashed, struck style
and displays both historical and still-in-river counts.

The HBB model is a deployment candidate, not the end of model work. OBB training/export and broader screenshot acceptance remain listed in the TODO.

## Future training TODO

The HBB candidate is now deployed and runnable. The following work is still
required before considering a broader model replacement or making YOLO the
default mode.

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
- separate original-frame hand/self-meld and warped-table river/opponent-meld YOLO overlays,
- accepted/rejected detections with class and confidence,
- final hand, self-meld, opponent-meld, and river outputs.

Generated debug images should stay out of commits unless explicitly requested.

The plugin panel exposes the first two coordinate spaces as separate previews:

- `原始截图`: the untouched captured Mahjong Soul window from `last_frame_path`.
- `变换后牌桌分区`: the `800 x 800` `warped_table` image used for the river/opponent-meld pass, with the adaptive center, four directional ownership sectors, warped-table detections, and opponent-meld group outlines. It never draws original-frame hand/self-meld zones.

The transformed preview is produced in memory by `mahjong_coach_table_region_preview`. If table-surface detection fails, the panel reports the failure and leaves this preview empty; it must never substitute the raw frame under the transformed-preview label.
