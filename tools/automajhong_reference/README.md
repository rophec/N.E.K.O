# AutoMajsoul table detection reference

This folder is for running the original AutoMajsoul table detector as a reference.

Source project:

- https://github.com/yuhao7370/AutoMajsoul
- Original detector file: `vendor/AutoMajsoul/Android/ingame_recognizer.py`

The runner does not change AutoMajsoul detection logic. It only:

1. imports `IngameRecognizer` from the original project,
2. creates debug/output directories,
3. calls `detect(image)`,
4. calls `warp_board(image, quad)`,
5. writes proof images and JSON summaries.

Run:

```powershell
uv run --with opencv-python-headless --with numpy --with pillow python tools/automajhong_reference/run_automajsoul_table_detection.py
```

Default input:

```text
<source-images>
```

Default output:

```text
<source-images>\automajhong_reference_output
```

Key output files per image:

- `*-source.png`: copied source screenshot.
- `automajsoul_debug/*_02_color_mask_before.png`: original AutoMajsoul color mask before morphology.
- `automajsoul_debug/*_02_color_mask_after.png`: original AutoMajsoul color mask after morphology.
- `automajsoul_debug/*_02_edges.png`: original AutoMajsoul edge fallback image, when used.
- `automajsoul_debug/*_02_quad_detection_color.png` or `*_edges.png`: original AutoMajsoul detected table quad.
- `*-automajsoul-warp.png`: original AutoMajsoul perspective warp.
- `*-proof-sheet.png`: contact sheet for visual checking.
- `*-quad.json`: detected quad coordinates and proof paths.

For source filenames that are not ASCII, the runner uses an ASCII `case_###`
output folder and debug timestamp. This avoids Windows `cv2.imwrite()` failures
without changing AutoMajsoul's recognition logic.

Important review note:

AutoMajsoul's original detector returns a broad table/content quad. It does not
detect the visible inner table seam lines or the central cross on the Mahjong
Soul table. The review overlay therefore draws only the detected quad boundary.
Passing this reference run does not mean the quad passes manual seam-alignment
validation. If seam alignment is required, add a separate inner-table seam
detector after this reference-only step.

Additional experiment notes:

- `tablecloth_edge_detection_flow.md` records the current Mahjong Soul
  tablecloth edge-detection flow: HSV mask, morphology, largest connected
  tablecloth component, external contour, support-line extension, intersections,
  and fixed `800x800` perspective warp.

Tile annotation constraint:

- The tablecloth warp output is only a preprocessing input for later tile
  recognition. It must not be treated as a tile-labeling result.
- Local hand recognition should stay on the original screenshot coordinate
  system. The bottom hand tiles are already near front-facing and get distorted
  by the tablecloth perspective warp.
- The tablecloth perspective warp is intended for table-area recognition such
  as river/discards, exposed melds, and other center-table visible tiles where
  perspective and window scaling cause larger coordinate drift.
- Tile annotations must be based only on the single-tile reference set under:

```text
<source-images>\单牌
<source-images>\单牌\rotations_0_90_180_270
```

- The reference set currently contains 34 tile classes, with 0/90/180/270
  rotations. Vertical hand tiles should use upright references; horizontal and
  upside-down table tiles must use the corresponding rotated references.
- Do not annotate generic white or bright candidates. UI panels, avatars, score
  digits, tile walls, and table borders may be white/bright, but they are not
  tile labels unless they match the single-tile reference set.
- 中文约束：后续标注对象只能来自 `单牌` 参考图及其旋转版本，不能再把白色候选块、
  头像、UI、数字、牌墙或桌面边缘当成麻将牌标注。
- 中文流程边界：自家手牌应在原始截图坐标系中识别；桌布透视变换主要给牌河、
  副露和桌面中部可见牌使用，不应用变换后的图去识别底部手牌。

River recognition diagnostic flow:

- Input: `09-warp-square-800.png` from the tablecloth perspective flow.
- Step 1: limit detection to the center table area and mask out the central
  score device.
- Step 2: build bright tile-surface regions. For river tiles this usually
  produces four river groups, not individual tile boxes.
- Step 3: assign each group to `self`, `opposite`, `left_opponent`, or
  `right_opponent` from its position relative to table center.
- Step 4: split each group by orientation into per-tile slots.
- Step 5: classify each slot against the single-tile reference set and its
  rotations. This step is currently diagnostic only; it should be replaced by
  the trained YOLO/recognition backend.
- 中文说明：牌河识别当前应先在透视变换图上找四家牌河组，再按方向拆成单张牌。
  当前参考模板分类只是诊断工具，不应当作为最终识别器。
