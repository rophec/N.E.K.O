"""Batch tablecloth warp for raw Mahjong Soul screenshots.

The output layout intentionally matches `prepare_river_yolo_dataset.py`: each
successful case contains `09-warp-square-800.png`.

中文说明：
这个脚本把“生截图”批量变成稳定的桌布透视图。输出目录里的每个成功样本都会有
`09-warp-square-800.png`，随后可以直接交给牌河预标注脚本生成 YOLO 草稿。
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


WARP_SIZE = 800


@dataclass(frozen=True)
class Line:
    name: str
    x1: int
    y1: int
    x2: int
    y2: int
    length: float
    angle: float


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to read image: {path}")
    return image


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"failed to encode png: {path}")
    encoded.tofile(str(path))


def case_id_from_path(path: Path, used: set[str]) -> str:
    stem = path.stem
    safe = "".join(ch if ch.isascii() and (ch.isalnum() or ch in "-_") else "_" for ch in stem)
    safe = safe.strip("_") or "case"
    candidate = safe
    index = 2
    while candidate in used:
        candidate = f"{safe}_{index}"
        index += 1
    used.add(candidate)
    return candidate


def hsv_tablecloth_mask(image: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    """Create the initial HSV mask from a sampled tablecloth color.

    English: This follows the AutoMajsoul idea: sample tablecloth color around
    the center-left table area and threshold nearby HSV colors.

    中文：参考 AutoMajsoul 思路，在牌桌中心偏左取桌布颜色，再用 HSV 邻域筛出桌布。
    """

    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sample_size = max(12, int(min(width, height) * 0.035))
    sample_points = [
        (0.32, 0.34),
        (0.50, 0.30),
        (0.68, 0.34),
        (0.28, 0.55),
        (0.72, 0.55),
        (0.42, 0.68),
        (0.58, 0.68),
    ]
    candidates: list[tuple[int, np.ndarray, dict[str, object]]] = []
    for frac_x, frac_y in sample_points:
        center_x = int(width * frac_x)
        center_y = int(height * frac_y)
        x0 = max(0, center_x - sample_size)
        x1 = min(width, center_x + sample_size)
        y0 = max(0, center_y - sample_size)
        y1 = min(height, center_y + sample_size)
        sample = hsv[y0:y1, x0:x1]
        mean_hsv = sample.reshape(-1, 3).mean(axis=0)
        lower = np.array(
            [max(0, mean_hsv[0] - 16), max(0, mean_hsv[1] - 85), max(0, mean_hsv[2] - 85)],
            dtype=np.uint8,
        )
        upper = np.array(
            [min(179, mean_hsv[0] + 16), min(255, mean_hsv[1] + 85), min(255, mean_hsv[2] + 85)],
            dtype=np.uint8,
        )
        mask = cv2.inRange(hsv, lower, upper)
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=3)
        try:
            _, component_info = largest_component(closed)
            area = int(component_info["area"])
        except ValueError:
            area = 0
        info = {
            "sample_point": [frac_x, frac_y],
            "sample_region": [x0, y0, x1, y1],
            "mean_hsv": [float(v) for v in mean_hsv],
            "lower_hsv": [int(v) for v in lower],
            "upper_hsv": [int(v) for v in upper],
            "largest_area": area,
        }
        candidates.append((area, mask, info))

    area, mask, info = max(candidates, key=lambda item: item[0])
    if area < width * height * 0.08:
        raise ValueError(f"no plausible tablecloth HSV sample; best_area={area}")
    info = dict(info)
    info["all_sample_candidates"] = [dict(candidate_info) for _, _, candidate_info in candidates]
    return mask, info


def largest_component(mask: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        raise ValueError("no foreground components")
    areas = stats[1:, cv2.CC_STAT_AREA]
    index = int(np.argmax(areas)) + 1
    component = np.where(labels == index, 255, 0).astype(np.uint8)
    return component, {
        "x": int(stats[index, cv2.CC_STAT_LEFT]),
        "y": int(stats[index, cv2.CC_STAT_TOP]),
        "width": int(stats[index, cv2.CC_STAT_WIDTH]),
        "height": int(stats[index, cv2.CC_STAT_HEIGHT]),
        "area": int(stats[index, cv2.CC_STAT_AREA]),
    }


def external_contour_image(component: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("no external contour")
    largest = max(contours, key=cv2.contourArea)
    contour_image = np.zeros_like(component)
    cv2.drawContours(contour_image, [largest], -1, 255, 3)
    return contour_image


def normalize_line(raw: np.ndarray) -> tuple[int, int, int, int, float, float]:
    x1, y1, x2, y2 = [int(v) for v in raw]
    length = float(math.hypot(x2 - x1, y2 - y1))
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    if angle > 90:
        angle -= 180
    if angle < -90:
        angle += 180
    return x1, y1, x2, y2, length, angle


def line_x_at_y(line: Line, y: float) -> float:
    if line.y2 == line.y1:
        return (line.x1 + line.x2) / 2
    t = (y - line.y1) / (line.y2 - line.y1)
    return line.x1 + t * (line.x2 - line.x1)


def select_support_lines(contour_image: np.ndarray) -> dict[str, Line]:
    height, width = contour_image.shape[:2]
    raw_lines = cv2.HoughLinesP(
        contour_image,
        rho=1,
        theta=np.pi / 180,
        threshold=60,
        minLineLength=max(60, int(min(width, height) * 0.08)),
        maxLineGap=28,
    )
    if raw_lines is None:
        raise ValueError("no Hough lines")

    candidates: list[Line] = []
    for raw in raw_lines[:, 0, :]:
        x1, y1, x2, y2, length, angle = normalize_line(raw)
        if length < min(width, height) * 0.08:
            continue
        candidates.append(Line("", x1, y1, x2, y2, length, angle))

    horizontal = [line for line in candidates if abs(line.angle) <= 12]
    negative = [line for line in candidates if -85 <= line.angle <= -35]
    positive = [line for line in candidates if 35 <= line.angle <= 85]

    top_pool = [line for line in horizontal if (line.y1 + line.y2) / 2 < height * 0.22]
    bottom_pool = [line for line in horizontal if (line.y1 + line.y2) / 2 > height * 0.72]
    left_pool = [line for line in negative if (line.x1 + line.x2) / 2 < width * 0.48]
    right_pool = [line for line in positive if (line.x1 + line.x2) / 2 > width * 0.52]

    if not top_pool:
        top_pool = horizontal
    if not bottom_pool:
        bottom_pool = horizontal
    if not left_pool:
        left_pool = negative
    if not right_pool:
        right_pool = positive

    if not (top_pool and bottom_pool and left_pool and right_pool):
        raise ValueError(
            f"missing support line pool: top={len(top_pool)} bottom={len(bottom_pool)} "
            f"left={len(left_pool)} right={len(right_pool)}"
        )

    top = min(top_pool, key=lambda line: ((line.y1 + line.y2) / 2, -line.length))
    bottom = max(bottom_pool, key=lambda line: ((line.y1 + line.y2) / 2, line.length))
    left = min(left_pool, key=lambda line: (line_x_at_y(line, height * 0.5), -line.length))
    right = max(right_pool, key=lambda line: (line_x_at_y(line, height * 0.5), line.length))

    return {
        "top": Line("top", top.x1, top.y1, top.x2, top.y2, top.length, top.angle),
        "bottom": Line("bottom", bottom.x1, bottom.y1, bottom.x2, bottom.y2, bottom.length, bottom.angle),
        "left": Line("left", left.x1, left.y1, left.x2, left.y2, left.length, left.angle),
        "right": Line("right", right.x1, right.y1, right.x2, right.y2, right.length, right.angle),
    }


def line_coeff(line: Line) -> np.ndarray:
    x1, y1, x2, y2 = line.x1, line.y1, line.x2, line.y2
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    return np.array([a, b, c], dtype=np.float64)


def intersect(a: Line, b: Line) -> tuple[float, float]:
    cross = np.cross(line_coeff(a), line_coeff(b))
    if abs(cross[2]) < 1e-6:
        raise ValueError(f"parallel lines: {a.name}, {b.name}")
    return float(cross[0] / cross[2]), float(cross[1] / cross[2])


def draw_lines(image: np.ndarray, lines: dict[str, Line], extended: bool = False) -> np.ndarray:
    output = image.copy()
    colors = {
        "top": (0, 255, 255),
        "bottom": (255, 255, 0),
        "left": (0, 255, 0),
        "right": (255, 0, 255),
    }
    height, width = image.shape[:2]
    for name, line in lines.items():
        color = colors[name]
        if extended:
            coeff = line_coeff(line)
            points: list[tuple[int, int]] = []
            for x in (-width, width * 2):
                if abs(coeff[1]) > 1e-6:
                    y = int(round(-(coeff[0] * x + coeff[2]) / coeff[1]))
                    points.append((int(x), y))
            for y in (-height, height * 2):
                if abs(coeff[0]) > 1e-6:
                    x = int(round(-(coeff[1] * y + coeff[2]) / coeff[0]))
                    points.append((x, int(y)))
            visible = [
                point
                for point in points
                if -width <= point[0] <= width * 2 and -height <= point[1] <= height * 2
            ]
            if len(visible) >= 2:
                cv2.line(output, visible[0], visible[1], color, 4)
        else:
            cv2.line(output, (line.x1, line.y1), (line.x2, line.y2), color, 4)
        cv2.putText(output, name, (line.x1, line.y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return output


def warp_table(image: np.ndarray, lines: dict[str, Line]) -> tuple[np.ndarray, dict[str, list[float]]]:
    points = {
        "top_left": intersect(lines["top"], lines["left"]),
        "top_right": intersect(lines["top"], lines["right"]),
        "bottom_left": intersect(lines["bottom"], lines["left"]),
        "bottom_right": intersect(lines["bottom"], lines["right"]),
    }
    src = np.array(
        [points["top_left"], points["top_right"], points["bottom_left"], points["bottom_right"]],
        dtype=np.float32,
    )
    dst = np.array(
        [[0, 0], [WARP_SIZE, 0], [0, WARP_SIZE], [WARP_SIZE, WARP_SIZE]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, matrix, (WARP_SIZE, WARP_SIZE))
    return warped, {name: [float(x), float(y)] for name, (x, y) in points.items()}


def contact_sheet(images: Iterable[tuple[str, np.ndarray]]) -> np.ndarray:
    cells = []
    for title, image in images:
        thumb = image.copy()
        if thumb.ndim == 2:
            thumb = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
        thumb = cv2.resize(thumb, (320, 180), interpolation=cv2.INTER_AREA)
        cv2.putText(thumb, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cells.append(thumb)
    if not cells:
        return np.zeros((1, 1, 3), np.uint8)
    rows = []
    for index in range(0, len(cells), 2):
        row = cells[index : index + 2]
        if len(row) == 1:
            row.append(np.full_like(row[0], 255))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def process_image(path: Path, case_dir: Path) -> dict[str, object]:
    image = read_image(path)
    write_png(case_dir / "01-source.png", image)
    mask_before, hsv_info = hsv_tablecloth_mask(image)
    write_png(case_dir / "02-mask-before.png", mask_before)
    mask_after = cv2.morphologyEx(
        mask_before,
        cv2.MORPH_CLOSE,
        np.ones((7, 7), np.uint8),
        iterations=3,
    )
    write_png(case_dir / "03-mask-after.png", mask_after)
    component, component_info = largest_component(mask_after)
    write_png(case_dir / "04-largest-component.png", component)
    contour = external_contour_image(component)
    write_png(case_dir / "05-external-contour.png", contour)
    lines = select_support_lines(contour)
    selected = np.zeros_like(image)
    selected[:] = (0, 0, 0)
    selected_lines = draw_lines(selected, lines)
    write_png(case_dir / "06-selected-lines.png", selected_lines)
    write_png(case_dir / "07-lines-on-source.png", draw_lines(image, lines))
    write_png(case_dir / "08-extended-lines.png", draw_lines(image, lines, extended=True))
    warped, intersections = warp_table(image, lines)
    write_png(case_dir / "09-warp-square-800.png", warped)
    sheet = contact_sheet(
        [
            ("01 source", image),
            ("03 mask", mask_after),
            ("05 contour", contour),
            ("07 lines", draw_lines(image, lines)),
            ("08 extended", draw_lines(image, lines, extended=True)),
            ("09 warp", warped),
        ]
    )
    write_png(case_dir / "contact-sheet.png", sheet)
    summary = {
        "source": str(path),
        "image_size": [int(image.shape[1]), int(image.shape[0])],
        **hsv_info,
        "largest_component": component_info,
        "selected_lines": {name: asdict(line) for name, line in lines.items()},
        "intersections": intersections,
        "warp_output": str(case_dir / "09-warp-square-800.png"),
    }
    with open(case_dir / "summary.json", "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, ensure_ascii=False, indent=2)
    return summary


def list_images(input_root: Path) -> list[Path]:
    if input_root.is_file():
        return [input_root]
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    return sorted(path for path in input_root.iterdir() if path.is_file() and path.suffix.lower() in suffixes)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    used: set[str] = set()
    summaries = []
    failures = []
    for image_path in list_images(args.input_root):
        case_id = case_id_from_path(image_path, used)
        case_dir = args.output_root / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        try:
            summary = process_image(image_path, case_dir)
            summaries.append({"case_id": case_id, "ok": True, **summary})
            print(f"ok {case_id}: {image_path}")
        except Exception as exc:  # noqa: BLE001 - keep batch running for review.
            failures.append({"case_id": case_id, "source": str(image_path), "error": str(exc)})
            print(f"fail {case_id}: {exc}")

    with open(args.output_root / "summary_all.json", "w", encoding="utf-8") as summary_file:
        json.dump({"ok": summaries, "failed": failures}, summary_file, ensure_ascii=False, indent=2)
    print(f"processed={len(summaries)} failed={len(failures)} output={args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
