from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


DEFAULT_WARP_SIZE = (800, 800)
SAMPLE_FRAC = 0.10
DELTA_H = 15
DELTA_SV = 60
MIN_TABLE_AREA_RATIO = 0.22
APPROX_EPS_FRAC = 0.02


@dataclass(frozen=True)
class TableSurfaceResult:
    ok: bool = False
    reason: str = ""
    elapsed_ms: float = 0.0
    method: str = ""
    quad: list[list[float]] = field(default_factory=list)
    content_bbox: list[float] = field(default_factory=list)
    warped_size: tuple[int, int] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    warped_image: Image.Image | None = field(default=None, repr=False, compare=False)

    def to_hints(self) -> dict[str, Any]:
        return {
            "table_surface_ok": self.ok,
            "table_surface_reason": self.reason,
            "table_surface_method": self.method,
            "table_surface_quad": self.quad,
            "table_surface_bbox": self.content_bbox,
            "table_surface_warped_size": list(self.warped_size or []),
            "table_surface_elapsed_ms": round(float(self.elapsed_ms), 1),
        }


def detect_table_surface(
    image: Image.Image,
    *,
    diagnostics_dir: Path | None = None,
    diagnostics_stem: str = "frame",
    warp_size: tuple[int, int] = DEFAULT_WARP_SIZE,
) -> TableSurfaceResult:
    started = time.perf_counter()
    np = _load_numpy()
    if np is None:
        return TableSurfaceResult(reason="numpy_unavailable", elapsed_ms=(time.perf_counter() - started) * 1000.0)

    rgb_image = image.convert("RGB")
    rgb = np.asarray(rgb_image)
    cv2 = _load_cv2()
    detection: tuple[str, list[list[float]], Any, dict[str, Any]] | None = None
    if cv2 is not None:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        detection = _detect_by_table_color(cv2, np, bgr)
        if detection is None:
            detection = _detect_by_edges(cv2, np, bgr)
    if detection is None:
        detection = _detect_by_numpy_color(np, rgb)
    if detection is None:
        reason = "table_surface_not_found" if cv2 is not None else "table_surface_not_found_without_opencv"
        return TableSurfaceResult(reason=reason, elapsed_ms=(time.perf_counter() - started) * 1000.0)

    method, quad, mask, extra_diagnostics = detection
    warped = _warp_quad_cv2(cv2, np, rgb, quad, warp_size) if cv2 is not None else _warp_quad_pil(np, rgb_image, quad, warp_size)
    bbox = _quad_bbox(quad)
    diagnostics = _write_diagnostics(
        image,
        quad=quad,
        warped=warped,
        mask=mask,
        extra=extra_diagnostics,
        diagnostics_dir=diagnostics_dir,
        diagnostics_stem=diagnostics_stem,
    )
    return TableSurfaceResult(
        ok=True,
        reason="table_surface_detected",
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
        method=method,
        quad=_round_quad(quad),
        content_bbox=[round(value, 1) for value in bbox],
        warped_size=warp_size,
        diagnostics=diagnostics,
        warped_image=warped,
    )


def _load_numpy() -> Any | None:
    try:
        import numpy as np  # type: ignore[import-not-found]
    except Exception:
        return None
    return np


def _load_cv2() -> Any | None:
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception:
        return None
    return cv2


def _detect_by_table_color(cv2: Any, np: Any, bgr: Any) -> tuple[str, list[list[float]], Any, dict[str, Any]] | None:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    height, width = hsv.shape[:2]
    sample = _sample_automajsoul_table_color(hsv, width=width, height=height)
    if sample is None:
        return None
    mask, mask_before = _build_color_mask(cv2, np, hsv, sample)
    candidate = _quad_from_mask(cv2, mask, min_area=width * height * MIN_TABLE_AREA_RATIO)
    if candidate is None:
        return None
    quad, _area = candidate
    return "automajsoul_opencv_color", quad, mask, {"mask_before": mask_before}


def _detect_by_edges(cv2: Any, np: Any, bgr: Any) -> tuple[str, list[list[float]], Any, dict[str, Any]] | None:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, None, iterations=2)
    height, width = gray.shape[:2]
    candidate = _quad_from_mask(cv2, edges, min_area=width * height * MIN_TABLE_AREA_RATIO)
    if candidate is None:
        return None
    quad, _area = candidate
    return "automajsoul_opencv_edges", quad, edges, {}


def _detect_by_numpy_color(np: Any, rgb: Any) -> tuple[str, list[list[float]], Any, dict[str, Any]] | None:
    height, width = rgb.shape[:2]
    hsv = _rgb_to_hsv_opencv_scale(np, rgb)
    sample = _sample_automajsoul_table_color(hsv, width=width, height=height)
    if sample is None:
        return None
    h, s, v = sample
    lower = np.array([max(0, h - DELTA_H), max(0, s - DELTA_SV), max(0, v - DELTA_SV)], dtype="float32")
    upper = np.array([min(179, h + DELTA_H), min(255, s + DELTA_SV), min(255, v + DELTA_SV)], dtype="float32")
    mask_bool = (
        (hsv[:, :, 0] >= lower[0])
        & (hsv[:, :, 0] <= upper[0])
        & (hsv[:, :, 1] >= lower[1])
        & (hsv[:, :, 1] <= upper[1])
        & (hsv[:, :, 2] >= lower[2])
        & (hsv[:, :, 2] <= upper[2])
    )
    coords = np.column_stack(np.nonzero(mask_bool))
    if float(coords.shape[0]) < width * height * MIN_TABLE_AREA_RATIO:
        return None
    ys = coords[:, 0].astype("float32")
    xs = coords[:, 1].astype("float32")
    quad = _extreme_quad_from_arrays(np, xs, ys)
    mask = mask_bool.astype("uint8") * 255
    return "automajsoul_numpy_color", quad, mask, {"mask_before": mask}


def _sample_automajsoul_table_color(hsv: Any, *, width: int, height: int) -> tuple[float, float, float] | None:
    # 中文：对齐 AutoMajsoul：避开中心计分盘，取中心向左 1/4 的桌布区域。
    # English: Match AutoMajsoul: avoid the score panel and sample the cloth left of center.
    center_y, center_x = height // 2, width // 2
    sample_x = int(center_x - width * 0.25)
    sample_y = center_y
    half_h = int(height * SAMPLE_FRAC / 2)
    half_w = int(width * SAMPLE_FRAC / 2)
    top, bottom = max(0, sample_y - half_h), min(height, sample_y + half_h)
    left, right = max(0, sample_x - half_w), min(width, sample_x + half_w)
    patch = hsv[top:bottom, left:right]
    if patch.size == 0:
        top, bottom = max(0, center_y - half_h), min(height, center_y + half_h)
        left, right = max(0, center_x - half_w), min(width, center_x + half_w)
        patch = hsv[top:bottom, left:right]
    if patch.size == 0:
        return None
    h, s, v = patch.reshape(-1, 3).mean(axis=0)
    return float(h), float(s), float(v)


def _rgb_to_hsv_opencv_scale(np: Any, rgb: Any) -> Any:
    rgb_float = rgb.astype("float32") / 255.0
    r = rgb_float[:, :, 0]
    g = rgb_float[:, :, 1]
    b = rgb_float[:, :, 2]
    maxc = rgb_float.max(axis=2)
    minc = rgb_float.min(axis=2)
    delta = maxc - minc

    hue = np.zeros_like(maxc)
    nonzero = delta > 1e-6
    red = (maxc == r) & nonzero
    green = (maxc == g) & nonzero
    blue = (maxc == b) & nonzero
    hue[red] = ((g[red] - b[red]) / delta[red]) % 6.0
    hue[green] = ((b[green] - r[green]) / delta[green]) + 2.0
    hue[blue] = ((r[blue] - g[blue]) / delta[blue]) + 4.0
    hue = hue * 30.0

    saturation = np.zeros_like(maxc)
    has_value = maxc > 1e-6
    saturation[has_value] = delta[has_value] / maxc[has_value] * 255.0
    value = maxc * 255.0
    return np.stack([hue, saturation, value], axis=2)


def _build_color_mask(cv2: Any, np: Any, hsv: Any, sample: tuple[float, float, float]) -> tuple[Any, Any]:
    h, s, v = sample
    lower = np.array([max(0, h - DELTA_H), max(0, s - DELTA_SV), max(0, v - DELTA_SV)], dtype=np.uint8)
    upper = np.array([min(179, h + DELTA_H), min(255, s + DELTA_SV), min(255, v + DELTA_SV)], dtype=np.uint8)
    mask_before = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask_before, cv2.MORPH_CLOSE, kernel, iterations=3)
    return mask, mask_before


def _quad_from_mask(cv2: Any, mask: Any, *, min_area: float) -> tuple[list[list[float]], float] | None:
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < min_area:
        return None
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, APPROX_EPS_FRAC * perimeter, True)
    if len(approx) > 4:
        approx = cv2.approxPolyDP(contour, 0.05 * perimeter, True)
    if len(approx) != 4:
        return None
    return _order_quad_points([[float(x), float(y)] for [[x, y]] in approx.tolist()]), area


def _warp_quad_cv2(cv2: Any, np: Any, rgb: Any, quad: list[list[float]], warp_size: tuple[int, int]) -> Image.Image:
    ordered = _order_quad_points(quad)
    width, height = warp_size
    source = np.array(ordered, dtype=np.float32)
    target = np.array([[0, 0], [width, 0], [0, height], [width, height]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(source, target)
    warped = cv2.warpPerspective(rgb, matrix, (width, height))
    return Image.fromarray(warped)


def _warp_quad_pil(np: Any, image: Image.Image, quad: list[list[float]], warp_size: tuple[int, int]) -> Image.Image:
    width, height = warp_size
    source = [[0, 0], [width, 0], [0, height], [width, height]]
    target = _order_quad_points(quad)
    coeffs = _perspective_coefficients(np, source, target)
    return image.transform(warp_size, Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BICUBIC)


def _perspective_coefficients(np: Any, source: list[list[float]], target: list[list[float]]) -> list[float]:
    matrix = []
    vector = []
    for (x, y), (u, v) in zip(source, target):
        matrix.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        matrix.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        vector.extend([u, v])
    coeffs = np.linalg.solve(np.array(matrix, dtype="float64"), np.array(vector, dtype="float64"))
    return [float(value) for value in coeffs]


def _write_diagnostics(
    image: Image.Image,
    *,
    quad: list[list[float]],
    warped: Image.Image,
    mask: Any,
    extra: dict[str, Any],
    diagnostics_dir: Path | None,
    diagnostics_stem: str,
) -> dict[str, Any]:
    if diagnostics_dir is None:
        return {}
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    ordered = _order_quad_points(quad)
    overlay = _draw_automajsoul_quad_overlay(image, ordered)

    quad_path = diagnostics_dir / f"{diagnostics_stem}-table-quad.png"
    warp_path = diagnostics_dir / f"{diagnostics_stem}-table-warp.png"
    mask_path = diagnostics_dir / f"{diagnostics_stem}-table-mask.png"
    mask_before_path = diagnostics_dir / f"{diagnostics_stem}-table-mask-before.png"
    overlay.save(quad_path)
    warped.save(warp_path)
    Image.fromarray(mask).save(mask_path)
    diagnostics = {
        "table_quad_path": str(quad_path),
        "table_warp_path": str(warp_path),
        "table_mask_path": str(mask_path),
    }
    mask_before = extra.get("mask_before")
    if mask_before is not None:
        Image.fromarray(mask_before).save(mask_before_path)
        diagnostics["table_mask_before_path"] = str(mask_before_path)
    return diagnostics


def _draw_automajsoul_quad_overlay(image: Image.Image, ordered: list[list[float]]) -> Image.Image:
    cv2 = _load_cv2()
    np = _load_numpy()
    if cv2 is not None and np is not None:
        bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        quad_int = np.array(ordered, dtype=np.int32)
        cv2.polylines(bgr, [quad_int], True, (0, 255, 0), 3)
        for label, (x, y) in zip(("TL", "TR", "BL", "BR"), quad_int):
            cv2.circle(bgr, (int(x), int(y)), 10, (255, 0, 0), -1)
            cv2.putText(bgr, label, (int(x) + 15, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    polygon = [tuple(point) for point in ordered]
    draw.line([*polygon, polygon[0]], fill="lime", width=3)
    for label, (x, y) in zip(("TL", "TR", "BL", "BR"), polygon):
        draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill="blue")
        draw.text((x + 15, y), label, fill="blue")
    return overlay


def _order_quad_points(points: list[list[float]]) -> list[list[float]]:
    if len(points) != 4:
        raise ValueError("quad requires exactly four points")
    ordered_points = [[float(x), float(y)] for x, y in points]
    top_left = min(ordered_points, key=lambda point: point[0] + point[1])
    bottom_right = max(ordered_points, key=lambda point: point[0] + point[1])
    top_right = max(ordered_points, key=lambda point: point[0] - point[1])
    bottom_left = min(ordered_points, key=lambda point: point[0] - point[1])
    return [top_left, top_right, bottom_left, bottom_right]


def _extreme_quad_from_points(points: list[list[float]]) -> list[list[float]]:
    if len(points) < 4:
        raise ValueError("at least four points are required")
    return _order_quad_points(
        [
            min(points, key=lambda point: point[0] + point[1]),
            max(points, key=lambda point: point[0] - point[1]),
            max(points, key=lambda point: point[0] + point[1]),
            min(points, key=lambda point: point[0] - point[1]),
        ]
    )


def _extreme_quad_from_arrays(np: Any, xs: Any, ys: Any) -> list[list[float]]:
    sums = xs + ys
    diffs = xs - ys
    return _order_quad_points(
        [
            [float(xs[int(np.argmin(sums))]), float(ys[int(np.argmin(sums))])],
            [float(xs[int(np.argmax(diffs))]), float(ys[int(np.argmax(diffs))])],
            [float(xs[int(np.argmax(sums))]), float(ys[int(np.argmax(sums))])],
            [float(xs[int(np.argmin(diffs))]), float(ys[int(np.argmin(diffs))])],
        ]
    )


def _quad_bbox(quad: list[list[float]]) -> list[float]:
    xs = [float(point[0]) for point in quad]
    ys = [float(point[1]) for point in quad]
    return [min(xs), min(ys), max(xs), max(ys)]


def _round_quad(quad: list[list[float]]) -> list[list[float]]:
    return [[round(float(x), 1), round(float(y), 1)] for x, y in _order_quad_points(quad)]
