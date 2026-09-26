from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


DEFAULT_WARP_SIZE = (800, 800)
DEFAULT_DETECTION_MODE = "support_lines"
LEGACY_DETECTION_MODE = "legacy"
SUPPORTED_DETECTION_MODES = frozenset({DEFAULT_DETECTION_MODE, LEGACY_DETECTION_MODE})

# The support-line path is the validated preprocessing domain used to prepare
# the river-training images. Multiple samples keep a tile, avatar, or table
# decoration from poisoning the single color estimate.
SUPPORT_SAMPLE_POINTS = (
    (0.32, 0.34),
    (0.50, 0.30),
    (0.68, 0.34),
    (0.28, 0.55),
    (0.72, 0.55),
    (0.42, 0.68),
    (0.58, 0.68),
)
SUPPORT_SAMPLE_FRAC = 0.035
SUPPORT_DELTA_H = 16
SUPPORT_DELTA_SV = 85
MIN_SUPPORT_COMPONENT_RATIO = 0.08
MIN_SUPPORT_QUAD_RATIO = 0.50
MAX_SUPPORT_QUAD_RATIO = 1.35
MAX_SUPPORT_EXTRAPOLATION_RATIO = 0.30
SUPPORT_CLUSTER_ANGLE_DEGREES = 3.5
SUPPORT_CLUSTER_OFFSET_RATIO = 0.025
SUPPORT_CLUSTER_GAP_RATIO = 0.035
MIN_HORIZONTAL_SUPPORT_COVERAGE = 0.25
MIN_SIDE_SUPPORT_COVERAGE = 0.14

# Kept intact for the explicit legacy mode and as the final compatibility
# fallback when the support-line detector cannot form a trustworthy quad.
SAMPLE_FRAC = 0.10
DELTA_H = 15
DELTA_SV = 60
MIN_TABLE_AREA_RATIO = 0.22
APPROX_EPS_FRAC = 0.02


@dataclass(frozen=True)
class _SupportLine:
    name: str
    x1: int
    y1: int
    x2: int
    y2: int
    length: float
    angle: float


@dataclass(frozen=True)
class TableSurfaceResult:
    ok: bool = False
    reason: str = ""
    elapsed_ms: float = 0.0
    method: str = ""
    quad: list[list[float]] = field(default_factory=list)
    content_bbox: list[float] = field(default_factory=list)
    warped_size: tuple[int, int] | None = None
    quality_score: float = 0.0
    component_area_ratio: float = 0.0
    quad_area_ratio: float = 0.0
    support_coverage: dict[str, float] = field(default_factory=dict)
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
            "table_surface_quality_score": round(float(self.quality_score), 4),
            "table_surface_component_area_ratio": round(float(self.component_area_ratio), 4),
            "table_surface_quad_area_ratio": round(float(self.quad_area_ratio), 4),
            "table_surface_support_coverage": {
                side: round(float(value), 4) for side, value in self.support_coverage.items()
            },
            "table_surface_elapsed_ms": round(float(self.elapsed_ms), 1),
        }


def detect_table_surface(
    image: Image.Image,
    *,
    diagnostics_dir: Path | None = None,
    diagnostics_stem: str = "frame",
    warp_size: tuple[int, int] = DEFAULT_WARP_SIZE,
    mode: str = DEFAULT_DETECTION_MODE,
    legacy_fallback: bool = False,
) -> TableSurfaceResult:
    started = time.perf_counter()
    normalized_mode = str(mode or DEFAULT_DETECTION_MODE).strip().lower()
    if normalized_mode not in SUPPORTED_DETECTION_MODES:
        raise ValueError(f"unsupported table-surface mode: {mode}")
    np = _load_numpy()
    if np is None:
        return TableSurfaceResult(reason="numpy_unavailable", elapsed_ms=(time.perf_counter() - started) * 1000.0)

    rgb_image = image.convert("RGB")
    rgb = np.asarray(rgb_image)
    cv2 = _load_cv2()
    detection: tuple[str, list[list[float]], Any, dict[str, Any]] | None = None
    if cv2 is not None:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if normalized_mode == DEFAULT_DETECTION_MODE:
            detection = _detect_by_support_lines(cv2, np, bgr)
        if detection is None and (normalized_mode == LEGACY_DETECTION_MODE or legacy_fallback):
            detection = _detect_by_table_color(cv2, np, bgr)
            if detection is None:
                detection = _detect_by_edges(cv2, np, bgr)
            if detection is not None and normalized_mode == DEFAULT_DETECTION_MODE:
                detection[3]["fallback_from"] = DEFAULT_DETECTION_MODE
    if detection is None and (normalized_mode == LEGACY_DETECTION_MODE or legacy_fallback):
        detection = _detect_by_numpy_color(np, rgb)
    if detection is None:
        if cv2 is not None and normalized_mode == DEFAULT_DETECTION_MODE:
            reason = "table_surface_support_lines_not_found"
        else:
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
        quality_score=float(extra_diagnostics.get("quality_score") or 0.0),
        component_area_ratio=float(extra_diagnostics.get("component_area_ratio") or 0.0),
        quad_area_ratio=float(extra_diagnostics.get("quad_area_ratio") or 0.0),
        support_coverage={
            str(side): float(value)
            for side, value in dict(extra_diagnostics.get("support_line_coverage") or {}).items()
        },
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


def _detect_by_support_lines(
    cv2: Any,
    np: Any,
    bgr: Any,
) -> tuple[str, list[list[float]], Any, dict[str, Any]] | None:
    """Detect the outer table from its cloth component and four support lines.

    This is deliberately different from approximating the largest contour to
    four points. Mahjong Soul's central panel and rivers form convincing inner
    quadrilaterals; the outer support lines remain identifiable even where a
    wall, avatar, or UI element interrupts the visible cloth boundary.
    """

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    height, width = hsv.shape[:2]
    sample_size = max(12, int(min(width, height) * SUPPORT_SAMPLE_FRAC))
    candidates: list[dict[str, Any]] = []

    for frac_x, frac_y in SUPPORT_SAMPLE_POINTS:
        center_x = int(width * frac_x)
        center_y = int(height * frac_y)
        x0 = max(0, center_x - sample_size)
        x1 = min(width, center_x + sample_size)
        y0 = max(0, center_y - sample_size)
        y1 = min(height, center_y + sample_size)
        sample = hsv[y0:y1, x0:x1]
        if sample.size == 0:
            continue
        mean_hsv = sample.reshape(-1, 3).mean(axis=0)
        mask_before = _color_mask_from_hsv(
            cv2,
            np,
            hsv,
            mean_hsv,
            delta_h=SUPPORT_DELTA_H,
            delta_sv=SUPPORT_DELTA_SV,
        )
        closed = cv2.morphologyEx(
            mask_before,
            cv2.MORPH_CLOSE,
            np.ones((7, 7), np.uint8),
            iterations=3,
        )
        component_result = _largest_component(cv2, np, closed)
        component_area = component_result[1] if component_result is not None else 0
        candidates.append(
            {
                "sample_point": [frac_x, frac_y],
                "sample_region": [x0, y0, x1, y1],
                "mean_hsv": [float(value) for value in mean_hsv],
                "component_area": int(component_area),
                "mask_before": mask_before,
                "closed_mask": closed,
                "component": component_result[0] if component_result is not None else None,
            }
        )

    minimum_component_area = width * height * MIN_SUPPORT_COMPONENT_RATIO
    viable = [item for item in candidates if item["component_area"] >= minimum_component_area]
    if not viable:
        return None

    # Preserve the validated reference behavior by trying the sample with the
    # largest connected cloth region first. Other samples are real fallbacks,
    # rather than silently accepting a geometrically bad first result.
    for candidate in sorted(viable, key=lambda item: item["component_area"], reverse=True):
        component = candidate["component"]
        contour_image = _external_contour_image(cv2, np, component)
        if contour_image is None:
            continue
        support_result = _select_support_lines(cv2, np, contour_image)
        if support_result is None:
            continue
        support_lines, support_selection = support_result
        quad = _support_line_intersections(np, support_lines)
        if quad is None:
            continue
        quality = _support_quad_quality(
            cv2,
            np,
            quad,
            support_lines=support_lines,
            support_selection=support_selection,
            width=width,
            height=height,
            component_area=float(candidate["component_area"]),
        )
        if not quality["accepted"]:
            continue
        sample_diagnostics = [
            {
                "sample_point": item["sample_point"],
                "sample_region": item["sample_region"],
                "mean_hsv": item["mean_hsv"],
                "component_area": item["component_area"],
            }
            for item in candidates
        ]
        return (
            "tablecloth_support_lines",
            quad,
            component,
            {
                **quality,
                "mask_before": candidate["mask_before"],
                "closed_mask": candidate["closed_mask"],
                "component": component,
                "contour": contour_image,
                "support_lines": support_lines,
                "support_selection": support_selection,
                "selected_sample_point": candidate["sample_point"],
                "sample_candidates": sample_diagnostics,
            },
        )
    return None


def _color_mask_from_hsv(
    cv2: Any,
    np: Any,
    hsv: Any,
    sample: Any,
    *,
    delta_h: int,
    delta_sv: int,
) -> Any:
    h, s, v = [float(value) for value in sample]
    lower = np.array(
        [max(0, h - delta_h), max(0, s - delta_sv), max(0, v - delta_sv)],
        dtype=np.uint8,
    )
    upper = np.array(
        [min(179, h + delta_h), min(255, s + delta_sv), min(255, v + delta_sv)],
        dtype=np.uint8,
    )
    return cv2.inRange(hsv, lower, upper)


def _largest_component(cv2: Any, np: Any, mask: Any) -> tuple[Any, int] | None:
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    index = int(np.argmax(areas)) + 1
    component = np.where(labels == index, 255, 0).astype(np.uint8)
    return component, int(stats[index, cv2.CC_STAT_AREA])


def _external_contour_image(cv2: Any, np: Any, component: Any) -> Any | None:
    contours, _hierarchy = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    contour_image = np.zeros_like(component)
    cv2.drawContours(contour_image, [largest], -1, 255, 3)
    return contour_image


def _normalize_support_line(raw: Any) -> _SupportLine:
    x1, y1, x2, y2 = [int(value) for value in raw]
    length = float(math.hypot(x2 - x1, y2 - y1))
    angle = float(math.degrees(math.atan2(y2 - y1, x2 - x1)))
    if angle > 90:
        angle -= 180
    if angle < -90:
        angle += 180
    return _SupportLine("", x1, y1, x2, y2, length, angle)


def _line_x_at_y(line: _SupportLine, y: float) -> float:
    if line.y2 == line.y1:
        return (line.x1 + line.x2) / 2
    scale = (y - line.y1) / (line.y2 - line.y1)
    return float(line.x1 + scale * (line.x2 - line.x1))


def _line_y_at_x(line: _SupportLine, x: float) -> float:
    if line.x2 == line.x1:
        return (line.y1 + line.y2) / 2
    scale = (x - line.x1) / (line.x2 - line.x1)
    return float(line.y1 + scale * (line.y2 - line.y1))


def _named_support_line(name: str, line: _SupportLine) -> _SupportLine:
    return _SupportLine(name, line.x1, line.y1, line.x2, line.y2, line.length, line.angle)


def _line_reference_offset(line: _SupportLine, *, side: str, width: int, height: int) -> float:
    if side in {"top", "bottom"}:
        return _line_y_at_x(line, width * 0.5)
    return _line_x_at_y(line, height * 0.5)


def _line_axis_interval(line: _SupportLine, *, side: str) -> tuple[float, float]:
    values = (line.x1, line.x2) if side in {"top", "bottom"} else (line.y1, line.y2)
    return float(min(values)), float(max(values))


def _merged_interval_coverage(
    intervals: list[tuple[float, float]],
    *,
    dimension: float,
    max_gap: float,
) -> float:
    if not intervals or dimension <= 0:
        return 0.0
    ordered = sorted(intervals)
    merged: list[list[float]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1] + max_gap:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return min(1.0, sum(max(0.0, end - start) for start, end in merged) / dimension)


def _support_outer_score(offset: float, *, side: str, width: int, height: int) -> float:
    if side == "top":
        return min(1.0, max(0.0, 1.0 - offset / max(1.0, height * 0.22)))
    if side == "bottom":
        return min(1.0, max(0.0, (offset - height * 0.72) / max(1.0, height * 0.28)))
    if side == "left":
        return min(1.0, max(0.0, 1.0 - offset / max(1.0, width * 0.48)))
    return min(1.0, max(0.0, (offset - width * 0.52) / max(1.0, width * 0.48)))


def _weighted_median(values: list[tuple[float, float]]) -> float:
    ordered = sorted(values, key=lambda item: item[0])
    threshold = sum(max(0.0, weight) for _value, weight in ordered) / 2
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += max(0.0, weight)
        if cumulative >= threshold:
            return float(value)
    return float(ordered[-1][0])


def _fit_support_line(
    members: list[_SupportLine],
    *,
    side: str,
    width: int,
    height: int,
) -> _SupportLine:
    angle = _weighted_median([(line.angle, line.length) for line in members])
    offset = _weighted_median(
        [
            (_line_reference_offset(line, side=side, width=width, height=height), line.length)
            for line in members
        ]
    )
    if side in {"top", "bottom"}:
        start = min(min(line.x1, line.x2) for line in members)
        end = max(max(line.x1, line.x2) for line in members)
        slope = math.tan(math.radians(angle))
        y1 = offset + (start - width * 0.5) * slope
        y2 = offset + (end - width * 0.5) * slope
        raw = [round(start), round(y1), round(end), round(y2)]
    else:
        start = min(min(line.y1, line.y2) for line in members)
        end = max(max(line.y1, line.y2) for line in members)
        tangent = math.tan(math.radians(angle))
        x1 = offset if abs(tangent) < 1e-6 else offset + (start - height * 0.5) / tangent
        x2 = offset if abs(tangent) < 1e-6 else offset + (end - height * 0.5) / tangent
        raw = [round(x1), round(start), round(x2), round(end)]
    return _named_support_line(side, _normalize_support_line(raw))


def _select_supported_line(
    candidates: list[_SupportLine],
    *,
    side: str,
    width: int,
    height: int,
) -> tuple[_SupportLine, dict[str, Any]] | None:
    if not candidates:
        return None
    axis_dimension = float(width if side in {"top", "bottom"} else height)
    offset_dimension = float(height if side in {"top", "bottom"} else width)
    offset_tolerance = offset_dimension * SUPPORT_CLUSTER_OFFSET_RATIO
    max_gap = axis_dimension * SUPPORT_CLUSTER_GAP_RATIO
    hypotheses: list[tuple[float, list[_SupportLine], dict[str, Any]]] = []

    for seed in candidates:
        seed_offset = _line_reference_offset(seed, side=side, width=width, height=height)
        members = [
            line
            for line in candidates
            if abs(line.angle - seed.angle) <= SUPPORT_CLUSTER_ANGLE_DEGREES
            and abs(_line_reference_offset(line, side=side, width=width, height=height) - seed_offset)
            <= offset_tolerance
        ]
        intervals = [_line_axis_interval(line, side=side) for line in members]
        coverage_ratio = _merged_interval_coverage(intervals, dimension=axis_dimension, max_gap=max_gap)
        spans = [(end - start) / axis_dimension for start, end in intervals]
        longest_span_ratio = max(spans, default=0.0)
        total_span_ratio = min(1.0, sum(spans))
        weighted_offset = sum(
            _line_reference_offset(line, side=side, width=width, height=height) * line.length for line in members
        ) / max(1.0, sum(line.length for line in members))
        outer_score = _support_outer_score(weighted_offset, side=side, width=width, height=height)
        score = (
            0.50 * coverage_ratio
            + 0.25 * longest_span_ratio
            + 0.15 * total_span_ratio
            + 0.10 * outer_score
        )
        hypotheses.append(
            (
                score,
                members,
                {
                    "score": round(float(score), 6),
                    "coverage_ratio": round(float(coverage_ratio), 6),
                    "longest_span_ratio": round(float(longest_span_ratio), 6),
                    "total_span_ratio": round(float(total_span_ratio), 6),
                    "outer_score": round(float(outer_score), 6),
                    "reference_offset": round(float(weighted_offset), 3),
                    "member_count": len(members),
                },
            )
        )

    _score, members, diagnostics = max(hypotheses, key=lambda item: item[0])
    fitted = _fit_support_line(members, side=side, width=width, height=height)
    diagnostics["fitted_angle"] = round(float(fitted.angle), 3)
    diagnostics["fitted_length"] = round(float(fitted.length), 3)
    return fitted, diagnostics


def _select_support_lines(
    cv2: Any,
    np: Any,
    contour_image: Any,
) -> tuple[dict[str, _SupportLine], dict[str, dict[str, Any]]] | None:
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
        return None
    candidates = [_normalize_support_line(raw) for raw in raw_lines[:, 0, :]]
    candidates = [line for line in candidates if line.length >= min(width, height) * 0.08]
    horizontal = [line for line in candidates if abs(line.angle) <= 12]
    negative = [line for line in candidates if -85 <= line.angle <= -35]
    positive = [line for line in candidates if 35 <= line.angle <= 85]

    top_pool = [line for line in horizontal if (line.y1 + line.y2) / 2 < height * 0.22]
    bottom_pool = [line for line in horizontal if (line.y1 + line.y2) / 2 > height * 0.72]
    left_pool = [line for line in negative if (line.x1 + line.x2) / 2 < width * 0.48]
    right_pool = [line for line in positive if (line.x1 + line.x2) / 2 > width * 0.52]

    top_pool = top_pool or horizontal
    bottom_pool = bottom_pool or horizontal
    left_pool = left_pool or negative
    right_pool = right_pool or positive
    if not (top_pool and bottom_pool and left_pool and right_pool):
        return None

    pools = {"top": top_pool, "bottom": bottom_pool, "left": left_pool, "right": right_pool}
    selected = {
        side: _select_supported_line(pool, side=side, width=width, height=height)
        for side, pool in pools.items()
    }
    if any(result is None for result in selected.values()):
        return None
    lines = {side: result[0] for side, result in selected.items() if result is not None}
    diagnostics = {side: result[1] for side, result in selected.items() if result is not None}
    return lines, diagnostics


def _line_coefficients(np: Any, line: _SupportLine) -> Any:
    return np.array(
        [line.y1 - line.y2, line.x2 - line.x1, line.x1 * line.y2 - line.x2 * line.y1],
        dtype=np.float64,
    )


def _intersect_support_lines(np: Any, first: _SupportLine, second: _SupportLine) -> list[float] | None:
    cross = np.cross(_line_coefficients(np, first), _line_coefficients(np, second))
    if abs(float(cross[2])) < 1e-6:
        return None
    return [float(cross[0] / cross[2]), float(cross[1] / cross[2])]


def _support_line_intersections(np: Any, lines: dict[str, _SupportLine]) -> list[list[float]] | None:
    top_left = _intersect_support_lines(np, lines["top"], lines["left"])
    top_right = _intersect_support_lines(np, lines["top"], lines["right"])
    bottom_left = _intersect_support_lines(np, lines["bottom"], lines["left"])
    bottom_right = _intersect_support_lines(np, lines["bottom"], lines["right"])
    if any(point is None for point in (top_left, top_right, bottom_left, bottom_right)):
        return None
    return [top_left, top_right, bottom_left, bottom_right]  # type: ignore[list-item]


def _support_quad_quality(
    cv2: Any,
    np: Any,
    quad: list[list[float]],
    *,
    support_lines: dict[str, _SupportLine],
    support_selection: dict[str, dict[str, Any]],
    width: int,
    height: int,
    component_area: float,
) -> dict[str, Any]:
    ordered = _order_quad_points(quad)
    polygon = np.asarray([ordered[0], ordered[1], ordered[3], ordered[2]], dtype=np.float32)
    quad_area_ratio = abs(float(cv2.contourArea(polygon))) / max(1.0, float(width * height))
    component_area_ratio = component_area / max(1.0, float(width * height))
    top_y = (ordered[0][1] + ordered[1][1]) / (2 * height)
    bottom_y = (ordered[2][1] + ordered[3][1]) / (2 * height)
    top_width_ratio = abs(ordered[1][0] - ordered[0][0]) / width
    bottom_width_ratio = abs(ordered[3][0] - ordered[2][0]) / width
    vertical_span_ratio = bottom_y - top_y
    extrapolation_ratio = max(
        max(0.0, -x, x - (width - 1), -y, y - (height - 1)) / max(width, height)
        for x, y in ordered
    )
    convex = bool(cv2.isContourConvex(polygon))
    support_coverage = {
        side: float(diagnostics.get("coverage_ratio") or 0.0)
        for side, diagnostics in support_selection.items()
    }
    support_accepted = bool(
        support_coverage.get("top", 0.0) >= MIN_HORIZONTAL_SUPPORT_COVERAGE
        and support_coverage.get("bottom", 0.0) >= MIN_HORIZONTAL_SUPPORT_COVERAGE
        and support_coverage.get("left", 0.0) >= MIN_SIDE_SUPPORT_COVERAGE
        and support_coverage.get("right", 0.0) >= MIN_SIDE_SUPPORT_COVERAGE
    )
    accepted = bool(
        convex
        and support_accepted
        and MIN_SUPPORT_QUAD_RATIO <= quad_area_ratio <= MAX_SUPPORT_QUAD_RATIO
        and component_area_ratio >= MIN_SUPPORT_COMPONENT_RATIO
        and top_y <= 0.40
        and bottom_y >= 0.60
        and top_width_ratio >= 0.35
        and bottom_width_ratio >= 0.45
        and vertical_span_ratio >= 0.50
        and extrapolation_ratio <= MAX_SUPPORT_EXTRAPOLATION_RATIO
    )
    area_score = max(0.0, 1.0 - abs(quad_area_ratio - 0.9) / 0.9)
    span_score = min(1.0, max(0.0, vertical_span_ratio))
    line_score = sum(support_coverage.values()) / max(1, len(support_coverage))
    extrapolation_score = max(0.0, 1.0 - extrapolation_ratio / MAX_SUPPORT_EXTRAPOLATION_RATIO)
    quality_score = 0.40 * area_score + 0.30 * span_score + 0.20 * line_score + 0.10 * extrapolation_score
    return {
        "accepted": accepted,
        "quality_score": round(float(quality_score), 6),
        "component_area_ratio": round(float(component_area_ratio), 6),
        "quad_area_ratio": round(float(quad_area_ratio), 6),
        "top_y_ratio": round(float(top_y), 6),
        "bottom_y_ratio": round(float(bottom_y), 6),
        "vertical_span_ratio": round(float(vertical_span_ratio), 6),
        "corner_extrapolation_ratio": round(float(extrapolation_ratio), 6),
        "support_line_coverage": {side: round(value, 6) for side, value in support_coverage.items()},
        "support_selection": support_selection,
        "support_line_angles": {name: round(float(line.angle), 3) for name, line in support_lines.items()},
    }


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
    scalar_keys = (
        "quality_score",
        "component_area_ratio",
        "quad_area_ratio",
        "top_y_ratio",
        "bottom_y_ratio",
        "vertical_span_ratio",
        "corner_extrapolation_ratio",
        "support_line_angles",
        "support_line_coverage",
        "support_selection",
        "selected_sample_point",
        "sample_candidates",
        "fallback_from",
    )
    diagnostics = {key: extra[key] for key in scalar_keys if key in extra}
    if diagnostics_dir is None:
        return diagnostics
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    ordered = _order_quad_points(quad)
    overlay = _draw_automajsoul_quad_overlay(image, ordered)

    quad_path = diagnostics_dir / f"{diagnostics_stem}-table-quad.png"
    warp_path = diagnostics_dir / f"{diagnostics_stem}-table-warp.png"
    mask_path = diagnostics_dir / f"{diagnostics_stem}-table-mask.png"
    mask_before_path = diagnostics_dir / f"{diagnostics_stem}-table-mask-before.png"
    component_path = diagnostics_dir / f"{diagnostics_stem}-table-component.png"
    contour_path = diagnostics_dir / f"{diagnostics_stem}-table-contour.png"
    support_path = diagnostics_dir / f"{diagnostics_stem}-table-support-lines.png"
    overlay.save(quad_path)
    warped.save(warp_path)
    Image.fromarray(mask).save(mask_path)
    diagnostics.update(
        {
            "table_quad_path": str(quad_path),
            "table_warp_path": str(warp_path),
            "table_mask_path": str(mask_path),
        }
    )
    mask_before = extra.get("mask_before")
    if mask_before is not None:
        Image.fromarray(mask_before).save(mask_before_path)
        diagnostics["table_mask_before_path"] = str(mask_before_path)
    component = extra.get("component")
    if component is not None:
        Image.fromarray(component).save(component_path)
        diagnostics["table_component_path"] = str(component_path)
    contour = extra.get("contour")
    if contour is not None:
        Image.fromarray(contour).save(contour_path)
        diagnostics["table_contour_path"] = str(contour_path)
    support_lines = extra.get("support_lines")
    if support_lines:
        _draw_support_lines_overlay(image, support_lines).save(support_path)
        diagnostics["table_support_lines_path"] = str(support_path)
    return diagnostics


def _draw_support_lines_overlay(image: Image.Image, lines: dict[str, _SupportLine]) -> Image.Image:
    cv2 = _load_cv2()
    np = _load_numpy()
    if cv2 is None or np is None:
        return image.copy()
    bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    colors = {
        "top": (0, 255, 255),
        "bottom": (255, 255, 0),
        "left": (0, 255, 0),
        "right": (255, 0, 255),
    }
    for name, line in lines.items():
        color = colors[name]
        cv2.line(bgr, (line.x1, line.y1), (line.x2, line.y2), color, 4, cv2.LINE_AA)
        cv2.putText(bgr, name, (line.x1, line.y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _draw_automajsoul_quad_overlay(image: Image.Image, ordered: list[list[float]]) -> Image.Image:
    cv2 = _load_cv2()
    np = _load_numpy()
    if cv2 is not None and np is not None:
        bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        quad_int = np.array(ordered, dtype=np.int32)
        polygon_int = quad_int[[0, 1, 3, 2]]
        cv2.polylines(bgr, [polygon_int], True, (0, 255, 0), 3)
        for label, (x, y) in zip(("TL", "TR", "BL", "BR"), quad_int):
            cv2.circle(bgr, (int(x), int(y)), 10, (255, 0, 0), -1)
            cv2.putText(bgr, label, (int(x) + 15, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    polygon = [tuple(point) for point in ordered]
    perimeter = [polygon[0], polygon[1], polygon[3], polygon[2], polygon[0]]
    draw.line(perimeter, fill="lime", width=3)
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
