from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .image_source import ImageSource, open_rgb, source_exists

SETTLEMENT_KINDS = {"none", "win", "exhaustive_draw", "abortive_draw", "unknown"}
SETTLEMENT_PHASES = {"playing", "settlement_candidate", "settlement_latched", "awaiting_next_round"}


@dataclass(frozen=True)
class SettlementFrameResult:
    detected: bool = False
    kind: str = "none"
    confidence: float = 0.0
    reason: str = "not_detected"
    evidence: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["confidence"] = round(float(self.confidence), 4)
        payload["elapsed_ms"] = round(float(self.elapsed_ms), 1)
        return payload


@dataclass(frozen=True)
class SettlementTransition:
    phase: str = "playing"
    changed: bool = False
    confirmation_frames: int = 0
    confirmation_elapsed_ms: float = 0.0
    last_frame_gap_ms: float = 0.0
    confirm_max_gap_ms: int = 2500
    result: SettlementFrameResult = field(default_factory=SettlementFrameResult)

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "changed": self.changed,
            "confirmation_frames": int(self.confirmation_frames),
            "confirmation_elapsed_ms": round(float(self.confirmation_elapsed_ms), 1),
            "last_frame_gap_ms": round(float(self.last_frame_gap_ms), 1),
            "confirm_max_gap_ms": int(self.confirm_max_gap_ms),
            **self.result.to_dict(),
        }


class SettlementTracker:
    """Confirm settlement overlays across frames and latch the round boundary."""

    def __init__(
        self,
        *,
        confirm_frames: int = 2,
        confirm_max_gap_ms: int = 2500,
    ) -> None:
        self.confirm_frames = max(1, int(confirm_frames))
        self.confirm_max_gap_ms = max(1, int(confirm_max_gap_ms))
        self.phase = "playing"
        self.confirmation_frames = 0
        self._candidate_kind = "none"
        self._latched_result = SettlementFrameResult()
        self._candidate_started_at: float | None = None
        self._last_candidate_at: float | None = None
        self._confirmation_elapsed_ms = 0.0
        self._last_frame_gap_ms = 0.0

    @property
    def latched_result(self) -> SettlementFrameResult:
        return self._latched_result

    def reset(self) -> None:
        self.phase = "playing"
        self.confirmation_frames = 0
        self._candidate_kind = "none"
        self._latched_result = SettlementFrameResult()
        self._candidate_started_at = None
        self._last_candidate_at = None
        self._confirmation_elapsed_ms = 0.0
        self._last_frame_gap_ms = 0.0

    def observe(
        self,
        result: SettlementFrameResult,
        *,
        round_active: bool,
        observed_at: float | None = None,
    ) -> SettlementTransition:
        now = time.monotonic() if observed_at is None else float(observed_at)
        # 中文：没有正在进行的旧局时不凭一张结算图创建“幽灵上一局”。
        # English: Do not create a ghost previous round when observation starts on a result screen.
        if self.phase == "playing" and not round_active:
            self._clear_candidate()
            return self._transition(self.phase, result)

        previous_phase = self.phase
        if result.detected:
            if self.phase == "awaiting_next_round":
                self.phase = "settlement_latched"
                return self._transition(previous_phase, self._latched_result)

            if self.phase == "settlement_latched":
                if result.confidence >= self._latched_result.confidence:
                    self._latched_result = result
                return self._transition(previous_phase, self._latched_result)

            compatible = _compatible_settlement_kinds(self._candidate_kind, result.kind)
            gap_ms = self._candidate_gap_ms(now)
            gap_expired = gap_ms is not None and (
                gap_ms < 0.0 or gap_ms > float(self.confirm_max_gap_ms)
            )
            if self.phase != "settlement_candidate" or not compatible or gap_expired:
                self.phase = "settlement_candidate"
                self.confirmation_frames = 1
                self._candidate_kind = result.kind
                self._candidate_started_at = now
                self._last_frame_gap_ms = max(0.0, float(gap_ms or 0.0))
            else:
                self.confirmation_frames += 1
                self._last_frame_gap_ms = max(0.0, float(gap_ms or 0.0))
                if self._candidate_kind == "unknown" and result.kind != "unknown":
                    self._candidate_kind = result.kind
            self._last_candidate_at = now
            self._confirmation_elapsed_ms = self._candidate_elapsed_ms(now)

            if self.confirmation_frames >= self.confirm_frames:
                self.phase = "settlement_latched"
                self._latched_result = result
            return self._transition(previous_phase, result)

        if self.phase == "settlement_candidate":
            self.phase = "playing"
            self._clear_candidate()
        elif self.phase == "settlement_latched":
            self.phase = "awaiting_next_round"

        visible_result = self._latched_result if self.phase == "awaiting_next_round" else result
        return self._transition(previous_phase, visible_result)

    def _transition(
        self,
        previous_phase: str,
        result: SettlementFrameResult,
    ) -> SettlementTransition:
        return SettlementTransition(
            phase=self.phase,
            changed=self.phase != previous_phase,
            confirmation_frames=self.confirmation_frames,
            confirmation_elapsed_ms=self._confirmation_elapsed_ms,
            last_frame_gap_ms=self._last_frame_gap_ms,
            confirm_max_gap_ms=self.confirm_max_gap_ms,
            result=result,
        )

    def _candidate_gap_ms(self, now: float) -> float | None:
        if self._last_candidate_at is None:
            return None
        return (now - self._last_candidate_at) * 1000.0

    def _candidate_elapsed_ms(self, now: float) -> float:
        if self._candidate_started_at is None:
            return 0.0
        return max(0.0, (now - self._candidate_started_at) * 1000.0)

    def _clear_candidate(self) -> None:
        self.confirmation_frames = 0
        self._candidate_kind = "none"
        self._candidate_started_at = None
        self._last_candidate_at = None
        self._confirmation_elapsed_ms = 0.0
        self._last_frame_gap_ms = 0.0


def detect_settlement_path(
    image_path: ImageSource,
    *,
    min_confidence: float = 0.72,
) -> SettlementFrameResult:
    started = time.perf_counter()
    if not source_exists(image_path):
        return SettlementFrameResult(
            reason="image_missing",
            elapsed_ms=_elapsed_ms(started),
        )
    try:
        with open_rgb(image_path) as opened:
            image = opened.copy()
    except (OSError, ValueError):
        return SettlementFrameResult(
            reason="image_unreadable",
            elapsed_ms=_elapsed_ms(started),
        )
    return detect_settlement_image(image, min_confidence=min_confidence, _started=started)


def detect_settlement_image(
    image: Image.Image,
    *,
    min_confidence: float = 0.72,
    _started: float | None = None,
) -> SettlementFrameResult:
    """Detect the dimmed Mahjong Soul round-result overlay without OCR."""

    started = _started if _started is not None else time.perf_counter()
    rgb = np.asarray(image.convert("RGB").resize((480, 270), Image.Resampling.BILINEAR))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, saturation, value = cv2.split(hsv)

    blue = (hue >= 82) & (hue <= 132) & (saturation >= 42) & (value >= 28)
    dark = value <= 72
    neutral_bright = (value >= 145) & (saturation <= 92)
    orange_border = (
        (hue >= 4)
        & (hue <= 28)
        & (saturation >= 90)
        & (value >= 80)
    )

    diagonal_ratio, diagonal_line = _longest_diagonal(orange_border)
    metrics: dict[str, Any] = {
        "bottom_dark_ratio": _region_ratio(dark, 0.0, 0.72, 1.0, 1.0),
        "bottom_right_dark_ratio": _region_ratio(dark, 0.64, 0.62, 1.0, 1.0),
        "corner_dark_ratio": _corner_ratio(dark),
        "center_blue_ratio": _region_ratio(blue, 0.20, 0.18, 0.80, 0.72),
        "top_bright_ratio": _region_ratio(neutral_bright, 0.08, 0.10, 0.92, 0.48),
        "result_diagonal_border_length": diagonal_ratio,
        "result_diagonal_line": diagonal_line,
    }
    components = _tile_like_components(neutral_bright)
    upper_components = [item for item in components if float(item["center_y"]) <= 0.58]
    aligned_count = _largest_aligned_row(upper_components)
    metrics["upper_tile_component_count"] = len(upper_components)
    metrics["upper_aligned_tile_count"] = aligned_count
    metrics["upper_tile_components"] = upper_components

    dim_score = (
        _unit(metrics["bottom_dark_ratio"], 0.22, 0.54) * 0.32
        + _unit(metrics["corner_dark_ratio"], 0.28, 0.54) * 0.24
        + _unit(metrics["bottom_right_dark_ratio"], 0.25, 0.55) * 0.16
    )
    panel_score = _unit(metrics["center_blue_ratio"], 0.18, 0.43) * 0.14
    result_content_score = (
        _unit(metrics["top_bright_ratio"], 0.07, 0.20) * 0.07
        + _unit(float(aligned_count), 3.0, 9.0) * 0.07
    )
    confidence = max(0.0, min(1.0, dim_score + panel_score + result_content_score))

    dimmed_result_layer = (
        metrics["bottom_dark_ratio"] >= 0.32
        and metrics["corner_dark_ratio"] >= 0.34
        and metrics["bottom_right_dark_ratio"] >= 0.30
    )
    blue_result_panel = metrics["center_blue_ratio"] >= 0.20
    diagonal_result_panel = metrics["result_diagonal_border_length"] >= 0.34
    result_content = aligned_count >= 4 or (
        len(upper_components) >= 6 and metrics["top_bright_ratio"] >= 0.11
    )
    detected = (
        dimmed_result_layer
        and blue_result_panel
        and diagonal_result_panel
        and result_content
        and confidence >= max(0.0, min(1.0, float(min_confidence)))
    )

    evidence: list[str] = []
    if dimmed_result_layer:
        evidence.append("dimmed_table_and_lower_screen")
    if blue_result_panel:
        evidence.append("large_blue_result_panel")
    if diagonal_result_panel:
        evidence.append("long_diagonal_result_border")
    if aligned_count >= 4:
        evidence.append(f"upper_aligned_tile_row:{aligned_count}")
    elif result_content:
        evidence.append(f"upper_result_content:{len(upper_components)}")

    # 中文：只有清晰的上方和牌牌列才标为 win；缺乏文字模型时流局不做武断细分。
    # English: Call it a win only with a clear winning-hand row; draw subtypes need explicit evidence.
    winning_hand_area = aligned_count >= 5 or (
        len(upper_components) >= 8 and metrics["top_bright_ratio"] >= 0.18
    )
    kind = "win" if detected and winning_hand_area else ("unknown" if detected else "none")
    reason = f"{kind}_settlement_signature" if detected else _failure_reason(
        dimmed_result_layer,
        blue_result_panel,
        diagonal_result_panel,
        result_content,
        confidence,
        min_confidence,
    )
    metrics["score"] = round(confidence, 4)
    metrics["image_width"] = int(image.width)
    metrics["image_height"] = int(image.height)
    return SettlementFrameResult(
        detected=detected,
        kind=kind,
        confidence=confidence,
        reason=reason,
        evidence=evidence,
        metrics=metrics,
        elapsed_ms=_elapsed_ms(started),
    )


def render_settlement_diagnostic_image(
    image: Image.Image,
    *,
    result: SettlementFrameResult | None = None,
    min_confidence: float = 0.72,
) -> Image.Image:
    """Draw the exact normalized regions and evidence used by settlement detection."""

    source = image.convert("RGB")
    frame_result = result or detect_settlement_image(source, min_confidence=min_confidence)
    canvas = source.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = canvas.size
    line_width = max(2, round(min(width, height) / 240))
    font_size = max(12, round(min(width, height) / 34))
    small_font_size = max(10, round(min(width, height) / 44))
    font = _diagnostic_font(font_size)
    small_font = _diagnostic_font(small_font_size)

    regions = (
        ("DIM", (0.0, 0.72, 1.0, 1.0), (73, 150, 255, 205)),
        ("DIM-R", (0.64, 0.62, 1.0, 1.0), (141, 114, 255, 210)),
        ("PANEL", (0.20, 0.18, 0.80, 0.72), (34, 211, 238, 215)),
        ("CONTENT", (0.08, 0.10, 0.92, 0.48), (250, 204, 21, 215)),
    )
    for label, normalized_box, color in regions:
        box = _pixel_box(normalized_box, width, height)
        draw.rectangle(box, outline=color, width=line_width)
        draw.text((box[0] + line_width, box[1] + line_width), label, fill=color, font=small_font)

    components = frame_result.metrics.get("upper_tile_components")
    if isinstance(components, list):
        for component in components:
            if not isinstance(component, dict):
                continue
            left = float(component.get("left") or 0.0)
            top = float(component.get("top") or 0.0)
            component_width = float(component.get("width") or 0.0)
            component_height = float(component.get("height") or 0.0)
            draw.rectangle(
                _pixel_box(
                    (left, top, left + component_width, top + component_height),
                    width,
                    height,
                ),
                outline=(65, 220, 126, 235),
                width=line_width,
            )

    diagonal_line = frame_result.metrics.get("result_diagonal_line")
    if isinstance(diagonal_line, list) and len(diagonal_line) == 4:
        x1, y1, x2, y2 = (float(value) for value in diagonal_line)
        draw.line(
            (
                round(x1 * width),
                round(y1 * height),
                round(x2 * width),
                round(y2 * height),
            ),
            fill=(255, 82, 170, 255),
            width=max(line_width + 1, 3),
        )

    detected_color = (65, 220, 126, 245) if frame_result.detected else (255, 107, 107, 245)
    status = (
        f"SETTLEMENT {frame_result.kind.upper()} {frame_result.confidence:.0%}"
        if frame_result.detected
        else f"NO SETTLEMENT {frame_result.confidence:.0%}"
    )
    metrics = frame_result.metrics
    thresholds = (
        f"dim {float(metrics.get('bottom_dark_ratio') or 0.0):.2f}/0.32  "
        f"panel {float(metrics.get('center_blue_ratio') or 0.0):.2f}/0.20  "
        f"diag {float(metrics.get('result_diagonal_border_length') or 0.0):.2f}/0.34  "
        f"row {int(metrics.get('upper_aligned_tile_count') or 0)}/4"
    )
    text_height = max(56, round(height * 0.105))
    status_top = max(8, height - text_height - 8)
    draw.rectangle(
        (8, status_top, max(9, width - 8), min(height - 1, status_top + text_height)),
        fill=(7, 12, 15, 215),
        outline=detected_color,
        width=line_width,
    )
    draw.text((16, status_top + 4), status, fill=detected_color, font=font)
    draw.text(
        (16, status_top + 6 + font_size),
        thresholds,
        fill=(236, 240, 239, 245),
        font=small_font,
    )
    draw.text(
        (16, status_top + 8 + font_size + small_font_size),
        frame_result.reason,
        fill=(185, 196, 193, 245),
        font=small_font,
    )
    return Image.alpha_composite(canvas, overlay).convert("RGB")


def _tile_like_components(mask: np.ndarray) -> list[dict[str, float | int]]:
    binary = np.asarray(mask, dtype=np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = binary.shape
    components: list[dict[str, float | int]] = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        area = float(cv2.contourArea(contour))
        box_area = max(1, box_width * box_height)
        aspect = box_width / max(1, box_height)
        if not (
            18 <= area <= 1450
            and 4 <= box_width <= 42
            and 7 <= box_height <= 70
            and 0.12 <= aspect <= 2.0
            and area / box_area >= 0.22
        ):
            continue
        components.append(
            {
                "left": x / width,
                "top": y / height,
                "width": box_width / width,
                "height": box_height / height,
                "center_x": (x + box_width / 2.0) / width,
                "center_y": (y + box_height / 2.0) / height,
                "area": int(area),
            }
        )
    return components


def _largest_aligned_row(components: list[dict[str, float | int]]) -> int:
    if not components:
        return 0
    centers = sorted(float(item["center_y"]) for item in components)
    best = 0
    for center in centers:
        count = sum(abs(candidate - center) <= 0.035 for candidate in centers)
        best = max(best, count)
    return best


def _longest_diagonal(mask: np.ndarray) -> tuple[float, list[float]]:
    binary = np.asarray(mask, dtype=np.uint8) * 255
    lines = cv2.HoughLinesP(
        binary,
        1,
        np.pi / 180.0,
        threshold=30,
        minLineLength=80,
        maxLineGap=30,
    )
    if lines is None:
        return 0.0, []
    width = max(1, binary.shape[1])
    height = max(1, binary.shape[0])
    best = 0.0
    best_line: list[float] = []
    for line in lines:
        x1, y1, x2, y2 = (int(value) for value in line[0])
        angle = abs(float(np.degrees(np.arctan2(y2 - y1, x2 - x1))))
        midpoint_y = (y1 + y2) / 2.0 / height
        reaches_right_edge = max(x1, x2) >= width * 0.88
        if not (4.0 <= angle <= 15.0 and 0.35 <= midpoint_y <= 0.88 and reaches_right_edge):
            continue
        length = float(np.hypot(x2 - x1, y2 - y1)) / width
        if length > best:
            best = length
            best_line = [
                round(x1 / width, 4),
                round(y1 / height, 4),
                round(x2 / width, 4),
                round(y2 / height, 4),
            ]
    return round(best, 4), best_line


def _longest_diagonal_ratio(mask: np.ndarray) -> float:
    ratio, _line = _longest_diagonal(mask)
    return ratio


def _region_ratio(
    mask: np.ndarray,
    left: float,
    top: float,
    right: float,
    bottom: float,
) -> float:
    height, width = mask.shape
    x1 = max(0, min(width - 1, int(width * left)))
    y1 = max(0, min(height - 1, int(height * top)))
    x2 = max(x1 + 1, min(width, int(width * right)))
    y2 = max(y1 + 1, min(height, int(height * bottom)))
    return round(float(mask[y1:y2, x1:x2].mean()), 4)


def _corner_ratio(mask: np.ndarray) -> float:
    regions = (
        _region_ratio(mask, 0.0, 0.0, 0.18, 0.20),
        _region_ratio(mask, 0.82, 0.0, 1.0, 0.20),
        _region_ratio(mask, 0.0, 0.80, 0.18, 1.0),
        _region_ratio(mask, 0.82, 0.80, 1.0, 1.0),
    )
    return round(sum(regions) / len(regions), 4)


def _failure_reason(
    dimmed: bool,
    blue_panel: bool,
    diagonal_panel: bool,
    result_content: bool,
    confidence: float,
    min_confidence: float,
) -> str:
    if not dimmed:
        return "settlement_dim_layer_missing"
    if not blue_panel:
        return "settlement_panel_missing"
    if not diagonal_panel:
        return "settlement_diagonal_border_missing"
    if not result_content:
        return "settlement_content_missing"
    if confidence < min_confidence:
        return "settlement_confidence_low"
    return "not_detected"


def _compatible_settlement_kinds(previous: str, current: str) -> bool:
    if previous in {"", "none"} or current in {"", "none"}:
        return False
    return previous == current or "unknown" in {previous, current}


def _pixel_box(
    normalized_box: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = normalized_box
    return (
        max(0, min(width - 1, round(left * width))),
        max(0, min(height - 1, round(top * height))),
        max(0, min(width - 1, round(right * width))),
        max(0, min(height - 1, round(bottom * height))),
    )


def _diagnostic_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for font_name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _unit(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (float(value) - low) / (high - low)))


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 1)
