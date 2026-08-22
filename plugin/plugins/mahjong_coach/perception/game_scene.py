from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .image_source import ImageSource, open_rgb, source_exists
from .table_surface import TableSurfaceResult, detect_table_surface


_CENTER_ROI = (0.36, 0.30, 0.64, 0.65)
_MIN_CYAN_RATIO = 0.003
_MIN_GOLD_RATIO = 0.001
_MIN_DARK_RATIO = 0.08
_MIN_TABLE_QUALITY = 0.65


@dataclass(frozen=True)
class GameSceneResult:
    detected: bool = False
    confidence: float = 0.0
    reason: str = ""
    elapsed_ms: float = 0.0
    evidence: list[str] = field(default_factory=list)
    center_metrics: dict[str, float] = field(default_factory=dict)
    center_roi: dict[str, int] = field(default_factory=dict)
    table_hints: dict[str, Any] = field(default_factory=dict)
    table_surface: TableSurfaceResult | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "detected": self.detected,
            "confidence": round(float(self.confidence), 4),
            "reason": self.reason,
            "elapsed_ms": round(float(self.elapsed_ms), 1),
            "evidence": list(self.evidence),
            "center_metrics": dict(self.center_metrics),
            "center_roi": dict(self.center_roi),
            "table_hints": dict(self.table_hints),
        }


def detect_game_scene_path(image_path: ImageSource) -> GameSceneResult:
    started = time.perf_counter()
    if not source_exists(image_path):
        return GameSceneResult(
            reason="image_missing",
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )
    try:
        with open_rgb(image_path) as opened:
            image = opened.copy()
    except Exception:
        return GameSceneResult(
            reason="image_unreadable",
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )
    return detect_game_scene_image(image, started=started)


def detect_game_scene_image(
    image: Image.Image,
    *,
    started: float | None = None,
) -> GameSceneResult:
    started = time.perf_counter() if started is None else started
    rgb = image.convert("RGB")
    width, height = rgb.size
    if width < 320 or height < 180:
        return GameSceneResult(
            reason="frame_too_small",
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

    left = int(width * _CENTER_ROI[0])
    top = int(height * _CENTER_ROI[1])
    right = int(width * _CENTER_ROI[2])
    bottom = int(height * _CENTER_ROI[3])
    center_roi = {
        "left": left,
        "top": top,
        "width": right - left,
        "height": bottom - top,
    }
    metrics = _center_panel_metrics(rgb.crop((left, top, right, bottom)))
    evidence: list[str] = []
    if metrics["cyan_ratio"] >= _MIN_CYAN_RATIO:
        evidence.append("center_cyan_round_text")
    if metrics["gold_ratio"] >= _MIN_GOLD_RATIO:
        evidence.append("center_gold_score_text")
    if metrics["dark_ratio"] >= _MIN_DARK_RATIO:
        evidence.append("center_dark_score_panel")

    center_ok = len(evidence) == 3
    if not center_ok:
        return GameSceneResult(
            reason="center_score_panel_not_found",
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            evidence=evidence,
            center_metrics=metrics,
            center_roi=center_roi,
        )

    table_surface = detect_table_surface(rgb)
    table_hints = table_surface.to_hints()
    if not table_surface.ok:
        return GameSceneResult(
            reason=table_surface.reason or "table_surface_not_found",
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            evidence=evidence,
            center_metrics=metrics,
            center_roi=center_roi,
            table_hints=table_hints,
            table_surface=table_surface,
        )
    if float(table_surface.quality_score) < _MIN_TABLE_QUALITY:
        return GameSceneResult(
            reason="table_surface_quality_too_low",
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            evidence=[*evidence, "table_surface_detected"],
            center_metrics=metrics,
            center_roi=center_roi,
            table_hints=table_hints,
            table_surface=table_surface,
        )

    evidence.append("table_surface_detected")
    cyan_score = min(1.0, metrics["cyan_ratio"] / 0.006)
    gold_score = min(1.0, metrics["gold_ratio"] / 0.015)
    dark_score = min(1.0, metrics["dark_ratio"] / 0.20)
    confidence = (
        0.45 * float(table_surface.quality_score)
        + 0.30 * cyan_score
        + 0.15 * gold_score
        + 0.10 * dark_score
    )
    return GameSceneResult(
        detected=True,
        confidence=max(0.0, min(1.0, confidence)),
        reason="active_mahjong_table",
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
        evidence=evidence,
        center_metrics=metrics,
        center_roi=center_roi,
        table_hints=table_hints,
        table_surface=table_surface,
    )


def _center_panel_metrics(crop: Image.Image) -> dict[str, float]:
    hsv = np.asarray(crop.convert("HSV"), dtype=np.uint8)
    hue = hsv[..., 0]
    saturation = hsv[..., 1]
    value = hsv[..., 2]

    # PIL hue uses 0..255 instead of OpenCV's 0..179. JPEG compression and
    # seasonal table skins materially lower text saturation, so the hue band
    # carries more weight than a brittle high-saturation threshold. The outer
    # perspective geometry remains mandatory and keeps lobby artwork out.
    cyan = (
        (hue >= 106)
        & (hue <= 149)
        & (saturation >= 80)
        & (value >= 100)
    )
    gold = (
        (hue >= 14)
        & (hue <= 50)
        & (saturation >= 90)
        & (value >= 130)
    )
    dark = (value <= 105) & (saturation <= 120)
    return {
        "cyan_ratio": round(float(cyan.mean()), 6),
        "gold_ratio": round(float(gold.mean()), 6),
        "dark_ratio": round(float(dark.mean()), 6),
    }
