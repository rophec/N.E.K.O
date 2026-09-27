from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CalibrationOffsets:
    x_px: int = 0
    y_px: int = 0
    width_px: int = 0
    height_px: int = 0
    gap_px: int = 0
    draw_gap_px: int = 0


@dataclass(frozen=True)
class CalibrationProfile:
    profile_id: str = "default"
    enabled: bool = False
    screen_width: int = 0
    screen_height: int = 0
    content_left: int = 0
    content_top: int = 0
    content_width: int = 0
    content_height: int = 0
    confidence: float = 0.0
    hand_offsets: CalibrationOffsets = field(default_factory=CalibrationOffsets)
    hand_tile_templates: dict[str, Any] = field(default_factory=dict)


def resolve_calibration_profile(
    width: int,
    height: int,
    *,
    calibration_dir: Path | None = None,
) -> CalibrationProfile:
    if calibration_dir is not None and calibration_dir.exists():
        exact = _find_profile(calibration_dir, width, height)
        if exact is not None:
            return exact
        scaled = _find_scaled_profile(calibration_dir, width, height)
        if scaled is not None:
            return scaled
    return CalibrationProfile(
        profile_id=f"default-{width}x{height}",
        screen_width=max(0, int(width)),
        screen_height=max(0, int(height)),
    )


def _find_profile(calibration_dir: Path, width: int, height: int) -> CalibrationProfile | None:
    candidates: list[CalibrationProfile] = []
    for path in sorted(calibration_dir.glob("*.json")):
        try:
            profile = load_calibration_profile(path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if not profile.enabled or not profile.hand_tile_templates:
            continue
        if profile.screen_width == int(width) and profile.screen_height == int(height):
            candidates.append(profile)
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.confidence)


def _find_scaled_profile(calibration_dir: Path, width: int, height: int) -> CalibrationProfile | None:
    target_width = max(1, int(width))
    target_height = max(1, int(height))
    candidates: list[tuple[float, CalibrationProfile]] = []
    for path in sorted(calibration_dir.glob("*.json")):
        try:
            profile = load_calibration_profile(path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if not profile.enabled or not profile.hand_tile_templates:
            continue
        if profile.screen_width <= 0 or profile.screen_height <= 0:
            continue
        scaled = _scale_profile_to_capture(profile, target_width, target_height)
        if scaled is None:
            continue
        scale = scaled.content_width / max(1, profile.screen_width)
        coverage = (scaled.content_width * scaled.content_height) / float(target_width * target_height)
        scale_closeness = 1.0 / (1.0 + abs(math.log(max(scale, 0.001))))
        score = (coverage * 10.0) + scale_closeness + float(profile.confidence)
        candidates.append((score, scaled))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _scale_profile_to_capture(profile: CalibrationProfile, width: int, height: int) -> CalibrationProfile | None:
    scale = min(width / float(profile.screen_width), height / float(profile.screen_height))
    if scale <= 0:
        return None
    content_width = max(1, int(round(profile.screen_width * scale)))
    content_height = max(1, int(round(profile.screen_height * scale)))
    coverage = (content_width * content_height) / float(max(1, width * height))
    if coverage < 0.65:
        return None
    content_left = int(round((width - content_width) / 2.0))
    content_top = int(round((height - content_height) / 2.0))
    return CalibrationProfile(
        profile_id=f"{profile.profile_id}-scaled-{width}x{height}",
        enabled=True,
        screen_width=width,
        screen_height=height,
        content_left=max(0, content_left),
        content_top=max(0, content_top),
        content_width=content_width,
        content_height=content_height,
        confidence=float(profile.confidence) * 0.9,
        hand_offsets=CalibrationOffsets(
            x_px=_scale_int(profile.hand_offsets.x_px, scale),
            y_px=_scale_int(profile.hand_offsets.y_px, scale),
            width_px=_scale_int(profile.hand_offsets.width_px, scale),
            height_px=_scale_int(profile.hand_offsets.height_px, scale),
            gap_px=_scale_int(profile.hand_offsets.gap_px, scale),
            draw_gap_px=_scale_int(profile.hand_offsets.draw_gap_px, scale),
        ),
        hand_tile_templates=dict(profile.hand_tile_templates),
    )


def load_calibration_profile(path: Path) -> CalibrationProfile:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("calibration profile must be a JSON object")
    return CalibrationProfile(
        profile_id=str(payload.get("profile_id") or path.stem),
        enabled=bool(payload.get("enabled", True)),
        screen_width=int(payload.get("screen_width") or 0),
        screen_height=int(payload.get("screen_height") or 0),
        confidence=float(payload.get("confidence") or 0.0),
        hand_offsets=_load_offsets(payload.get("hand_offsets")),
        hand_tile_templates=_load_template_payload(payload.get("hand_tile_templates")),
    )


def _load_offsets(value: Any) -> CalibrationOffsets:
    if not isinstance(value, dict):
        return CalibrationOffsets()
    return CalibrationOffsets(
        x_px=int(value.get("x_px") or 0),
        y_px=int(value.get("y_px") or 0),
        width_px=int(value.get("width_px") or 0),
        height_px=int(value.get("height_px") or 0),
        gap_px=int(value.get("gap_px") or 0),
        draw_gap_px=int(value.get("draw_gap_px") or 0),
    )


def _load_template_payload(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _scale_int(value: int, scale: float) -> int:
    return int(round(int(value) * float(scale)))
