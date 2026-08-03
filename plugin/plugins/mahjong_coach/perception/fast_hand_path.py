from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .calibration import resolve_calibration_profile
from .hand_layout import build_hand_layout
from .image_source import ImageSource, open_rgb, source_exists
from .roi import RoiBox, collect_region_metrics
from .tile_classifier_dispatch import classify_hand_tile
from .tile_templates import is_probably_occupied_hand_slot


MIN_FAST_HAND_CONFIDENCE = 0.12

# These bands are expressed in normalized coordinates of the perspective-
# corrected table, not the original screenshot.  They cover the four inner
# discard lanes while leaving the outer hand/meld shelves out of the cheap
# fingerprint gate.
RIVER_FINGERPRINT_ROIS: dict[str, tuple[float, float, float, float]] = {
    "self": (0.28, 0.56, 0.72, 0.76),
    "left_opponent": (0.24, 0.28, 0.44, 0.72),
    "top_opponent": (0.28, 0.24, 0.72, 0.44),
    "right_opponent": (0.56, 0.28, 0.76, 0.72),
}


@dataclass(frozen=True)
class FastHandResult:
    ok: bool = False
    hand_tiles: list[str] = field(default_factory=list)
    confidence: float = 0.0
    reason: str = ""
    elapsed_ms: float = 0.0
    raw_detections: list[dict[str, Any]] = field(default_factory=list)
    draw_slot_index: int = 14
    analysis_hints: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["elapsed_ms"] = round(float(self.elapsed_ms), 1)
        payload["confidence"] = round(float(self.confidence), 4)
        return payload


def detect_fast_hand_path(
    image_path: ImageSource,
    *,
    calibration_dir: Path | None = None,
    min_hand_tiles: int = 12,
    max_hand_tiles: int = 14,
    use_onnx_hand: bool | None = None,
) -> FastHandResult:
    started = time.perf_counter()
    if not source_exists(image_path):
        return FastHandResult(reason="image_missing")
    min_tiles = max(1, min(14, int(min_hand_tiles or 12)))
    max_tiles = max(min_tiles, min(14, int(max_hand_tiles or 14)))

    with open_rgb(image_path) as image:
        calibration = resolve_calibration_profile(*image.size, calibration_dir=calibration_dir)
        template_payload = calibration.hand_tile_templates
        if not calibration.enabled or not template_payload:
            return FastHandResult(
                reason="missing_hand_tile_templates",
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )

        layout = build_hand_layout(*image.size, calibration=calibration)
        hand_tiles: list[str] = []
        confidences: list[float] = []
        raw_detections: list[dict[str, Any]] = []
        empty_streak_after_hand = 0
        for slot in layout["hand"][:14]:
            metrics = collect_region_metrics(image, slot.box, sample_step=6)
            occupied = is_probably_occupied_hand_slot(
                {
                    "slot_mean_luma": metrics.get("mean_luma"),
                    "slot_bright_ratio": metrics.get("bright_ratio"),
                    "slot_dark_ratio": metrics.get("dark_ratio"),
                    "slot_stddev": metrics.get("stddev"),
                }
            )
            detection = {
                "slot_id": slot.slot_id,
                "candidate_tile": "",
                "confidence": 0.0,
                "box": slot.box.to_dict(),
                "occupied": occupied,
                "source": "legacy_fast_hand_path",
            }
            if not occupied:
                raw_detections.append(detection)
                if hand_tiles:
                    empty_streak_after_hand += 1
                    if empty_streak_after_hand >= 2:
                        break
                continue
            empty_streak_after_hand = 0
            crop = image.crop((slot.box.left, slot.box.top, slot.box.right, slot.box.bottom))
            match = classify_hand_tile(
                crop,
                template_payload,
                use_onnx=use_onnx_hand,
                fallback_to_onnx=True,
            )
            if match is None:
                raw_detections.append(detection)
                continue
            confidence = float(match.confidence)
            detection.update(
                {
                    "candidate_tile": match.tile,
                    "confidence": confidence,
                    "template_distance": match.distance,
                    "runner_up_tile": match.runner_up_tile,
                    "runner_up_distance": match.runner_up_distance,
                }
            )
            if confidence < MIN_FAST_HAND_CONFIDENCE:
                detection["accepted"] = False
                detection["rejection_reason"] = "low_confidence"
                raw_detections.append(detection)
                continue
            detection["accepted"] = True
            hand_tiles.append(match.tile)
            confidences.append(confidence)
            raw_detections.append(detection)

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    mean_confidence = sum(confidences) / max(1, len(confidences))
    hand_count = len(hand_tiles)
    if not (min_tiles <= hand_count <= max_tiles):
        return FastHandResult(
            hand_tiles=hand_tiles,
            confidence=mean_confidence,
            reason="unstable_hand_count",
            elapsed_ms=elapsed_ms,
            raw_detections=raw_detections,
        )
    confidence = mean_confidence
    if hand_count < 13:
        confidence = round(mean_confidence * (0.92 if hand_count >= 12 else 0.84), 4)
    reason_prefix = "matched_open" if hand_count < 12 else "matched"
    return FastHandResult(
        ok=True,
        hand_tiles=hand_tiles,
        confidence=confidence,
        reason=f"{reason_prefix}_{hand_count}_hand_tiles",
        elapsed_ms=elapsed_ms,
        raw_detections=raw_detections,
    )


def quick_frame_fingerprint(
    image_path: ImageSource,
    last_hashes: dict[str, bytes] | None = None,
    *,
    warped_table: Image.Image | None = None,
) -> dict[str, Any]:
    """Cheap pixel hash of action, hand, and four warped river regions.

    Returns dict with:
      - action_changed: bool
      - hand_changed: bool
      - river_changed: bool
      - river_changes: dict[str, bool]
      - hashes: dict[str, bytes]  (for next call's last_hashes)
    """
    last_hashes = last_hashes or {}
    if not source_exists(image_path):
        return {
            "action_changed": True,
            "hand_changed": True,
            "river_changed": True,
            "river_changes": {owner: True for owner in RIVER_FINGERPRINT_ROIS},
            "hashes": {},
        }

    with open_rgb(image_path) as opened:
        image = opened.copy()
    w, h = image.width, image.height
    arr = np.asarray(image, dtype=np.int16)

    # Action bar region (same as action_detector.py uses)
    ax = int(w * 0.18)
    ay = int(h * 0.54)
    aw = int(w * 0.68)
    ah = int(h * 0.28)
    action_crop = arr[ay:ay + ah:8, ax:ax + aw:8]
    action_hash = _row_hash(action_crop)

    # Hand region (same as hand_layout.py uses)
    hx = int(w * 0.14)
    hy = int(h * 0.72)
    hw = int(w * 0.54)
    hh = int(h * 0.15)
    hand_crop = arr[hy:hy + hh:8, hx:hx + hw:8]
    hand_hash = _row_hash(hand_crop)

    hashes = {"action": action_hash, "hand": hand_hash}
    river_changes: dict[str, bool] = {}
    if warped_table is None:
        # Missing or rejected perspective geometry must fail open.  Treating
        # it as unchanged would suppress opponent discards and riichi turns.
        river_changes = {owner: True for owner in RIVER_FINGERPRINT_ROIS}
    else:
        table = warped_table.convert("RGB")
        for owner, bounds in RIVER_FINGERPRINT_ROIS.items():
            key = f"river:{owner}"
            value = _region_hash(table, bounds)
            hashes[key] = value
            river_changes[owner] = value != last_hashes.get(key, b"")

    return {
        "action_changed": action_hash != last_hashes.get("action", b""),
        "hand_changed": hand_hash != last_hashes.get("hand", b""),
        "river_changed": any(river_changes.values()),
        "river_changes": river_changes,
        "hashes": hashes,
    }


def _row_hash(crop: np.ndarray) -> bytes:
    """Compact hash of a 2D pixel array — sum of each row's mean."""
    if crop.size == 0:
        return b""
    row_means = crop.reshape(crop.shape[0], -1).mean(axis=1)
    return row_means.astype(np.float32).tobytes()


def _region_hash(image: Image.Image, bounds: tuple[float, float, float, float]) -> bytes:
    """Return a compact spatial hash while retaining tile orientation changes."""
    width, height = image.size
    left = max(0, min(width - 1, int(round(width * bounds[0]))))
    top = max(0, min(height - 1, int(round(height * bounds[1]))))
    right = max(left + 1, min(width, int(round(width * bounds[2]))))
    bottom = max(top + 1, min(height, int(round(height * bounds[3]))))
    crop = image.crop((left, top, right, bottom)).resize((48, 24), Image.Resampling.BILINEAR)
    # Five-bit color is enough to preserve a newly placed or rotated tile and
    # avoids rerunning YOLO for insignificant one-level capture noise.
    quantized = (np.asarray(crop, dtype=np.uint8) >> 3).tobytes()
    return hashlib.blake2s(quantized, digest_size=16).digest()
