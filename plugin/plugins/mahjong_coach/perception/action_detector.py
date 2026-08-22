from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .image_source import ImageSource, open_rgb, source_exists
from .roi import RoiBox, collect_region_metrics

CALL_BUTTONS = {"chi", "pon", "kan"}
WIN_BUTTONS = {"ron", "tsumo"}
BUTTON_ORDER = ("ron", "tsumo", "riichi", "kan", "pon", "chi", "skip")
DEFAULT_MATCH_THRESHOLD = 0.58
MAX_SEARCH_WIDTH = 520
MATCH_NMS_IOU_THRESHOLD = 0.45
RGB_MATCH_WEIGHT = 0.55
HSV_MATCH_WEIGHT = 0.45
CONTEXTUAL_SKIP_THRESHOLD = 0.44
STRONG_NEIGHBOR_THRESHOLD = 0.72


def detect_action_buttons_fast(image_path: ImageSource) -> tuple[list[str], dict[str, Any]]:
    """Very light action-window hinting.

    This is intentionally conservative and exists as a fast interrupt layer.
    Call sites can also pass observed buttons from another detector.
    """
    started = time.perf_counter()
    if not source_exists(image_path):
        return [], {"error": 1.0, "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 1)}
    with open_rgb(image_path) as image:
        box = RoiBox(
            "bottom_action_bar",
            left=int(image.width * 0.24),
            top=int(image.height * 0.61),
            width=int(image.width * 0.52),
            height=int(image.height * 0.17),
        )
        metrics = collect_region_metrics(image, box, sample_step=5)
        if not _has_action_button_candidate(metrics):
            # This is only a cheap pre-filter. Color metrics are deliberately not
            # treated as action evidence by themselves.
            return [], {
                "metrics": metrics,
                "templates": {"available": True, "skipped": "no_action_candidate"},
                "button_filter": {"input_buttons": [], "rejected": False, "reasons": []},
                "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 1),
            }
        template_buttons, template_meta = _detect_template_buttons(image)
    buttons, filter_meta = _filter_plausible_buttons(template_buttons)
    return buttons, {
        "metrics": metrics,
        "templates": template_meta,
        "button_filter": filter_meta,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 1),
    }


def _has_action_button_candidate(metrics: dict[str, Any]) -> bool:
    color_ratio = sum(
        float(metrics.get(name) or 0.0)
        for name in ("red_ratio", "green_ratio", "gold_ratio", "orange_ratio", "colorful_ratio")
    )
    return (
        float(metrics.get("bright_ratio") or 0.0) >= 0.01
        or color_ratio >= 0.012
        or float(metrics.get("stddev") or 0.0) >= 42.0
    )


def _detect_template_buttons(image: Image.Image) -> tuple[list[str], dict[str, Any]]:
    template_root = Path(__file__).resolve().parent / "templates"
    meta_path = template_root / "meta.json"
    if not meta_path.exists():
        return [], {"available": False}
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return [], {"available": False, "error": "meta_unreadable"}
    templates = payload.get("templates")
    if not isinstance(templates, dict):
        return [], {"available": False, "error": "templates_missing"}

    search_box = (
        int(image.width * 0.18),
        int(image.height * 0.54),
        int(image.width * 0.86),
        int(image.height * 0.82),
    )
    search = image.crop(search_box).convert("RGB")
    matches: list[dict[str, Any]] = []
    for item in templates.values():
        if not isinstance(item, dict):
            continue
        button_type = str(item.get("button_type") or "").strip()
        rel_file = str(item.get("file") or "").strip()
        resolution = item.get("resolution") if isinstance(item.get("resolution"), list) else []
        if not button_type or not rel_file or len(resolution) != 2:
            continue
        template_path = template_root / rel_file
        if not template_path.exists():
            continue
        with Image.open(template_path) as opened:
            template = opened.convert("RGB")
        scale = min(image.width / max(1, int(resolution[0])), image.height / max(1, int(resolution[1])))
        if abs(scale - 1.0) > 0.05:
            template = template.resize(
                (max(8, int(template.width * scale)), max(8, int(template.height * scale))),
                Image.Resampling.BILINEAR,
            )
        match = _coarse_template_match(search, template)
        threshold = _match_threshold(item, payload)
        relative_box = match.get("box")
        frame_box = None
        if isinstance(relative_box, list) and len(relative_box) == 4:
            frame_box = [
                int(relative_box[0]) + search_box[0],
                int(relative_box[1]) + search_box[1],
                int(relative_box[2]) + search_box[0],
                int(relative_box[3]) + search_box[1],
            ]
        matches.append(
            {
                "button_type": button_type,
                "score": round(float(match["score"]), 4),
                "rgb_score": round(float(match["rgb_score"]), 4),
                "hsv_score": round(float(match["hsv_score"]), 4),
                "box": frame_box,
                "threshold": round(threshold, 4),
                "above_threshold": float(match["score"]) >= threshold,
                "accepted": False,
            }
        )

    _recover_contextual_skip(matches)
    _suppress_overlapping_matches(matches)
    detected = [item["button_type"] for item in matches if item["accepted"]]
    return detected, {
        "available": True,
        "matcher": "fused_rgb_hsv_tm_ccoeff_normed_v2",
        "search_box": list(search_box),
        "matches": sorted(matches, key=lambda item: -item["score"])[:8],
    }


def _coarse_template_score(search: Image.Image, template: Image.Image) -> float:
    """Return the fused normalized-correlation score for compatibility."""
    return float(_coarse_template_match(search, template)["score"])


def _coarse_template_match(search: Image.Image, template: Image.Image) -> dict[str, Any]:
    """Locate a button with color-aware normalized correlation.

    The old grayscale mean-absolute-error matcher assigned high scores to
    ordinary blue table regions because most template pixels are also table
    background. Normalized correlation compares spatial structure instead of
    average brightness. Fusing RGB and HSV at the same location also prevents a
    gray tile or table seam from satisfying a colorful action-button template.
    """
    downscale = min(1.0, MAX_SEARCH_WIDTH / max(1, search.width))
    search_small = search.resize(
        (max(1, int(search.width * downscale)), max(1, int(search.height * downscale))),
        Image.Resampling.BILINEAR,
    )
    template_small = template.resize(
        (max(4, int(template.width * downscale)), max(4, int(template.height * downscale))),
        Image.Resampling.BILINEAR,
    )
    if template_small.width > search_small.width or template_small.height > search_small.height:
        return {"score": 0.0, "rgb_score": 0.0, "hsv_score": 0.0, "box": None}

    search_rgb = np.asarray(search_small.convert("RGB"), dtype=np.uint8)
    template_rgb = np.asarray(template_small.convert("RGB"), dtype=np.uint8)
    rgb_response = cv2.matchTemplate(search_rgb, template_rgb, cv2.TM_CCOEFF_NORMED)
    search_hsv = cv2.cvtColor(search_rgb, cv2.COLOR_RGB2HSV)
    template_hsv = cv2.cvtColor(template_rgb, cv2.COLOR_RGB2HSV)
    hsv_response = cv2.matchTemplate(search_hsv, template_hsv, cv2.TM_CCOEFF_NORMED)

    fused_response = (RGB_MATCH_WEIGHT * rgb_response) + (HSV_MATCH_WEIGHT * hsv_response)
    _min_score, fused_score, _min_location, location = cv2.minMaxLoc(fused_response)
    x, y = location
    rgb_score = float(rgb_response[y, x])
    hsv_score = float(hsv_response[y, x])
    scale_x = search.width / max(1, search_small.width)
    scale_y = search.height / max(1, search_small.height)
    box = [
        round(x * scale_x),
        round(y * scale_y),
        round((x + template_small.width) * scale_x),
        round((y + template_small.height) * scale_y),
    ]
    return {
        "score": max(0.0, min(1.0, float(fused_score))),
        "rgb_score": max(-1.0, min(1.0, rgb_score)),
        "hsv_score": max(-1.0, min(1.0, hsv_score)),
        "box": box,
    }


def _match_threshold(item: dict[str, Any], payload: dict[str, Any]) -> float:
    raw_threshold = item.get("match_threshold") or payload.get("default_match_threshold") or DEFAULT_MATCH_THRESHOLD
    try:
        threshold = float(raw_threshold)
    except (TypeError, ValueError):
        threshold = DEFAULT_MATCH_THRESHOLD
    return max(0.35, min(0.99, threshold))


def _suppress_overlapping_matches(matches: list[dict[str, Any]]) -> None:
    """Keep only the strongest label when templates claim the same button."""
    kept: list[dict[str, Any]] = []
    candidates = sorted(
        (item for item in matches if item.get("above_threshold") and item.get("box")),
        key=lambda item: -float(item["score"]),
    )
    for candidate in candidates:
        overlap_iou, conflicting = max(
            ((_box_iou(candidate["box"], accepted["box"]), accepted) for accepted in kept),
            default=(0.0, None),
            key=lambda pair: pair[0],
        )
        if conflicting is not None and overlap_iou >= MATCH_NMS_IOU_THRESHOLD:
            candidate["suppressed_by"] = conflicting["button_type"]
            candidate["suppressed_iou"] = round(overlap_iou, 4)
            continue
        candidate["accepted"] = True
        kept.append(candidate)


def _recover_contextual_skip(matches: list[dict[str, Any]]) -> None:
    """Recover a partly occluded skip button beside one unambiguous action.

    Character effects can cover the wide right-side skip button. A lower score
    is accepted only when it is horizontally aligned to the right of a strong,
    independently matched action button. It can never create a standalone
    action-window result.
    """
    skip = next((item for item in matches if item.get("button_type") == "skip"), None)
    if skip is None or skip.get("above_threshold") or float(skip.get("score") or 0.0) < CONTEXTUAL_SKIP_THRESHOLD:
        return
    strong_actions = [
        item
        for item in matches
        if item.get("button_type") != "skip"
        and item.get("box")
        and float(item.get("score") or 0.0) >= max(float(item.get("threshold") or 0.0), STRONG_NEIGHBOR_THRESHOLD)
    ]
    if not strong_actions or not skip.get("box"):
        return
    neighbor = max(strong_actions, key=lambda item: float(item["score"]))
    if not _is_right_aligned_neighbor(neighbor["box"], skip["box"]):
        return
    skip["above_threshold"] = True
    skip["contextual_recovery"] = {
        "reason": "strong_left_action",
        "neighbor": neighbor["button_type"],
        "minimum_score": CONTEXTUAL_SKIP_THRESHOLD,
    }


def _is_right_aligned_neighbor(action_box: list[int], skip_box: list[int]) -> bool:
    action_center_x = (action_box[0] + action_box[2]) / 2.0
    action_center_y = (action_box[1] + action_box[3]) / 2.0
    skip_center_x = (skip_box[0] + skip_box[2]) / 2.0
    skip_center_y = (skip_box[1] + skip_box[3]) / 2.0
    max_height = max(action_box[3] - action_box[1], skip_box[3] - skip_box[1])
    max_width = max(action_box[2] - action_box[0], skip_box[2] - skip_box[0])
    return (
        skip_center_x > action_center_x
        and abs(skip_center_y - action_center_y) <= max_height * 0.55
        and skip_box[0] - action_box[2] <= max_width * 0.45
    )


def _box_iou(left: list[int], right: list[int]) -> float:
    intersection_width = max(0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    left_area = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    right_area = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _filter_plausible_buttons(buttons: list[str]) -> tuple[list[str], dict[str, Any]]:
    unique = {str(button).strip() for button in buttons if str(button).strip()}
    conflicts: list[str] = []
    if "ron" in unique and "tsumo" in unique:
        conflicts.append("ron_with_tsumo")
    if "tsumo" in unique and any(button in unique for button in CALL_BUTTONS):
        conflicts.append("tsumo_with_call")
    if "riichi" in unique and any(button in unique for button in CALL_BUTTONS):
        conflicts.append("riichi_with_call")
    if len([button for button in unique if button != "skip"]) > 4:
        conflicts.append("too_many_action_buttons")
    if conflicts:
        return [], {"input_buttons": sorted(unique), "rejected": True, "reasons": conflicts}
    ordered = [button for button in BUTTON_ORDER if button in unique]
    return ordered, {"input_buttons": sorted(unique), "rejected": False, "reasons": []}
