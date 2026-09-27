from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .roi import RoiBox, collect_region_metrics

CALL_BUTTONS = {"chi", "pon", "kan"}
WIN_BUTTONS = {"ron", "tsumo"}
BUTTON_ORDER = ("ron", "tsumo", "riichi", "kan", "pon", "chi", "skip")


def detect_action_buttons_fast(image_path: Path) -> tuple[list[str], dict[str, Any]]:
    """Very light action-window hinting.

    This is intentionally conservative and exists as a fast interrupt layer.
    Call sites can also pass observed buttons from another detector.
    """
    started = time.perf_counter()
    if not image_path.exists():
        return [], {"error": 1.0, "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 1)}
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
        box = RoiBox(
            "bottom_action_bar",
            left=int(image.width * 0.24),
            top=int(image.height * 0.61),
            width=int(image.width * 0.52),
            height=int(image.height * 0.17),
        )
        metrics = collect_region_metrics(image, box, sample_step=5)
        if not _has_action_button_candidate(metrics):
            # 无明显按钮色块时跳过昂贵模板匹配，避免观察态每帧被拖慢。
            # Skip expensive template matching when the action area has no obvious
            # button-like highlights, so normal observation frames stay cheap.
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
    # 中文：蓝色牌桌本身会让 colorful_ratio 接近 1，不能作为按钮出现的依据。
    # English: The blue table makes colorful_ratio approach 1, so it cannot signal an action window.
    accent_ratio = max(
        float(metrics.get(name) or 0.0)
        for name in ("red_ratio", "green_ratio", "gold_ratio", "orange_ratio")
    )
    return (
        float(metrics.get("bright_ratio") or 0.0) >= 0.02
        or accent_ratio >= 0.008
        or float(metrics.get("stddev") or 0.0) >= 26.0
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
        score, match_location = _precise_template_match(search, template)
        threshold = max(0.86, min(0.99, float(item.get("match_threshold") or payload.get("default_match_threshold") or 0.9)))
        matches.append(
            {
                "button_type": button_type,
                "score": round(score, 4),
                "threshold": round(threshold, 4),
                "accepted": False,
                "match_box": [
                    int(search_box[0] + match_location[0]),
                    int(search_box[1] + match_location[1]),
                    int(template.width),
                    int(template.height),
                ],
            }
        )

    for item in matches:
        item["accepted"] = item["score"] >= item["threshold"]

    detected = [item["button_type"] for item in matches if item["accepted"]]
    return detected, {"available": True, "matches": sorted(matches, key=lambda item: -item["score"])[:8]}


def _precise_template_match(search: Image.Image, template: Image.Image) -> tuple[float, tuple[int, int]]:
    # 中文：保留按钮颜色与完整轮廓；旧灰度内容掩码会把蓝桌和白色麻将牌误认成所有按钮。
    # English: Keep full button color/shape; the old grayscale mask confused table tiles with every button.
    max_search_width = 420
    downscale = min(1.0, max_search_width / max(1, search.width))
    search_small = search.resize(
        (max(1, int(search.width * downscale)), max(1, int(search.height * downscale))),
        Image.Resampling.BILINEAR,
    )
    template_small = template.resize(
        (max(4, int(template.width * downscale)), max(4, int(template.height * downscale))),
        Image.Resampling.BILINEAR,
    )
    if template_small.width > search_small.width or template_small.height > search_small.height:
        return 0.0, (0, 0)

    search_arr = np.asarray(search_small.convert("RGB"), dtype=np.uint8)
    template_arr = np.asarray(template_small.convert("RGB"), dtype=np.uint8)
    scores = cv2.matchTemplate(search_arr, template_arr, cv2.TM_CCOEFF_NORMED)
    _minimum, best_score, _minimum_location, best_location = cv2.minMaxLoc(scores)
    if not np.isfinite(best_score):
        return 0.0, (0, 0)
    inverse_scale = 1.0 / max(downscale, 1e-6)
    location = (
        int(round(best_location[0] * inverse_scale)),
        int(round(best_location[1] * inverse_scale)),
    )
    return max(0.0, min(1.0, float(best_score))), location


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
