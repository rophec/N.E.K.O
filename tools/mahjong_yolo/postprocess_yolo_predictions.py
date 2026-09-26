from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Any


@dataclass(frozen=True)
class PostprocessConfig:
    min_confidence: float = 0.05
    min_width: float = 14.0
    min_height: float = 14.0
    max_width: float = 95.0
    max_height: float = 105.0
    min_area: float = 320.0
    max_area: float = 9000.0
    min_aspect: float = 0.35
    max_aspect: float = 3.20
    duplicate_iou: float = 0.28
    duplicate_center_ratio: float = 0.42
    duplicate_center_px: float = 10.0
    image_width: float = 800.0
    image_height: float = 800.0
    river_min_x: float = 210.0
    river_max_x: float = 590.0
    river_min_y: float = 210.0
    river_max_y: float = 590.0
    self_hand_min_y: float = 640.0


def box_width(pred: dict[str, Any]) -> float:
    x1, _y1, x2, _y2 = pred["xyxy"]
    return max(0.0, float(x2) - float(x1))


def box_height(pred: dict[str, Any]) -> float:
    _x1, y1, _x2, y2 = pred["xyxy"]
    return max(0.0, float(y2) - float(y1))


def box_area(pred: dict[str, Any]) -> float:
    return box_width(pred) * box_height(pred)


def box_center(pred: dict[str, Any]) -> tuple[float, float]:
    x1, y1, x2, y2 = pred["xyxy"]
    return ((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0)


def iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a["xyxy"]]
    bx1, by1, bx2, by2 = [float(v) for v in b["xyxy"]]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    union = box_area(a) + box_area(b) - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def geometry_reject_reason(pred: dict[str, Any], config: PostprocessConfig) -> str | None:
    # EN: Mahjong visible faces in the warped 800x800 view have a bounded size range.
    # ZH: 在 800x800 透视变换图里，可见牌面尺寸应该落在稳定范围内。
    width = box_width(pred)
    height = box_height(pred)
    area = width * height
    aspect = width / height if height > 0 else 999.0
    confidence = float(pred["confidence"])
    if confidence < config.min_confidence:
        return "low_confidence"
    if width < config.min_width or height < config.min_height:
        return "too_small"
    if width > config.max_width or height > config.max_height:
        return "too_large"
    if area < config.min_area:
        return "area_too_small"
    if area > config.max_area:
        return "area_too_large"
    if aspect < config.min_aspect or aspect > config.max_aspect:
        return "bad_aspect"
    return None


def same_tile_candidate(a: dict[str, Any], b: dict[str, Any], config: PostprocessConfig) -> bool:
    # EN: Low thresholds often produce several class guesses for one physical tile.
    # ZH: 低阈值下，同一张实体牌常常会产生多个类别猜测。
    overlap = iou(a, b)
    if overlap >= config.duplicate_iou:
        return True
    acx, acy = box_center(a)
    bcx, bcy = box_center(b)
    center_distance = hypot(acx - bcx, acy - bcy)
    smaller_side = min(box_width(a), box_height(a), box_width(b), box_height(b))
    center_limit = max(config.duplicate_center_px, smaller_side * config.duplicate_center_ratio)
    return center_distance <= center_limit and overlap > 0.08


def sort_for_review(preds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        preds,
        key=lambda p: (
            round(float(p["xyxy"][1]) / 20.0),
            float(p["xyxy"][0]),
            -float(p["confidence"]),
        ),
    )


def assign_area(pred: dict[str, Any], config: PostprocessConfig) -> dict[str, Any]:
    # EN: First-pass spatial ownership for the warped 800x800 review image.
    # ZH: 针对 800x800 变换图的第一版区域/归属判断。
    cx, cy = box_center(pred)
    out = dict(pred)

    if cy >= config.self_hand_min_y:
        out["area_kind"] = "self_hand"
        out["owner"] = "self"
        return out

    in_river = (
        config.river_min_x <= cx <= config.river_max_x
        and config.river_min_y <= cy <= config.river_max_y
    )
    if in_river:
        out["area_kind"] = "river"
        center_x = config.image_width / 2.0
        center_y = config.image_height / 2.0
        dx = cx - center_x
        dy = cy - center_y
        if abs(dx) > abs(dy):
            out["owner"] = "right_opponent" if dx > 0 else "left_opponent"
        else:
            out["owner"] = "self" if dy > 0 else "across_opponent"
        return out

    out["area_kind"] = "other_visible_tile"
    out["owner"] = "unknown"
    return out


def postprocess_predictions(
    predictions: list[dict[str, Any]],
    config: PostprocessConfig | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    config = config or PostprocessConfig()
    rejected: list[dict[str, Any]] = []
    geometry_kept: list[dict[str, Any]] = []

    for pred in predictions:
        reason = geometry_reject_reason(pred, config)
        if reason:
            rejected.append({**pred, "reject_reason": reason})
            continue
        geometry_kept.append(dict(pred))

    selected: list[dict[str, Any]] = []
    for pred in sorted(geometry_kept, key=lambda p: float(p["confidence"]), reverse=True):
        duplicate_of = next(
            (kept for kept in selected if same_tile_candidate(pred, kept, config)),
            None,
        )
        if duplicate_of is not None:
            rejected.append(
                {
                    **pred,
                    "reject_reason": "duplicate_same_tile",
                    "duplicate_of": {
                        "class_name": duplicate_of["class_name"],
                        "confidence": duplicate_of["confidence"],
                        "xyxy": duplicate_of["xyxy"],
                    },
                    "duplicate_iou": round(iou(pred, duplicate_of), 4),
                }
            )
            continue
        selected.append(pred)

    selected_with_area = [assign_area(pred, config) for pred in selected]
    rejected_with_area = [assign_area(pred, config) for pred in rejected]
    return sort_for_review(selected_with_area), sort_for_review(rejected_with_area)
