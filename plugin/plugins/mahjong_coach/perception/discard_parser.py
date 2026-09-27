from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from .discard_layout import DiscardSlot, build_discard_layout
from .roi import collect_region_metrics
from .tile_classifier_dispatch import classify_discard_tiles_batch, onnx_discard_available
from .tile_templates import TileTemplateMatch


DEFAULT_MIN_DISCARD_CONFIDENCE = 0.90


@dataclass
class _DiscardCandidate:
    player: str
    slot: DiscardSlot
    detection: dict[str, Any]
    match: TileTemplateMatch


@dataclass
class DiscardParseResult:
    discard_piles: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    visible_tiles: list[str] = field(default_factory=list)
    raw_detections: list[dict[str, Any]] = field(default_factory=list)
    analysis_hints: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "discard_piles": self.discard_piles,
            "visible_tiles": self.visible_tiles,
            "raw_detections": self.raw_detections,
            "analysis_hints": self.analysis_hints,
        }


def parse_discards_from_image(
    image: Image.Image,
    template_payload: dict[str, Any] | None = None,
    *,
    layout: dict[str, list[DiscardSlot]] | None = None,
    min_confidence: float = DEFAULT_MIN_DISCARD_CONFIDENCE,
    include_empty_detections: bool = False,
    require_local_support: bool = False,
) -> DiscardParseResult:
    template_payload = template_payload or {}
    onnx_available = onnx_discard_available()
    if not onnx_available and not template_payload:
        return DiscardParseResult(
            analysis_hints={
                "discard_parser_available": False,
                "discard_parser_reason": "missing_onnx_model_and_templates",
                "discard_parser_source": "unavailable",
            }
        )

    layout = layout or build_discard_layout(*image.size)
    plans: list[tuple[str, DiscardSlot, dict[str, Any], Image.Image]] = []
    raw_detections: list[dict[str, Any]] = []
    occupied_turns_by_player: dict[str, set[int]] = {}
    occupied_count = 0

    for player, slots in layout.items():
        for slot in slots:
            metrics = _collect_discard_slot_metrics(image, slot)
            occupied = is_probably_occupied_discard_slot(metrics)
            detection = _base_detection(slot, slot_metrics=metrics, occupied=occupied)
            if not occupied:
                if include_empty_detections:
                    raw_detections.append(detection)
                continue
            occupied_count += 1
            occupied_turns_by_player.setdefault(player, set()).add(slot.turn_index)
            plans.append((player, slot, detection, crop_discard_slot(image, slot)))

    matches = classify_discard_tiles_batch(
        [crop for _, _, _, crop in plans],
        template_payload,
    )
    discard_piles: dict[str, list[dict[str, Any]]] = {}
    visible_tiles: list[str] = []
    confidences: list[float] = []
    accepted_candidates: list[_DiscardCandidate] = []

    for (player, slot, detection, _crop), match in zip(plans, matches, strict=True):
        if match is None:
            detection["accepted"] = False
            detection["rejection_reason"] = "classification_failed"
            raw_detections.append(detection)
            continue

        detection.update(
            {
                "candidate_tile": match.tile,
                "confidence": match.confidence,
                "template_distance": match.distance,
                "runner_up_tile": match.runner_up_tile,
                "runner_up_distance": match.runner_up_distance,
            }
        )
        rejection_reason = _discard_match_rejection_reason(match, min_confidence=min_confidence)
        if rejection_reason:
            detection["accepted"] = False
            detection["rejection_reason"] = rejection_reason
            raw_detections.append(detection)
            continue

        accepted_candidates.append(_DiscardCandidate(player=player, slot=slot, detection=detection, match=match))

    non_contiguous_candidates: list[_DiscardCandidate] = []
    if require_local_support:
        accepted_candidates, non_contiguous_candidates = _split_contiguous_discard_candidates(
            accepted_candidates,
            occupied_turns_by_player=occupied_turns_by_player,
        )
        for candidate in non_contiguous_candidates:
            candidate.detection["accepted"] = False
            candidate.detection["rejection_reason"] = "non_contiguous_discard_slot"
            raw_detections.append(candidate.detection)

    for candidate in accepted_candidates:
        player = candidate.player
        slot = candidate.slot
        match = candidate.match
        detection = candidate.detection
        detection["accepted"] = True
        raw_detections.append(detection)
        item = {
            "tile": match.tile,
            "player": player,
            "turn_index": slot.turn_index,
            "bbox": slot.bbox,
            "quad": [[x, y] for x, y in slot.quad],
            "confidence": match.confidence,
            "orientation": slot.orientation,
            "source": "onnx_discard_model" if onnx_available else "discard_template_profile",
            "slot_id": slot.slot_id,
        }
        discard_piles.setdefault(player, []).append(item)
        visible_tiles.append(match.tile)
        confidences.append(match.confidence)

    recognized_count = len(visible_tiles)
    confidence = round(sum(confidences) / max(1, len(confidences)), 4) if confidences else 0.0
    analysis_hints = {
        "discard_parser_available": True,
        "discard_parser_source": "onnx_discard_model" if onnx_available else "discard_template_profile",
        "discard_slot_count": sum(len(slots) for slots in layout.values()),
        "occupied_discard_slot_count": occupied_count,
        "recognized_discard_tile_count": recognized_count,
        "discard_analysis_confidence": confidence,
    }
    if require_local_support:
        analysis_hints["classified_discard_candidate_count"] = len(accepted_candidates) + len(non_contiguous_candidates)
        analysis_hints["rejected_non_contiguous_discard_count"] = len(non_contiguous_candidates)
    return DiscardParseResult(
        discard_piles=discard_piles,
        visible_tiles=visible_tiles,
        raw_detections=raw_detections,
        analysis_hints=analysis_hints,
    )


def crop_discard_slot(image: Image.Image, slot: DiscardSlot) -> Image.Image:
    return crop_discard_quad(
        image,
        slot.quad,
        output_size=(slot.box.width, slot.box.height),
        orientation=slot.orientation,
    )


def crop_discard_quad(
    image: Image.Image,
    quad: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]],
    *,
    output_size: tuple[int, int],
    orientation: str,
) -> Image.Image:
    width = max(1, int(output_size[0]))
    height = max(1, int(output_size[1]))
    data = tuple(float(value) for point in quad for value in point)
    transform_quad = getattr(getattr(Image, "Transform", Image), "QUAD")
    resampling = getattr(getattr(Image, "Resampling", Image), "BICUBIC")
    crop = image.transform((width, height), transform_quad, data, resample=resampling)
    return normalize_discard_crop(crop, orientation)


def normalize_discard_crop(crop: Image.Image, orientation: str) -> Image.Image:
    if orientation == "top":
        return crop.rotate(180, expand=True)
    if orientation == "left":
        return crop.rotate(270, expand=True)
    if orientation == "right":
        return crop.rotate(90, expand=True)
    return crop


def is_probably_occupied_discard_slot(slot_metrics: dict[str, Any]) -> bool:
    mean_luma = _float_metric(slot_metrics, "slot_mean_luma", "mean_luma")
    bright_ratio = _float_metric(slot_metrics, "slot_bright_ratio", "bright_ratio")
    dark_ratio = _float_metric(slot_metrics, "slot_dark_ratio", "dark_ratio")
    stddev = _float_metric(slot_metrics, "slot_stddev", "stddev")
    return mean_luma >= 88.0 and bright_ratio >= 0.12 and dark_ratio <= 0.62 and stddev >= 14.0


def _collect_discard_slot_metrics(image: Image.Image, slot: DiscardSlot) -> dict[str, Any]:
    metrics = collect_region_metrics(image, slot.box, sample_step=4)
    return {
        "slot_mean_luma": metrics["mean_luma"],
        "slot_bright_ratio": metrics["bright_ratio"],
        "slot_dark_ratio": metrics["dark_ratio"],
        "slot_colorful_ratio": metrics["colorful_ratio"],
        "slot_stddev": metrics["stddev"],
    }


def _base_detection(
    slot: DiscardSlot,
    *,
    slot_metrics: dict[str, Any],
    occupied: bool,
) -> dict[str, Any]:
    return {
        "slot_id": slot.slot_id,
        "group": "discard",
        "player": slot.player,
        "turn_index": slot.turn_index,
        "candidate_tile": "",
        "confidence": 0.0,
        "box": slot.box.to_dict(),
        "bbox": slot.bbox,
        "quad": [[x, y] for x, y in slot.quad],
        "orientation": slot.orientation,
        "occupied": occupied,
        "source": "discard_parser",
        **slot_metrics,
    }


def _discard_match_rejection_reason(match: TileTemplateMatch, *, min_confidence: float) -> str:
    if match.tile == "empty":
        return "empty_tile_class"
    if match.confidence < min_confidence:
        return "low_confidence"
    return ""


def _split_contiguous_discard_candidates(
    candidates: list[_DiscardCandidate],
    *,
    occupied_turns_by_player: dict[str, set[int]],
) -> tuple[list[_DiscardCandidate], list[_DiscardCandidate]]:
    accepted: list[_DiscardCandidate] = []
    rejected: list[_DiscardCandidate] = []
    for candidate in candidates:
        if _has_local_occupied_support(candidate.slot, occupied_turns_by_player.get(candidate.player, set())):
            accepted.append(candidate)
        else:
            rejected.append(candidate)
    accepted.sort(key=lambda item: (item.player, item.slot.turn_index))
    rejected.sort(key=lambda item: (item.player, item.slot.turn_index))
    return accepted, rejected


def _has_local_occupied_support(slot: DiscardSlot, occupied_turns: set[int]) -> bool:
    if slot.turn_index <= 1:
        return True
    # 中文：新 guarded 模式保留成片出现的牌河格子，过滤孤立的后段噪声格。
    # English: New guarded mode keeps clustered discard slots and drops isolated late-slot noise.
    return (slot.turn_index - 1) in occupied_turns or (slot.turn_index + 1) in occupied_turns


def _float_metric(metrics: dict[str, Any], primary: str, fallback: str) -> float:
    value = metrics.get(primary, metrics.get(fallback, 0.0))
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
