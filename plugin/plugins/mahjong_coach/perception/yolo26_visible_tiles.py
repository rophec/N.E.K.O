from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .fast_hand_path import FastHandResult
from .meld_state import MeldStateResult
from .river_state import RiverStateResult
from .table_surface import detect_table_surface


DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models" / "yolo26_mahjong"
DEFAULT_CONFIDENCE = 0.25

# River ownership is inferred from the current warped-frame detections instead
# of fixed screen boxes. The table center can drift materially when the visible
# table quad is clipped, so hard-coded polygons are not a stable boundary.
_RIVER_OWNER_ORDER = ("self", "right_opponent", "top_opponent", "left_opponent")
_RIVER_INTERIOR_X = (0.18, 0.82)
_RIVER_INTERIOR_Y = (0.16, 0.80)
_RIVER_CENTER_X_LIMITS = (0.34, 0.66)
_RIVER_CENTER_Y_LIMITS = (0.28, 0.66)


@dataclass(frozen=True)
class YoloTileDetection:
    tile: str
    confidence: float
    bbox: list[float]
    obb: list[list[float]] = field(default_factory=list)
    area_kind: str = "unknown"
    owner: str = ""
    source: str = "yolo26_lightweight"
    coordinate_space: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Yolo26TableStateResult:
    ok: bool = False
    hand_tiles: list[str] = field(default_factory=list)
    melds: list[dict[str, Any]] = field(default_factory=list)
    meld_tiles: list[str] = field(default_factory=list)
    discard_piles: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    visible_tiles: list[str] = field(default_factory=list)
    riichi_players: list[str] = field(default_factory=list)
    confidence: float = 0.0
    reason: str = ""
    elapsed_ms: float = 0.0
    raw_detections: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    analysis_hints: dict[str, Any] = field(default_factory=dict)
    original_inference_ok: bool | None = None
    river_inference_ok: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["elapsed_ms"] = round(float(self.elapsed_ms), 1)
        payload["confidence"] = round(float(self.confidence), 4)
        return payload

    def to_hand_result(self, *, min_hand_tiles: int = 12) -> FastHandResult:
        original_ok = self.ok if self.original_inference_ok is None else self.original_inference_ok
        hints = {
            **self.analysis_hints,
            "tile_recognition_mode": "yolo26",
            "yolo26_reason": self.reason,
            "yolo26_fallback_needed": not original_ok,
        }
        if not original_ok:
            reason = str(self.analysis_hints.get("yolo26_original_reason") or self.reason or "yolo26_unavailable")
            return FastHandResult(reason=reason, elapsed_ms=self.elapsed_ms, analysis_hints=hints)
        # Each exposed meld removes three tiles from the concealed-hand display.
        # 每组副露会让屏幕上的暗手减少三张，因此稳定门槛也要同步下降。
        required_hand_tiles = max(1, int(min_hand_tiles or 1) - 3 * len(self.melds))
        ok = len(self.hand_tiles) >= required_hand_tiles
        return FastHandResult(
            ok=ok,
            hand_tiles=list(self.hand_tiles),
            confidence=self.confidence,
            reason="recognized_yolo26_hand" if ok else "unstable_yolo26_hand_count",
            elapsed_ms=self.elapsed_ms,
            raw_detections=[item for item in self.raw_detections if item.get("area_kind") == "hand"],
            analysis_hints=hints,
        )

    def to_meld_result(self) -> MeldStateResult:
        hints = {
            **self.analysis_hints,
            "tile_recognition_mode": "yolo26",
            "yolo26_reason": self.reason,
            "tile_identity_reliable": bool(self.ok),
        }
        original_ok = self.ok if self.original_inference_ok is None else self.original_inference_ok
        if not original_ok:
            reason = str(self.analysis_hints.get("yolo26_original_reason") or self.reason or "yolo26_unavailable")
            return MeldStateResult(reason=reason, elapsed_ms=self.elapsed_ms, analysis_hints=hints)
        open_meld_count = len(self.melds)
        return MeldStateResult(
            ok=bool(open_meld_count),
            open_meld_count=open_meld_count,
            melds=[dict(item) for item in self.melds],
            tiles=list(self.meld_tiles),
            confidence=self.confidence,
            reason="recognized_yolo26_self_melds" if open_meld_count else "no_self_melds",
            elapsed_ms=self.elapsed_ms,
            raw_detections=[item for item in self.raw_detections if item.get("area_kind") == "self_meld"],
            analysis_hints=hints,
        )

    def to_river_result(self) -> RiverStateResult:
        hints = {
            **self.analysis_hints,
            "tile_recognition_mode": "yolo26",
            "yolo26_reason": self.reason,
        }
        river_ok = self.ok if self.river_inference_ok is None else self.river_inference_ok
        if not river_ok:
            reason = str(self.analysis_hints.get("yolo26_river_reason") or self.reason or "yolo26_river_unavailable")
            return RiverStateResult(reason=reason, elapsed_ms=self.elapsed_ms, analysis_hints=hints)
        return RiverStateResult(
            ok=True,
            discard_piles={player: [dict(item) for item in items] for player, items in self.discard_piles.items()},
            visible_tiles=list(self.visible_tiles),
            confidence=self.confidence,
            reason="recognized_yolo26_discards" if self.visible_tiles else "no_visible_discards",
            elapsed_ms=self.elapsed_ms,
            raw_detections=[item for item in self.raw_detections if item.get("area_kind") == "river"],
            analysis_hints=hints,
        )


def detect_yolo26_table_state_path(
    image_path: Path,
    *,
    model_dir: Path | None = None,
    min_confidence: float = DEFAULT_CONFIDENCE,
    diagnostics_dir: Path | None = None,
) -> Yolo26TableStateResult:
    started = time.perf_counter()
    if not image_path.exists():
        return Yolo26TableStateResult(reason="image_missing")
    selected_model_dir = model_dir or DEFAULT_MODEL_DIR
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
        table_surface = detect_table_surface(
            image,
            diagnostics_dir=diagnostics_dir,
            diagnostics_stem=image_path.stem,
        )
        backend = load_yolo26_backend(selected_model_dir)
        if not backend.available:
            return Yolo26TableStateResult(
                reason=backend.reason,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                diagnostics=dict(table_surface.diagnostics),
                analysis_hints={
                    **_base_hints(selected_model_dir, backend.reason, backend.runtime),
                    **table_surface.to_hints(),
                },
            )

        original_detections: list[YoloTileDetection] = []
        river_detections: list[YoloTileDetection] = []
        original_ok = False
        river_ok = False
        original_reason = ""
        river_reason = ""

        # The near-facing hand and self melds must stay in screenshot space.
        # Perspective warping stretches or clips this bottom row.
        try:
            original_detections = backend.detect(image, min_confidence=min_confidence)
            original_ok = True
        except Exception as exc:
            original_reason = f"yolo26_original_inference_failed:{type(exc).__name__}"

        # Rivers use only the normalized table image. A failed table warp does
        # not invalidate a successful original-frame hand result.
        warped_image = table_surface.warped_image if table_surface.ok else None
        if warped_image is None:
            river_reason = table_surface.reason or "table_surface_unavailable"
        else:
            try:
                river_detections = backend.detect(warped_image, min_confidence=min_confidence)
                river_ok = True
            except Exception as exc:
                river_reason = f"yolo26_river_inference_failed:{type(exc).__name__}"

        if not original_ok and not river_ok:
            reason = original_reason or river_reason or "yolo26_inference_failed"
            return Yolo26TableStateResult(
                reason=reason,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                diagnostics=dict(table_surface.diagnostics),
                analysis_hints={
                    **_base_hints(selected_model_dir, reason, backend.runtime),
                    **table_surface.to_hints(),
                    "yolo26_input_spaces": {
                        "hand_meld": "original_frame",
                        "river": "warped_table",
                    },
                    "yolo26_original_reason": original_reason,
                    "yolo26_river_reason": river_reason,
                },
                original_inference_ok=False,
                river_inference_ok=False,
            )
        grouped = postprocess_yolo26_detections(
            original_detections,
            image_size=image.size,
            river_detections=river_detections if river_ok else [],
            river_image_size=warped_image.size if warped_image is not None else (800, 800),
            min_confidence=min_confidence,
        )
        diagnostics: dict[str, Any] = {}
        if original_ok:
            diagnostics.update(
                _write_diagnostics(
                    image,
                    grouped["original_detections"],
                    image_path=image_path,
                    diagnostics_dir=diagnostics_dir,
                    suffix="original",
                )
            )
        if river_ok and warped_image is not None:
            diagnostics.update(
                _write_diagnostics(
                    warped_image,
                    grouped["river_detections"],
                    image_path=image_path,
                    diagnostics_dir=diagnostics_dir,
                    suffix="warped-rivers",
                )
            )

    return Yolo26TableStateResult(
        ok=original_ok or river_ok,
        hand_tiles=grouped["hand_tiles"],
        melds=grouped["melds"],
        meld_tiles=grouped["meld_tiles"],
        discard_piles=grouped["discard_piles"],
        visible_tiles=grouped["visible_tiles"],
        riichi_players=grouped["riichi_players"],
        confidence=grouped["confidence"],
        reason="recognized_yolo26_visible_tiles" if original_ok and river_ok else "recognized_yolo26_partial",
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
        raw_detections=[item.to_dict() for item in grouped["detections"]],
        diagnostics={**table_surface.diagnostics, **diagnostics},
        analysis_hints={
            **_base_hints(selected_model_dir, "", backend.runtime),
            **table_surface.to_hints(),
            "yolo26_input_spaces": {
                "hand_meld": "original_frame",
                "river": "warped_table",
            },
            "yolo26_original_inference_ok": original_ok,
            "yolo26_river_inference_ok": river_ok,
            "yolo26_original_reason": original_reason,
            "yolo26_river_reason": river_reason,
            "yolo26_river_center": grouped["river_center"],
            "yolo26_detection_count": len(grouped["detections"]),
            "yolo26_hand_count": len(grouped["hand_tiles"]),
            "yolo26_self_meld_count": len(grouped["melds"]),
            "yolo26_river_tile_count": len(grouped["visible_tiles"]),
            "yolo26_opponent_meld_count": grouped["opponent_meld_count"],
            "yolo26_excluded_visible_count": grouped["excluded_visible_count"],
            "yolo26_riichi_players": list(grouped["riichi_players"]),
        },
        original_inference_ok=original_ok,
        river_inference_ok=river_ok,
    )


def postprocess_yolo26_detections(
    detections: list[YoloTileDetection],
    *,
    image_size: tuple[int, int],
    river_detections: list[YoloTileDetection] | None = None,
    river_image_size: tuple[int, int] = (800, 800),
    min_confidence: float = DEFAULT_CONFIDENCE,
) -> dict[str, Any]:
    original_accepted = _dedupe_detections(
        [item for item in detections if item.tile and item.tile != "empty" and item.confidence >= min_confidence]
    )
    width, height = image_size
    original_items = [
        _with_detection_area(item, "original_other", "", coordinate_space="original_frame")
        for item in original_accepted
    ]
    provisional_hand = _select_original_bottom_row(
        original_items,
        width=max(1, int(width)),
        height=max(1, int(height)),
    )
    provisional_hand = sorted(
        provisional_hand,
        key=lambda item: (_center(item)[0], _center(item)[1]),
    )
    hand, split_meld = _split_bottom_hand_and_meld(provisional_hand, image_width=max(1, int(width)))
    hand = [_with_detection_area(item, "hand", "self", coordinate_space="original_frame") for item in hand]
    self_meld = sorted(split_meld, key=lambda item: (_center(item)[0], _center(item)[1]))
    selected_original_ids = {id(item) for item in provisional_hand}
    enriched_original = [item for item in original_items if id(item) not in selected_original_ids] + hand + self_meld

    warped_accepted = _dedupe_detections(
        [
            item
            for item in river_detections or []
            if item.tile and item.tile != "empty" and item.confidence >= min_confidence
        ]
    )
    river_width, river_height = river_image_size
    enriched_river, river_center = _assign_river_detection_areas(
        warped_accepted,
        width=max(1, int(river_width)),
        height=max(1, int(river_height)),
    )
    river = sorted(
        [item for item in enriched_river if item.area_kind == "river"],
        key=lambda item: _river_sort_key(item, center=river_center),
    )
    enriched = enriched_original + enriched_river

    discard_piles: dict[str, list[dict[str, Any]]] = {}
    for item in river:
        owner = item.owner or "unknown"
        pile = discard_piles.setdefault(owner, [])
        pile.append(
            {
                "tile": item.tile,
                "player": owner,
                "turn_index": len(pile) + 1,
                "bbox": item.bbox,
                "quad": item.obb,
                "confidence": item.confidence,
                "source": item.source,
            }
        )

    melds = _group_self_melds(self_meld)
    riichi_players = _detect_riichi_declarations(discard_piles)
    confidences = [item.confidence for item in enriched]
    opponent_meld_count = sum(
        item.area_kind == "excluded_table_tile" and item.owner not in {"", "self"}
        for item in enriched_river
    )
    excluded_visible_count = sum(item.area_kind != "river" for item in enriched_river)
    return {
        "detections": enriched,
        "original_detections": enriched_original,
        "river_detections": enriched_river,
        "river_center": [round(river_center[0], 4), round(river_center[1], 4)],
        "hand_tiles": [item.tile for item in hand],
        "melds": melds,
        "meld_tiles": [item.tile for item in self_meld],
        "discard_piles": discard_piles,
        "visible_tiles": [item.tile for item in river],
        "riichi_players": riichi_players,
        "opponent_meld_count": opponent_meld_count,
        "excluded_visible_count": excluded_visible_count,
        "confidence": round(sum(confidences) / max(1, len(confidences)), 4) if confidences else 0.0,
    }


@dataclass(frozen=True)
class _Backend:
    available: bool
    reason: str
    runtime: str
    labels: list[str] = field(default_factory=list)

    def detect(self, image: Image.Image, *, min_confidence: float) -> list[YoloTileDetection]:
        return []


def load_yolo26_backend(model_dir: Path) -> _Backend:
    metadata_path = model_dir / "metadata.json"
    labels_path = model_dir / "labels.json"
    labels = _load_labels(labels_path)
    if not model_dir.exists():
        return _Backend(False, "yolo26_model_dir_missing", "none", labels)
    metadata = _load_metadata(metadata_path)
    runtime = str(metadata.get("runtime") or metadata.get("format") or "onnxruntime").lower()
    model_name = str(metadata.get("model_file") or "model.onnx")
    model_path = model_dir / model_name
    if not labels:
        return _Backend(False, "yolo26_labels_missing", runtime, labels)
    if not model_path.exists():
        return _Backend(False, "yolo26_model_missing", runtime, labels)
    if runtime not in {"onnx", "onnxruntime"}:
        return _Backend(False, f"yolo26_runtime_unimplemented:{runtime}", runtime, labels)
    raw_size = metadata.get("input_size") or [800, 800]
    if not isinstance(raw_size, list) or len(raw_size) != 2:
        raw_size = [800, 800]
    input_size = (max(1, int(raw_size[0])), max(1, int(raw_size[1])))
    return _OnnxYolo26Backend(model_path=model_path, labels=labels, input_size=input_size)


@dataclass(frozen=True)
class _OnnxYolo26Backend(_Backend):
    model_path: Path = Path()
    input_size: tuple[int, int] = (800, 800)

    def __init__(self, *, model_path: Path, labels: list[str], input_size: tuple[int, int]) -> None:
        object.__setattr__(self, "available", True)
        object.__setattr__(self, "reason", "")
        object.__setattr__(self, "runtime", "onnxruntime")
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "model_path", model_path)
        object.__setattr__(self, "input_size", input_size)

    def detect(self, image: Image.Image, *, min_confidence: float) -> list[YoloTileDetection]:
        # 中文：这里执行 ONNX 推理，并解码 YOLO26 的端到端检测输出。
        # English: This runs ONNX inference and decodes YOLO26 end-to-end detections.
        # 中文：插件仅加载导出的轻量模型，不导入训练框架。
        # English: The plugin loads only the lightweight export and never imports the training framework.
        session = _load_onnx_session(str(self.model_path.resolve()))
        tensor, scale, pad_x, pad_y = _prepare_onnx_input(image, input_size=self.input_size)
        outputs = session.run(None, {session.get_inputs()[0].name: tensor})
        if not outputs:
            return []
        return _decode_end2end_output(
            outputs[0],
            labels=self.labels,
            image_size=image.size,
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
            min_confidence=min_confidence,
        )


@lru_cache(maxsize=2)
def _load_onnx_session(model_path: str) -> Any:
    import onnxruntime as ort  # type: ignore[import-not-found]

    available = set(ort.get_available_providers())
    providers = [
        provider
        for provider in ("CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider")
        if provider in available
    ]
    if not providers:
        providers = ["CPUExecutionProvider"]
    try:
        return ort.InferenceSession(model_path, providers=providers)
    except Exception:
        if providers == ["CPUExecutionProvider"]:
            raise
        return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


def _prepare_onnx_input(
    image: Image.Image,
    *,
    input_size: tuple[int, int],
) -> tuple[Any, float, float, float]:
    import numpy as np  # type: ignore[import-not-found]

    input_width, input_height = input_size
    source = image.convert("RGB")
    source_width, source_height = source.size
    scale = min(input_width / max(1, source_width), input_height / max(1, source_height))
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    resized = source.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    pad_x = float((input_width - resized_width) // 2)
    pad_y = float((input_height - resized_height) // 2)
    letterboxed = Image.new("RGB", (input_width, input_height), (114, 114, 114))
    letterboxed.paste(resized, (int(pad_x), int(pad_y)))
    tensor = np.asarray(letterboxed, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    return np.ascontiguousarray(tensor), float(scale), pad_x, pad_y


def _decode_end2end_output(
    output: Any,
    *,
    labels: list[str],
    image_size: tuple[int, int],
    scale: float,
    pad_x: float,
    pad_y: float,
    min_confidence: float,
) -> list[YoloTileDetection]:
    import numpy as np  # type: ignore[import-not-found]

    rows = np.asarray(output)
    if rows.ndim == 3:
        rows = rows[0]
    if rows.ndim != 2:
        raise ValueError(f"unexpected_yolo26_output_rank:{rows.ndim}")
    if rows.shape[-1] != 6 and rows.shape[0] == 6:
        rows = rows.transpose(1, 0)
    if rows.shape[-1] < 6:
        raise ValueError(f"unexpected_yolo26_output_shape:{tuple(rows.shape)}")

    image_width, image_height = image_size
    safe_scale = max(float(scale), 1e-6)
    detections: list[YoloTileDetection] = []
    for row in rows:
        confidence = float(row[4])
        if confidence < min_confidence:
            continue
        class_id = int(round(float(row[5])))
        if class_id < 0 or class_id >= len(labels):
            continue
        left = max(0.0, min(float(image_width), (float(row[0]) - pad_x) / safe_scale))
        top = max(0.0, min(float(image_height), (float(row[1]) - pad_y) / safe_scale))
        right = max(0.0, min(float(image_width), (float(row[2]) - pad_x) / safe_scale))
        bottom = max(0.0, min(float(image_height), (float(row[3]) - pad_y) / safe_scale))
        if right <= left or bottom <= top:
            continue
        detections.append(
            YoloTileDetection(
                tile=labels[class_id],
                confidence=confidence,
                bbox=[left, top, right, bottom],
                source="yolo26_onnxruntime",
            )
        )
    return detections


def _select_original_bottom_row(
    items: list[YoloTileDetection],
    *,
    width: int,
    height: int,
) -> list[YoloTileDetection]:
    """Select the detected near-facing bottom row without a fixed hand box."""
    candidates = [item for item in items if _center(item)[1] >= height * 0.58]
    if not candidates:
        return []
    heights = [max(1.0, _ordered_bbox(item.bbox)[3] - _ordered_bbox(item.bbox)[1]) for item in candidates]
    median_height = sorted(heights)[len(heights) // 2]
    baseline = max(_ordered_bbox(item.bbox)[3] for item in candidates)
    baseline_tolerance = max(height * 0.055, median_height * 0.85)
    selected = [
        item
        for item in candidates
        if baseline - _ordered_bbox(item.bbox)[3] <= baseline_tolerance
        and _center(item)[1] >= height * 0.64
    ]
    if selected:
        hand_right = max(_center(item)[0] for item in selected)
        side_candidates = [
            item
            for item in candidates
            if item not in selected and _center(item)[0] - hand_right >= width * 0.055
        ]
        if len(side_candidates) >= 3:
            side_bottoms = sorted(_ordered_bbox(item.bbox)[3] for item in side_candidates)
            side_baseline = side_bottoms[len(side_bottoms) // 2]
            selected.extend(
                item
                for item in side_candidates
                if abs(_ordered_bbox(item.bbox)[3] - side_baseline) <= baseline_tolerance
            )
    return [
        item
        for item in selected
        if (_ordered_bbox(item.bbox)[2] - _ordered_bbox(item.bbox)[0]) <= width * 0.12
    ]


def _assign_river_detection_areas(
    items: list[YoloTileDetection],
    *,
    width: int,
    height: int,
) -> tuple[list[YoloTileDetection], tuple[float, float]]:
    normalized = [
        _with_detection_area(item, "excluded_table_tile", "", coordinate_space="warped_table")
        for item in items
    ]
    center = _estimate_river_center(normalized, width=width, height=height)
    if not normalized:
        return [], center

    tile_scales = []
    for item in normalized:
        left, top, right, bottom = _ordered_bbox(item.bbox)
        tile_scales.append(max((right - left) / width, (bottom - top) / height))
    median_tile_scale = sorted(tile_scales)[len(tile_scales) // 2]
    minimum_radius = max(0.045, median_tile_scale * 1.05)
    maximum_radius = 0.40

    provisional: dict[str, list[tuple[float, YoloTileDetection]]] = {
        owner: [] for owner in _RIVER_OWNER_ORDER
    }
    for item in normalized:
        nx, ny = _normalized_center(item, width=width, height=height)
        if not (_RIVER_INTERIOR_X[0] <= nx <= _RIVER_INTERIOR_X[1]):
            continue
        if not (_RIVER_INTERIOR_Y[0] <= ny <= _RIVER_INTERIOR_Y[1]):
            continue
        dx, dy = nx - center[0], ny - center[1]
        radius = (dx * dx + dy * dy) ** 0.5
        if radius < minimum_radius or radius > maximum_radius:
            continue
        owner = _river_owner_from_vector(dx, dy)
        provisional[owner].append((radius, item))

    accepted_ids: dict[int, str] = {}
    for owner, candidates in provisional.items():
        if not candidates:
            continue
        nearest_radius = min(radius for radius, _item in candidates)
        # Keep the connected inner river band and reject farther side melds or
        # animation tiles that happen to share the same angular sector.
        outer_limit = min(maximum_radius, nearest_radius + max(0.13, median_tile_scale * 3.2))
        for radius, item in candidates:
            if radius <= outer_limit:
                accepted_ids[id(item)] = owner

    enriched: list[YoloTileDetection] = []
    for item in normalized:
        owner = accepted_ids.get(id(item), "")
        if owner:
            enriched.append(_with_detection_area(item, "river", owner, coordinate_space="warped_table"))
            continue
        nx, ny = _normalized_center(item, width=width, height=height)
        enriched.append(
            _with_detection_area(
                item,
                "excluded_table_tile",
                _river_owner_from_vector(nx - center[0], ny - center[1]),
                coordinate_space="warped_table",
            )
        )
    return enriched, center


def _estimate_river_center(
    items: list[YoloTileDetection],
    *,
    width: int,
    height: int,
) -> tuple[float, float]:
    centers = [
        _normalized_center(item, width=width, height=height)
        for item in items
    ]
    interior = [
        point
        for point in centers
        if _RIVER_INTERIOR_X[0] <= point[0] <= _RIVER_INTERIOR_X[1]
        and _RIVER_INTERIOR_Y[0] <= point[1] <= _RIVER_INTERIOR_Y[1]
    ]
    if len(interior) < 4:
        return 0.5, 0.5
    xs = [point[0] for point in interior]
    ys = [point[1] for point in interior]
    if max(xs) - min(xs) < 0.16 or max(ys) - min(ys) < 0.16:
        return 0.5, 0.5
    center_x = _clamp_float((min(xs) + max(xs)) / 2.0, *_RIVER_CENTER_X_LIMITS)
    center_y = _clamp_float((min(ys) + max(ys)) / 2.0, *_RIVER_CENTER_Y_LIMITS)
    return center_x, center_y


def _river_owner_from_vector(dx: float, dy: float) -> str:
    if abs(dx) >= abs(dy):
        return "right_opponent" if dx >= 0.0 else "left_opponent"
    return "self" if dy >= 0.0 else "top_opponent"


def _normalized_center(item: YoloTileDetection, *, width: int, height: int) -> tuple[float, float]:
    cx, cy = _center(item)
    return cx / max(1, width), cy / max(1, height)


def _river_sort_key(item: YoloTileDetection, *, center: tuple[float, float]) -> tuple[Any, ...]:
    cx, cy = _center(item)
    owner_index = _RIVER_OWNER_ORDER.index(item.owner) if item.owner in _RIVER_OWNER_ORDER else len(_RIVER_OWNER_ORDER)
    if item.owner == "self":
        position = (cy, cx)
    elif item.owner == "top_opponent":
        position = (-cy, -cx)
    elif item.owner == "left_opponent":
        position = (-cx, cy)
    else:
        position = (cx, -cy)
    return owner_index, *position


def _clamp_float(value: float, minimum: float, maximum: float) -> float:
    return max(float(minimum), min(float(maximum), float(value)))


def _detect_riichi_declarations(discard_piles: dict[str, list[dict[str, Any]]]) -> list[str]:
    detected: list[str] = []
    for owner, pile in discard_piles.items():
        if owner == "self" or len(pile) < 4:
            continue
        ratios: list[float] = []
        for item in pile:
            left, top, right, bottom = _ordered_bbox(list(item.get("bbox") or [0, 0, 0, 0]))
            ratios.append((right - left) / max(1e-6, bottom - top))
        if owner in {"left_opponent", "right_opponent"}:
            normal_count = sum(ratio >= 1.05 for ratio in ratios)
            has_declaration = any(ratio <= 0.90 for ratio in ratios)
        else:
            normal_count = sum(ratio <= 0.95 for ratio in ratios)
            has_declaration = any(ratio >= 1.10 for ratio in ratios)
        if normal_count >= max(3, len(ratios) - 2) and has_declaration:
            detected.append(owner)
    return sorted(detected)


def _split_bottom_hand_and_meld(
    items: list[YoloTileDetection],
    *,
    image_width: int,
) -> tuple[list[YoloTileDetection], list[YoloTileDetection]]:
    if len(items) < 5:
        return items, []
    centers = [_center(item)[0] for item in items]
    gaps = [centers[index + 1] - centers[index] for index in range(len(centers) - 1)]
    if not gaps:
        return items, []
    median_gap = sorted(gaps)[len(gaps) // 2]
    threshold = max(image_width * 0.065, median_gap * 1.8)
    candidates: list[tuple[float, int]] = []
    for index, gap in enumerate(gaps):
        right_count = len(items) - index - 1
        if _plausible_meld_tile_count(right_count) and gap >= threshold:
            candidates.append((gap, index))
    if not candidates:
        return items, []
    _gap, split_index = max(candidates)
    hand = [_with_detection_area(item, "hand", "self", coordinate_space="original_frame") for item in items[: split_index + 1]]
    meld = [_with_detection_area(item, "self_meld", "self", coordinate_space="original_frame") for item in items[split_index + 1 :]]
    return hand, meld


def _plausible_meld_tile_count(count: int) -> bool:
    if count < 3 or count > 16:
        return False
    return any(3 * meld_count <= count <= 4 * meld_count for meld_count in range(1, 5))


def _with_detection_area(
    item: YoloTileDetection,
    area_kind: str,
    owner: str,
    *,
    coordinate_space: str | None = None,
) -> YoloTileDetection:
    return YoloTileDetection(
        tile=item.tile,
        confidence=item.confidence,
        bbox=list(item.bbox),
        obb=[list(point) for point in item.obb],
        area_kind=area_kind,
        owner=owner,
        source=item.source,
        coordinate_space=coordinate_space if coordinate_space is not None else item.coordinate_space,
    )


def _group_self_melds(items: list[YoloTileDetection]) -> list[dict[str, Any]]:
    ordered = sorted(items, key=lambda item: _center(item)[0])
    groups = _split_self_meld_groups(ordered)
    melds: list[dict[str, Any]] = []
    for group in groups:
        if not group:
            continue
        melds.append(
            {
                "meld_index": len(melds) + 1,
                "tiles": [item.tile for item in group],
                "confidence": round(sum(item.confidence for item in group) / max(1, len(group)), 4),
                "source": "yolo26_visible_tiles",
                "detections": [item.to_dict() for item in group],
            }
        )
    return melds


def _split_self_meld_groups(items: list[YoloTileDetection]) -> list[list[YoloTileDetection]]:
    if not items:
        return []
    if len(items) <= 4:
        return [items]
    centers = [_center(item)[0] for item in items]
    widths = [_ordered_bbox(item.bbox)[2] - _ordered_bbox(item.bbox)[0] for item in items]
    median_width = sorted(widths)[len(widths) // 2]
    gaps = [centers[index + 1] - centers[index] for index in range(len(centers) - 1)]
    median_gap = sorted(gaps)[len(gaps) // 2]
    split_after = {
        index
        for index, gap in enumerate(gaps)
        if gap >= max(median_gap * 1.55, median_width * 1.15)
    }
    if split_after:
        groups: list[list[YoloTileDetection]] = []
        start = 0
        for index in sorted(split_after):
            group = items[start : index + 1]
            if len(group) >= 3:
                groups.append(group)
                start = index + 1
        tail = items[start:]
        if len(tail) >= 3:
            groups.append(tail)
        if groups and sum(len(group) for group in groups) == len(items):
            return groups

    group_count = max(1, min(4, int(round(len(items) / 3.5))))
    base_size, extra = divmod(len(items), group_count)
    groups = []
    start = 0
    for index in range(group_count):
        size = base_size + (1 if index < extra else 0)
        groups.append(items[start : start + size])
        start += size
    return groups


def _dedupe_detections(items: list[YoloTileDetection]) -> list[YoloTileDetection]:
    selected: list[YoloTileDetection] = []
    for item in sorted(items, key=lambda value: value.confidence, reverse=True):
        if all(_iou(item.bbox, other.bbox) < 0.5 for other in selected):
            selected.append(item)
    return selected


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = _ordered_bbox(a)
    bx1, by1, bx2, by2 = _ordered_bbox(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return intersection / max(1e-6, area_a + area_b - intersection)


def _ordered_bbox(box: list[float]) -> tuple[float, float, float, float]:
    left, top, right, bottom = (float(value) for value in box[:4])
    return min(left, right), min(top, bottom), max(left, right), max(top, bottom)


def _center(item: YoloTileDetection) -> tuple[float, float]:
    left, top, right, bottom = _ordered_bbox(item.bbox)
    return (left + right) / 2.0, (top + bottom) / 2.0


def _load_labels(path: Path) -> list[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(payload, list):
        return [str(item) for item in payload if str(item).strip()]
    if isinstance(payload, dict):
        values = payload.get("labels") or payload.get("names") or []
        if isinstance(values, dict):
            return [str(values[key]) for key in sorted(values, key=lambda value: int(value) if str(value).isdigit() else str(value))]
        if isinstance(values, list):
            return [str(item) for item in values if str(item).strip()]
    return []


def _load_metadata(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_diagnostics(
    image: Image.Image,
    detections: list[YoloTileDetection],
    *,
    image_path: Path,
    diagnostics_dir: Path | None,
    suffix: str,
) -> dict[str, Any]:
    if diagnostics_dir is None:
        return {}
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for item in detections:
        left, top, right, bottom = _ordered_bbox(item.bbox)
        draw.rectangle((left, top, right, bottom), outline="red", width=2)
        draw.text((left, max(0, top - 14)), f"{item.tile} {item.area_kind}", fill="red")
    target = diagnostics_dir / f"{image_path.stem}-yolo26-{suffix}-overlay.jpg"
    overlay.save(target)
    key = f"yolo26_{suffix.replace('-', '_')}_overlay_path"
    payload = {key: str(target)}
    if suffix == "warped-rivers":
        payload["yolo26_overlay_path"] = str(target)
    return payload


def render_yolo26_region_diagnostic_image(
    image: Image.Image,
    *,
    raw_detections: list[dict[str, Any]] | None = None,
) -> Image.Image:
    """Render adaptive river ownership on the already warped table image."""
    preview = image.convert("RGB")
    width, height = preview.size
    draw = ImageDraw.Draw(preview, "RGBA")
    region_colors: dict[str, tuple[int, int, int, int]] = {
        "top_opponent": (255, 205, 68, 230),
        "left_opponent": (73, 214, 126, 230),
        "right_opponent": (235, 93, 178, 230),
        "self": (57, 196, 226, 230),
    }

    # 中文：这些坐标必须画在透视变换后的标准牌桌上，不能叠到原始窗口截图上。
    # English: These normalized regions belong to warped-table space, never to the original window frame.
    warped_items: list[dict[str, Any]] = []
    center_items: list[YoloTileDetection] = []
    for item in raw_detections or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("coordinate_space") or "warped_table") != "warped_table":
            continue
        bbox = item.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        warped_items.append(item)
        center_items.append(
            YoloTileDetection(
                tile=str(item.get("tile") or "?"),
                confidence=float(item.get("confidence") or 0.0),
                bbox=[float(value) for value in bbox],
                coordinate_space="warped_table",
            )
        )

    center = _estimate_river_center(center_items, width=max(1, width), height=max(1, height))
    center_x, center_y = round(center[0] * width), round(center[1] * height)
    line_width = max(2, round(width / 320))
    draw.line((0, 0, center_x, center_y, width - 1, height - 1), fill=(225, 232, 236, 135), width=line_width)
    draw.line((width - 1, 0, center_x, center_y, 0, height - 1), fill=(225, 232, 236, 135), width=line_width)
    center_radius = max(5, round(width / 100))
    draw.ellipse(
        (center_x - center_radius, center_y - center_radius, center_x + center_radius, center_y + center_radius),
        outline=(245, 248, 250, 230),
        width=line_width,
    )
    label_offsets = {
        "top_opponent": (0.0, -0.17),
        "left_opponent": (-0.22, 0.0),
        "right_opponent": (0.08, 0.0),
        "self": (0.0, 0.16),
    }
    for owner, (offset_x, offset_y) in label_offsets.items():
        draw.text(
            (center_x + round(offset_x * width), center_y + round(offset_y * height)),
            owner,
            fill=region_colors[owner],
        )

    detection_colors = {
        "river": (98, 238, 160, 255),
        "excluded_table_tile": (210, 216, 222, 255),
    }
    for item in warped_items:
        bbox = item["bbox"]
        left, top, right, bottom = _ordered_bbox([float(value) for value in bbox])
        area_kind = str(item.get("area_kind") or "unknown")
        color = detection_colors.get(area_kind, (225, 225, 225, 255))
        draw.rectangle((left, top, right, bottom), outline=color, width=max(2, round(width / 320)))
        tile = str(item.get("tile") or "?")
        confidence = float(item.get("confidence") or 0.0)
        draw.text((left, max(28, top - 13)), f"{tile} {confidence:.2f}", fill=color)

    draw.rectangle((0, 0, width, 24), fill=(8, 12, 14, 205))
    draw.text((8, 6), "WARPED TABLE | adaptive four-river ownership only", fill=(242, 245, 244, 255))
    return preview


def _base_hints(model_dir: Path, reason: str, runtime: str) -> dict[str, Any]:
    return {
        "tile_recognition_mode": "yolo26",
        "yolo26_runtime": runtime,
        "yolo26_model_dir": str(model_dir),
        "yolo26_reason": reason,
        "yolo26_backend_available": not bool(reason),
    }
