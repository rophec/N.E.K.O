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

# 中文：YOLO 在正视牌桌坐标中只把中央四块区域认作牌河，区域外的副露和动画牌不会污染弃牌历史。
# English: Only the four central polygons are rivers in warped-table space; outer melds and animation tiles stay excluded.
_RIVER_REGION_POLYGONS: dict[str, tuple[tuple[float, float], ...]] = {
    "top_opponent": ((0.30, 0.25), (0.70, 0.25), (0.64, 0.46), (0.36, 0.46)),
    "left_opponent": ((0.24, 0.30), (0.40, 0.34), (0.44, 0.68), (0.28, 0.72)),
    "right_opponent": ((0.60, 0.34), (0.76, 0.30), (0.72, 0.72), (0.56, 0.68)),
    "self": ((0.36, 0.54), (0.64, 0.54), (0.70, 0.74), (0.30, 0.74)),
}
_RIVER_REGION_ANCHORS: dict[str, tuple[float, float]] = {
    "top_opponent": (0.50, 0.35),
    "left_opponent": (0.35, 0.51),
    "right_opponent": (0.65, 0.51),
    "self": (0.50, 0.65),
}
_TABLE_SIDE_ANCHORS: dict[str, tuple[float, float]] = {
    "top_opponent": (0.50, 0.08),
    "left_opponent": (0.08, 0.50),
    "right_opponent": (0.92, 0.50),
    "self": (0.50, 0.92),
}


@dataclass(frozen=True)
class YoloTileDetection:
    tile: str
    confidence: float
    bbox: list[float]
    obb: list[list[float]] = field(default_factory=list)
    area_kind: str = "unknown"
    owner: str = ""
    source: str = "yolo26_lightweight"

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

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["elapsed_ms"] = round(float(self.elapsed_ms), 1)
        payload["confidence"] = round(float(self.confidence), 4)
        return payload

    def to_hand_result(self, *, min_hand_tiles: int = 12) -> FastHandResult:
        hints = {
            **self.analysis_hints,
            "tile_recognition_mode": "yolo26",
            "yolo26_reason": self.reason,
            "yolo26_fallback_needed": not self.ok,
        }
        if not self.ok:
            return FastHandResult(reason=self.reason or "yolo26_unavailable", elapsed_ms=self.elapsed_ms, analysis_hints=hints)
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
        if not self.ok:
            return MeldStateResult(reason=self.reason or "yolo26_unavailable", elapsed_ms=self.elapsed_ms, analysis_hints=hints)
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
        if not self.ok:
            return RiverStateResult(reason=self.reason or "yolo26_unavailable", elapsed_ms=self.elapsed_ms, analysis_hints=hints)
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

        inference_image = table_surface.warped_image if table_surface.ok and table_surface.warped_image is not None else image
        input_space = "warped_table" if inference_image is not image else "original_frame"
        try:
            detections = backend.detect(inference_image, min_confidence=min_confidence)
        except Exception as exc:
            reason = f"yolo26_inference_failed:{type(exc).__name__}"
            return Yolo26TableStateResult(
                reason=reason,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                diagnostics=dict(table_surface.diagnostics),
                analysis_hints={
                    **_base_hints(selected_model_dir, reason, backend.runtime),
                    **table_surface.to_hints(),
                    "yolo26_input_space": input_space,
                },
            )
        if not detections:
            reason = "yolo26_empty_detections"
            return Yolo26TableStateResult(
                reason=reason,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                diagnostics=dict(table_surface.diagnostics),
                analysis_hints={
                    **_base_hints(selected_model_dir, reason, backend.runtime),
                    **table_surface.to_hints(),
                    "yolo26_input_space": input_space,
                },
            )
        grouped = postprocess_yolo26_detections(
            detections,
            image_size=inference_image.size,
            min_confidence=min_confidence,
        )
        diagnostics = _write_diagnostics(
            inference_image,
            grouped["detections"],
            image_path=image_path,
            diagnostics_dir=diagnostics_dir,
        )

    return Yolo26TableStateResult(
        ok=True,
        hand_tiles=grouped["hand_tiles"],
        melds=grouped["melds"],
        meld_tiles=grouped["meld_tiles"],
        discard_piles=grouped["discard_piles"],
        visible_tiles=grouped["visible_tiles"],
        riichi_players=grouped["riichi_players"],
        confidence=grouped["confidence"],
        reason="recognized_yolo26_visible_tiles",
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
        raw_detections=[item.to_dict() for item in grouped["detections"]],
        diagnostics={**table_surface.diagnostics, **diagnostics},
        analysis_hints={
            **_base_hints(selected_model_dir, "", backend.runtime),
            **table_surface.to_hints(),
            "yolo26_input_space": input_space,
            "yolo26_detection_count": len(grouped["detections"]),
            "yolo26_hand_count": len(grouped["hand_tiles"]),
            "yolo26_self_meld_count": len(grouped["melds"]),
            "yolo26_river_tile_count": len(grouped["visible_tiles"]),
            "yolo26_opponent_meld_count": grouped["opponent_meld_count"],
            "yolo26_excluded_visible_count": grouped["excluded_visible_count"],
            "yolo26_riichi_players": list(grouped["riichi_players"]),
        },
    )


def postprocess_yolo26_detections(
    detections: list[YoloTileDetection],
    *,
    image_size: tuple[int, int],
    min_confidence: float = DEFAULT_CONFIDENCE,
) -> dict[str, Any]:
    accepted = _dedupe_detections(
        [item for item in detections if item.tile and item.tile != "empty" and item.confidence >= min_confidence]
    )
    width, height = image_size
    enriched = [
        _assign_detection_area(item, width=max(1, int(width)), height=max(1, int(height)))
        for item in accepted
    ]
    provisional_hand = sorted(
        [item for item in enriched if item.area_kind == "hand"],
        key=lambda item: (_center(item)[0], _center(item)[1]),
    )
    hand, split_meld = _split_bottom_hand_and_meld(provisional_hand, image_width=max(1, int(width)))
    self_meld = sorted(
        [item for item in enriched if item.area_kind == "self_meld"] + split_meld,
        key=lambda item: (_center(item)[0], _center(item)[1]),
    )
    river = sorted(
        [item for item in enriched if item.area_kind == "river"],
        key=lambda item: (item.owner, _center(item)[1], _center(item)[0]),
    )
    enriched = [item for item in enriched if item.area_kind not in {"hand", "self_meld"}] + hand + self_meld

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
    opponent_meld_count = sum(item.area_kind == "opponent_meld" for item in enriched)
    excluded_visible_count = sum(
        item.area_kind not in {"hand", "self_meld", "river"}
        for item in enriched
    )
    return {
        "detections": enriched,
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


def _assign_detection_area(item: YoloTileDetection, *, width: int, height: int) -> YoloTileDetection:
    cx, cy = _center(item)
    nx = cx / width
    ny = cy / height
    if ny >= 0.78:
        area_kind = "hand"
        owner = "self"
    elif ny >= 0.62 and (nx <= 0.28 or nx >= 0.72):
        area_kind = "self_meld"
        owner = "self"
    elif river_owner := _river_owner(nx, ny):
        area_kind = "river"
        owner = river_owner
    else:
        owner = _nearest_owner(nx, ny, _TABLE_SIDE_ANCHORS)
        area_kind = "opponent_meld" if owner != "self" else "other_visible"
    return YoloTileDetection(
        tile=item.tile,
        confidence=item.confidence,
        bbox=list(item.bbox),
        obb=[list(point) for point in item.obb],
        area_kind=area_kind,
        owner=owner,
        source=item.source,
    )


def _river_owner(nx: float, ny: float) -> str | None:
    candidates = [
        owner
        for owner, polygon in _RIVER_REGION_POLYGONS.items()
        if _point_in_polygon(nx, ny, polygon)
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda owner: _squared_distance((nx, ny), _RIVER_REGION_ANCHORS[owner]),
    )


def _nearest_owner(
    nx: float,
    ny: float,
    anchors: dict[str, tuple[float, float]],
) -> str:
    return min(anchors, key=lambda owner: _squared_distance((nx, ny), anchors[owner]))


def _point_in_polygon(
    x: float,
    y: float,
    polygon: tuple[tuple[float, float], ...],
) -> bool:
    inside = False
    previous_x, previous_y = polygon[-1]
    for current_x, current_y in polygon:
        crosses = (current_y > y) != (previous_y > y)
        if crosses:
            boundary_x = (previous_x - current_x) * (y - current_y) / (previous_y - current_y) + current_x
            if x <= boundary_x:
                inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside


def _squared_distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return (left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2


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
        if right_count >= 3 and right_count % 3 == 0 and gap >= threshold:
            candidates.append((gap, index))
    if not candidates:
        return items, []
    _gap, split_index = max(candidates)
    hand = [_with_detection_area(item, "hand", "self") for item in items[: split_index + 1]]
    meld = [_with_detection_area(item, "self_meld", "self") for item in items[split_index + 1 :]]
    return hand, meld


def _with_detection_area(item: YoloTileDetection, area_kind: str, owner: str) -> YoloTileDetection:
    return YoloTileDetection(
        tile=item.tile,
        confidence=item.confidence,
        bbox=list(item.bbox),
        obb=[list(point) for point in item.obb],
        area_kind=area_kind,
        owner=owner,
        source=item.source,
    )


def _group_self_melds(items: list[YoloTileDetection]) -> list[dict[str, Any]]:
    melds: list[dict[str, Any]] = []
    for index in range(0, len(items), 3):
        group = items[index : index + 3]
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
    target = diagnostics_dir / f"{image_path.stem}-yolo26-overlay.jpg"
    overlay.save(target)
    return {"yolo26_overlay_path": str(target)}


def render_yolo26_region_diagnostic_image(
    image: Image.Image,
    *,
    raw_detections: list[dict[str, Any]] | None = None,
) -> Image.Image:
    """Render YOLO ownership regions on the already warped table image."""
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
    for owner, polygon in _RIVER_REGION_POLYGONS.items():
        points = [(round(x * width), round(y * height)) for x, y in polygon]
        color = region_colors[owner]
        draw.polygon(points, fill=(*color[:3], 28), outline=color)
        draw.line(points + [points[0]], fill=color, width=max(2, round(width / 260)))
        label_x, label_y = points[0]
        draw.text((label_x + 4, label_y + 4), owner, fill=color)

    hand_color = (255, 154, 61, 225)
    meld_color = (82, 141, 255, 225)
    draw.rectangle((0, round(height * 0.78), width - 1, height - 1), outline=hand_color, width=max(2, round(width / 260)))
    draw.text((8, round(height * 0.78) + 6), "hand", fill=hand_color)
    for left, right in ((0.0, 0.28), (0.72, 1.0)):
        box = (round(width * left), round(height * 0.62), round(width * right), round(height * 0.78))
        draw.rectangle(box, outline=meld_color, width=max(2, round(width / 260)))
        draw.text((box[0] + 6, box[1] + 6), "self_meld", fill=meld_color)

    detection_colors = {
        "river": (98, 238, 160, 255),
        "hand": (255, 190, 82, 255),
        "self_meld": (100, 164, 255, 255),
        "opponent_meld": (255, 103, 126, 255),
        "other_visible": (210, 216, 222, 255),
    }
    for item in raw_detections or []:
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        left, top, right, bottom = _ordered_bbox([float(value) for value in bbox])
        area_kind = str(item.get("area_kind") or "unknown")
        color = detection_colors.get(area_kind, (225, 225, 225, 255))
        draw.rectangle((left, top, right, bottom), outline=color, width=max(2, round(width / 320)))
        tile = str(item.get("tile") or "?")
        confidence = float(item.get("confidence") or 0.0)
        draw.text((left, max(28, top - 13)), f"{tile} {confidence:.2f}", fill=color)

    draw.rectangle((0, 0, width, 24), fill=(8, 12, 14, 205))
    draw.text((8, 6), "WARPED TABLE | four rivers + hand/meld zones", fill=(242, 245, 244, 255))
    return preview


def _base_hints(model_dir: Path, reason: str, runtime: str) -> dict[str, Any]:
    return {
        "tile_recognition_mode": "yolo26",
        "yolo26_runtime": runtime,
        "yolo26_model_dir": str(model_dir),
        "yolo26_reason": reason,
        "yolo26_backend_available": not bool(reason),
    }
