from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .fast_hand_path import FastHandResult
from .meld_state import MeldStateResult
from .river_state import RiverStateResult
from .table_surface import detect_table_surface


DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models" / "yolo26_mahjong"
DEFAULT_CONFIDENCE = 0.25


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
        ok = len(self.hand_tiles) >= max(1, int(min_hand_tiles or 1))
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

        detections = backend.detect(image, min_confidence=min_confidence)
        grouped = postprocess_yolo26_detections(detections, image_size=image.size, min_confidence=min_confidence)
        diagnostics = _write_diagnostics(image, grouped["detections"], image_path=image_path, diagnostics_dir=diagnostics_dir)

    return Yolo26TableStateResult(
        ok=True,
        hand_tiles=grouped["hand_tiles"],
        melds=grouped["melds"],
        meld_tiles=grouped["meld_tiles"],
        discard_piles=grouped["discard_piles"],
        visible_tiles=grouped["visible_tiles"],
        confidence=grouped["confidence"],
        reason="recognized_yolo26_visible_tiles",
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
        raw_detections=[item.to_dict() for item in grouped["detections"]],
        diagnostics={**table_surface.diagnostics, **diagnostics},
        analysis_hints={
            **_base_hints(selected_model_dir, "", backend.runtime),
            **table_surface.to_hints(),
            "yolo26_detection_count": len(grouped["detections"]),
            "yolo26_hand_count": len(grouped["hand_tiles"]),
            "yolo26_self_meld_count": len(grouped["melds"]),
            "yolo26_river_tile_count": len(grouped["visible_tiles"]),
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
    hand = sorted([item for item in enriched if item.area_kind == "hand"], key=lambda item: (_center(item)[0], _center(item)[1]))
    self_meld = sorted([item for item in enriched if item.area_kind == "self_meld"], key=lambda item: (_center(item)[0], _center(item)[1]))
    river = sorted([item for item in enriched if item.area_kind == "river"], key=lambda item: (item.owner, _center(item)[1], _center(item)[0]))

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
    confidences = [item.confidence for item in enriched]
    return {
        "detections": enriched,
        "hand_tiles": [item.tile for item in hand],
        "melds": melds,
        "meld_tiles": [item.tile for item in self_meld],
        "discard_piles": discard_piles,
        "visible_tiles": [item.tile for item in river],
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
    return _Backend(False, "yolo26_decoder_unimplemented", "onnxruntime", labels)


@dataclass(frozen=True)
class _OnnxYolo26Backend(_Backend):
    model_path: Path = Path()

    def __init__(self, *, model_path: Path, labels: list[str]) -> None:
        object.__setattr__(self, "available", True)
        object.__setattr__(self, "reason", "")
        object.__setattr__(self, "runtime", "onnxruntime")
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "model_path", model_path)

    def detect(self, image: Image.Image, *, min_confidence: float) -> list[YoloTileDetection]:
        # 中文：这里是轻量部署后端的稳定接口，实际 YOLO26 输出解码会在模型产物确定后补齐。
        # English: This is the stable lightweight backend seam; YOLO26 output decoding is filled in after the artifact is fixed.
        return []


def _assign_detection_area(item: YoloTileDetection, *, width: int, height: int) -> YoloTileDetection:
    cx, cy = _center(item)
    nx = cx / width
    ny = cy / height
    area_kind = "river"
    owner = _river_owner(nx, ny)
    if ny >= 0.78:
        area_kind = "hand"
        owner = "self"
    elif ny >= 0.62 and (nx <= 0.28 or nx >= 0.72):
        area_kind = "self_meld"
        owner = "self"
    return YoloTileDetection(
        tile=item.tile,
        confidence=item.confidence,
        bbox=list(item.bbox),
        obb=[list(point) for point in item.obb],
        area_kind=area_kind,
        owner=owner,
        source=item.source,
    )


def _river_owner(nx: float, ny: float) -> str:
    if ny >= 0.55:
        return "self"
    if nx < 0.38:
        return "left_opponent"
    if nx > 0.62:
        return "right_opponent"
    return "top_opponent"


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


def _base_hints(model_dir: Path, reason: str, runtime: str) -> dict[str, Any]:
    return {
        "tile_recognition_mode": "yolo26",
        "yolo26_runtime": runtime,
        "yolo26_model_dir": str(model_dir),
        "yolo26_reason": reason,
        "yolo26_backend_available": not bool(reason),
    }
