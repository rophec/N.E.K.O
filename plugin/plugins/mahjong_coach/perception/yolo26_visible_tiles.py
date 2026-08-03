from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .fast_hand_path import FastHandResult
from .image_source import ImageSource, open_rgb, source_exists, source_stem
from .meld_state import MeldStateResult
from .river_state import RiverStateResult
from .table_surface import TableSurfaceResult, detect_table_surface


DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models" / "yolo26_mahjong"
DEFAULT_CONFIDENCE = 0.25
_ORIGINAL_GEOMETRY_RECOVERY_CONFIDENCE = 0.01
_ORIGINAL_HAND_RECOVERY_CONFIDENCE = 0.15
_VALID_EFFECTIVE_HAND_COUNTS = {13, 14}
_GEOMETRIC_RECOVERY_SOURCE_SUFFIX = ":geometric_recovery"

# River ownership is inferred from the current warped-frame detections instead
# of fixed screen boxes. The table center can drift materially when the visible
# table quad is clipped, so hard-coded polygons are not a stable boundary.
_RIVER_OWNER_ORDER = ("self", "right_opponent", "top_opponent", "left_opponent")
_RIVER_INTERIOR_X = (0.18, 0.82)
_RIVER_INTERIOR_Y = (0.16, 0.80)
_RIVER_CENTER_X_LIMITS = (0.34, 0.66)
_RIVER_CENTER_Y_LIMITS = (0.28, 0.66)
_OPPONENT_OWNERS = ("left_opponent", "top_opponent", "right_opponent")
_OPPONENT_MELD_MIN_OUTWARD_DISTANCE = 0.29
_OPPONENT_MELD_CORNER_ALLOWANCE = 0.08


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
    opponent_melds: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    opponent_meld_tiles: list[str] = field(default_factory=list)
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
        tile_identity_reliable = bool(
            self.analysis_hints.get("yolo26_meld_identity_reliable", self.ok)
        )
        hints = {
            **self.analysis_hints,
            "tile_recognition_mode": "yolo26",
            "yolo26_reason": self.reason,
            "tile_identity_reliable": tile_identity_reliable,
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
            opponent_melds={
                owner: [dict(item) for item in items]
                for owner, items in self.opponent_melds.items()
            },
            opponent_meld_tiles=list(self.opponent_meld_tiles),
            confidence=self.confidence,
            reason="recognized_yolo26_discards" if self.visible_tiles else "no_visible_discards",
            elapsed_ms=self.elapsed_ms,
            raw_detections=[
                item
                for item in self.raw_detections
                if item.get("area_kind") in {"river", "opponent_meld"}
            ],
            analysis_hints=hints,
        )


def detect_yolo26_table_state_path(
    image_path: ImageSource,
    *,
    model_dir: Path | None = None,
    min_confidence: float = DEFAULT_CONFIDENCE,
    diagnostics_dir: Path | None = None,
    table_surface_result: TableSurfaceResult | None = None,
) -> Yolo26TableStateResult:
    started = time.perf_counter()
    if not source_exists(image_path):
        return Yolo26TableStateResult(reason="image_missing")
    selected_model_dir = model_dir or DEFAULT_MODEL_DIR
    with open_rgb(image_path) as image:
        table_surface = table_surface_result or detect_table_surface(
            image,
            diagnostics_dir=diagnostics_dir,
            diagnostics_stem=source_stem(image_path),
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
            original_detections = backend.detect(
                image,
                min_confidence=min(min_confidence, _ORIGINAL_GEOMETRY_RECOVERY_CONFIDENCE),
            )
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
                    opponent_melds=grouped["opponent_melds"],
                )
            )

    return Yolo26TableStateResult(
        ok=original_ok or river_ok,
        hand_tiles=grouped["hand_tiles"],
        melds=grouped["melds"],
        meld_tiles=grouped["meld_tiles"],
        opponent_melds={
            owner: [dict(item) for item in items]
            for owner, items in grouped["opponent_melds"].items()
        },
        opponent_meld_tiles=list(grouped["opponent_meld_tiles"]),
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
            "yolo26_opponent_meld_count": grouped["opponent_meld_count"],
            "yolo26_opponent_meld_tile_count": len(grouped["opponent_meld_tiles"]),
            "yolo26_original_recovered_count": grouped["original_recovered_count"],
            "yolo26_hand_recovered_count": grouped["hand_recovered_count"],
            "yolo26_meld_recovered_count": grouped["meld_recovered_count"],
            "yolo26_meld_identity_reliable": grouped["meld_identity_reliable"],
            "yolo26_original_inference_floor": min(
                min_confidence,
                _ORIGINAL_GEOMETRY_RECOVERY_CONFIDENCE,
            ),
            "yolo26_river_tile_count": len(grouped["visible_tiles"]),
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
    original_recovery_accepted = _dedupe_detections(
        [
            item
            for item in detections
            if item.tile
            and item.tile != "empty"
            and item.confidence >= min(min_confidence, _ORIGINAL_GEOMETRY_RECOVERY_CONFIDENCE)
        ]
    )
    width, height = image_size
    original_items = [
        _with_detection_area(item, "original_other", "", coordinate_space="original_frame")
        for item in original_accepted
    ]
    original_recovery_items = [
        _with_detection_source(
            _with_detection_area(item, "original_other", "", coordinate_space="original_frame"),
            (
                item.source
                if item.confidence >= min_confidence
                else f"{item.source}{_GEOMETRIC_RECOVERY_SOURCE_SUFFIX}"
            ),
        )
        for item in original_recovery_accepted
    ]
    hand_items, meld_items = _select_original_hand_and_meld_with_recovery(
        original_items,
        original_recovery_items,
        width=max(1, int(width)),
        height=max(1, int(height)),
        min_confidence=min_confidence,
    )
    hand = [
        _with_detection_area(item, "hand", "self", coordinate_space="original_frame")
        for item in hand_items
    ]
    self_meld = [
        _with_detection_area(item, "self_meld", "self", coordinate_space="original_frame")
        for item in meld_items
    ]
    selected_bottom = hand + self_meld
    enriched_original = [
        item
        for item in original_items
        if all(_iou(item.bbox, selected.bbox) < 0.5 for selected in selected_bottom)
    ] + selected_bottom
    hand_recovered_count = sum(_is_geometric_recovery(item) for item in hand)
    meld_recovered_count = sum(_is_geometric_recovery(item) for item in self_meld)
    original_recovered_count = hand_recovered_count + meld_recovered_count

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
    enriched_river, opponent_melds = _group_opponent_melds(
        enriched_river,
        center=river_center,
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
    opponent_meld_tiles = [
        str(tile)
        for owner in _OPPONENT_OWNERS
        for meld in opponent_melds.get(owner, [])
        if meld.get("tile_identity_reliable") is not False
        for tile in meld.get("tiles", [])
        if str(tile).strip()
    ]
    riichi_players = _detect_riichi_declarations(discard_piles)
    confidences = [item.confidence for item in enriched]
    opponent_meld_count = sum(len(items) for items in opponent_melds.values())
    excluded_visible_count = sum(item.area_kind != "river" for item in enriched_river)
    return {
        "detections": enriched,
        "original_detections": enriched_original,
        "river_detections": enriched_river,
        "river_center": [round(river_center[0], 4), round(river_center[1], 4)],
        "hand_tiles": [item.tile for item in hand],
        "melds": melds,
        "meld_tiles": [item.tile for item in self_meld],
        "opponent_melds": opponent_melds,
        "opponent_meld_tiles": opponent_meld_tiles,
        "discard_piles": discard_piles,
        "visible_tiles": [item.tile for item in river],
        "riichi_players": riichi_players,
        "original_recovered_count": original_recovered_count,
        "hand_recovered_count": hand_recovered_count,
        "meld_recovered_count": meld_recovered_count,
        "meld_identity_reliable": meld_recovered_count == 0,
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
    if not model_dir.exists():
        return _Backend(False, "yolo26_model_dir_missing", "none")
    metadata_path = model_dir / "metadata.json"
    labels_path = model_dir / "labels.json"
    labels = _load_labels(labels_path)
    metadata = _load_metadata(metadata_path)
    runtime = str(metadata.get("runtime") or metadata.get("format") or "onnxruntime").lower()
    model_name = str(metadata.get("model_file") or "model.onnx")
    model_path = _resolve_model_path(model_dir, model_name)
    if not labels:
        return _Backend(False, "yolo26_labels_missing", runtime, labels)
    if model_path is None:
        return _Backend(False, "yolo26_model_path_invalid", runtime, labels)
    if not model_path.exists():
        return _Backend(False, "yolo26_model_missing", runtime, labels)
    expected_class_count = metadata.get("class_count")
    if expected_class_count is not None:
        try:
            class_count = int(expected_class_count)
        except (TypeError, ValueError):
            return _Backend(False, "yolo26_class_count_invalid", runtime, labels)
        if class_count != len(labels):
            return _Backend(False, "yolo26_class_count_mismatch", runtime, labels)
    try:
        model_identity = _model_file_identity(model_path)
    except OSError:
        return _Backend(False, "yolo26_model_unreadable", runtime, labels)
    validation_reason = _validate_model_artifact(
        model_identity,
        str(metadata.get("sha256") or "").strip().lower(),
    )
    if validation_reason:
        return _Backend(False, validation_reason, runtime, labels)
    if runtime not in {"onnx", "onnxruntime"}:
        return _Backend(False, f"yolo26_runtime_unimplemented:{runtime}", runtime, labels)
    raw_size = metadata.get("input_size") or [800, 800]
    if not isinstance(raw_size, list) or len(raw_size) != 2:
        raw_size = [800, 800]
    input_size = (max(1, int(raw_size[0])), max(1, int(raw_size[1])))
    return _OnnxYolo26Backend(
        model_path=model_path,
        model_identity=model_identity,
        labels=labels,
        input_size=input_size,
    )


def _resolve_model_path(model_dir: Path, model_name: str) -> Path | None:
    relative = Path(model_name)
    if relative.is_absolute():
        return None
    try:
        root = model_dir.resolve()
        candidate = (root / relative).resolve()
        candidate.relative_to(root)
    except (OSError, ValueError):
        return None
    return candidate


def _model_file_identity(model_path: Path) -> tuple[str, int, int, str]:
    resolved = model_path.resolve()
    stat = resolved.stat()
    sample_size = 64 * 1024
    offsets = {
        0,
        max(0, stat.st_size // 2 - sample_size // 2),
        max(0, stat.st_size - sample_size),
    }
    digest = hashlib.blake2s(digest_size=16)
    with resolved.open("rb") as model_file:
        for offset in sorted(offsets):
            model_file.seek(offset)
            digest.update(offset.to_bytes(8, "little", signed=False))
            digest.update(model_file.read(sample_size))
    return str(resolved), int(stat.st_size), int(stat.st_mtime_ns), digest.hexdigest()


@lru_cache(maxsize=8)
def _validate_model_artifact(
    model_identity: tuple[str, int, int, str],
    expected_sha256: str,
) -> str:
    model_path = Path(model_identity[0])
    try:
        with model_path.open("rb") as model_file:
            prefix = model_file.read(200)
            if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
                return "yolo26_model_lfs_pointer"
            if expected_sha256:
                model_file.seek(0)
                actual_sha256 = hashlib.file_digest(model_file, "sha256").hexdigest()
                if actual_sha256.lower() != expected_sha256:
                    return "yolo26_model_checksum_mismatch"
    except OSError:
        return "yolo26_model_unreadable"
    return ""


@dataclass(frozen=True)
class _OnnxYolo26Backend(_Backend):
    model_path: Path = Path()
    model_identity: tuple[str, int, int, str] = ("", 0, 0, "")
    input_size: tuple[int, int] = (800, 800)

    def __init__(
        self,
        *,
        model_path: Path,
        model_identity: tuple[str, int, int, str],
        labels: list[str],
        input_size: tuple[int, int],
    ) -> None:
        object.__setattr__(self, "available", True)
        object.__setattr__(self, "reason", "")
        object.__setattr__(self, "runtime", "onnxruntime")
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "model_path", model_path)
        object.__setattr__(self, "model_identity", model_identity)
        object.__setattr__(self, "input_size", input_size)

    def detect(self, image: Image.Image, *, min_confidence: float) -> list[YoloTileDetection]:
        # 中文：这里执行 ONNX 推理，并解码 YOLO26 的端到端检测输出。
        # English: This runs ONNX inference and decodes YOLO26 end-to-end detections.
        # 中文：插件仅加载导出的轻量模型，不导入训练框架。
        # English: The plugin loads only the lightweight export and never imports the training framework.
        session = _load_onnx_session(*self.model_identity)
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


@lru_cache(maxsize=4)
def _load_onnx_session(
    model_path: str,
    _model_size: int,
    _model_mtime_ns: int,
    _model_content_digest: str,
) -> Any:
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


def _select_original_hand_and_meld_with_recovery(
    accepted_items: list[YoloTileDetection],
    recovery_items: list[YoloTileDetection],
    *,
    width: int,
    height: int,
    min_confidence: float,
) -> tuple[list[YoloTileDetection], list[YoloTileDetection]]:
    """Recover only a geometrically complete bottom-row hand or meld layout."""

    def partition(items: list[YoloTileDetection]) -> tuple[list[YoloTileDetection], list[YoloTileDetection]]:
        bottom = sorted(
            _select_original_bottom_row(items, width=width, height=height),
            key=lambda item: (_center(item)[0], _center(item)[1]),
        )
        return _split_bottom_hand_and_meld(bottom, image_width=width)

    accepted_hand, accepted_meld = partition(accepted_items)
    recovery_hand, recovery_meld = partition(recovery_items)
    recovered_hand = [item for item in recovery_hand if item.confidence < min_confidence]
    recovered_meld = [item for item in recovery_meld if item.confidence < min_confidence]
    if not recovered_hand and not recovered_meld:
        return accepted_hand, accepted_meld

    recovery_selected = recovery_hand + recovery_meld
    if any(
        all(_iou(item.bbox, candidate.bbox) < 0.5 for candidate in recovery_selected)
        for item in accepted_hand + accepted_meld
    ):
        return accepted_hand, accepted_meld

    if recovered_hand and not _is_coherent_hand_recovery(
        recovery_hand,
        recovered_hand,
        min_confidence=min_confidence,
    ):
        return accepted_hand, accepted_meld

    meld_groups = _split_self_meld_groups(recovery_meld)
    if recovery_meld and (
        not meld_groups
        or any(len(group) not in {3, 4} for group in meld_groups)
        or not _is_coherent_meld_recovery(meld_groups, width=width, height=height)
    ):
        return accepted_hand, accepted_meld

    effective_count = len(recovery_hand) + 3 * len(meld_groups)
    if effective_count not in _VALID_EFFECTIVE_HAND_COUNTS:
        return accepted_hand, accepted_meld
    return recovery_hand, recovery_meld


def _is_coherent_hand_recovery(
    hand: list[YoloTileDetection],
    recovered: list[YoloTileDetection],
    *,
    min_confidence: float,
) -> bool:
    if any(item.confidence < _ORIGINAL_HAND_RECOVERY_CONFIDENCE for item in recovered):
        return False
    accepted = [item for item in hand if item.confidence >= min_confidence]
    if len(accepted) < 2:
        return False
    accepted_centers = sorted(_center(item)[0] for item in accepted)
    accepted_gaps = [
        accepted_centers[index + 1] - accepted_centers[index]
        for index in range(len(accepted_centers) - 1)
        if accepted_centers[index + 1] > accepted_centers[index]
    ]
    if not accepted_gaps:
        return False
    median_gap = sorted(accepted_gaps)[len(accepted_gaps) // 2]
    ordered = sorted(hand, key=lambda item: _center(item)[0])
    recovered_ids = {id(item) for item in recovered}
    for index, item in enumerate(ordered):
        if id(item) not in recovered_ids:
            continue
        if index == 0 or index == len(ordered) - 1:
            return False
        left = ordered[index - 1]
        right = ordered[index + 1]
        if left.confidence < min_confidence or right.confidence < min_confidence:
            return False
        left_gap = _center(item)[0] - _center(left)[0]
        right_gap = _center(right)[0] - _center(item)[0]
        if not (
            median_gap * 0.35 <= left_gap <= median_gap * 1.75
            and median_gap * 0.35 <= right_gap <= median_gap * 1.75
        ):
            return False
    return True


def _is_coherent_meld_recovery(
    groups: list[list[YoloTileDetection]],
    *,
    width: int,
    height: int,
) -> bool:
    for group in groups:
        boxes = [_ordered_bbox(item.bbox) for item in group]
        widths = [max(1.0, right - left) for left, _top, right, _bottom in boxes]
        heights = [max(1.0, bottom - top) for _left, top, _right, bottom in boxes]
        median_width = sorted(widths)[len(widths) // 2]
        median_height = sorted(heights)[len(heights) // 2]
        if max(widths) / min(widths) > 1.8 or max(heights) / min(heights) > 1.8:
            return False
        centers = sorted(_center(item) for item in group)
        center_ys = [center[1] for center in centers]
        bottoms = [box[3] for box in boxes]
        vertical_tolerance = max(height * 0.045, median_height * 0.70)
        if max(center_ys) - min(center_ys) > vertical_tolerance:
            return False
        if max(bottoms) - min(bottoms) > vertical_tolerance:
            return False
        gaps = [centers[index + 1][0] - centers[index][0] for index in range(len(centers) - 1)]
        if not gaps or min(gaps) <= 0:
            return False
        if max(gaps) > max(width * 0.075, median_width * 1.9):
            return False
    return True


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
    diagonal_ambiguity = max(0.02, median_tile_scale * 0.65)

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
        owner = _river_owner_from_detection(
            item,
            dx=dx,
            dy=dy,
            diagonal_ambiguity=diagonal_ambiguity,
        )
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
                _river_owner_from_detection(
                    item,
                    dx=nx - center[0],
                    dy=ny - center[1],
                    diagonal_ambiguity=diagonal_ambiguity,
                ),
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


def _river_owner_from_detection(
    item: YoloTileDetection,
    *,
    dx: float,
    dy: float,
    diagonal_ambiguity: float,
) -> str:
    angular_owner = _river_owner_from_vector(dx, dy)
    if abs(abs(dx) - abs(dy)) > diagonal_ambiguity:
        return angular_owner
    left, top, right, bottom = _ordered_bbox(item.bbox)
    aspect_ratio = (right - left) / max(1e-6, bottom - top)
    if aspect_ratio >= 1.05:
        return "right_opponent" if dx >= 0.0 else "left_opponent"
    if aspect_ratio <= 0.95:
        return "self" if dy >= 0.0 else "top_opponent"
    return angular_owner


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


def _group_opponent_melds(
    items: list[YoloTileDetection],
    *,
    center: tuple[float, float],
    width: int,
    height: int,
) -> tuple[list[YoloTileDetection], dict[str, list[dict[str, Any]]]]:
    """Promote coherent outer-seat tile shelves into structured opponent melds.

    The river pass already rejects the outer band from ``discard_piles``. This
    second stage is deliberately stricter: a tile must be on the correct
    player's meld-side corner, far outside the river band, aligned with that
    seat's shelf, and part of a geometrically complete group of three or four.
    Isolated animation/UI detections remain ``excluded_table_tile``.
    """

    grouped_items: dict[str, list[list[YoloTileDetection]]] = {}
    for owner in _OPPONENT_OWNERS:
        candidates = [
            item
            for item in items
            if item.area_kind == "excluded_table_tile"
            and item.owner == owner
            and _is_opponent_meld_shelf_candidate(
                item,
                owner=owner,
                center=center,
                width=width,
                height=height,
            )
        ]
        owner_groups: list[list[YoloTileDetection]] = []
        for cluster in _cluster_opponent_meld_candidates(
            candidates,
            owner=owner,
            width=width,
            height=height,
        ):
            owner_groups.extend(
                _partition_opponent_meld_cluster(
                    cluster,
                    owner=owner,
                    width=width,
                    height=height,
                )
            )
        if owner_groups:
            grouped_items[owner] = owner_groups[:4]

    promoted_ids = {
        id(item)
        for groups in grouped_items.values()
        for group in groups
        for item in group
    }
    promoted = {
        id(item): _with_detection_area(
            item,
            "opponent_meld",
            item.owner,
            coordinate_space="warped_table",
        )
        for item in items
        if id(item) in promoted_ids
    }
    enriched = [promoted.get(id(item), item) for item in items]

    opponent_melds: dict[str, list[dict[str, Any]]] = {}
    for owner in _OPPONENT_OWNERS:
        payloads: list[dict[str, Any]] = []
        for index, group in enumerate(grouped_items.get(owner, []), start=1):
            marked_group = [promoted[id(item)] for item in group]
            payloads.append(
                _opponent_meld_payload(
                    marked_group,
                    owner=owner,
                    meld_index=index,
                )
            )
        if payloads:
            opponent_melds[owner] = payloads
    return enriched, opponent_melds


def _is_opponent_meld_shelf_candidate(
    item: YoloTileDetection,
    *,
    owner: str,
    center: tuple[float, float],
    width: int,
    height: int,
) -> bool:
    nx, ny = _normalized_center(item, width=width, height=height)
    if owner == "top_opponent":
        outward_distance = center[1] - ny
        on_meld_side = nx <= center[0] + _OPPONENT_MELD_CORNER_ALLOWANCE
    elif owner == "right_opponent":
        outward_distance = nx - center[0]
        on_meld_side = ny <= center[1] + _OPPONENT_MELD_CORNER_ALLOWANCE
    elif owner == "left_opponent":
        outward_distance = center[0] - nx
        on_meld_side = ny >= center[1] - _OPPONENT_MELD_CORNER_ALLOWANCE
    else:
        return False
    return on_meld_side and outward_distance >= _OPPONENT_MELD_MIN_OUTWARD_DISTANCE


def _cluster_opponent_meld_candidates(
    items: list[YoloTileDetection],
    *,
    owner: str,
    width: int,
    height: int,
) -> list[list[YoloTileDetection]]:
    if not items:
        return []
    ordered = sorted(items, key=lambda item: _opponent_meld_progress(item, owner=owner))
    major_sizes = [_opponent_meld_box_axes(item, owner=owner)[0] for item in ordered]
    cross_sizes = [_opponent_meld_box_axes(item, owner=owner)[1] for item in ordered]
    median_major = sorted(major_sizes)[len(major_sizes) // 2]
    median_cross = sorted(cross_sizes)[len(cross_sizes) // 2]
    major_dimension = width if owner == "top_opponent" else height
    cross_dimension = height if owner == "top_opponent" else width
    maximum_major_gap = max(major_dimension * 0.095, median_major * 2.35)
    maximum_cross_gap = max(cross_dimension * 0.055, median_cross * 1.35)

    clusters: list[list[YoloTileDetection]] = [[ordered[0]]]
    for item in ordered[1:]:
        previous = clusters[-1][-1]
        major_gap = abs(
            _opponent_meld_progress(item, owner=owner)
            - _opponent_meld_progress(previous, owner=owner)
        )
        cross_gap = abs(
            _opponent_meld_cross_position(item, owner=owner)
            - _opponent_meld_cross_position(previous, owner=owner)
        )
        if major_gap <= maximum_major_gap and cross_gap <= maximum_cross_gap:
            clusters[-1].append(item)
        else:
            clusters.append([item])
    return [cluster for cluster in clusters if len(cluster) >= 3]


def _partition_opponent_meld_cluster(
    items: list[YoloTileDetection],
    *,
    owner: str,
    width: int,
    height: int,
) -> list[list[YoloTileDetection]]:
    """Split one contiguous shelf with a small DP over legal meld sizes."""

    ordered = sorted(items, key=lambda item: _opponent_meld_progress(item, owner=owner))

    @lru_cache(maxsize=None)
    def solve(index: int) -> tuple[float, tuple[tuple[int, int], ...], int]:
        if index >= len(ordered):
            return 0.0, (), 0

        skipped_score, skipped_groups, skipped_count = solve(index + 1)
        best = (skipped_score - 2.5, skipped_groups, skipped_count + 1)

        for size in (3, 4):
            end = index + size
            if end > len(ordered):
                continue
            group = ordered[index:end]
            if not _is_coherent_opponent_meld_group(
                group,
                owner=owner,
                width=width,
                height=height,
            ):
                continue
            classification = _classify_opponent_meld_group(group, owner=owner)
            kind = str(classification["kind"])
            if kind == "unknown" or classification["called_tile_index"] is None:
                # Open chi/pon/kan must expose one called tile in the seat's
                # turned orientation. This keeps aligned dora indicators or
                # other UI tiles from becoming a meld on geometry alone.
                continue
            kind_bonus = {
                "chi": 5.0,
                "pon": 5.0,
                "kan": 6.0,
            }.get(kind, 0.0)
            size_bonus = 1.5 if size == 3 else 0.5
            called_bonus = 0.45 if classification["called_tile_index"] is not None else 0.0
            detection_bonus = sum(item.confidence for item in group) / size
            tail_score, tail_groups, tail_skips = solve(end)
            candidate = (
                tail_score + kind_bonus + size_bonus + called_bonus + detection_bonus,
                ((index, end), *tail_groups),
                tail_skips,
            )
            if candidate[0] > best[0] or (
                abs(candidate[0] - best[0]) < 1e-6
                and candidate[2] < best[2]
            ):
                best = candidate
        return best

    _score, ranges, _skipped = solve(0)
    return [ordered[start:end] for start, end in ranges]


def _is_coherent_opponent_meld_group(
    items: list[YoloTileDetection],
    *,
    owner: str,
    width: int,
    height: int,
) -> bool:
    if len(items) not in {3, 4}:
        return False
    major_positions = [_opponent_meld_progress(item, owner=owner) for item in items]
    cross_positions = [_opponent_meld_cross_position(item, owner=owner) for item in items]
    major_sizes = [_opponent_meld_box_axes(item, owner=owner)[0] for item in items]
    cross_sizes = [_opponent_meld_box_axes(item, owner=owner)[1] for item in items]
    median_major = sorted(major_sizes)[len(major_sizes) // 2]
    median_cross = sorted(cross_sizes)[len(cross_sizes) // 2]
    major_dimension = width if owner == "top_opponent" else height
    cross_dimension = height if owner == "top_opponent" else width
    if max(cross_positions) - min(cross_positions) > max(
        cross_dimension * 0.05,
        median_cross * 0.9,
    ):
        return False
    gaps = [
        major_positions[index + 1] - major_positions[index]
        for index in range(len(major_positions) - 1)
    ]
    if not gaps or min(gaps) <= 0.0:
        return False
    if max(gaps) > max(major_dimension * 0.08, median_major * 2.45):
        return False
    return True


def _opponent_meld_payload(
    items: list[YoloTileDetection],
    *,
    owner: str,
    meld_index: int,
) -> dict[str, Any]:
    classification = _classify_opponent_meld_group(items, owner=owner)
    lefts, tops, rights, bottoms = zip(*(_ordered_bbox(item.bbox) for item in items))
    return {
        "owner": owner,
        "meld_index": meld_index,
        "kind": classification["kind"],
        "kind_reason": classification["kind_reason"],
        "classification_confidence": classification["classification_confidence"],
        "tiles": list(classification["tiles"]),
        "observed_tiles": [item.tile for item in items],
        "called_tile_index": classification["called_tile_index"],
        "corrections": list(classification["corrections"]),
        "tile_identity_reliable": classification["kind"] != "unknown",
        "confidence": round(sum(item.confidence for item in items) / len(items), 4),
        "bbox": [
            round(min(lefts), 2),
            round(min(tops), 2),
            round(max(rights), 2),
            round(max(bottoms), 2),
        ],
        "source": "yolo26_opponent_meld_geometry",
        "coordinate_space": "warped_table",
        "detections": [item.to_dict() for item in items],
    }


def _classify_opponent_meld_group(
    items: list[YoloTileDetection],
    *,
    owner: str,
) -> dict[str, Any]:
    observed = [str(item.tile) for item in items]
    tiles = list(observed)
    called_index = _opponent_meld_called_tile_index(items, owner=owner)
    corrections: list[dict[str, Any]] = []
    kind = "unknown"
    kind_reason = "geometry_only"
    classification_confidence = 0.45

    counts = Counter(observed)
    majority_tile, majority_count = counts.most_common(1)[0]
    if len(items) == 3 and len(counts) == 1:
        kind = "pon"
        kind_reason = "exact_triplet"
        classification_confidence = 0.99
    elif len(items) == 3 and _is_exact_chi(observed):
        kind = "chi"
        kind_reason = "exact_sequence"
        classification_confidence = 0.98
    elif (
        len(items) == 3
        and majority_count == 2
        and called_index is not None
        and observed[called_index] != majority_tile
    ):
        corrections.append(
            {
                "tile_index": called_index,
                "from": observed[called_index],
                "to": majority_tile,
                "reason": "called_tile_legal_pon_recovery",
            }
        )
        tiles[called_index] = majority_tile
        kind = "pon"
        kind_reason = "called_tile_legal_pon_recovery"
        classification_confidence = 0.82
    elif len(items) == 4 and len(counts) == 1:
        kind = "kan"
        kind_reason = "exact_quad"
        classification_confidence = 0.99
    elif (
        len(items) == 4
        and majority_count == 3
        and called_index is not None
        and observed[called_index] != majority_tile
    ):
        corrections.append(
            {
                "tile_index": called_index,
                "from": observed[called_index],
                "to": majority_tile,
                "reason": "called_tile_legal_kan_recovery",
            }
        )
        tiles[called_index] = majority_tile
        kind = "kan"
        kind_reason = "called_tile_legal_kan_recovery"
        classification_confidence = 0.82

    return {
        "kind": kind,
        "kind_reason": kind_reason,
        "classification_confidence": round(classification_confidence, 4),
        "tiles": tiles,
        "called_tile_index": called_index,
        "corrections": corrections,
    }


def _is_exact_chi(tiles: list[str]) -> bool:
    if len(tiles) != 3:
        return False
    suits = {tile[-1:] for tile in tiles}
    if len(suits) != 1 or next(iter(suits), "") not in {"m", "p", "s"}:
        return False
    try:
        ranks = sorted(int(tile[:-1]) for tile in tiles)
    except ValueError:
        return False
    return ranks[0] >= 1 and ranks[2] <= 9 and ranks[1] == ranks[0] + 1 and ranks[2] == ranks[1] + 1


def _opponent_meld_called_tile_index(
    items: list[YoloTileDetection],
    *,
    owner: str,
) -> int | None:
    if len(items) < 3:
        return None
    turnedness = []
    for item in items:
        left, top, right, bottom = _ordered_bbox(item.bbox)
        width = max(1e-6, right - left)
        height = max(1e-6, bottom - top)
        score = width / height if owner == "top_opponent" else height / width
        turnedness.append(score)
    ranked = sorted(range(len(turnedness)), key=lambda index: turnedness[index], reverse=True)
    best_index = ranked[0]
    best_score = turnedness[best_index]
    runner_up = turnedness[ranked[1]]
    if best_score >= 0.95 or (
        best_score >= runner_up * 1.2
        and best_score - runner_up >= 0.12
    ):
        return best_index
    return None


def _opponent_meld_progress(item: YoloTileDetection, *, owner: str) -> float:
    center_x, center_y = _center(item)
    if owner == "top_opponent":
        return center_x
    if owner == "right_opponent":
        return center_y
    return -center_y


def _opponent_meld_cross_position(item: YoloTileDetection, *, owner: str) -> float:
    center_x, center_y = _center(item)
    return center_y if owner == "top_opponent" else center_x


def _opponent_meld_box_axes(item: YoloTileDetection, *, owner: str) -> tuple[float, float]:
    left, top, right, bottom = _ordered_bbox(item.bbox)
    if owner == "top_opponent":
        return right - left, bottom - top
    return bottom - top, right - left


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
    return items[: split_index + 1], items[split_index + 1 :]


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


def _with_detection_source(item: YoloTileDetection, source: str) -> YoloTileDetection:
    return YoloTileDetection(
        tile=item.tile,
        confidence=item.confidence,
        bbox=list(item.bbox),
        obb=[list(point) for point in item.obb],
        area_kind=item.area_kind,
        owner=item.owner,
        source=source,
        coordinate_space=item.coordinate_space,
    )


def _is_geometric_recovery(item: YoloTileDetection) -> bool:
    return item.source.endswith(_GEOMETRIC_RECOVERY_SOURCE_SUFFIX)


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
    image_path: ImageSource,
    diagnostics_dir: Path | None,
    suffix: str,
    opponent_melds: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    if diagnostics_dir is None:
        return {}
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    area_colors = {
        "hand": "#36d889",
        "self_meld": "#ff9f32",
        "river": "#62eea0",
        "opponent_meld": "#58b8ff",
        "excluded_table_tile": "#aeb5bb",
    }
    for item in detections:
        left, top, right, bottom = _ordered_bbox(item.bbox)
        color = area_colors.get(item.area_kind, "red")
        draw.rectangle((left, top, right, bottom), outline=color, width=2)
        draw.text((left, max(0, top - 14)), f"{item.tile} {item.area_kind}", fill=color)
    for owner, melds in (opponent_melds or {}).items():
        for meld in melds:
            bbox = meld.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            left, top, right, bottom = _ordered_bbox([float(value) for value in bbox])
            label = f"{owner} #{meld.get('meld_index')} {meld.get('kind')}"
            draw.rectangle((left - 3, top - 3, right + 3, bottom + 3), outline="#ffe062", width=3)
            draw.text((left, min(image.height - 14, bottom + 3)), label, fill="#ffe062")
    target = diagnostics_dir / f"{source_stem(image_path)}-yolo26-{suffix}-overlay.jpg"
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
    opponent_melds: dict[str, list[dict[str, Any]]] | None = None,
) -> Image.Image:
    """Render adaptive river and opponent-meld ownership on the warped table."""
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
        "opponent_meld": (88, 184, 255, 255),
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
        owner = str(item.get("owner") or "")
        label = f"{tile} {confidence:.2f}" if area_kind == "river" else f"{tile} {owner}"
        draw.text((left, max(28, top - 13)), label, fill=color)

    for owner, melds in (opponent_melds or {}).items():
        owner_color = region_colors.get(owner, (255, 224, 98, 255))
        for meld in melds:
            bbox = meld.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            left, top, right, bottom = _ordered_bbox([float(value) for value in bbox])
            group_width = max(3, round(width / 220))
            draw.rectangle(
                (left - group_width, top - group_width, right + group_width, bottom + group_width),
                outline=owner_color,
                width=group_width,
            )
            kind = str(meld.get("kind") or "unknown")
            index = int(meld.get("meld_index") or 0)
            draw.text(
                (left, min(height - 14, bottom + 4)),
                f"{owner} M{index} {kind}",
                fill=owner_color,
            )

    draw.rectangle((0, 0, width, 24), fill=(8, 12, 14, 205))
    meld_count = sum(len(items) for items in (opponent_melds or {}).values())
    draw.text(
        (8, 6),
        f"WARPED TABLE | four rivers + opponent melds ({meld_count})",
        fill=(242, 245, 244, 255),
    )
    return preview


def _base_hints(model_dir: Path, reason: str, runtime: str) -> dict[str, Any]:
    metadata = _load_metadata(model_dir / "metadata.json")
    model_source = "bundled" if _same_resolved_path(model_dir, DEFAULT_MODEL_DIR) else "override"
    raw_model_id = str(
        metadata.get("training_run")
        or metadata.get("model_variant")
        or metadata.get("model_family")
        or "yolo26"
    ).strip()
    model_id = raw_model_id.replace("\\", "/").rsplit("/", 1)[-1][:128] or "yolo26"
    model_hash = str(metadata.get("sha256") or "").strip().lower()[:12]
    if not model_hash:
        model_path = _resolve_model_path(model_dir, str(metadata.get("model_file") or "model.onnx"))
        if model_path is not None and model_path.is_file():
            try:
                model_hash = _model_file_identity(model_path)[3][:12]
            except OSError:
                model_hash = ""
    return {
        "tile_recognition_mode": "yolo26",
        "yolo26_runtime": runtime,
        "model_source": model_source,
        "model_id": model_id,
        "model_hash": model_hash,
        "yolo26_reason": reason,
        "yolo26_backend_available": not bool(reason),
    }


def _same_resolved_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return False
