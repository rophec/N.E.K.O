from __future__ import annotations

import importlib.util
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from .image_source import ImageSource, open_rgb, source_exists
from .table_surface import TableSurfaceResult, detect_table_surface


_SCORE_CROPS = {
    "self": ((0.4375, 0.5475, 0.5625, 0.58375), 0),
    "top_opponent": ((0.4375, 0.42875, 0.5625, 0.46375), 180),
    "left_opponent": ((0.4275, 0.46875, 0.46375, 0.5625), 90),
    "right_opponent": ((0.53625, 0.46875, 0.5725, 0.5625), 270),
}

# Mahjong Soul's upper-left counters use a red-dot 1000-point stick for
# riichi deposits and a black-dot 100-point stick for honba. The crops include
# the multiplier and digits, not the icon, so two-digit counters still fit.
_COUNTER_CROPS = {
    "riichi_stick_count": (0.043, 0.141, 0.085, 0.195),
    "honba_count": (0.108, 0.141, 0.150, 0.195),
}

_MIN_SCORE_CONFIDENCE = 0.60
_MIN_COUNTER_CONFIDENCE = 0.42
_SCORE_PATTERN = re.compile(r"-?\d{3,6}")
_COUNTER_PATTERN = re.compile(r"\d{1,2}")

NumericRecognizer = Callable[[Image.Image, str], tuple[str, float]]


@dataclass(frozen=True)
class TableContextResult:
    ok: bool = False
    scores: dict[str, int] = field(default_factory=dict)
    ranks: dict[str, int] = field(default_factory=dict)
    honba_count: int | None = None
    riichi_stick_count: int | None = None
    confidence: float = 0.0
    reason: str = ""
    score_reads: dict[str, dict[str, Any]] = field(default_factory=dict)
    counter_reads: dict[str, dict[str, Any]] = field(default_factory=dict)
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "scores": dict(self.scores),
            "ranks": dict(self.ranks),
            "honba_count": self.honba_count,
            "riichi_stick_count": self.riichi_stick_count,
            "confidence": round(float(self.confidence), 4),
            "reason": self.reason,
            "score_reads": {key: dict(value) for key, value in self.score_reads.items()},
            "counter_reads": {key: dict(value) for key, value in self.counter_reads.items()},
            "elapsed_ms": round(float(self.elapsed_ms), 1),
        }


def detect_table_context(
    image_source: ImageSource,
    *,
    table_surface_result: TableSurfaceResult | None = None,
    recognizer: NumericRecognizer | None = None,
    model_path: str | Path | None = None,
) -> TableContextResult:
    """Read all four scores plus the honba/deposit counters from one frame.

    Four valid scores are mandatory. Counters are deliberately optional: a
    table skin or a temporary effect may cover the upper-left HUD while the
    central score panel is still reliable. The engine applies its own
    consecutive-frame confirmation before these values affect strategy.
    """
    started = time.perf_counter()
    if not source_exists(image_source):
        return _result(reason="image_missing", started=started)
    try:
        with open_rgb(image_source) as opened:
            frame = opened.copy()
    except Exception:
        return _result(reason="image_unreadable", started=started)

    surface = table_surface_result
    if surface is None:
        surface = detect_table_surface(frame)
    if not surface.ok or surface.warped_image is None:
        return _result(
            reason=surface.reason or "table_surface_unavailable",
            started=started,
        )

    if recognizer is None:
        try:
            recognizer = _recognizer_for_model(_resolve_rec_model_path(model_path))
        except Exception as exc:
            return _result(
                reason=f"score_ocr_unavailable:{type(exc).__name__}",
                started=started,
            )

    score_reads: dict[str, dict[str, Any]] = {}
    scores: dict[str, int] = {}
    score_confidences: list[float] = []
    warped = surface.warped_image.convert("RGB")
    for player, (crop_box, rotation) in _SCORE_CROPS.items():
        crop = _normalized_crop(warped, crop_box)
        if rotation:
            crop = crop.rotate(rotation, expand=True)
        text, confidence = _safe_recognize(recognizer, crop, f"score:{player}")
        score = _parse_score(text, confidence)
        score_reads[player] = {
            "text": text,
            "confidence": round(float(confidence), 4),
            "value": score,
            "rotation": rotation,
        }
        if score is not None:
            scores[player] = score
            score_confidences.append(float(confidence))

    counter_reads: dict[str, dict[str, Any]] = {}
    counters: dict[str, int | None] = {}
    for field_name, crop_box in _COUNTER_CROPS.items():
        crop = _normalized_crop(frame, crop_box)
        text, confidence = _safe_recognize(recognizer, crop, f"counter:{field_name}")
        value = _parse_counter(text, confidence)
        counters[field_name] = value
        counter_reads[field_name] = {
            "text": text,
            "confidence": round(float(confidence), 4),
            "value": value,
        }

    if len(scores) != 4:
        return TableContextResult(
            reason="four_scores_not_confirmed",
            scores=scores,
            ranks=_score_ranks(scores),
            honba_count=counters.get("honba_count"),
            riichi_stick_count=counters.get("riichi_stick_count"),
            confidence=min(score_confidences, default=0.0),
            score_reads=score_reads,
            counter_reads=counter_reads,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

    if not _score_total_is_plausible(scores):
        return TableContextResult(
            reason="score_total_implausible",
            scores=scores,
            ranks=_score_ranks(scores),
            honba_count=counters.get("honba_count"),
            riichi_stick_count=counters.get("riichi_stick_count"),
            confidence=min(score_confidences, default=0.0),
            score_reads=score_reads,
            counter_reads=counter_reads,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

    confidence = min(score_confidences)
    return TableContextResult(
        ok=True,
        scores=scores,
        ranks=_score_ranks(scores),
        honba_count=counters.get("honba_count"),
        riichi_stick_count=counters.get("riichi_stick_count"),
        confidence=confidence,
        reason="table_context_detected",
        score_reads=score_reads,
        counter_reads=counter_reads,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
    )


def _result(*, reason: str, started: float) -> TableContextResult:
    return TableContextResult(
        reason=reason,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
    )


def _normalized_crop(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    width, height = image.size
    left, top, right, bottom = box
    return image.crop(
        (
            max(0, int(round(width * left))),
            max(0, int(round(height * top))),
            min(width, int(round(width * right))),
            min(height, int(round(height * bottom))),
        )
    )


def _safe_recognize(
    recognizer: NumericRecognizer,
    crop: Image.Image,
    field_name: str,
) -> tuple[str, float]:
    try:
        text, confidence = recognizer(crop, field_name)
        return str(text or "").strip(), max(0.0, min(1.0, float(confidence or 0.0)))
    except Exception:
        return "", 0.0


def _parse_score(text: str, confidence: float) -> int | None:
    if confidence < _MIN_SCORE_CONFIDENCE:
        return None
    normalized = str(text or "").translate(str.maketrans("０１２３４５６７８９−", "0123456789-"))
    match = _SCORE_PATTERN.search(normalized.replace(",", "").replace(" ", ""))
    if match is None:
        return None
    value = int(match.group(0))
    if value < -100_000 or value > 200_000 or value % 100:
        return None
    return value


def _parse_counter(text: str, confidence: float) -> int | None:
    if confidence < _MIN_COUNTER_CONFIDENCE:
        return None
    normalized = str(text or "").translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    matches = _COUNTER_PATTERN.findall(normalized)
    if not matches:
        return None
    value = int(matches[-1])
    return value if 0 <= value <= 99 else None


def _score_ranks(scores: dict[str, int]) -> dict[str, int]:
    if not scores:
        return {}
    ordered_values = sorted(set(scores.values()), reverse=True)
    return {player: ordered_values.index(score) + 1 for player, score in scores.items()}


def _score_total_is_plausible(scores: dict[str, int]) -> bool:
    total = sum(int(score) for score in scores.values())
    # Points move between players in 100-point units, while riichi deposits
    # remove exactly 1000 points from the four displayed scores. Ranked and
    # custom four-player starting totals are therefore still 1000-aligned.
    return 40_000 <= total <= 200_000 and total % 1000 == 0


def _resolve_rec_model_path(explicit_path: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())

    try:
        spec = importlib.util.find_spec("rapidocr_onnxruntime")
    except (ImportError, ModuleNotFoundError, ValueError):
        spec = None
    if spec is not None and spec.origin:
        candidates.append(Path(spec.origin).resolve().parent / "models" / "ch_PP-OCRv4_rec_infer.onnx")

    repo_root = Path(__file__).resolve().parents[4]
    candidates.append(
        repo_root
        / "deps"
        / "rapidocr_pillow"
        / "rapidocr_onnxruntime"
        / "models"
        / "ch_PP-OCRv4_rec_infer.onnx"
    )
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.is_file():
            return resolved
    raise FileNotFoundError("ch_PP-OCRv4_rec_infer.onnx")


@lru_cache(maxsize=2)
def _recognizer_for_model(model_path: Path) -> NumericRecognizer:
    try:
        import cv2  # type: ignore[import-not-found]
        import onnxruntime as ort  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError("onnx_ocr_runtime_unavailable") from exc

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    metadata = session.get_modelmeta().custom_metadata_map
    characters = ["blank", *str(metadata.get("character") or "").splitlines(), " "]

    def recognize(crop: Image.Image, _field_name: str) -> tuple[str, float]:
        rgb = np.asarray(crop.convert("RGB"), dtype=np.uint8)
        height, width = rgb.shape[:2]
        resized_width = min(320, max(1, int(round(48 * width / max(1, height)))))
        resized = cv2.resize(rgb, (resized_width, 48), interpolation=cv2.INTER_LINEAR)
        normalized = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
        tensor = np.zeros((1, 3, 48, 320), dtype=np.float32)
        tensor[0, :, :, :resized_width] = normalized.transpose(2, 0, 1)
        prediction = session.run(None, {input_name: tensor})[0][0]
        indices = prediction.argmax(axis=1)
        probabilities = prediction.max(axis=1)
        output: list[str] = []
        confidence_values: list[float] = []
        previous = -1
        for raw_index, probability in zip(indices, probabilities):
            index = int(raw_index)
            if index and index != previous and index < len(characters):
                output.append(characters[index])
                confidence_values.append(float(probability))
            previous = index
        confidence = (
            sum(confidence_values) / len(confidence_values)
            if confidence_values
            else 0.0
        )
        return "".join(output), confidence

    return recognize
