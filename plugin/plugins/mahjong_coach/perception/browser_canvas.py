from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from PIL import Image


_BROWSER_TITLE_FRAGMENTS = (
    "microsoft edge",
    "google chrome",
    "mozilla firefox",
    "chromium",
    "brave",
    "vivaldi",
    "opera",
)
_GAME_ASPECT_RATIO = 16.0 / 9.0


@dataclass(frozen=True)
class BrowserCanvasResult:
    box: tuple[int, int, int, int]
    normalized_box: tuple[float, float, float, float]
    confidence: float
    reason: str = "active_mahjong_canvas"


def is_browser_window(window_title: str = "", app_name: str = "") -> bool:
    """Return whether a bound Mahjong window is hosted by a common browser."""

    haystack = " ".join((str(window_title or ""), str(app_name or ""))).casefold()
    # Some Chromium window titles contain zero-width formatting characters.
    haystack = "".join(character for character in haystack if character.isprintable())
    return any(fragment in haystack for fragment in _BROWSER_TITLE_FRAGMENTS)


def find_browser_game_canvas(
    image: Image.Image,
    *,
    scene_detector: Callable[[Image.Image], Any] | None = None,
) -> BrowserCanvasResult | None:
    """Find the centered 16:9 Mahjong canvas inside a captured browser window.

    Browser chrome and the web client's decorative side bars make the complete
    HWND unsuitable for table geometry. Candidate vertical boundaries come
    from strong horizontal transitions in the captured bitmap; every candidate
    must then pass the same active-table detector used by the normal live loop.
    No browser DOM, extension, or fixed DPI coordinate is involved.
    """

    width, height = image.size
    if width < 640 or height < 360:
        return None
    if scene_detector is None:
        # Keep NumPy/OpenCV-backed perception out of plugin metadata imports.
        from .game_scene import detect_game_scene_image

        scene_detector = detect_game_scene_image

    frame_aspect = width / max(1, height)
    if abs(frame_aspect - _GAME_ASPECT_RATIO) <= 0.08:
        detected, confidence = _evaluate_scene(image, scene_detector)
        if detected and confidence >= 0.94:
            full_box = (0, 0, width, height)
            return BrowserCanvasResult(
                box=full_box,
                normalized_box=(0.0, 0.0, 1.0, 1.0),
                confidence=confidence,
            )

    candidates = _candidate_boxes(image)
    best: BrowserCanvasResult | None = None
    for box in candidates:
        crop = image.crop(box)
        try:
            detected, confidence = _evaluate_scene(crop, scene_detector)
        finally:
            crop.close()
        if not detected:
            continue
        candidate = BrowserCanvasResult(
            box=box,
            normalized_box=_normalize_box(box, image.size),
            confidence=max(0.0, min(1.0, confidence)),
        )
        if best is None or candidate.confidence > best.confidence:
            best = candidate
        # A strong boundary pair plus the full scene detector is enough. This
        # normally accepts the first candidate and keeps discovery below 100ms.
        if candidate.confidence >= 0.95:
            return candidate
    return best


def normalized_box_matches_edges(
    image: Image.Image,
    normalized_box: tuple[float, float, float, float],
) -> bool:
    """Cheaply verify that a cached canvas still follows browser boundaries."""

    _width, height = image.size
    _left, top_ratio, _right, bottom_ratio = normalized_box
    top = round(top_ratio * height)
    bottom = round(bottom_ratio * height)
    scores = _horizontal_edge_scores(image)
    top_edges = _rank_edge_positions(
        scores,
        start=max(1, round(height * 0.05)),
        end=max(2, round(height * 0.48)),
        limit=12,
    )
    bottom_edges = _rank_edge_positions(
        scores,
        start=max(1, round(height * 0.68)),
        end=height - 1,
        limit=12,
    )
    top_ok = top <= 3 or any(abs(top - position) <= 5 for position, _score in top_edges)
    bottom_ok = bottom >= height - 3 or any(
        abs(bottom - position) <= 5 for position, _score in bottom_edges
    )
    return top_ok and bottom_ok


def crop_from_normalized_box(
    image: Image.Image,
    normalized_box: tuple[float, float, float, float],
) -> Image.Image:
    width, height = image.size
    left, top, right, bottom = normalized_box
    box = (
        max(0, min(width - 1, round(left * width))),
        max(0, min(height - 1, round(top * height))),
        max(1, min(width, round(right * width))),
        max(1, min(height, round(bottom * height))),
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        return image.copy()
    return image.crop(box)


def _candidate_boxes(image: Image.Image) -> list[tuple[int, int, int, int]]:
    width, height = image.size
    edge_scores = _horizontal_edge_scores(image)
    top_edges = _rank_edge_positions(
        edge_scores,
        start=max(1, round(height * 0.06)),
        end=max(2, round(height * 0.45)),
        limit=7,
    )
    bottom_edges = _rank_edge_positions(
        edge_scores,
        start=max(1, round(height * 0.72)),
        end=height - 1,
        limit=7,
    )

    ranked: list[tuple[float, tuple[int, int, int, int]]] = []
    # Fullscreen browsers and installed web apps can already be clean 16:9
    # frames, so retain the outer boundary as a candidate as well.
    vertical_pairs = [
        (top, bottom, top_score + bottom_score)
        for top, top_score in top_edges
        for bottom, bottom_score in bottom_edges
    ]
    vertical_pairs.extend(
        (0, bottom, bottom_score) for bottom, bottom_score in bottom_edges
    )
    vertical_pairs.extend(
        (top, height, top_score) for top, top_score in top_edges
    )
    vertical_pairs.append((0, height, 0.0))

    for top, bottom, boundary_score in vertical_pairs:
        canvas_height = int(bottom - top)
        if canvas_height < round(height * 0.45):
            continue
        canvas_width = round(canvas_height * _GAME_ASPECT_RATIO)
        if canvas_width > width or canvas_width < round(width * 0.55):
            continue
        left = (width - canvas_width) // 2
        box = (left, int(top), left + canvas_width, int(bottom))
        # Strong visual boundaries come first; near-maximum content area breaks
        # ties without assuming a fixed browser toolbar height.
        ranked.append((float(boundary_score) + canvas_height / max(1, height), box))

    ranked.sort(key=lambda item: item[0], reverse=True)
    unique: list[tuple[int, int, int, int]] = []
    for _score, box in ranked:
        if any(_boxes_nearly_equal(box, existing) for existing in unique):
            continue
        unique.append(box)
        if len(unique) >= 18:
            break
    return unique


def _evaluate_scene(
    image: Image.Image,
    scene_detector: Callable[[Image.Image], Any],
) -> tuple[bool, float]:
    try:
        result = scene_detector(image)
    except Exception:
        return False, 0.0
    detected = bool(getattr(result, "detected", False))
    confidence = float(getattr(result, "confidence", 0.0) or 0.0)
    table_surface = getattr(result, "table_surface", None)
    warped_image = getattr(table_surface, "warped_image", None)
    if warped_image is not None:
        try:
            warped_image.close()
        except Exception:
            pass
    return detected, confidence


def _horizontal_edge_scores(image: Image.Image) -> list[float]:
    # NumPy is imported only during browser discovery, never while N.E.K.O
    # scans plugin metadata in its long-lived parent process.
    import numpy as np

    sample_width = min(384, image.width)
    sample = image.convert("RGB").resize((sample_width, image.height), Image.Resampling.BILINEAR)
    try:
        pixels = np.asarray(sample, dtype=np.int16)
        differences = np.abs(pixels[1:] - pixels[:-1]).mean(axis=(1, 2))
        return [float(value) for value in differences]
    finally:
        sample.close()


def _rank_edge_positions(
    scores: list[float],
    *,
    start: int,
    end: int,
    limit: int,
) -> list[tuple[int, float]]:
    ranked = sorted(
        (
            (index + 1, float(scores[index]))
            for index in range(max(0, start - 1), min(len(scores), end))
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    selected: list[tuple[int, float]] = []
    for position, score in ranked:
        if any(abs(position - existing) <= 3 for existing, _value in selected):
            continue
        selected.append((position, score))
        if len(selected) >= limit:
            break
    return selected


def _normalize_box(
    box: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    width, height = image_size
    return (
        box[0] / max(1, width),
        box[1] / max(1, height),
        box[2] / max(1, width),
        box[3] / max(1, height),
    )


def _boxes_nearly_equal(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> bool:
    return all(abs(first - second) <= 3 for first, second in zip(left, right))
