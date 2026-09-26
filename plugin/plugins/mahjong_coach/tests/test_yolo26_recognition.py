from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from plugin.plugins.mahjong_coach.perception.table_surface import (
    _order_quad_points,
    _quad_bbox,
    detect_table_surface,
)
from plugin.plugins.mahjong_coach.perception.yolo26_visible_tiles import (
    YoloTileDetection,
    detect_yolo26_table_state_path,
    load_yolo26_backend,
    postprocess_yolo26_detections,
)


def detection(tile: str, confidence: float, bbox: list[float]) -> YoloTileDetection:
    return YoloTileDetection(tile=tile, confidence=confidence, bbox=bbox, source="test_yolo26")


def test_yolo26_postprocess_splits_hand_meld_and_river() -> None:
    result = postprocess_yolo26_detections(
        [
            detection("3m", 0.91, [220, 820, 260, 900]),
            detection("1m", 0.93, [120, 820, 160, 900]),
            detection("5z", 0.88, [1480, 690, 1520, 760]),
            detection("5z", 0.87, [1530, 690, 1570, 760]),
            detection("5z", 0.86, [1580, 690, 1620, 760]),
            detection("7p", 0.89, [880, 430, 920, 500]),
            detection("2s", 0.90, [300, 390, 340, 460]),
            detection("9m", 0.92, [1500, 390, 1540, 460]),
        ],
        image_size=(1920, 1080),
        min_confidence=0.25,
    )

    assert result["hand_tiles"] == ["1m", "3m"]
    assert result["meld_tiles"] == ["5z", "5z", "5z"]
    assert len(result["melds"]) == 1
    assert result["discard_piles"]["top_opponent"][0]["tile"] == "7p"
    assert result["discard_piles"]["left_opponent"][0]["tile"] == "2s"
    assert result["discard_piles"]["right_opponent"][0]["tile"] == "9m"


def test_yolo26_postprocess_filters_low_confidence_and_dedupes() -> None:
    result = postprocess_yolo26_detections(
        [
            detection("1m", 0.95, [100, 850, 150, 930]),
            detection("2m", 0.80, [104, 852, 154, 932]),
            detection("3m", 0.10, [220, 850, 270, 930]),
            detection("empty", 0.99, [320, 850, 370, 930]),
        ],
        image_size=(1920, 1080),
        min_confidence=0.25,
    )

    assert result["hand_tiles"] == ["1m"]
    assert len(result["detections"]) == 1


def test_yolo26_backend_reports_missing_model_dir(tmp_path: Path) -> None:
    backend = load_yolo26_backend(tmp_path / "missing")

    assert backend.available is False
    assert backend.reason == "yolo26_model_dir_missing"


def test_yolo26_backend_requires_decoder_after_model_exists(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "labels.json").write_text('["1m"]', encoding="utf-8")
    (model_dir / "metadata.json").write_text('{"runtime":"onnxruntime","model_file":"model.onnx"}', encoding="utf-8")
    (model_dir / "model.onnx").write_bytes(b"placeholder")

    backend = load_yolo26_backend(model_dir)

    assert backend.available is False
    assert backend.reason == "yolo26_decoder_unimplemented"


def test_table_surface_orders_quad_and_bbox() -> None:
    quad = _order_quad_points([[500, 400], [100, 100], [520, 120], [80, 420]])

    assert quad == [[100.0, 100.0], [520.0, 120.0], [80.0, 420.0], [500.0, 400.0]]
    assert _quad_bbox(quad) == [80.0, 100.0, 520.0, 420.0]


def test_table_surface_detects_synthetic_table() -> None:
    image = Image.new("RGB", (640, 360), "black")
    draw = ImageDraw.Draw(image)
    draw.polygon([(90, 50), (550, 40), (610, 315), (30, 320)], fill=(38, 95, 155))

    result = detect_table_surface(image)

    assert result.ok is True
    assert result.reason == "table_surface_detected"
    assert result.quad
    assert result.warped_size == (800, 800)


def test_yolo26_missing_backend_still_reports_table_surface_hints(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (320, 180), "black").save(image_path)

    result = detect_yolo26_table_state_path(image_path, model_dir=tmp_path / "missing-model")

    assert result.ok is False
    assert result.reason == "yolo26_model_dir_missing"
    assert "table_surface_ok" in result.analysis_hints
