from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw

from plugin.plugins.mahjong_coach.perception import yolo26_visible_tiles as visible_tiles
from plugin.plugins.mahjong_coach.perception.table_surface import (
    _order_quad_points,
    _quad_bbox,
    detect_table_surface,
)
from plugin.plugins.mahjong_coach.perception.yolo26_visible_tiles import (
    Yolo26TableStateResult,
    YoloTileDetection,
    _decode_end2end_output,
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
        ],
        image_size=(1920, 1080),
        river_detections=[
            detection("7p", 0.89, [420, 170, 460, 230]),
            detection("2s", 0.90, [250, 325, 290, 375]),
            detection("9m", 0.92, [610, 325, 650, 375]),
            detection("4s", 0.94, [430, 550, 470, 610]),
        ],
        river_image_size=(800, 800),
        min_confidence=0.25,
    )

    assert result["hand_tiles"] == ["1m", "3m"]
    assert result["meld_tiles"] == ["5z", "5z", "5z"]
    assert len(result["melds"]) == 1
    assert result["discard_piles"]["top_opponent"][0]["tile"] == "7p"
    assert result["discard_piles"]["left_opponent"][0]["tile"] == "2s"
    assert result["discard_piles"]["right_opponent"][0]["tile"] == "9m"
    assert result["discard_piles"]["self"][0]["tile"] == "4s"
    assert {item.coordinate_space for item in result["original_detections"]} == {"original_frame"}
    assert {item.coordinate_space for item in result["river_detections"]} == {"warped_table"}


def test_yolo26_postprocess_excludes_outer_visible_tiles_from_river() -> None:
    result = postprocess_yolo26_detections(
        [],
        image_size=(1920, 1080),
        river_detections=[
            detection("5z", 0.95, [60, 60, 100, 140]),
            detection("6p", 0.94, [690, 180, 730, 250]),
            detection("3m", 0.93, [355, 265, 395, 330]),
        ],
        river_image_size=(800, 800),
    )

    assert result["visible_tiles"] == ["3m"]
    assert result["opponent_meld_count"] == 2
    assert result["excluded_visible_count"] == 2
    assert sorted(item.area_kind for item in result["detections"]) == [
        "excluded_table_tile",
        "excluded_table_tile",
        "river",
    ]


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


def test_yolo26_backend_accepts_export_after_model_exists(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "labels.json").write_text('["1m"]', encoding="utf-8")
    (model_dir / "metadata.json").write_text('{"runtime":"onnxruntime","model_file":"model.onnx"}', encoding="utf-8")
    (model_dir / "model.onnx").write_bytes(b"placeholder")

    backend = load_yolo26_backend(model_dir)

    assert backend.available is True
    assert backend.reason == ""
    assert backend.runtime == "onnxruntime"


def test_yolo26_decodes_end_to_end_rows() -> None:
    import numpy as np

    detections = _decode_end2end_output(
        np.array([[[10.0, 20.0, 30.0, 40.0, 0.9, 1.0], [1.0, 2.0, 3.0, 4.0, 0.1, 0.0]]]),
        labels=["1m", "2m"],
        image_size=(100, 100),
        scale=1.0,
        pad_x=0.0,
        pad_y=0.0,
        min_confidence=0.25,
    )

    assert len(detections) == 1
    assert detections[0].tile == "2m"
    assert detections[0].bbox == [10.0, 20.0, 30.0, 40.0]


def test_yolo26_bottom_gap_splits_closed_tiles_from_self_meld() -> None:
    detections = [
        detection(f"{(index % 9) + 1}m", 0.95, [100 + index * 38, 720, 134 + index * 38, 790])
        for index in range(11)
    ]
    detections.extend(
        [
            detection("1s", 0.95, [640, 720, 670, 790]),
            detection("2s", 0.95, [672, 720, 702, 790]),
            detection("3s", 0.95, [704, 720, 734, 790]),
        ]
    )

    result = postprocess_yolo26_detections(detections, image_size=(800, 800))

    assert len(result["hand_tiles"]) == 11
    assert result["meld_tiles"] == ["1s", "2s", "3s"]
    assert len(result["melds"]) == 1


def test_yolo26_groups_four_detected_self_meld_tiles_as_one_kan() -> None:
    detections = [
        detection(f"{(index % 9) + 1}m", 0.95, [100 + index * 38, 720, 134 + index * 38, 790])
        for index in range(11)
    ]
    detections.extend(
        detection(f"{index + 1}s", 0.95, [640 + index * 32, 720, 670 + index * 32, 790])
        for index in range(4)
    )

    result = postprocess_yolo26_detections(detections, image_size=(800, 800))

    assert len(result["hand_tiles"]) == 11
    assert result["meld_tiles"] == ["1s", "2s", "3s", "4s"]
    assert len(result["melds"]) == 1
    assert result["melds"][0]["tiles"] == ["1s", "2s", "3s", "4s"]


def test_yolo26_hand_threshold_accounts_for_recognized_self_melds() -> None:
    result = Yolo26TableStateResult(
        ok=True,
        hand_tiles=["1m"] * 11,
        melds=[{"tiles": ["1s", "2s", "3s"]}],
        confidence=0.95,
        reason="recognized_yolo26_visible_tiles",
    )

    hand = result.to_hand_result(min_hand_tiles=12)

    assert hand.ok is True
    assert hand.reason == "recognized_yolo26_hand"


def test_yolo26_low_right_river_tile_stays_with_right_player() -> None:
    result = postprocess_yolo26_detections(
        [],
        image_size=(1920, 1080),
        river_detections=[detection("2m", 0.95, [530, 448, 561, 488])],
        river_image_size=(800, 800),
    )

    assert result["discard_piles"]["right_opponent"][0]["tile"] == "2m"


def test_yolo26_detects_right_player_riichi_declaration_by_orientation() -> None:
    detections = [
        detection(f"{index + 1}p", 0.95, [500, 300 + index * 32, 540, 330 + index * 32])
        for index in range(5)
    ]
    detections.append(detection("2m", 0.95, [546, 430, 576, 472]))

    result = postprocess_yolo26_detections(
        [],
        image_size=(1920, 1080),
        river_detections=detections,
        river_image_size=(800, 800),
    )

    assert result["riichi_players"] == ["right_opponent"]


def test_yolo26_estimates_shifted_river_center_instead_of_fixed_polygons() -> None:
    result = postprocess_yolo26_detections(
        [],
        image_size=(1920, 1080),
        river_detections=[
            detection("1m", 0.95, [430, 175, 470, 225]),
            detection("2m", 0.95, [280, 320, 320, 370]),
            detection("3m", 0.95, [590, 320, 630, 370]),
            detection("4m", 0.95, [430, 535, 470, 585]),
        ],
        river_image_size=(800, 800),
    )

    center_x, center_y = result["river_center"]
    assert center_x == 0.5687
    assert center_y == 0.475
    assert {owner: pile[0]["tile"] for owner, pile in result["discard_piles"].items()} == {
        "self": "4m",
        "right_opponent": "3m",
        "top_opponent": "1m",
        "left_opponent": "2m",
    }


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


def test_yolo26_runtime_runs_original_hand_and_warped_river_as_separate_passes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (1920, 1080), "black").save(image_path)
    calls: list[tuple[int, int]] = []

    class FakeBackend:
        available = True
        reason = ""
        runtime = "test"

        def detect(self, image: Image.Image, *, min_confidence: float) -> list[YoloTileDetection]:
            calls.append(image.size)
            if image.size == (1920, 1080):
                return [
                    detection(f"{(index % 9) + 1}m", 0.95, [100 + index * 70, 850, 150 + index * 70, 1010])
                    for index in range(14)
                ]
            return [
                detection("1m", 0.95, [430, 175, 470, 225]),
                detection("2m", 0.95, [280, 320, 320, 370]),
                detection("3m", 0.95, [590, 320, 630, 370]),
                detection("4m", 0.95, [430, 535, 470, 585]),
            ]

    surface = SimpleNamespace(
        ok=True,
        reason="table_surface_detected",
        warped_image=Image.new("RGB", (800, 800), "black"),
        diagnostics={},
        to_hints=lambda: {"table_surface_ok": True},
    )
    monkeypatch.setattr(visible_tiles, "load_yolo26_backend", lambda _model_dir: FakeBackend())
    monkeypatch.setattr(visible_tiles, "detect_table_surface", lambda *_args, **_kwargs: surface)

    result = detect_yolo26_table_state_path(image_path, model_dir=tmp_path / "model")

    assert calls == [(1920, 1080), (800, 800)]
    assert result.original_inference_ok is True
    assert result.river_inference_ok is True
    assert len(result.hand_tiles) == 14
    assert set(result.discard_piles) == {"self", "right_opponent", "top_opponent", "left_opponent"}
    assert {item["coordinate_space"] for item in result.raw_detections} == {"original_frame", "warped_table"}


def test_yolo26_table_warp_failure_keeps_original_hand_and_marks_river_unavailable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (1280, 720), "black").save(image_path)

    class FakeBackend:
        available = True
        reason = ""
        runtime = "test"

        def detect(self, _image: Image.Image, *, min_confidence: float) -> list[YoloTileDetection]:
            return [
                detection(f"{(index % 9) + 1}m", 0.95, [80 + index * 60, 570, 125 + index * 60, 690])
                for index in range(14)
            ]

    surface = SimpleNamespace(
        ok=False,
        reason="table_surface_not_found",
        warped_image=None,
        diagnostics={},
        to_hints=lambda: {"table_surface_ok": False},
    )
    monkeypatch.setattr(visible_tiles, "load_yolo26_backend", lambda _model_dir: FakeBackend())
    monkeypatch.setattr(visible_tiles, "detect_table_surface", lambda *_args, **_kwargs: surface)

    result = detect_yolo26_table_state_path(image_path, model_dir=tmp_path / "model")

    assert result.ok is True
    assert result.original_inference_ok is True
    assert result.river_inference_ok is False
    assert result.to_hand_result(min_hand_tiles=12).ok is True
    assert result.to_river_result().ok is False
    assert result.to_river_result().reason == "table_surface_not_found"
