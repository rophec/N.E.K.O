from __future__ import annotations

from PIL import Image, ImageDraw

from plugin.plugins.mahjong_coach.perception import game_scene
from plugin.plugins.mahjong_coach.perception.game_scene import detect_game_scene_image
from plugin.plugins.mahjong_coach.perception.table_surface import TableSurfaceResult


def _scene_image(*, center_panel: bool = True) -> Image.Image:
    image = Image.new("RGB", (1000, 600), (26, 65, 105))
    if not center_panel:
        return image
    draw = ImageDraw.Draw(image)
    draw.rectangle((420, 215, 580, 385), fill=(55, 57, 66))
    draw.rectangle((470, 260, 495, 272), fill=(20, 210, 220))
    draw.rectangle((505, 305, 545, 325), fill=(240, 175, 30))
    return image


def test_game_scene_requires_center_score_panel_even_when_outer_table_exists(monkeypatch) -> None:
    monkeypatch.setattr(
        game_scene,
        "detect_table_surface",
        lambda _image: (_ for _ in ()).throw(AssertionError("outer table scan should be skipped")),
    )

    result = detect_game_scene_image(_scene_image(center_panel=False))

    assert result.detected is False
    assert result.reason == "center_score_panel_not_found"


def test_game_scene_requires_outer_table_after_center_signature(monkeypatch) -> None:
    monkeypatch.setattr(
        game_scene,
        "detect_table_surface",
        lambda _image: TableSurfaceResult(reason="table_surface_support_lines_not_found"),
    )

    result = detect_game_scene_image(_scene_image())

    assert result.detected is False
    assert result.reason == "table_surface_support_lines_not_found"
    assert "center_cyan_round_text" in result.evidence


def test_game_scene_accepts_center_panel_and_high_quality_table(monkeypatch) -> None:
    monkeypatch.setattr(
        game_scene,
        "detect_table_surface",
        lambda _image: TableSurfaceResult(
            ok=True,
            reason="table_surface_detected",
            method="support_lines",
            quality_score=0.9,
            quad_area_ratio=0.86,
        ),
    )

    result = detect_game_scene_image(_scene_image())

    assert result.detected is True
    assert result.reason == "active_mahjong_table"
    assert result.confidence >= 0.8
    assert result.evidence == [
        "center_cyan_round_text",
        "center_gold_score_text",
        "center_dark_score_panel",
        "table_surface_detected",
    ]
