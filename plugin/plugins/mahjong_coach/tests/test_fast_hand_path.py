from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from plugin.plugins.mahjong_coach.perception.fast_hand_path import quick_frame_fingerprint


@pytest.fixture
def original_frame(tmp_path: Path) -> Path:
    path = tmp_path / "frame.png"
    Image.new("RGB", (1920, 1080), (28, 65, 94)).save(path)
    return path


def test_quick_fingerprint_accepts_a_completely_static_table(original_frame: Path) -> None:
    warped = Image.new("RGB", (800, 800), (24, 62, 91))
    first = quick_frame_fingerprint(original_frame, warped_table=warped)
    second = quick_frame_fingerprint(original_frame, first["hashes"], warped_table=warped)

    assert second["action_changed"] is False
    assert second["hand_changed"] is False
    assert second["river_changed"] is False
    assert second["river_changes"] == {
        "self": False,
        "left_opponent": False,
        "top_opponent": False,
        "right_opponent": False,
    }


@pytest.mark.parametrize(
    ("owner", "center"),
    [
        ("self", (0.50, 0.65)),
        ("left_opponent", (0.35, 0.50)),
        ("top_opponent", (0.50, 0.35)),
        ("right_opponent", (0.65, 0.50)),
    ],
)
def test_quick_fingerprint_detects_each_warped_river(
    original_frame: Path,
    owner: str,
    center: tuple[float, float],
) -> None:
    baseline = Image.new("RGB", (800, 800), (24, 62, 91))
    first = quick_frame_fingerprint(original_frame, warped_table=baseline)
    changed = baseline.copy()
    draw = ImageDraw.Draw(changed)
    x, y = int(center[0] * 800), int(center[1] * 800)
    draw.rectangle((x - 22, y - 30, x + 22, y + 30), fill=(238, 234, 218), outline=(213, 129, 22), width=4)

    result = quick_frame_fingerprint(original_frame, first["hashes"], warped_table=changed)

    assert result["river_changed"] is True
    assert result["river_changes"][owner] is True


def test_quick_fingerprint_detects_a_rotated_riichi_tile(original_frame: Path) -> None:
    baseline = Image.new("RGB", (800, 800), (24, 62, 91))
    first = quick_frame_fingerprint(original_frame, warped_table=baseline)
    changed = baseline.copy()
    draw = ImageDraw.Draw(changed)
    draw.rectangle((480, 326, 570, 360), fill=(240, 237, 220), outline=(210, 126, 20), width=4)

    result = quick_frame_fingerprint(original_frame, first["hashes"], warped_table=changed)

    assert result["river_changed"] is True
    assert result["river_changes"]["top_opponent"] is True


def test_quick_fingerprint_fails_open_without_a_warp(original_frame: Path) -> None:
    first = quick_frame_fingerprint(original_frame, warped_table=None)
    second = quick_frame_fingerprint(original_frame, first["hashes"], warped_table=None)

    assert second["action_changed"] is False
    assert second["hand_changed"] is False
    assert second["river_changed"] is True
    assert all(second["river_changes"].values())
