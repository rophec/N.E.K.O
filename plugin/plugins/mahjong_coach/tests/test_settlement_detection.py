from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageChops, ImageDraw

from plugin.plugins.mahjong_coach.coach import RoundCoachEngine
from plugin.plugins.mahjong_coach.models import MahjongCoachConfig
from plugin.plugins.mahjong_coach.perception.fast_hand_path import FastHandResult
from plugin.plugins.mahjong_coach.perception.meld_state import MeldStateResult
from plugin.plugins.mahjong_coach.perception.settlement_detector import (
    SettlementFrameResult,
    SettlementTracker,
    SettlementTransition,
    detect_settlement_image,
    detect_settlement_path,
    render_settlement_diagnostic_image,
)


OLD_HAND = ["1m", "2m", "3m", "4p", "5p", "6p", "2s", "3s", "4s", "5s", "6s", "7s", "1z"]
NEW_HAND = ["2m", "3m", "4m", "2p", "3p", "4p", "6s", "7s", "8s", "5z", "5z", "6z", "7z"]


def test_settlement_config_defaults_and_bounds() -> None:
    default = MahjongCoachConfig.from_payload({})
    configured = MahjongCoachConfig.from_payload(
        {
            "perception": {
                "settlement_recognition_enabled": False,
                "settlement_min_confidence": 2.0,
                "settlement_confirm_frames": 0,
                "settlement_confirm_max_gap_ms": 99_999,
            }
        }
    )

    assert default.settlement_recognition_enabled is True
    assert default.settlement_min_confidence == pytest.approx(0.72)
    assert default.settlement_confirm_frames == 2
    assert default.settlement_confirm_max_gap_ms == 2500
    assert configured.settlement_recognition_enabled is False
    assert configured.settlement_min_confidence == pytest.approx(1.0)
    assert configured.settlement_confirm_frames == 2
    assert configured.settlement_confirm_max_gap_ms == 10_000


def test_visual_detector_accepts_win_overlay(tmp_path: Path) -> None:
    image_path = tmp_path / "win-settlement.png"
    _synthetic_win_settlement().save(image_path)

    result = detect_settlement_path(image_path)

    assert result.detected is True
    assert result.kind == "win"
    assert result.confidence >= 0.72
    assert "long_diagonal_result_border" in result.evidence
    assert len(result.metrics["result_diagonal_line"]) == 4
    assert result.metrics["upper_tile_components"]


def test_settlement_diagnostic_draws_regions_and_evidence() -> None:
    image = _synthetic_win_settlement()
    result = detect_settlement_image(image)

    diagnostic = render_settlement_diagnostic_image(image, result=result)
    difference = ImageChops.difference(image, diagnostic)

    assert diagnostic.size == image.size
    assert difference.getbbox() is not None


@pytest.mark.parametrize("kind", ["normal_table", "menu"])
def test_visual_detector_rejects_non_settlement_frames(tmp_path: Path, kind: str) -> None:
    image_path = tmp_path / f"{kind}.png"
    image = _synthetic_normal_table()
    if kind == "menu":
        draw = ImageDraw.Draw(image)
        draw.rectangle((360, 120, 920, 600), fill=(226, 229, 232), outline=(40, 45, 54), width=8)
        draw.rectangle((430, 210, 850, 285), fill=(70, 79, 92))
        draw.rectangle((430, 330, 850, 405), fill=(70, 79, 92))
    image.save(image_path)

    result = detect_settlement_path(image_path)

    assert result.detected is False
    assert result.kind == "none"


def test_settlement_tracker_confirms_latches_and_waits() -> None:
    tracker = SettlementTracker(confirm_frames=2)
    hit = _settlement_hit()
    miss = SettlementFrameResult(reason="not_detected")

    first = tracker.observe(hit, round_active=True)
    second = tracker.observe(hit, round_active=True)
    waiting = tracker.observe(miss, round_active=True)
    visible_again = tracker.observe(hit, round_active=True)

    assert first.phase == "settlement_candidate"
    assert first.confirmation_frames == 1
    assert second.phase == "settlement_latched"
    assert second.confirmation_frames == 2
    assert waiting.phase == "awaiting_next_round"
    assert waiting.result.kind == "win"
    assert visible_again.phase == "settlement_latched"


def test_settlement_tracker_does_not_invent_previous_round() -> None:
    tracker = SettlementTracker(confirm_frames=2)

    transition = tracker.observe(_settlement_hit(), round_active=False)

    assert transition.phase == "playing"
    assert transition.confirmation_frames == 0


def test_settlement_tracker_restarts_candidate_after_time_gap() -> None:
    tracker = SettlementTracker(confirm_frames=2, confirm_max_gap_ms=500)
    hit = _settlement_hit()

    first = tracker.observe(hit, round_active=True, observed_at=10.0)
    expired = tracker.observe(hit, round_active=True, observed_at=10.7)
    confirmed = tracker.observe(hit, round_active=True, observed_at=10.9)

    assert first.phase == "settlement_candidate"
    assert expired.phase == "settlement_candidate"
    assert expired.confirmation_frames == 1
    assert expired.last_frame_gap_ms == pytest.approx(700.0)
    assert confirmed.phase == "settlement_latched"
    assert confirmed.confirmation_frames == 2
    assert confirmed.confirmation_elapsed_ms == pytest.approx(200.0)


def test_engine_freezes_old_round_and_reopens_after_two_stable_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = RoundCoachEngine(MahjongCoachConfig())
    engine.state.opening_emitted = True
    engine.state.round_id = "old-round"
    engine.state.round_phase = "normal_tracking"
    engine.state.last_update_reason = "scheduled_checkpoint"
    engine.state.current_plan = "keep-old-plan"
    engine.state.last_hand_tiles = list(OLD_HAND)
    engine.state.last_hand_confidence = 0.94
    engine.state.last_discard_piles = {
        "self": [{"tile": "1m", "turn_index": 1}],
        "right_opponent": [{"tile": "9p", "turn_index": 1}],
    }
    frame_results = iter(
        [
            _settlement_hit(),
            _settlement_hit(),
            SettlementFrameResult(reason="settlement_gone"),
            SettlementFrameResult(reason="settlement_gone"),
        ]
    )
    monkeypatch.setattr(engine, "_detect_settlement", lambda _path: next(frame_results))
    monkeypatch.setattr(
        engine,
        "_detect_hand",
        lambda _path: FastHandResult(
            ok=True,
            hand_tiles=list(NEW_HAND),
            confidence=0.97,
            reason="stable_new_hand",
        ),
    )
    monkeypatch.setattr(
        engine,
        "_detect_melds",
        lambda _path, hand_result=None: MeldStateResult(
            reason="closed_hand_count_no_melds",
        ),
    )

    candidate = engine.analyze_frame("frame.png")
    settled = engine.analyze_frame("frame.png")
    waiting = engine.analyze_frame("frame.png")

    assert candidate.decision_type == "settlement_candidate"
    assert settled.decision_type == "round_settlement"
    assert waiting.decision_type == "awaiting_next_round"
    assert engine.state.current_plan == "keep-old-plan"
    assert engine.state.last_hand_tiles == OLD_HAND
    assert sum(len(items) for items in engine.state.last_discard_piles.values()) == 2
    assert settled.perception["settlement"]["kind"] == "win"
    assert settled.perception["settlement"]["round_archive_id"] == "round-archive-1"
    assert waiting.perception["settlement"]["new_hand_confirmation_frames"] == 1

    archived = engine.last_round_archive
    assert archived["round_id"] == "old-round"
    assert archived["settlement"]["kind"] == "win"
    assert archived["state"]["round_phase"] == "normal_tracking"
    assert archived["state"]["current_plan"] == "keep-old-plan"
    assert archived["state"]["last_hand_tiles"] == OLD_HAND
    assert archived["state"]["last_discard_piles"] == {
        "self": [{"tile": "1m", "turn_index": 1}],
        "right_opponent": [{"tile": "9p", "turn_index": 1}],
    }

    reopened = engine.analyze_frame("frame.png")

    assert reopened.decision_type == "opening_plan"
    assert "settlement_closed" in reopened.reason_codes
    assert "auto_new_round_detected" in reopened.reason_codes
    assert reopened.engine_meta["round_transition"] == "settlement_new_hand_confirmed"
    assert engine.state.round_id == "auto-settlement-round-1"
    assert engine.state.last_hand_tiles == NEW_HAND
    assert engine.state.last_discard_piles == {}
    assert engine.state.settlement_phase == "playing"
    assert reopened.engine_meta["previous_round_archive_id"] == "round-archive-1"
    assert len(engine.round_history) == 1

    archived["state"]["last_hand_tiles"].clear()
    assert engine.last_round_archive["state"]["last_hand_tiles"] == OLD_HAND


def test_round_history_keeps_only_latest_two_rounds() -> None:
    engine = RoundCoachEngine(MahjongCoachConfig())
    transition = SettlementTransition(
        phase="settlement_latched",
        result=_settlement_hit(),
    )

    for index in range(4):
        engine.state.round_id = f"round-{index}"
        engine.state.last_hand_tiles = [f"{index % 9 + 1}m"]
        engine._archive_current_round(transition)
        engine.reset_round(f"next-{index}")

    history = engine.round_history
    assert len(history) == 2
    assert history[0]["round_id"] == "round-2"
    assert history[-1]["round_id"] == "round-3"
    assert history[-1]["archive_id"] == "round-archive-4"


def _settlement_hit() -> SettlementFrameResult:
    return SettlementFrameResult(
        detected=True,
        kind="win",
        confidence=0.93,
        reason="win_settlement_signature",
        evidence=[
            "dimmed_table_and_lower_screen",
            "large_blue_result_panel",
            "long_diagonal_result_border",
        ],
        elapsed_ms=4.2,
    )


def _synthetic_normal_table() -> Image.Image:
    image = Image.new("RGB", (1280, 720), (25, 74, 118))
    draw = ImageDraw.Draw(image)
    for index in range(13):
        x = 410 + index * 36
        draw.rounded_rectangle((x, 25, x + 31, 82), radius=3, fill=(206, 121, 24))
    for index in range(13):
        x = 320 + index * 48
        draw.rounded_rectangle((x, 620, x + 43, 705), radius=3, fill=(226, 225, 215))
    draw.rectangle((510, 250, 770, 500), fill=(34, 52, 74), outline=(12, 22, 36), width=8)
    return image


def _synthetic_win_settlement() -> Image.Image:
    image = Image.new("RGB", (1280, 720), (12, 17, 30))
    draw = ImageDraw.Draw(image)
    draw.polygon(
        [(30, 170), (1280, 0), (1280, 510), (30, 650)],
        fill=(38, 83, 126),
    )
    draw.line((40, 622, 1279, 485), fill=(207, 92, 32), width=9)
    for index in range(14):
        x = 430 + index * 51
        y = 55 - int(index * 4.5)
        draw.rounded_rectangle((x, y, x + 45, y + 78), radius=3, fill=(235, 234, 226))
        draw.rectangle((x + 8, y + 15, x + 35, y + 56), fill=(48, 103, 72))
    for index in range(4):
        draw.rectangle((680, 230 + index * 48, 925, 252 + index * 48), fill=(232, 228, 208))
    draw.rectangle((970, 580, 1215, 685), fill=(28, 76, 132), outline=(211, 144, 45), width=6)
    return image
