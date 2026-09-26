from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from plugin.plugins.mahjong_coach import (
    MahjongCoachPlugin,
    _build_frame_preview_payload,
    _build_settlement_diagnostic_preview_payload,
    _build_table_region_preview_payload,
)
from plugin.plugins.mahjong_coach import coach as coach_module
from plugin.plugins.mahjong_coach.coach import (
    ORPHAN_TYPES,
    RoundCoachEngine,
    _efficiency_analysis,
    _riichi_waits_after_discard,
    _visible_counts,
    analyze_call_options,
    build_round_plan,
    rank_discard_decisions,
)
from plugin.plugins.mahjong_coach.models import LiveSessionState, MahjongCoachConfig
from plugin.plugins.mahjong_coach.overlay import _overlay_geometry, overlay_detail_text_from_payload, overlay_text_from_payload
from plugin.plugins.mahjong_coach.perception.fast_hand_path import FastHandResult
from plugin.plugins.mahjong_coach.perception.meld_state import MeldStateResult
from plugin.plugins.mahjong_coach.perception.river_state import RiverStateResult
from plugin.plugins.mahjong_coach.perception.settlement_detector import (
    SettlementFrameResult,
    SettlementTransition,
)
from plugin.plugins.mahjong_coach.perception.yolo26_visible_tiles import Yolo26TableStateResult
from plugin.plugins.mahjong_coach.tile_labels import hand_signature


HAND = ["1m", "2m", "3m", "4p", "5p", "6p", "2s", "3s", "4s", "5s", "6s", "7s", "1z", "1z"]


def test_frame_preview_payload_is_compact_jpeg(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (1920, 1080), (20, 54, 83)).save(image_path)

    payload = _build_frame_preview_payload(image_path)

    assert payload["image_path"] == str(image_path)
    assert payload["data_url"].startswith("data:image/jpeg;base64,")
    assert payload["width"] == 960
    assert payload["height"] == 540


def test_settlement_diagnostic_preview_is_in_memory_jpeg(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (1920, 1080), (20, 54, 83)).save(image_path)

    payload = _build_settlement_diagnostic_preview_payload(image_path)

    assert payload["image_path"] == str(image_path)
    assert payload["data_url"].startswith("data:image/jpeg;base64,")
    assert payload["width"] == 960
    assert payload["height"] == 540
    assert payload["detected"] is False
    assert payload["reason"]


def test_table_region_preview_uses_warped_table_space(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    image = Image.new("RGB", (640, 360), "black")
    draw = ImageDraw.Draw(image)
    draw.polygon([(90, 50), (550, 40), (610, 315), (30, 320)], fill=(38, 95, 155))
    image.save(image_path)

    payload = _build_table_region_preview_payload(
        image_path,
        raw_detections=[
            {
                "tile": "3m",
                "confidence": 0.94,
                "bbox": [350.0, 260.0, 390.0, 330.0],
                "area_kind": "river",
                "owner": "top_opponent",
            }
        ],
    )

    assert payload["transformed"] is True
    assert payload["input_space"] == "warped_table"
    assert payload["width"] == 800
    assert payload["height"] == 800
    assert payload["detection_count"] == 1
    assert payload["data_url"].startswith("data:image/jpeg;base64,")


def test_runtime_settlement_config_is_bounded() -> None:
    plugin = MahjongCoachPlugin.__new__(MahjongCoachPlugin)
    plugin._cfg = MahjongCoachConfig()
    plugin._engine = RoundCoachEngine(plugin._cfg)
    plugin.logger = SimpleNamespace(info=lambda *_args, **_kwargs: None)

    plugin._apply_runtime_settlement_config(
        enabled=False,
        min_confidence=3.0,
        confirm_frames=99,
        confirm_max_gap_ms=10,
    )

    assert plugin._cfg.settlement_recognition_enabled is False
    assert plugin._cfg.settlement_min_confidence == 1.0
    assert plugin._cfg.settlement_confirm_frames == 8
    assert plugin._cfg.settlement_confirm_max_gap_ms == 200
    assert plugin._engine.config == plugin._cfg


@pytest.mark.asyncio
async def test_status_exposes_complete_round_archive() -> None:
    plugin = MahjongCoachPlugin.__new__(MahjongCoachPlugin)
    plugin.ctx = SimpleNamespace(plugin_id="mahjong_coach")
    plugin._cfg = MahjongCoachConfig()
    plugin._engine = RoundCoachEngine(plugin._cfg)
    plugin._engine.state.round_id = "archived-round"
    plugin._engine.state.last_hand_tiles = ["1m", "2m", "3m"]
    plugin._engine._archive_current_round(
        SettlementTransition(
            phase="settlement_latched",
            result=SettlementFrameResult(detected=True, kind="win", confidence=0.91),
        )
    )
    plugin._last_decision = {}
    plugin._live_state = LiveSessionState()
    plugin._live_timing_log = []

    payload = (await plugin.mahjong_coach_status()).unwrap()

    assert len(payload["round_history"]) == 1
    assert payload["last_round_archive"]["round_id"] == "archived-round"
    assert payload["last_round_archive"]["state"]["last_hand_tiles"] == ["1m", "2m", "3m"]


@pytest.mark.asyncio
async def test_status_panel_reflects_two_frame_river_correction() -> None:
    plugin = MahjongCoachPlugin.__new__(MahjongCoachPlugin)
    plugin.ctx = SimpleNamespace(plugin_id="mahjong_coach")
    plugin._cfg = MahjongCoachConfig(river_tracking_mode="live", tile_recognition_mode="yolo26")
    plugin._engine = RoundCoachEngine(plugin._cfg)
    plugin._last_decision = {}
    plugin._live_state = LiveSessionState()
    plugin._live_timing_log = []

    # 中文：模拟错误的 4p 已经进入后端状态和网页面板。
    # English: Simulate a wrong 4p that has already reached backend state and the web panel.
    plugin._engine.state.last_discard_piles = {
        "right_opponent": [
            {
                "tile": "4p",
                "player": "right_opponent",
                "turn_index": 1,
                "bbox": [500, 300, 540, 360],
                "confidence": 0.61,
                "source": "injected_false_river",
            }
        ]
    }
    plugin._engine.state.last_visible_discards = ["4p"]
    plugin._engine.state.river_tracking_initialized = True

    injected_payload = (await plugin.mahjong_coach_status()).unwrap()
    assert injected_payload["round_state"]["last_discard_piles"]["right_opponent"][0]["tile"] == "4p"

    corrected_snapshot = RiverStateResult(
        ok=True,
        discard_piles={
            "right_opponent": [
                {
                    "tile": "2p",
                    "player": "right_opponent",
                    "turn_index": 1,
                    "bbox": [502, 302, 542, 362],
                    "confidence": 0.94,
                    "source": "test_yolo26",
                }
            ]
        },
        visible_tiles=["2p"],
        confidence=0.94,
        reason="recognized_yolo26_discards",
    )

    first = plugin._engine._reconcile_full_river(corrected_snapshot)
    plugin._engine._remember_river(first)
    plugin._last_decision = {"perception": {"river": first.to_dict()}}
    pending_payload = (await plugin.mahjong_coach_status()).unwrap()
    assert pending_payload["round_state"]["last_discard_piles"]["right_opponent"][0]["tile"] == "4p"
    assert pending_payload["last_decision"]["perception"]["river"]["analysis_hints"][
        "river_pending_corrections"
    ] == 1

    second = plugin._engine._reconcile_full_river(corrected_snapshot)
    plugin._engine._remember_river(second)
    plugin._last_decision = {"perception": {"river": second.to_dict()}}
    corrected_payload = (await plugin.mahjong_coach_status()).unwrap()
    corrected_item = corrected_payload["round_state"]["last_discard_piles"]["right_opponent"][0]
    correction_hints = corrected_payload["last_decision"]["perception"]["river"]["analysis_hints"]

    assert corrected_item["tile"] == "2p"
    assert corrected_item["corrected_from"] == "4p"
    assert correction_hints["river_corrected_count"] == 1
    assert correction_hints["river_correction_events"] == [
        {
            "player": "right_opponent",
            "turn_index": 1,
            "from": "4p",
            "to": "2p",
            "confidence": 0.94,
        }
    ]


def test_opening_scan_ignores_impossible_buttons_and_uses_checkpoint_river_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = RoundCoachEngine(MahjongCoachConfig())
    monkeypatch.setattr(
        engine,
        "_detect_hand",
        lambda _path: FastHandResult(ok=True, hand_tiles=list(HAND), confidence=0.91, reason="test_hand"),
    )
    monkeypatch.setattr(
        engine,
        "_resolve_buttons",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("opening scan should not inspect buttons")),
    )
    monkeypatch.setattr(
        engine,
        "_detect_river",
        lambda _path: (_ for _ in ()).throw(AssertionError("checkpoint river mode should not scan river during opening")),
    )

    decision = engine.analyze_frame("frame.png", observed_buttons=["ron", "riichi"])

    assert decision.decision_type == "opening_plan"
    assert decision.perception["action"]["source"] == "opening_hand_scan"
    assert decision.perception["river"]["reason"] == "opening_skips_river_scan"
    assert decision.coach_state["round_phase"] == "opening_strategy"


def test_live_river_mode_tracks_river_during_opening(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = RoundCoachEngine(MahjongCoachConfig(river_tracking_mode="live"))
    river_calls: list[Path | None] = []
    monkeypatch.setattr(
        engine,
        "_detect_hand",
        lambda _path: FastHandResult(ok=True, hand_tiles=list(HAND), confidence=0.91, reason="test_hand"),
    )
    monkeypatch.setattr(
        engine,
        "_resolve_buttons",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("opening scan should not inspect buttons")),
    )
    monkeypatch.setattr(
        engine,
        "_detect_river",
        lambda path: river_calls.append(path)
        or RiverStateResult(ok=True, visible_tiles=["1m"], confidence=0.95, reason="recognized_discards"),
    )

    decision = engine.analyze_frame("frame.png", observed_buttons=["ron", "riichi"])

    assert decision.decision_type == "opening_plan"
    assert decision.perception["river"]["reason"] == "recognized_discards"
    assert decision.engine_meta["river_tracking_mode"] == "live"
    assert decision.coach_state["last_visible_discards"] == ["1m"]
    assert len(river_calls) == 1


def test_fast_style_after_opening_uses_open_hand_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = RoundCoachEngine(MahjongCoachConfig(play_style="fast"))
    engine.state.opening_emitted = True
    seen: dict[str, int] = {}

    def fake_detect(
        _path: Path,
        *,
        calibration_dir=None,
        min_hand_tiles: int = 12,
        max_hand_tiles: int = 14,
        use_onnx_hand: bool | None = None,
    ) -> FastHandResult:
        seen["min_hand_tiles"] = min_hand_tiles
        seen["max_hand_tiles"] = max_hand_tiles
        return FastHandResult(ok=True, hand_tiles=["1m", "2m", "3m", "4p"], confidence=0.77, reason="test_open_hand")

    monkeypatch.setattr("plugin.plugins.mahjong_coach.coach.detect_fast_hand_path", fake_detect)

    result = engine._detect_hand(Path("frame.png"))

    assert result.ok is True
    assert seen["min_hand_tiles"] == 4
    assert seen["max_hand_tiles"] == 14


def test_fast_style_late_open_hand_can_track_two_tiles(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = RoundCoachEngine(MahjongCoachConfig(play_style="fast"))
    engine.state.opening_emitted = True
    engine.state.last_hand_tiles = ["1m", "2m", "3m", "4p", "5p"]
    seen: dict[str, int] = {}

    def fake_detect(
        _path: Path,
        *,
        calibration_dir=None,
        min_hand_tiles: int = 12,
        max_hand_tiles: int = 14,
        use_onnx_hand: bool | None = None,
    ) -> FastHandResult:
        seen["min_hand_tiles"] = min_hand_tiles
        return FastHandResult(ok=True, hand_tiles=["1m", "2m"], confidence=0.64, reason="test_late_open_hand")

    monkeypatch.setattr("plugin.plugins.mahjong_coach.coach.detect_fast_hand_path", fake_detect)

    result = engine._detect_hand(Path("frame.png"))

    assert result.ok is True
    assert seen["min_hand_tiles"] == 2


def test_fast_style_opening_still_requires_full_starting_hand(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = RoundCoachEngine(MahjongCoachConfig(play_style="fast"))
    seen: dict[str, int] = {}

    def fake_detect(
        _path: Path,
        *,
        calibration_dir=None,
        min_hand_tiles: int = 12,
        max_hand_tiles: int = 14,
        use_onnx_hand: bool | None = None,
    ) -> FastHandResult:
        seen["min_hand_tiles"] = min_hand_tiles
        return FastHandResult(reason="unstable_hand_count")

    monkeypatch.setattr("plugin.plugins.mahjong_coach.coach.detect_fast_hand_path", fake_detect)

    result = engine._detect_hand(Path("frame.png"))

    assert result.ok is False
    assert seen["min_hand_tiles"] == 12


def test_opening_accepts_plausible_open_hand_when_meld_scan_misses(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = RoundCoachEngine(MahjongCoachConfig())
    open_hand = ["1m", "1m", "2m", "3m", "6m", "6m", "4s", "0s", "8s", "9s", "5z"]

    monkeypatch.setattr(
        engine,
        "_detect_hand",
        lambda _path: FastHandResult(
            ok=False,
            hand_tiles=list(open_hand),
            confidence=0.91,
            reason="unstable_hand_count",
            raw_detections=[{"accepted": True, "occupied": True} for _ in open_hand],
        ),
    )
    monkeypatch.setattr(
        engine,
        "_detect_melds",
        lambda _path, **_kwargs: MeldStateResult(reason="no_self_melds"),
    )
    monkeypatch.setattr(engine, "_detect_riichi_players", lambda _path: [])

    decision = engine.analyze_frame("frame.png")

    assert decision.decision_type == "opening_plan"
    assert decision.hand_tiles == open_hand
    assert decision.perception["hand"]["reason"] == "inferred_open_11_hand_tiles"
    assert decision.coach_state["last_open_meld_count"] == 1
    assert decision.coach_state["round_phase"] == "opening_strategy"


def test_win_window_interrupts_before_hand_scan_after_opening(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = RoundCoachEngine(MahjongCoachConfig())
    engine.state.opening_emitted = True

    def fail_if_called(_path: Path | None) -> FastHandResult:
        raise AssertionError("hand scan should not run for win windows")

    monkeypatch.setattr(engine, "_detect_hand", fail_if_called)

    decision = engine.analyze_frame(observed_buttons=["ron"])

    assert decision.decision_type == "win_window"
    assert decision.action_required is True
    assert decision.buttons == ["ron"]
    assert decision.reason_codes == ["critical_action_interrupt"]


def test_call_window_uses_hand_plan_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = RoundCoachEngine(MahjongCoachConfig())
    engine.state.opening_emitted = True
    engine.state.current_plan = "主线：断幺速度"
    monkeypatch.setattr(
        engine,
        "_detect_hand",
        lambda _path: FastHandResult(ok=True, hand_tiles=list(HAND), confidence=0.91, reason="test_hand"),
    )
    monkeypatch.setattr(
        engine,
        "_detect_river",
        lambda _path: (_ for _ in ()).throw(AssertionError("call windows should not wait for river scan")),
    )

    decision = engine.analyze_frame("frame.png", observed_buttons=["pon"])

    assert decision.decision_type == "call_window"
    assert decision.action_required is True
    assert decision.buttons == ["pon"]
    assert "默认跳过" in decision.suggestion
    assert decision.hand_tiles == HAND


def test_opening_plan_once_then_checkpoint_every_three_turns(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = RoundCoachEngine(MahjongCoachConfig(coach_checkpoint_self_turns=3))
    river_calls: list[Path | None] = []

    monkeypatch.setattr(
        engine,
        "_detect_hand",
        lambda _path: FastHandResult(ok=True, hand_tiles=list(HAND), confidence=0.91, reason="test_hand"),
    )
    monkeypatch.setattr(
        engine,
        "_detect_river",
        lambda path: river_calls.append(path) or RiverStateResult(reason="test_river"),
    )

    first = engine.analyze_frame("frame.png", self_turn_index=1)
    second = engine.analyze_frame("frame.png", self_turn_index=2)
    third = engine.analyze_frame("frame.png", self_turn_index=3)

    assert first.decision_type == "opening_plan"
    assert second.decision_type == "observe"
    assert "discard" not in second.suggestion.lower()
    assert third.decision_type == "coach_checkpoint"
    assert third.reason_codes == ["scheduled_checkpoint"]
    assert len(river_calls) == 1


def test_live_river_mode_tracks_river_every_normal_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = RoundCoachEngine(MahjongCoachConfig(coach_checkpoint_self_turns=3, river_tracking_mode="live"))
    river_calls: list[Path | None] = []
    incremental_calls: list[Path | None] = []

    monkeypatch.setattr(
        engine,
        "_detect_hand",
        lambda _path: FastHandResult(ok=True, hand_tiles=list(HAND), confidence=0.91, reason="test_hand"),
    )
    monkeypatch.setattr(
        engine,
        "_detect_river",
        lambda path: river_calls.append(path)
        or RiverStateResult(
            ok=True,
            discard_piles={"self": [{"tile": "1m", "turn_index": 1, "slot_id": "discard_self_01", "confidence": 0.95}]},
            visible_tiles=["1m"],
            reason="test_river",
        ),
    )
    monkeypatch.setattr(
        engine,
        "_detect_incremental_river",
        lambda path: incremental_calls.append(path)
        or RiverStateResult(ok=True, discard_piles={}, visible_tiles=[], reason="no_new_discards"),
    )

    engine.analyze_frame("frame.png", self_turn_index=1)
    engine.analyze_frame("frame.png", self_turn_index=2)
    engine.analyze_frame("frame.png", self_turn_index=3)

    assert len(river_calls) == 1
    assert len(incremental_calls) == 2
    assert engine.state.last_visible_discards == ["1m"]


def test_incremental_river_appends_only_the_new_slot() -> None:
    engine = RoundCoachEngine(MahjongCoachConfig(river_tracking_mode="live"))
    engine.state.last_discard_piles = {
        "right_opponent": [{"tile": "1m", "turn_index": 1, "slot_id": "discard_right_opponent_01", "confidence": 0.95}]
    }
    engine.state.last_visible_discards = ["1m"]
    engine.state.river_tracking_initialized = True
    delta = RiverStateResult(
        ok=True,
        discard_piles={
            "right_opponent": [
                {"tile": "2m", "turn_index": 2, "slot_id": "discard_right_opponent_02", "confidence": 0.96}
            ]
        },
        visible_tiles=["2m"],
        confidence=0.96,
        reason="recognized_new_discards",
        analysis_hints={"incremental": True},
    )

    merged = engine._merge_incremental_river(delta)
    engine._remember_river(merged)

    assert [item["tile"] for item in merged.discard_piles["right_opponent"]] == ["1m", "2m"]
    assert merged.analysis_hints["new_discard_count"] == 1
    assert engine.state.last_visible_discards == ["1m", "2m"]


def test_full_river_rescan_corrects_same_position_after_two_confirmations() -> None:
    engine = RoundCoachEngine(MahjongCoachConfig(river_tracking_mode="live", tile_recognition_mode="yolo26"))
    engine.state.last_discard_piles = {
        "right_opponent": [
            {
                "tile": "4p",
                "player": "right_opponent",
                "turn_index": 1,
                "bbox": [500, 300, 540, 360],
                "confidence": 0.61,
            }
        ]
    }
    engine.state.last_visible_discards = ["4p"]
    engine.state.river_tracking_initialized = True
    snapshot = RiverStateResult(
        ok=True,
        discard_piles={
            "right_opponent": [
                {
                    "tile": "2p",
                    "player": "right_opponent",
                    "turn_index": 1,
                    "bbox": [502, 302, 542, 362],
                    "confidence": 0.94,
                    "source": "test_yolo26",
                }
            ]
        },
        visible_tiles=["2p"],
        confidence=0.94,
        reason="recognized_yolo26_discards",
    )

    first = engine._reconcile_full_river(snapshot)
    engine._remember_river(first)
    second = engine._reconcile_full_river(snapshot)

    assert first.discard_piles["right_opponent"][0]["tile"] == "4p"
    assert first.analysis_hints["river_pending_corrections"] == 1
    assert second.discard_piles["right_opponent"][0]["tile"] == "2p"
    assert second.discard_piles["right_opponent"][0]["corrected_from"] == "4p"
    assert second.analysis_hints["river_corrected_count"] == 1
    assert second.analysis_hints["river_pending_corrections"] == 0


def test_full_river_rescan_requires_two_frames_before_appending_new_tile() -> None:
    engine = RoundCoachEngine(MahjongCoachConfig(river_tracking_mode="live", tile_recognition_mode="yolo26"))
    engine.state.last_discard_piles = {
        "self": [
            {"tile": "1m", "player": "self", "turn_index": 1, "bbox": [340, 500, 380, 560], "confidence": 0.95}
        ]
    }
    engine.state.last_visible_discards = ["1m"]
    engine.state.river_tracking_initialized = True
    snapshot = RiverStateResult(
        ok=True,
        discard_piles={
            "self": [
                {"tile": "1m", "player": "self", "turn_index": 1, "bbox": [341, 501, 381, 561], "confidence": 0.95},
                {"tile": "2m", "player": "self", "turn_index": 2, "bbox": [382, 500, 422, 560], "confidence": 0.93},
            ]
        },
        visible_tiles=["1m", "2m"],
        reason="recognized_yolo26_discards",
    )

    first = engine._reconcile_full_river(snapshot)
    engine._remember_river(first)
    second = engine._reconcile_full_river(snapshot)

    assert [item["tile"] for item in first.discard_piles["self"]] == ["1m"]
    assert first.analysis_hints["river_pending_corrections"] == 1
    assert [item["tile"] for item in second.discard_piles["self"]] == ["1m", "2m"]
    assert second.analysis_hints["new_discard_count"] == 1


def test_full_river_rescan_confirms_call_window_tile_and_preserves_history() -> None:
    engine = RoundCoachEngine(MahjongCoachConfig(river_tracking_mode="live", tile_recognition_mode="yolo26"))
    engine.state.river_tracking_initialized = True
    snapshot = RiverStateResult(
        ok=True,
        discard_piles={
            "left_opponent": [
                {"tile": "5z", "player": "left_opponent", "turn_index": 1, "bbox": [280, 360, 320, 420], "confidence": 0.96}
            ]
        },
        visible_tiles=["5z"],
        reason="recognized_yolo26_discards",
    )

    first = engine._reconcile_full_river(snapshot, call_buttons=["pon"])
    engine._remember_river(first)
    second = engine._reconcile_full_river(snapshot, call_buttons=["pon"])

    assert first.visible_tiles == []
    assert first.analysis_hints["river_pending_corrections"] == 1
    assert second.visible_tiles == ["5z"]
    assert second.analysis_hints["new_discard_count"] == 1
    assert second.analysis_hints["river_pending_corrections"] == 0
    assert second.discard_piles["left_opponent"][0]["preserve_history"] is True
    assert second.discard_piles["left_opponent"][0]["history_reason"] == "call_window_discard"


def test_full_river_rescan_retracts_injected_phantom_after_three_missing_audits() -> None:
    engine = RoundCoachEngine(MahjongCoachConfig(river_tracking_mode="live", tile_recognition_mode="yolo26"))
    engine.state.last_discard_piles = {
        "right_opponent": [
            {"tile": "1m", "player": "right_opponent", "turn_index": 1, "bbox": [500, 300, 540, 360]},
            {
                "tile": "9p",
                "player": "right_opponent",
                "turn_index": 2,
                "bbox": [545, 300, 585, 360],
                "confidence": 0.51,
                "source": "injected_phantom",
            },
        ]
    }
    engine.state.river_tracking_initialized = True
    snapshot = RiverStateResult(
        ok=True,
        discard_piles={
            "right_opponent": [
                {"tile": "1m", "player": "right_opponent", "turn_index": 1, "bbox": [501, 301, 541, 361]}
            ]
        },
        visible_tiles=["1m"],
        reason="recognized_yolo26_discards",
    )

    results: list[RiverStateResult] = []
    for _ in range(3):
        result = engine._reconcile_full_river(snapshot)
        engine._remember_river(result)
        results.append(result)

    assert [item["tile"] for item in results[0].discard_piles["right_opponent"]] == ["1m", "9p"]
    assert results[0].discard_piles["right_opponent"][1]["missing_confirmations"] == 1
    assert [item["tile"] for item in results[1].discard_piles["right_opponent"]] == ["1m", "9p"]
    assert [item["tile"] for item in results[2].discard_piles["right_opponent"]] == ["1m"]
    assert results[2].analysis_hints["river_retracted_count"] == 1
    assert results[2].analysis_hints["river_retraction_events"][0]["tile"] == "9p"


def test_full_river_rescan_migrates_wrong_owner_after_two_confirmations() -> None:
    engine = RoundCoachEngine(MahjongCoachConfig(river_tracking_mode="live", tile_recognition_mode="yolo26"))
    engine.state.last_discard_piles = {
        "right_opponent": [
            {
                "tile": "3s",
                "player": "right_opponent",
                "turn_index": 1,
                "bbox": [400, 300, 440, 360],
                "confidence": 0.62,
                "source": "injected_wrong_owner",
            }
        ]
    }
    engine.state.river_tracking_initialized = True
    snapshot = RiverStateResult(
        ok=True,
        discard_piles={
            "left_opponent": [
                {
                    "tile": "3s",
                    "player": "left_opponent",
                    "turn_index": 1,
                    "bbox": [401, 301, 441, 361],
                    "confidence": 0.96,
                }
            ]
        },
        visible_tiles=["3s"],
        reason="recognized_yolo26_discards",
    )

    first = engine._reconcile_full_river(snapshot)
    engine._remember_river(first)
    second = engine._reconcile_full_river(snapshot)

    assert [item["tile"] for item in first.discard_piles["right_opponent"]] == ["3s"]
    assert first.discard_piles["left_opponent"] == []
    assert second.discard_piles["right_opponent"] == []
    assert [item["tile"] for item in second.discard_piles["left_opponent"]] == ["3s"]
    assert second.discard_piles["left_opponent"][0]["corrected_player_from"] == "right_opponent"
    assert second.analysis_hints["river_owner_migrated_count"] == 1


def test_call_window_history_is_not_retracted_when_tile_disappears() -> None:
    engine = RoundCoachEngine(MahjongCoachConfig(river_tracking_mode="live", tile_recognition_mode="yolo26"))
    engine.state.river_tracking_initialized = True
    call_snapshot = RiverStateResult(
        ok=True,
        discard_piles={
            "left_opponent": [
                {"tile": "5z", "player": "left_opponent", "turn_index": 1, "bbox": [280, 360, 320, 420]}
            ]
        },
        visible_tiles=["5z"],
        reason="recognized_yolo26_discards",
    )
    engine._remember_river(engine._reconcile_full_river(call_snapshot, call_buttons=["pon"]))
    engine._remember_river(engine._reconcile_full_river(call_snapshot, call_buttons=["pon"]))
    empty_snapshot = RiverStateResult(ok=True, discard_piles={}, visible_tiles=[], reason="no_visible_discards")

    for _ in range(4):
        result = engine._reconcile_full_river(empty_snapshot)
        engine._remember_river(result)

    assert [item["tile"] for item in result.discard_piles["left_opponent"]] == ["5z"]
    assert result.discard_piles["left_opponent"][0]["currently_visible"] is False
    assert result.analysis_hints["river_retracted_count"] == 0
    assert result.analysis_hints["river_preserved_missing_count"] == 1


def test_meld_growth_preserves_the_disappearing_discard_as_history() -> None:
    engine = RoundCoachEngine(MahjongCoachConfig(river_tracking_mode="live", tile_recognition_mode="yolo26"))
    engine.state.last_discard_piles = {
        "left_opponent": [
            {"tile": "1m", "player": "left_opponent", "turn_index": 1, "bbox": [300, 300, 340, 360]},
            {"tile": "2m", "player": "left_opponent", "turn_index": 2, "bbox": [345, 300, 385, 360]},
        ]
    }
    engine.state.river_tracking_initialized = True
    engine._river_last_opponent_meld_count = 0
    snapshot = RiverStateResult(
        ok=True,
        discard_piles={
            "left_opponent": [
                {"tile": "1m", "player": "left_opponent", "turn_index": 1, "bbox": [301, 301, 341, 361]}
            ]
        },
        visible_tiles=["1m"],
        reason="recognized_yolo26_discards",
        analysis_hints={"yolo26_opponent_meld_count": 1},
    )

    first = engine._reconcile_full_river(snapshot)
    engine._remember_river(first)
    for _ in range(4):
        result = engine._reconcile_full_river(snapshot)
        engine._remember_river(result)

    preserved = result.discard_piles["left_opponent"][1]
    assert preserved["tile"] == "2m"
    assert preserved["preserve_history"] is True
    assert preserved["history_reason"] == "meld_growth"
    assert result.analysis_hints["river_retracted_count"] == 0


def test_live_yolo_call_window_exposes_claimed_tile_only_after_two_scans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = RoundCoachEngine(MahjongCoachConfig(river_tracking_mode="live", tile_recognition_mode="yolo26"))
    engine.state.last_discard_piles = {
        "left_opponent": [
            {"tile": "1m", "player": "left_opponent", "turn_index": 1, "bbox": [280, 360, 320, 420]}
        ]
    }
    engine.state.last_visible_discards = ["1m"]
    engine.state.river_tracking_initialized = True
    snapshot = RiverStateResult(
        ok=True,
        discard_piles={
            "left_opponent": [
                {"tile": "1m", "player": "left_opponent", "turn_index": 1, "bbox": [281, 361, 321, 421]},
                {"tile": "5z", "player": "left_opponent", "turn_index": 2, "bbox": [325, 360, 365, 420]},
            ]
        },
        visible_tiles=["1m", "5z"],
        reason="recognized_yolo26_discards",
    )
    monkeypatch.setattr(engine, "_detect_river", lambda _path: snapshot)

    first, first_claimed = engine._call_window_river(Path("call.png"), ["pon"])
    second, second_claimed = engine._call_window_river(Path("call.png"), ["pon"])

    assert [item["tile"] for item in first.discard_piles["left_opponent"]] == ["1m"]
    assert first_claimed == ""
    assert [item["tile"] for item in second.discard_piles["left_opponent"]] == ["1m", "5z"]
    assert second_claimed == "5z"


def test_full_river_rescan_does_not_delete_broad_obstruction() -> None:
    engine = RoundCoachEngine(MahjongCoachConfig(river_tracking_mode="live", tile_recognition_mode="yolo26"))
    engine.state.last_discard_piles = {
        "self": [
            {"tile": tile, "player": "self", "turn_index": index, "slot_id": f"self-{index}"}
            for index, tile in enumerate(["1m", "2m", "3m", "4m"], start=1)
        ]
    }
    engine.state.river_tracking_initialized = True
    empty_snapshot = RiverStateResult(ok=True, discard_piles={}, visible_tiles=[], reason="covered_table")

    for _ in range(5):
        result = engine._reconcile_full_river(empty_snapshot)
        engine._remember_river(result)

    assert [item["tile"] for item in result.discard_piles["self"]] == ["1m", "2m", "3m", "4m"]
    assert result.analysis_hints["river_absence_audit_stable"] is False
    assert result.analysis_hints["river_retracted_count"] == 0


def test_legacy_live_tracking_runs_periodic_full_river_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = RoundCoachEngine(MahjongCoachConfig(river_tracking_mode="live"))
    engine.state.last_discard_piles = {
        "self": [
            {"tile": "1m", "player": "self", "turn_index": 1, "slot_id": "discard_self_01", "confidence": 0.95}
        ]
    }
    engine.state.last_visible_discards = ["1m"]
    engine.state.river_tracking_initialized = True
    engine._river_frames_since_full_scan = coach_module._RIVER_FULL_RESCAN_INTERVAL - 1
    full_calls: list[Path | None] = []
    monkeypatch.setattr(
        engine,
        "_detect_river",
        lambda path: full_calls.append(path)
        or RiverStateResult(
            ok=True,
            discard_piles={
                "self": [
                    {"tile": "1m", "player": "self", "turn_index": 1, "slot_id": "discard_self_01", "confidence": 0.96}
                ]
            },
            visible_tiles=["1m"],
            reason="recognized_discards",
        ),
    )
    monkeypatch.setattr(
        engine,
        "_detect_incremental_river",
        lambda _path: (_ for _ in ()).throw(AssertionError("periodic audit should skip incremental detection")),
    )

    result = engine._track_river_for_frame(Path("frame.png"), "normal_river_tracking")

    assert len(full_calls) == 1
    assert result.analysis_hints["river_full_rescan"] is True
    assert engine._river_frames_since_full_scan == 0


def test_force_checkpoint_works_without_per_turn_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = RoundCoachEngine(MahjongCoachConfig(per_turn_discard_prompt=False))
    engine.state.opening_emitted = True
    monkeypatch.setattr(
        engine,
        "_detect_hand",
        lambda _path: FastHandResult(ok=True, hand_tiles=list(HAND), confidence=0.91, reason="test_hand"),
    )

    decision = engine.analyze_frame("frame.png", force_checkpoint=True)

    assert decision.decision_type == "coach_checkpoint"
    assert decision.reason_codes == ["forced_checkpoint"]
    assert decision.engine_meta["per_turn_discard_prompt"] is False


def test_riichi_window_uses_local_fast_path_without_river(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = RoundCoachEngine(MahjongCoachConfig())
    engine.state.opening_emitted = True
    monkeypatch.setattr(
        engine,
        "_detect_hand",
        lambda _path: FastHandResult(ok=True, hand_tiles=list(HAND), confidence=0.91, reason="test_hand"),
    )
    monkeypatch.setattr(
        engine,
        "_detect_river",
        lambda _path: (_ for _ in ()).throw(AssertionError("riichi window should not wait for river scan")),
    )

    decision = engine.analyze_frame("frame.png", observed_buttons=["riichi"])

    assert decision.decision_type == "riichi_window"
    assert decision.action_required is True
    assert "本地暂未算出" in decision.suggestion or "推荐立直" in decision.suggestion


def test_riichi_window_recommends_good_wait() -> None:
    engine = RoundCoachEngine(MahjongCoachConfig())
    hand = ["1m", "2m", "3m", "4p", "5p", "6p", "2s", "3s", "4s", "5s", "6s", "7s", "1z", "1z"]

    decision = engine._critical_decision(
        ["riichi"],
        {"source": "test"},
        0.0,
        hand_result=FastHandResult(ok=True, hand_tiles=hand, confidence=0.91, reason="test_hand"),
    )

    assert decision.decision_type == "riichi_window"
    assert "推荐立直" in decision.suggestion
    assert "听2索、5索、8索" in decision.suggestion
    assert "好形" in decision.suggestion


def test_riichi_window_warns_on_poor_wait() -> None:
    engine = RoundCoachEngine(MahjongCoachConfig())
    hand = ["1m", "1m", "1m", "2p", "3p", "4p", "5p", "6p", "7p", "2s", "3s", "4s", "7s", "9s"]

    decision = engine._critical_decision(
        ["riichi"],
        {"source": "test"},
        0.0,
        hand_result=FastHandResult(ok=True, hand_tiles=hand, confidence=0.91, reason="test_hand"),
    )

    assert decision.decision_type == "riichi_window"
    assert "谨慎立直" in decision.suggestion
    assert "听8索" in decision.suggestion
    assert "愚形" in decision.suggestion


def test_round_plan_gives_concrete_keep_and_cleanup_guidance() -> None:
    plan = build_round_plan(["2m", "7m", "0p", "8p", "1s", "2s", "2s", "3s", "3s", "4s", "3z", "4z", "5z"])

    assert "索" in plan["summary"]
    assert any("保留" in item and "2索" in item for item in plan["targets"])
    assert any("路线选择" in item and "打" in item for item in plan["cautions"])
    assert any("优先清理" in item and "西" in item for item in plan["cautions"])
    assert "鸣牌" in plan["cautions"][-1]


def test_round_plan_names_honor_cleanup_route() -> None:
    plan = build_round_plan(["1m", "4m", "8m", "2p", "5p", "7p", "3s", "6s", "1z", "2z", "5z", "6z", "7z"])

    assert any("路线选择" in item and "孤字先打" in item for item in plan["cautions"])
    assert any(name in " ".join(plan["cautions"]) for name in ["东", "南", "白", "发", "中"])


def test_round_plan_keeps_dora_out_of_discard_priority() -> None:
    plan = build_round_plan(
        ["2m", "7m", "0p", "8p", "1s", "2s", "2s", "3s", "3s", "4s", "3z", "4z", "5z"],
        MahjongCoachConfig(dora_tiles=["7m"]),
    )

    assert "7m" not in plan["discard_priority"]
    assert any("宝牌/红5" in item and "7万" in item for item in plan["targets"])


def test_call_window_opens_value_honor_pair() -> None:
    engine = RoundCoachEngine(MahjongCoachConfig())
    engine.state.opening_emitted = True
    hand = ["1m", "2m", "3m", "4p", "5p", "6p", "2s", "3s", "4s", "5s", "6s", "5z", "5z", "7m"]

    decision = engine._critical_decision(
        ["pon"],
        {"source": "test"},
        0.0,
        hand_result=FastHandResult(ok=True, hand_tiles=hand, confidence=0.91, reason="test_hand"),
    )

    assert decision.decision_type == "call_window"
    assert "役牌对子白" in decision.suggestion
    assert "可以开" in decision.suggestion


def test_call_analysis_evaluates_a_known_claimed_tile() -> None:
    hand = ["1m", "2m", "3m", "4m", "5m", "6m", "2p", "3p", "4p", "5z", "5z", "7s", "9s"]

    analysis = analyze_call_options(hand, MahjongCoachConfig(), ["pon"], claimed_tile="5z")

    assert analysis["claimed_tile_known"] is True
    assert analysis["claimed_tile"] == "5z"
    option = next(item for item in analysis["options"] if item["action"] == "pon")
    assert option["claimed_tile"] == "5z"
    assert option["status"] == "evaluated"
    assert option["discard"]
    assert option["post_shanten"] >= -1


def test_call_analysis_stays_conditional_without_the_claimed_tile() -> None:
    hand = ["1m", "2m", "3m", "4m", "5m", "6m", "2p", "3p", "4p", "5z", "5z", "7s", "9s"]

    analysis = analyze_call_options(hand, MahjongCoachConfig(), ["pon"])

    assert analysis["claimed_tile_known"] is False
    assert any(item["action"] == "pon" and item["claimed_tile"] == "5z" for item in analysis["options"])


def test_call_window_marks_unknown_discard_as_conditional() -> None:
    engine = RoundCoachEngine(MahjongCoachConfig())
    engine.state.opening_emitted = True
    hand = ["1m", "2m", "3m", "4m", "5m", "6m", "2p", "3p", "4p", "5z", "5z", "7s", "9s"]

    decision = engine._critical_decision(
        ["pon"],
        {"source": "test"},
        0.0,
        hand_result=FastHandResult(ok=True, hand_tiles=hand, confidence=0.91, reason="test_hand"),
    )

    assert "尚未识别本次被弃牌" in decision.suggestion


def test_live_call_window_uses_a_unique_river_delta(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = RoundCoachEngine(MahjongCoachConfig(river_tracking_mode="live"))
    engine.state.opening_emitted = True
    hand = ["1m", "2m", "3m", "4m", "5m", "6m", "2p", "3p", "4p", "5z", "5z", "7s", "9s"]
    engine.state.last_discard_piles = {"right_opponent": [{"tile": "1m", "confidence": 0.95}]}
    monkeypatch.setattr(
        engine,
        "_detect_hand",
        lambda _path: FastHandResult(ok=True, hand_tiles=hand, confidence=0.91, reason="test_hand"),
    )
    monkeypatch.setattr(
        engine,
        "_detect_river",
        lambda _path: RiverStateResult(
            ok=True,
            discard_piles={"right_opponent": [{"tile": "1m", "confidence": 0.95}, {"tile": "5z", "confidence": 0.96}]},
            visible_tiles=["1m", "5z"],
            confidence=0.955,
            reason="test_river",
        ),
    )

    decision = engine.analyze_frame("frame.png", observed_buttons=["pon"])

    assert decision.perception["action"]["claimed_tile"] == "5z"
    assert decision.perception["action"]["claimed_tile_source"] == "river_delta"
    assert "尚未识别本次被弃牌" not in decision.suggestion


def test_round_plan_includes_local_shanten_and_ukeire() -> None:
    plan = build_round_plan(["1m", "2m", "3m", "4p", "5p", "6p", "2s", "3s", "4s", "5s", "6s", "8s", "1z", "5z"])

    efficiency = plan["efficiency"]
    best_discard = efficiency["discard_options"][0]
    assert efficiency["best_path"] == "standard"
    assert efficiency["current_shanten"] == 1
    assert best_discard["tile"] in {"1z", "5z"}
    assert best_discard["effective_types"] > 0
    assert best_discard["effective_count"] > 0
    assert any("牌效" in item and "有效" in item for item in plan["cautions"])
    assert plan["discard_priority"][0] == best_discard["tile"]


def test_round_plan_subtracts_visible_river_from_ukeire() -> None:
    hand = ["1m", "2m", "3m", "4p", "5p", "6p", "2s", "3s", "4s", "5s", "6s", "8s", "1z", "5z"]
    plain = build_round_plan(hand)
    adjusted = build_round_plan(hand, visible_tiles=["1s", "2s", "4s", "5s", "7s", "8s"])

    plain_best = plain["efficiency"]["discard_options"][0]
    adjusted_best = adjusted["efficiency"]["discard_options"][0]
    assert plain_best["tile"] == adjusted_best["tile"]
    assert adjusted_best["effective_count"] < plain_best["effective_count"]
    assert adjusted["efficiency"]["visible_tile_count"] == 6
    assert any("已扣可见牌" in item for item in adjusted["cautions"])


def test_efficiency_counts_a_discarded_tile_as_available_again() -> None:
    hand = ["1m", "1m", "1m", "2p", "3p", "4p", "2s", "3s", "4s", "6s", "7s", "8s", "5z", "5z"]

    plan = build_round_plan(hand)
    option = next(item for item in plan["efficiency"]["discard_options"] if item["tile"] == "2p")

    assert option["effective_tiles"] == ["2p", "5p"]
    assert option["effective_count"] == 8


def test_riichi_waits_use_the_post_discard_tile_count() -> None:
    hand = ["1m", "1m", "1m", "2p", "3p", "4p", "2s", "3s", "4s", "6s", "7s", "8s", "5z", "5z"]

    waits, wait_count = _riichi_waits_after_discard(hand, ["2p", "5p"], "2p", [])

    assert waits == ["2p", "5p"]
    assert wait_count == 8


def test_efficiency_keeps_all_thirteen_orphans_waits() -> None:
    hand = [*ORPHAN_TYPES, "1m"]

    result = _efficiency_analysis(hand, ["1m"], Counter(hand), _visible_counts([]), "", set(), set())
    option = result["discard_options"][0]

    assert option["effective_types"] == 13
    assert len(option["effective_tiles"]) == 13
    assert option["effective_count"] == 39


def test_fast_open_hand_plan_focuses_on_closing_not_starting_route() -> None:
    plan = build_round_plan(
        ["4s", "5s", "6s", "2p", "3p", "7p", "5z", "5z"],
        MahjongCoachConfig(play_style="fast"),
    )

    assert plan["efficiency"]["open_melds"] == 2
    assert plan["direction"].startswith("副露")
    assert plan["direction"] != "役牌速攻"
    assert any("已副露2组" in item for item in plan["targets"])
    assert any("副露牌效" in item for item in plan["cautions"])


def test_fast_late_open_hand_shanten_counts_existing_melds() -> None:
    plan = build_round_plan(
        ["2m", "3m", "5p", "5p", "7s"],
        MahjongCoachConfig(play_style="fast"),
    )

    assert plan["efficiency"]["open_melds"] == 3
    assert plan["efficiency"]["current_shanten"] <= 1
    assert plan["direction"].startswith("副露")


def test_round_plan_uses_detected_meld_tiles_for_open_yaku() -> None:
    plan = build_round_plan(
        ["2m", "3m", "4p", "5p", "6s", "7s", "8s", "9s"],
        MahjongCoachConfig(play_style="fast"),
        open_melds=2,
        meld_tiles=["5z", "5z", "5z", "2s", "3s", "4s"],
    )

    assert plan["efficiency"]["open_melds"] == 2
    assert plan["direction"].startswith("副露")
    assert any("副露识别" in item and "白" in item for item in plan["targets"])
    assert any("役牌副露" in item and "白" in item for item in plan["targets"])
    assert not any("可能没役" in item for item in plan["cautions"])


def test_checkpoint_plan_uses_onnx_meld_state(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = RoundCoachEngine(MahjongCoachConfig(play_style="fast"))
    engine.state.opening_emitted = True
    monkeypatch.setattr(
        engine,
        "_detect_hand",
        lambda _path: FastHandResult(
            ok=True,
            hand_tiles=["2m", "3m", "4p", "5p", "6s", "7s", "8s", "9s"],
            confidence=0.86,
            reason="test_open_hand",
        ),
    )
    monkeypatch.setattr(
        engine,
        "_detect_melds",
        lambda _path, **_kwargs: MeldStateResult(
            ok=True,
            open_meld_count=2,
            melds=[{"player": "self", "meld_index": 1, "tiles": [{"tile": "5z"}]}],
            tiles=["5z", "5z", "5z", "2s", "3s", "4s"],
            confidence=0.93,
            reason="recognized_self_melds",
        ),
    )
    monkeypatch.setattr(engine, "_resolve_buttons", lambda *_args, **_kwargs: ([], {"source": "test"}))
    monkeypatch.setattr(engine, "_detect_river", lambda _path: RiverStateResult(ok=True, reason="test_river"))

    decision = engine.analyze_frame("frame.png", force_checkpoint=True)

    assert decision.decision_type == "coach_checkpoint"
    assert decision.perception["meld"]["open_meld_count"] == 2
    assert decision.coach_state["last_open_meld_count"] == 2
    assert any("役牌副露" in item for item in decision.coach_state["target_shapes"])


def test_round_plan_prefers_seven_pairs_when_pairs_are_dense() -> None:
    plan = build_round_plan(["1m", "1m", "2p", "2p", "3s", "3s", "4m", "4m", "5p", "5p", "6s", "7s", "8s", "9s"])

    assert plan["direction"] == "七对子"
    assert plan["efficiency"]["best_path"] == "seven_pairs"
    assert any("七对子胚子" in item for item in plan["targets"])
    assert any(item == "保留：1万、4万、2筒、5筒、3索" for item in plan["targets"])


def test_observe_explains_missing_stable_hand(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = RoundCoachEngine(MahjongCoachConfig())
    monkeypatch.setattr(
        engine,
        "_detect_hand",
        lambda _path: FastHandResult(
            ok=False,
            reason="unstable_hand_count",
            raw_detections=[{"occupied": True}, {"occupied": True}],
        ),
    )

    decision = engine.analyze_frame()

    assert decision.decision_type == "observe"
    assert "No stable hand tiles" in decision.detail
    assert "hand_unstable_hand_count" in decision.reason_codes


def test_riichi_players_trigger_defense_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = RoundCoachEngine(MahjongCoachConfig())
    engine.state.opening_emitted = True
    monkeypatch.setattr(
        engine,
        "_detect_hand",
        lambda _path: FastHandResult(ok=True, hand_tiles=list(HAND), confidence=0.91, reason="test_hand"),
    )

    decision = engine.analyze_frame("frame.png", riichi_players=["shimocha"])

    assert decision.decision_type == "defense_alert"
    assert decision.action_required is True
    assert decision.coach_state["attack_defense_bias"] == "defense"


def test_opponent_riichi_detection_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = RoundCoachEngine(MahjongCoachConfig(opponent_riichi_recognition_enabled=False))
    engine.state.opening_emitted = True
    engine.state.update_count = 2
    monkeypatch.setattr(
        engine,
        "_detect_hand",
        lambda _path: FastHandResult(ok=True, hand_tiles=list(HAND), confidence=0.91, reason="test_hand"),
    )
    monkeypatch.setattr(
        coach_module,
        "detect_riichi_sticks",
        lambda _path: (_ for _ in ()).throw(AssertionError("opponent riichi detector should be opt-in")),
    )

    decision = engine.analyze_frame("frame.png")

    assert decision.decision_type != "defense_alert"
    assert decision.engine_meta["opponent_riichi_recognition_enabled"] is False


def test_opponent_riichi_detection_ignores_existing_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = RoundCoachEngine(MahjongCoachConfig())
    monkeypatch.setattr(
        coach_module,
        "detect_riichi_sticks",
        lambda _path: SimpleNamespace(riichi_players=["unknown"], stick_count=1),
    )

    assert engine._detect_riichi_players(Path("frame.png")) == []
    assert engine.state.riichi_stick_baseline == 1


def test_opponent_riichi_detection_uses_counter_increase(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = RoundCoachEngine(MahjongCoachConfig())
    engine.state.riichi_stick_baseline = 0
    engine.state.riichi_pending = {"unknown": 1}
    monkeypatch.setattr(
        coach_module,
        "detect_riichi_sticks",
        lambda _path: SimpleNamespace(riichi_players=["unknown"], stick_count=1),
    )

    assert engine._detect_riichi_players(Path("frame.png")) == ["unknown"]


def test_riichi_defense_uses_recognized_riichi_player_river(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = RoundCoachEngine(MahjongCoachConfig())
    engine.state.opening_emitted = True
    monkeypatch.setattr(
        engine,
        "_detect_hand",
        lambda _path: FastHandResult(ok=True, hand_tiles=list(HAND), confidence=0.91, reason="test_hand"),
    )
    monkeypatch.setattr(
        engine,
        "_detect_river",
        lambda _path: RiverStateResult(
            ok=True,
            discard_piles={
                "right_opponent": [
                    {"tile": "7p", "confidence": 0.96},
                    {"tile": "2s", "confidence": 0.95},
                ]
            },
            visible_tiles=["7p", "2s"],
            confidence=0.955,
            reason="test_river",
        ),
    )

    decision = engine.analyze_frame("frame.png", riichi_players=["shimocha"])

    assert "现物" in decision.suggestion
    assert "2索" in decision.suggestion
    assert "7筒" not in decision.suggestion
    assert decision.coach_state["last_visible_discards"] == ["7p", "2s"]


def test_riichi_defense_only_recommends_safe_tiles_player_actually_holds(monkeypatch: pytest.MonkeyPatch) -> None:
    hand = ["1m", "2m", "3m", "4p", "5p", "6p", "2s", "3s", "4s", "5s", "6s", "7s", "1z", "1z"]
    engine = RoundCoachEngine(MahjongCoachConfig())
    engine.state.opening_emitted = True
    monkeypatch.setattr(
        engine,
        "_detect_hand",
        lambda _path: FastHandResult(ok=True, hand_tiles=hand, confidence=0.91, reason="test_hand"),
    )
    monkeypatch.setattr(
        engine,
        "_detect_river",
        lambda _path: RiverStateResult(
            ok=True,
            discard_piles={
                "right_opponent": [
                    {"tile": "3z", "confidence": 0.96},
                    {"tile": "4z", "confidence": 0.95},
                ]
            },
            visible_tiles=["3z", "4z"],
            confidence=0.955,
            reason="test_river",
        ),
    )

    decision = engine.analyze_frame("frame.png", riichi_players=["shimocha"])

    assert decision.decision_type == "defense_alert"
    assert "现物" in decision.suggestion
    assert "手里没有" in decision.suggestion


def test_riichi_defense_does_not_share_genbutsu_between_multiple_players() -> None:
    engine = RoundCoachEngine(MahjongCoachConfig())
    hand = ["3p", "4p", "5p", "1m"]
    river = RiverStateResult(
        ok=True,
        discard_piles={
            "left_opponent": [{"tile": "3p", "confidence": 0.96}],
            "right_opponent": [{"tile": "4p", "confidence": 0.95}],
        },
        visible_tiles=["3p", "4p"],
        confidence=0.955,
        reason="test_river",
    )

    suggestion = engine._defense_suggestion(["kamicha", "shimocha"], river, hand_tiles=hand)

    assert "对全部立直者都成立的现物" not in suggestion
    assert "没有对所有立直者共同成立的现物" in suggestion


def test_unified_discard_ranking_keeps_attack_and_risk_breakdown() -> None:
    hand = ["1m", "2m", "3m", "4p", "5p", "6p", "2s", "3s", "4s", "5s", "6s", "7s", "1z", "1z"]
    river = RiverStateResult(
        ok=True,
        discard_piles={"right_opponent": [{"tile": "1z", "confidence": 0.96}]},
        visible_tiles=["1z"],
        confidence=0.96,
        reason="test_river",
    )

    ranking = rank_discard_decisions(hand, MahjongCoachConfig(), ["shimocha"], river)
    east = next(item for item in ranking["candidates"] if item["tile"] == "1z")

    assert len(ranking["top_candidates"]) == 3
    assert east["defense_risk"] == 0.0
    assert east["risk_by_player"]["right_opponent"]["basis"] == "现物"
    assert "attack_score" in east
    assert "effective_count" in east


def test_unified_discard_ranking_does_not_treat_single_player_genbutsu_as_common() -> None:
    hand = ["1m", "2m", "3m", "3p", "4p", "5p", "2s", "3s", "4s", "5s", "6s", "7s", "1z", "1z"]
    river = RiverStateResult(
        ok=True,
        discard_piles={
            "left_opponent": [{"tile": "3p", "confidence": 0.96}],
            "right_opponent": [{"tile": "4p", "confidence": 0.96}],
        },
        visible_tiles=["3p", "4p"],
        confidence=0.96,
        reason="test_river",
    )

    ranking = rank_discard_decisions(hand, MahjongCoachConfig(), ["kamicha", "shimocha"], river)
    three_pin = next(item for item in ranking["candidates"] if item["tile"] == "3p")

    assert three_pin["risk_by_player"]["left_opponent"]["basis"] == "现物"
    assert three_pin["risk_by_player"]["right_opponent"]["risk"] > 0
    assert three_pin["defense_risk"] > 0
    assert three_pin["safety"] == "单家现物，另家有风险"


def test_defense_decision_exposes_unified_strategy_details(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = RoundCoachEngine(MahjongCoachConfig())
    engine.state.opening_emitted = True
    monkeypatch.setattr(
        engine,
        "_detect_hand",
        lambda _path: FastHandResult(ok=True, hand_tiles=list(HAND), confidence=0.91, reason="test_hand"),
    )
    monkeypatch.setattr(
        engine,
        "_detect_river",
        lambda _path: RiverStateResult(
            ok=True,
            discard_piles={"right_opponent": [{"tile": "2s", "confidence": 0.96}]},
            visible_tiles=["2s"],
            confidence=0.96,
            reason="test_river",
        ),
    )

    decision = engine.analyze_frame("frame.png", riichi_players=["shimocha"])

    assert decision.perception["strategy"]["source"] == "local_efficiency_plus_river_risk"
    assert len(decision.perception["strategy"]["top_candidates"]) == 3
    assert decision.engine_meta["timings_ms"]["strategy"] >= 0


def test_checkpoint_plan_uses_recognized_river_for_ukeire(monkeypatch: pytest.MonkeyPatch) -> None:
    hand = ["1m", "2m", "3m", "4p", "5p", "6p", "2s", "3s", "4s", "5s", "6s", "8s", "1z", "5z"]
    engine = RoundCoachEngine(MahjongCoachConfig())
    engine.state.opening_emitted = True
    monkeypatch.setattr(
        engine,
        "_detect_hand",
        lambda _path: FastHandResult(ok=True, hand_tiles=hand, confidence=0.91, reason="test_hand"),
    )
    monkeypatch.setattr(
        engine,
        "_detect_river",
        lambda _path: RiverStateResult(
            ok=True,
            visible_tiles=["1s", "2s", "4s", "5s", "7s", "8s"],
            confidence=0.95,
            reason="test_river",
        ),
    )

    decision = engine.analyze_frame("frame.png", self_turn_index=3)

    assert decision.decision_type == "coach_checkpoint"
    assert "已扣可见牌" in decision.detail
    assert any("已扣可见牌" in item for item in decision.coach_state["caution_points"])


def test_live_config_from_payload() -> None:
    cfg = MahjongCoachConfig.from_payload(
        {
            "live": {
                "window_keywords": ["Mahjong Soul"],
                "interval_ms": 900,
                "fast_interval_ms": 180,
                "keep_frames": 12,
                "checkpoint_interval_seconds": 14,
                "overlay_enabled": False,
                "save_format": "jpg",
            }
        }
    )

    assert cfg.live_window_keywords == ["Mahjong Soul"]
    assert cfg.live_interval_ms == 900
    assert cfg.live_fast_interval_ms == 180
    assert cfg.live_keep_frames == 12
    assert cfg.live_checkpoint_interval_seconds == 14
    assert cfg.live_overlay_enabled is False
    assert cfg.live_save_format == "jpg"


def test_live_config_defaults_keep_training_material() -> None:
    cfg = MahjongCoachConfig.from_payload({})

    assert cfg.live_interval_ms == 400
    assert cfg.live_keep_frames == 1000
    assert cfg.live_save_format == "jpg"


def test_show_overlay_restarts_config_window() -> None:
    plugin = MahjongCoachPlugin.__new__(MahjongCoachPlugin)
    overlay = _FakeOverlay()
    plugin._overlay = overlay

    assert plugin._show_overlay(strategy=False) is True
    assert overlay.calls == ["start", "config"]


@pytest.mark.asyncio
async def test_show_overlay_entry_uses_strategy_when_live_running() -> None:
    plugin = MahjongCoachPlugin.__new__(MahjongCoachPlugin)
    overlay = _FakeOverlay()
    plugin._overlay = overlay
    plugin._live_task = SimpleNamespace(done=lambda: False)
    plugin._live_state = LiveSessionState(running=True)

    result = await plugin.mahjong_coach_show_overlay()

    assert result.unwrap()["running"] is True
    assert overlay.calls == ["start", "strategy"]


@pytest.mark.asyncio
async def test_start_live_reopens_overlay_when_already_running() -> None:
    plugin = MahjongCoachPlugin.__new__(MahjongCoachPlugin)
    overlay = _FakeOverlay()
    plugin._overlay = overlay
    plugin._engine = RoundCoachEngine(MahjongCoachConfig())
    plugin._cfg = MahjongCoachConfig()
    plugin._live_task = SimpleNamespace(done=lambda: False)
    plugin._live_state = LiveSessionState(running=True)
    plugin.logger = SimpleNamespace(info=lambda *_args, **_kwargs: None, warning=lambda *_args, **_kwargs: None)

    result = await plugin._overlay_start_live(overlay=True)

    assert result.unwrap()["status"] == "already_running"
    assert overlay.calls == ["start", "strategy"]


@pytest.mark.asyncio
async def test_start_live_updates_style_when_already_running() -> None:
    plugin = MahjongCoachPlugin.__new__(MahjongCoachPlugin)
    overlay = _FakeOverlay()
    plugin._overlay = overlay
    plugin._cfg = MahjongCoachConfig(play_style="riichi")
    plugin._engine = RoundCoachEngine(plugin._cfg)
    plugin._engine.state.opening_emitted = True
    plugin._engine.state.current_plan = "old plan"
    plugin._live_task = SimpleNamespace(done=lambda: False)
    plugin._live_state = LiveSessionState(running=True)
    plugin._live_last_checkpoint_at = 123.0
    plugin.logger = SimpleNamespace(info=lambda *_args, **_kwargs: None, warning=lambda *_args, **_kwargs: None)

    result = await plugin._overlay_start_live(overlay=True, play_style="fast", river_tracking_mode="live")

    assert result.unwrap()["status"] == "already_running"
    assert plugin._cfg.play_style == "fast"
    assert plugin._cfg.river_tracking_mode == "live"
    assert plugin._engine.config.play_style == "fast"
    assert plugin._engine.config.river_tracking_mode == "live"
    assert plugin._engine.state.play_style == "fast"
    assert plugin._engine.state.opening_emitted is False
    assert plugin._engine.state.current_plan == ""
    assert plugin._engine.state.last_update_reason == "style_changed"
    assert plugin._live_last_checkpoint_at == 0.0
    assert overlay.calls == ["start", "strategy"]


def test_river_config_from_payload() -> None:
    cfg = MahjongCoachConfig.from_payload(
        {
            "perception": {
                "river_recognition_enabled": False,
                "river_tracking_mode": "live",
                "river_min_confidence": 0.72,
                "tile_recognition_mode": "yolo26",
            }
        }
    )

    assert cfg.river_recognition_enabled is False
    assert cfg.river_tracking_mode == "live"
    assert cfg.river_min_confidence == 0.72
    assert cfg.tile_recognition_mode == "yolo26"


def test_tile_recognition_mode_defaults_to_legacy() -> None:
    assert MahjongCoachConfig.from_payload({}).tile_recognition_mode == "legacy"
    assert MahjongCoachConfig.from_payload({"perception": {"tile_recognition_mode": "bad"}}).tile_recognition_mode == "legacy"


def test_yolo26_mode_falls_back_to_legacy_hand_scan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (640, 360), "black").save(image_path)
    engine = RoundCoachEngine(MahjongCoachConfig(tile_recognition_mode="yolo26"))

    monkeypatch.setattr(
        engine,
        "_detect_yolo26_table",
        lambda _path: Yolo26TableStateResult(reason="yolo26_model_dir_missing"),
    )
    monkeypatch.setattr(
        coach_module,
        "detect_fast_hand_path",
        lambda *_args, **_kwargs: FastHandResult(ok=True, hand_tiles=list(HAND), confidence=0.9, reason="legacy_test"),
    )

    result = engine._detect_hand(image_path)

    assert result.ok is True
    assert result.reason == "legacy_test"
    assert result.analysis_hints["tile_recognition_mode"] == "yolo26"
    assert result.analysis_hints["fallback_mode"] == "legacy"
    assert result.analysis_hints["fallback_reason"] == "yolo26_model_dir_missing"


def test_overlay_text_prioritizes_action_required() -> None:
    text = overlay_text_from_payload(
        {
            "overlay_display_mode": "beginner",
            "last_decision": {
                "decision_type": "call_window",
                "action_required": True,
                "summary": "Call window detected",
                "suggestion": "Evaluate chi/pon/kan quickly.",
            },
            "round_state": {"current_plan": "Play inside hand"},
        }
    )

    assert "本地鸣牌" in text
    assert "Evaluate chi/pon/kan quickly" in text


def test_overlay_call_window_gives_beginner_action() -> None:
    text = overlay_text_from_payload(
        {
            "overlay_display_mode": "beginner",
            "last_decision": {
                "decision_type": "call_window",
                "action_required": True,
                "suggestion": "默认跳过；只有役牌、直接进听、明显加速主线或安全和牌时才吃碰杠。",
            },
            "round_state": {},
        }
    )

    assert "本地鸣牌" in text
    assert "建议跳过" in text
    assert "不要随便碰/吃" in text


def test_overlay_strategy_uses_beginner_labels() -> None:
    text = overlay_text_from_payload(
        {
            "overlay_display_mode": "beginner",
            "last_decision": {"decision_type": "observe"},
            "round_state": {
                "local_direction": "牌效推进",
                "target_shapes": ["保留：1万、2万、3万、4万、5万、6万"],
                "caution_points": ["鸣牌：默认跳过，只有役牌、直接听牌、或明显加速主线才开。"],
            },
        }
    )

    assert "目标：先让手牌更快听牌" in text
    assert "先留这些：1万、2万、3万（共6张）" in text
    assert "吃碰杠规则：" in text
    assert "流程：" not in text
    assert "等6张" not in text
    assert "留：" not in text
    assert "开：" not in text


def test_overlay_strategy_supports_compact_labels() -> None:
    text = overlay_text_from_payload(
        {
            "overlay_display_mode": "compact",
            "last_decision": {"decision_type": "observe"},
            "round_state": {
                "local_direction": "牌效推进",
                "target_shapes": ["保留：1万、2万、3万、4万、5万、6万"],
                "caution_points": ["鸣牌：默认跳过，只有役牌、直接听牌、或明显加速主线才开。"],
            },
        }
    )

    assert "方向：牌效推进" in text
    assert "留：1万、2万、3万（共6张）" in text
    assert "开：默认跳过" in text
    assert "流程：" not in text
    assert "目标：" not in text
    assert "先留这些：" not in text


def test_overlay_detail_explains_pipeline_and_strategy_source() -> None:
    detail = overlay_detail_text_from_payload(
        {
            "live": {
                "last_frame_path": "frames/live-001.jpg",
                "last_window_title": "雀魂 - Mahjong Soul",
                "last_capture_source": "window_capture",
            },
            "last_decision": {
                "decision_type": "opening_plan",
                "suggestion": "已副露1组，优先打向听最低、有效牌最多的牌",
                "perception": {
                    "hand": {
                        "ok": True,
                        "reason": "inferred_open_11_hand_tiles",
                        "hand_tiles": ["1m", "1m", "2m", "3m", "6m", "6m", "4s", "0s", "8s", "9s", "5z"],
                    },
                    "meld": {"reason": "no_self_melds"},
                    "action": {"source": "opening_hand_scan"},
                    "river": {"ok": True, "reason": "recognized_discards", "visible_tiles": ["1m"]},
                },
                "engine_meta": {"source": "opening_plan"},
            },
            "round_state": {
                "last_update_reason": "opening_plan",
                "last_open_meld_count": 1,
                "local_direction": "副露加速",
                "target_shapes": ["保留：1万、2万、3万、6万、红5索"],
                "caution_points": [
                    "副露牌效：估算1组，当前2向听；打8索后2向听，有效6种18枚。",
                    "鸣牌：继续快攻，但只开能进听或明显增加有效牌的牌。",
                ],
            },
        }
    )

    assert "截图依据：frames/live-001.jpg" in detail
    assert "窗口：雀魂 - Mahjong Soul；来源：window_capture" in detail
    assert "识别流程：" in detail
    assert "识别逻辑：capture.py/capture_frame() → coach.py/analyze_frame()" in detail
    assert "手牌逻辑：perception/fast_hand_path.py/detect_fast_hand_path()；结果=inferred_open_11_hand_tiles" in detail
    assert "副露逻辑：perception/meld_state.py/detect_meld_state_path()；结果=no_self_melds" in detail
    assert "按钮逻辑：perception/action_detector.py/detect_action_buttons_fast()；结果=opening_hand_scan" in detail
    assert "牌河逻辑：perception/river_state.py/detect_river_state_path()；结果=recognized_discards" in detail
    assert "识别结果：手牌11张" in detail
    assert "手牌✓11" in detail
    assert "副露✓1推" in detail
    assert "策略功能：coach.py / build_round_plan()" in detail
    assert "策略来源：opening_plan" in detail
    assert "先留这些：来自“保留”目标或策略摘要" in detail
    assert "吃碰杠规则：来自风险点里的“鸣牌”" in detail


def test_overlay_text_shows_riichi_fast_judgement() -> None:
    text = overlay_text_from_payload(
        {
            "last_decision": {
                "decision_type": "riichi_window",
                "action_required": True,
                "summary": "立直窗口",
                "suggestion": "推荐立直：打7索听5索、8索，有效2种8枚；好形/枚数够；本地快判。",
            },
            "round_state": {"current_plan": "Play inside hand"},
        }
    )

    assert "本地立直" in text
    assert "推荐立直：打7索听5索、8索" in text
    assert "好形/枚数够" in text


def test_overlay_geometry_defaults_near_self_hand() -> None:
    x, y = _overlay_geometry(1920, 1080, 630, 138)

    assert 600 <= x <= 700
    assert 680 <= y <= 780


def test_overlay_text_uses_strategy_when_no_action() -> None:
    text = overlay_text_from_payload(
        {
            "overlay_display_mode": "compact",
            "last_decision": {"decision_type": "observe"},
            "round_state": {
                "current_plan": "主线：围绕索子 122334 推进",
                "attack_defense_bias": "attack",
                "last_update_reason": "opening_plan",
                "target_shapes": ["主线：围绕索子 122334 推进"],
                "caution_points": ["路线选择：主线打西；保守打7万", "优先清理：2万、7万、西、北"],
            },
        }
    )

    assert "本地" in text
    assert "方向：围绕索子 122334 推进" in text
    assert "…" not in text


def test_overlay_text_shows_detailed_open_hand_lines() -> None:
    text = overlay_text_from_payload(
        {
            "overlay_display_mode": "beginner",
            "last_decision": {"decision_type": "observe"},
            "round_state": {
                "local_direction": "副露一向听",
                "local_plan": "已副露2组，优先打向听最低、有效牌最多的牌；留6索、7索、8索、9索，先看打6索，当前1向听",
                "target_shapes": ["已成役：役牌副露 白", "保留：6索、7索、8索、9索"],
                "caution_points": [
                    "副露收束：主线打6索；不硬染打2万",
                    "副露牌效：估算2组，当前1向听；打6索后1向听，有效8种28枚，已扣可见牌（1万、2万）。",
                    "鸣牌：继续快攻，但只开能进听或明显增加有效牌的牌。",
                ],
            },
        }
    )

    assert "目标：已经接近听牌" in text
    assert "役：役牌副露 白" in text
    assert "先留这些：6索、7索、8索（共4张）" in text
    assert "吃碰杠规则：" in text
    assert "流程：" not in text
    assert "..." not in text


def test_overlay_text_shows_round_idle_without_old_plan() -> None:
    text = overlay_text_from_payload(
        {
            "last_decision": {"decision_type": "round_idle", "summary": "等待下一局"},
            "round_state": {"current_plan": "上一局旧主线"},
        }
    )

    assert "等待下一局" in text
    assert "上一局已结束" in text
    assert "上一局旧主线" not in text


def test_live_menu_obstruction_preserves_old_round_after_missing_hand_streak() -> None:
    plugin = MahjongCoachPlugin.__new__(MahjongCoachPlugin)
    plugin._engine = RoundCoachEngine(MahjongCoachConfig())
    plugin._engine.state.opening_emitted = True
    plugin._engine.state.current_plan = "当前局主线"
    plugin._engine.state.last_hand_tiles = list(HAND[:-1])
    plugin._engine.state.last_discard_piles = {"self": [{"tile": "1m"}, {"tile": "2m"}]}
    plugin._live_state = LiveSessionState(running=True)
    plugin._live_last_hand_signature = "old"
    plugin._live_last_checkpoint_at = 10.0
    plugin._last_decision = {}
    plugin.logger = SimpleNamespace(info=lambda *_args, **_kwargs: None)
    decision = SimpleNamespace(
        action_required=False,
        hand_tiles=[],
        reason_codes=["hand_unstable_hand_count"],
        perception={"hand": {"ok": False, "reason": "unstable_hand_count"}},
    )

    assert plugin._classify_live_hand_gap(decision) == "none"
    assert plugin._classify_live_hand_gap(decision) == "none"
    assert plugin._classify_live_hand_gap(decision) == "none"
    assert plugin._classify_live_hand_gap(decision) == "view_obstructed"
    assert plugin._engine.state.current_plan == "当前局主线"
    assert plugin._engine.state.last_discard_piles["self"] == [{"tile": "1m"}, {"tile": "2m"}]
    assert plugin._engine.state.opening_emitted is True
    assert plugin._live_state.missing_hand_frames == 4


def test_live_fingerprint_match_does_not_start_an_obstruction_gap() -> None:
    plugin = MahjongCoachPlugin.__new__(MahjongCoachPlugin)
    plugin._engine = RoundCoachEngine(MahjongCoachConfig())
    plugin._engine.state.opening_emitted = True
    plugin._engine.state.last_hand_tiles = list(HAND[:-1])
    plugin._live_state = LiveSessionState(running=True)
    plugin.logger = SimpleNamespace(info=lambda *_args, **_kwargs: None)
    decision = SimpleNamespace(
        action_required=False,
        hand_tiles=[],
        reason_codes=["coach_observe", "hand_fingerprint_match"],
        perception={"hand": {"ok": False, "reason": "fingerprint_match"}},
    )

    for _ in range(8):
        assert plugin._classify_live_hand_gap(decision) == "none"

    assert plugin._live_state.missing_hand_frames == 0


def test_live_static_menu_reaches_obstructed_state_through_fingerprint_frames() -> None:
    plugin = MahjongCoachPlugin.__new__(MahjongCoachPlugin)
    plugin._engine = RoundCoachEngine(MahjongCoachConfig())
    plugin._engine.state.opening_emitted = True
    plugin._engine.state.current_plan = "保留当前策略"
    plugin._engine.state.last_hand_tiles = list(HAND[:-1])
    plugin._live_state = LiveSessionState(running=True)
    plugin.logger = SimpleNamespace(info=lambda *_args, **_kwargs: None)
    first_missing = SimpleNamespace(
        action_required=False,
        hand_tiles=[],
        reason_codes=["hand_unstable_hand_count"],
        perception={"hand": {"ok": False, "reason": "unstable_hand_count"}},
    )
    unchanged_menu = SimpleNamespace(
        action_required=False,
        hand_tiles=[],
        reason_codes=["coach_observe", "hand_fingerprint_match"],
        perception={"hand": {"ok": False, "reason": "fingerprint_match"}},
    )

    assert plugin._classify_live_hand_gap(first_missing) == "none"
    assert plugin._classify_live_hand_gap(unchanged_menu) == "none"
    assert plugin._classify_live_hand_gap(unchanged_menu) == "none"
    assert plugin._classify_live_hand_gap(unchanged_menu) == "view_obstructed"
    assert plugin._engine.state.current_plan == "保留当前策略"


def test_live_menu_obstruction_resumes_same_round_when_hand_returns() -> None:
    plugin = MahjongCoachPlugin.__new__(MahjongCoachPlugin)
    plugin._engine = RoundCoachEngine(MahjongCoachConfig())
    plugin._engine.state.opening_emitted = True
    plugin._engine.state.current_plan = "保留当前策略"
    plugin._engine.state.last_hand_tiles = list(HAND[:-1])
    plugin._live_state = LiveSessionState(running=True, missing_hand_frames=4)
    plugin._live_gap_hand_tiles = list(HAND[:-1])
    plugin._live_gap_candidate_tiles = []
    plugin._live_gap_candidate_frames = 0
    plugin.logger = SimpleNamespace(info=lambda *_args, **_kwargs: None)
    resumed_hand = [*HAND[:-2], "4z"]
    decision = SimpleNamespace(
        action_required=False,
        hand_tiles=resumed_hand,
        reason_codes=["coach_observe"],
        perception={"hand": {"ok": True, "reason": "recognized_hand", "hand_tiles": resumed_hand}},
    )

    assert plugin._classify_live_hand_gap(decision) == "resumed"
    assert plugin._live_state.missing_hand_frames == 0
    assert plugin._engine.state.current_plan == "保留当前策略"


def test_live_menu_gap_requires_two_stable_new_hands_before_round_reset() -> None:
    plugin = MahjongCoachPlugin.__new__(MahjongCoachPlugin)
    plugin._engine = RoundCoachEngine(MahjongCoachConfig())
    old_hand = ["1m", "1m", "2m", "3m", "4p", "5p", "6p", "2s", "3s", "4s", "1z", "2z", "3z"]
    new_hand = ["4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "6s", "7s", "8s", "5z"]
    plugin._engine.state.opening_emitted = True
    plugin._engine.state.current_plan = "上一局策略"
    plugin._engine.state.last_hand_tiles = list(new_hand)
    plugin._live_state = LiveSessionState(running=True, missing_hand_frames=4)
    plugin._live_gap_hand_tiles = list(old_hand)
    plugin._live_gap_candidate_tiles = []
    plugin._live_gap_candidate_frames = 0
    plugin.logger = SimpleNamespace(info=lambda *_args, **_kwargs: None)
    decision = SimpleNamespace(
        action_required=False,
        hand_tiles=list(new_hand),
        reason_codes=["coach_observe"],
        perception={"hand": {"ok": True, "reason": "recognized_hand", "hand_tiles": list(new_hand)}},
    )

    assert plugin._classify_live_hand_gap(decision) == "verifying_new_round"
    assert plugin._engine.state.current_plan == "上一局策略"
    assert plugin._classify_live_hand_gap(decision) == "new_round"
    assert plugin._live_state.observed_hand_changes == 0
    assert plugin._live_state.missing_hand_frames == 0


def test_yolo_menu_gap_does_not_reset_from_changed_hand_without_river_reset() -> None:
    plugin = MahjongCoachPlugin.__new__(MahjongCoachPlugin)
    plugin._engine = RoundCoachEngine(MahjongCoachConfig(tile_recognition_mode="yolo26"))
    old_hand = ["1m", "1m", "2m", "3m", "4p", "5p", "6p", "2s", "3s", "4s", "1z", "2z", "3z"]
    changed_hand = ["4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "6s", "7s", "8s", "5z"]
    plugin._engine.state.opening_emitted = True
    plugin._engine.state.current_plan = "同一局策略"
    plugin._engine.state.last_hand_tiles = list(changed_hand)
    plugin._live_state = LiveSessionState(running=True, missing_hand_frames=4)
    plugin._live_gap_hand_tiles = list(old_hand)
    plugin._live_gap_candidate_tiles = []
    plugin._live_gap_candidate_frames = 0
    plugin.logger = SimpleNamespace(info=lambda *_args, **_kwargs: None)
    decision = SimpleNamespace(
        action_required=False,
        hand_tiles=list(changed_hand),
        reason_codes=["coach_observe"],
        perception={"hand": {"ok": True, "reason": "recognized_yolo26_hand", "hand_tiles": list(changed_hand)}},
    )

    assert plugin._classify_live_hand_gap(decision) == "resumed"
    assert plugin._engine.state.current_plan == "同一局策略"
    assert plugin._live_state.missing_hand_frames == 0


def test_yolo26_new_round_signal_resets_stale_round_after_two_frames() -> None:
    engine = RoundCoachEngine(MahjongCoachConfig(tile_recognition_mode="yolo26"))
    old_hand = ["1m", "1m", "2m", "3m", "4p", "5p", "6p", "2s", "3s", "4s", "1z", "2z", "3z"]
    new_hand = ["4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "6s", "7s", "8s", "5z"]
    engine.state.opening_emitted = True
    engine.state.current_plan = "old plan"
    engine.state.last_hand_tiles = list(old_hand)
    engine.state.last_discard_piles = {
        "self": [{"tile": "1m"}, {"tile": "2m"}],
        "right_opponent": [{"tile": "3m"}, {"tile": "4m"}],
    }
    engine.state.riichi_players = ["right_opponent"]
    hand_result = FastHandResult(ok=True, hand_tiles=list(new_hand), confidence=0.96, reason="recognized_yolo26_hand")
    meld_result = MeldStateResult(ok=False, open_meld_count=0, reason="no_self_melds")
    table_result = Yolo26TableStateResult(
        ok=True,
        hand_tiles=list(new_hand),
        discard_piles={},
        visible_tiles=[],
        confidence=0.95,
        reason="recognized_yolo26_table",
    )
    engine._last_yolo26_path = Path("new-round.png")
    engine._last_yolo26_result = table_result

    first = engine._maybe_confirm_yolo26_new_round(
        path=Path("new-round.png"),
        previous_hand_tiles=old_hand,
        hand_result=hand_result,
        meld_result=meld_result,
        started=0.0,
    )
    second = engine._maybe_confirm_yolo26_new_round(
        path=Path("new-round.png"),
        previous_hand_tiles=new_hand,
        hand_result=hand_result,
        meld_result=meld_result,
        started=0.0,
    )

    assert first is None
    assert second is not None
    assert second.decision_type == "opening_plan"
    assert "auto_new_round_detected" in second.reason_codes
    assert engine.state.round_id == "auto-round-1"
    assert engine.state.riichi_players == []
    assert engine.state.last_discard_piles == {}
    assert engine.state.last_update_reason == "auto_new_round_detected"


def test_yolo26_new_round_signal_ignores_normal_one_tile_hand_change() -> None:
    engine = RoundCoachEngine(MahjongCoachConfig(tile_recognition_mode="yolo26"))
    old_hand = ["1m", "1m", "2m", "3m", "4p", "5p", "6p", "2s", "3s", "4s", "1z", "2z", "3z"]
    next_hand = [*old_hand[:-1], "4z"]
    engine.state.opening_emitted = True
    engine.state.last_hand_tiles = list(old_hand)
    engine.state.last_discard_piles = {"self": [{"tile": str(index)} for index in range(4)]}
    hand_result = FastHandResult(ok=True, hand_tiles=list(next_hand), confidence=0.96, reason="recognized_yolo26_hand")
    meld_result = MeldStateResult(ok=False, open_meld_count=0, reason="no_self_melds")
    engine._last_yolo26_path = Path("normal-turn.png")
    engine._last_yolo26_result = Yolo26TableStateResult(
        ok=True,
        hand_tiles=list(next_hand),
        discard_piles={},
        visible_tiles=[],
        confidence=0.95,
    )

    decision = engine._maybe_confirm_yolo26_new_round(
        path=Path("normal-turn.png"),
        previous_hand_tiles=old_hand,
        hand_result=hand_result,
        meld_result=meld_result,
        started=0.0,
    )

    assert decision is None
    assert engine._new_round_candidate_frames == 0
    assert engine.state.round_id == "default"


def test_live_round_transition_resets_live_turn_counters() -> None:
    plugin = MahjongCoachPlugin.__new__(MahjongCoachPlugin)
    plugin._engine = RoundCoachEngine(MahjongCoachConfig())
    plugin._engine.state.round_id = "auto-round-1"
    plugin._engine.state.last_hand_signature = "1m|2m|3m"
    plugin._live_state = LiveSessionState(running=True, observed_hand_changes=9, missing_hand_frames=3)
    plugin._live_last_hand_signature = "old"
    plugin._live_last_checkpoint_at = 123.0
    logged: list[tuple] = []
    plugin.logger = SimpleNamespace(info=lambda *args, **_kwargs: logged.append(args))
    decision = SimpleNamespace(
        reason_codes=["first_stable_hand", "auto_new_round_detected"],
        engine_meta={"previous_river_count": 18, "current_river_count": 0},
    )

    assert plugin._observe_live_round_transition(decision) is True
    assert plugin._live_state.observed_hand_changes == 0
    assert plugin._live_state.missing_hand_frames == 0
    assert plugin._live_last_checkpoint_at == 0.0
    assert plugin._live_last_hand_signature == "1m|2m|3m"
    assert logged


class _FakeOverlay:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def start(self) -> bool:
        self.calls.append("start")
        return True

    def show_config(self) -> None:
        self.calls.append("config")

    def show_strategy(self) -> None:
        self.calls.append("strategy")

    def update_payload(self, *, text: str, detail: str = "", image_path: str = "") -> None:
        self.calls.append(f"payload:{bool(text)}:{bool(detail)}:{bool(image_path)}")
