from __future__ import annotations

import asyncio
from collections import Counter
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from plugin.plugins.mahjong_coach import (
    MahjongCoachPlugin,
    _build_frame_preview_payload,
    _build_settlement_diagnostic_preview_payload,
    _build_table_region_preview_payload,
    _resolve_preview_frame_path,
)
from plugin.plugins.mahjong_coach import coach as coach_module
from plugin.plugins.mahjong_coach.coach import (
    ORPHAN_TYPES,
    RoundCoachEngine,
    _efficiency_analysis,
    _risk_budget_breakdown,
    _riichi_waits_after_discard,
    _visible_counts,
    analyze_call_options,
    build_round_plan,
    rank_discard_decisions,
)
from plugin.plugins.mahjong_coach.models import LiveSessionState, MahjongCoachConfig
from plugin.plugins.mahjong_coach.overlay import _overlay_geometry, overlay_detail_text_from_payload, overlay_text_from_payload
from plugin.plugins.mahjong_coach.perception.fast_hand_path import FastHandResult
from plugin.plugins.mahjong_coach.perception.game_scene import GameSceneResult
from plugin.plugins.mahjong_coach.perception.meld_state import MeldStateResult
from plugin.plugins.mahjong_coach.perception.river_state import RiverStateResult
from plugin.plugins.mahjong_coach.perception.settlement_detector import (
    SettlementFrameResult,
    SettlementTransition,
)
from plugin.plugins.mahjong_coach.perception.table_context import TableContextResult
from plugin.plugins.mahjong_coach.perception.table_surface import TableSurfaceResult
from plugin.plugins.mahjong_coach.perception.yolo26_visible_tiles import Yolo26TableStateResult
from plugin.plugins.mahjong_coach.tile_labels import hand_signature


HAND = ["1m", "2m", "3m", "4p", "5p", "6p", "2s", "3s", "4s", "5s", "6s", "7s", "1z", "1z"]


def test_risk_budget_breakdown_exposes_every_threshold_adjustment() -> None:
    model = _risk_budget_breakdown(
        shanten=1,
        riichi_count=2,
        turn_number=13,
        estimated_value=3900,
        risk_tolerance="aggressive",
        room="jade",
        rank="saint",
    )

    assert model["final"] == 41.0
    assert model["calculation"] == (
        "一向听基础44 激进风格+12 2家立直压力−12 12巡后−7 "
        "预计3900点以上+4 玉之间−2 雀圣+2 = 上限41"
    )


def test_risk_budget_uses_score_rank_honba_and_riichi_sticks_transparently() -> None:
    model = _risk_budget_breakdown(
        shanten=1,
        riichi_count=1,
        turn_number=8,
        estimated_value=2000,
        risk_tolerance="balanced",
        room="unknown",
        rank="unknown",
        player_scores={
            "self": 12000,
            "left_opponent": 20000,
            "top_opponent": 31000,
            "right_opponent": 34000,
        },
        player_ranks={
            "self": 4,
            "left_opponent": 3,
            "top_opponent": 2,
            "right_opponent": 1,
        },
        honba_count=3,
        riichi_stick_count=1,
    )

    assert model["table_reward_bonus"] == 1900
    assert model["estimated_value_with_table_rewards"] == 3900
    assert model["placement_adjustment"] == 8.0
    assert model["final"] == 56.0
    assert "预计3900点以上（含本场/供托）+4" in model["calculation"]
    assert "四位追三位差8000点+8" in model["calculation"]


def test_table_context_scans_once_per_round_and_updates_strategy_state(monkeypatch) -> None:
    engine = RoundCoachEngine()
    surface = TableSurfaceResult(
        ok=True,
        warped_image=Image.new("RGB", (800, 800), "black"),
    )
    scene = GameSceneResult(detected=True, table_surface=surface)
    result = TableContextResult(
        ok=True,
        scores={"self": 25000, "left_opponent": 25000, "top_opponent": 25000, "right_opponent": 25000},
        ranks={"self": 1, "left_opponent": 1, "top_opponent": 1, "right_opponent": 1},
        honba_count=0,
        riichi_stick_count=0,
        confidence=0.9,
        reason="table_context_detected",
    )
    calls = []
    monkeypatch.setattr(engine, "_detect_game_scene", lambda _path: scene)
    monkeypatch.setattr(
        coach_module,
        "detect_table_context",
        lambda *_args, **_kwargs: calls.append("ocr") or result,
    )

    engine._observe_table_context(Image.new("RGB", (1000, 600), "black"))
    assert engine.state.player_scores["self"] == 25000
    assert engine.state.honba_count == 0
    assert engine.state.table_riichi_stick_count == 0
    assert engine.state.table_context_pending_frames == 0
    assert calls == ["ocr"]

    engine._observe_table_context(Image.new("RGB", (1000, 600), "black"))
    assert calls == ["ocr"]

    engine.reset_round("next")
    engine._observe_table_context(Image.new("RGB", (1000, 600), "black"))
    assert calls == ["ocr", "ocr"]


def test_round_plan_uses_fourth_place_gap_and_table_rewards() -> None:
    plan = build_round_plan(
        HAND,
        MahjongCoachConfig(),
        player_scores={
            "self": 12000,
            "left_opponent": 20000,
            "top_opponent": 31000,
            "right_opponent": 34000,
        },
        player_ranks={
            "self": 4,
            "left_opponent": 3,
            "top_opponent": 2,
            "right_opponent": 1,
        },
        honba_count=2,
        riichi_stick_count=1,
    )

    assert plan["bias"] == "attack"
    assert "当前四位" in plan["summary"]
    assert "距三位8000点" in plan["targets"][0]
    assert any("额外增加1600点" in target for target in plan["targets"])


def test_scene_and_yolo_caches_invalidate_when_same_path_is_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame_path = tmp_path / "live-frame.png"
    frame_path.write_bytes(b"frame-a")
    original_stat = frame_path.stat()
    scene_calls: list[Path] = []
    yolo_calls: list[Path] = []

    def fake_scene(path: Path) -> GameSceneResult:
        scene_calls.append(path)
        return GameSceneResult(reason=f"scene-{len(scene_calls)}")

    def fake_yolo(path: Path, *, table_surface_result=None) -> Yolo26TableStateResult:
        del table_surface_result
        yolo_calls.append(path)
        return Yolo26TableStateResult(reason=f"yolo-{len(yolo_calls)}")

    monkeypatch.setattr(coach_module, "detect_game_scene_path", fake_scene)
    monkeypatch.setattr(coach_module, "detect_yolo26_table_state_path", fake_yolo)
    engine = RoundCoachEngine(MahjongCoachConfig(tile_recognition_mode="yolo26"))

    first_scene = engine._detect_game_scene(frame_path)
    first_yolo = engine._detect_yolo26_table(frame_path)
    assert engine._detect_game_scene(frame_path) is first_scene
    assert engine._detect_yolo26_table(frame_path) is first_yolo

    frame_path.write_bytes(b"frame-b")
    os.utime(frame_path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    second_scene = engine._detect_game_scene(frame_path)
    second_yolo = engine._detect_yolo26_table(frame_path)

    assert second_scene is not first_scene
    assert second_yolo is not first_yolo
    assert len(scene_calls) == 2
    assert len(yolo_calls) == 2

    engine.reset_round("next")
    assert engine._last_game_scene_result is None
    assert engine._last_game_scene_identity is None
    assert engine._last_yolo26_result is None
    assert engine._last_yolo26_identity is None


def test_frame_preview_payload_is_compact_jpeg(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (1920, 1080), (20, 54, 83)).save(image_path)

    payload = _build_frame_preview_payload(image_path)

    assert payload["image_path"] == str(image_path)
    assert payload["data_url"].startswith("data:image/jpeg;base64,")
    assert payload["width"] == 960
    assert payload["height"] == 540


def test_preview_frame_path_accepts_only_existing_live_frames(tmp_path: Path) -> None:
    frames_dir = tmp_path / "live_frames"
    frames_dir.mkdir()
    current_frame = frames_dir / "current.jpg"
    current_frame.write_bytes(b"frame")
    outside_frame = tmp_path / "outside.jpg"
    outside_frame.write_bytes(b"outside")

    assert _resolve_preview_frame_path(str(current_frame), frames_dir) == current_frame.resolve()
    assert _resolve_preview_frame_path(str(outside_frame), frames_dir) is None
    assert _resolve_preview_frame_path(str(frames_dir / "missing.jpg"), frames_dir) is None


def test_live_scene_gate_blocks_hand_and_strategy_before_game(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = RoundCoachEngine(MahjongCoachConfig(tile_recognition_mode="yolo26"))
    monkeypatch.setattr(
        engine,
        "_handle_settlement_boundary",
        lambda _path, _started: (_ for _ in ()).throw(
            AssertionError("settlement recognition must stay paused before a game")
        ),
    )
    monkeypatch.setattr(
        engine,
        "_detect_game_scene",
        lambda _path: GameSceneResult(reason="center_score_panel_not_found"),
    )
    monkeypatch.setattr(
        engine,
        "_detect_hand",
        lambda _path: (_ for _ in ()).throw(AssertionError("hand recognition must stay paused")),
    )

    decision = engine.analyze_frame("lobby.png", require_game_scene=True)

    assert decision.decision_type == "waiting_for_game"
    assert decision.action_required is False
    assert decision.perception["action"]["skipped"] is True
    assert decision.perception["hand"]["reason"] == "game_scene_gate"
    assert engine.state.opening_emitted is False


def test_live_scene_gate_requires_two_frames_before_opening(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = RoundCoachEngine(MahjongCoachConfig())
    monkeypatch.setattr(engine, "_handle_settlement_boundary", lambda _path, _started: None)
    monkeypatch.setattr(
        engine,
        "_detect_game_scene",
        lambda _path: GameSceneResult(
            detected=True,
            confidence=0.94,
            reason="active_mahjong_table",
        ),
    )
    monkeypatch.setattr(
        engine,
        "_detect_hand",
        lambda _path: FastHandResult(
            ok=True,
            hand_tiles=list(HAND),
            confidence=0.96,
            reason="test_hand",
        ),
    )
    monkeypatch.setattr(
        engine,
        "_detect_melds",
        lambda _path, **_kwargs: MeldStateResult(reason="no_self_melds"),
    )
    monkeypatch.setattr(engine, "_detect_riichi_players", lambda _path: [])

    first = engine.analyze_frame("table.png", require_game_scene=True)
    second = engine.analyze_frame("table.png", require_game_scene=True)

    assert first.decision_type == "waiting_for_game"
    assert "game_scene_confirmation_pending" in first.reason_codes
    assert second.decision_type == "opening_plan"
    assert engine.state.opening_emitted is True


def test_live_scene_loss_freezes_existing_round(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = RoundCoachEngine(MahjongCoachConfig())
    engine.state.opening_emitted = True
    engine.state.current_plan = "保留当前局策略"
    engine.state.last_hand_tiles = list(HAND)
    engine._game_scene_confirmed = True
    monkeypatch.setattr(engine, "_handle_settlement_boundary", lambda _path, _started: None)
    monkeypatch.setattr(
        engine,
        "_detect_game_scene",
        lambda _path: GameSceneResult(reason="center_score_panel_not_found"),
    )
    monkeypatch.setattr(
        engine,
        "_resolve_buttons",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("button scan must stay paused")),
    )

    decision = engine.analyze_frame("menu.png", require_game_scene=True)

    assert decision.decision_type == "waiting_for_game"
    assert engine.state.opening_emitted is True
    assert engine.state.current_plan == "保留当前局策略"
    assert engine.state.last_hand_tiles == HAND


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
        opponent_melds={
            "top_opponent": [
                {
                    "owner": "top_opponent",
                    "meld_index": 1,
                    "kind": "chi",
                    "tiles": ["3m", "4m", "5m"],
                    "bbox": [60.0, 50.0, 160.0, 110.0],
                }
            ]
        },
    )

    assert payload["transformed"] is True
    assert payload["input_space"] == "warped_table"
    assert payload["width"] == 800
    assert payload["height"] == 800
    assert payload["detection_count"] == 1
    assert payload["opponent_meld_count"] == 1
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
async def test_status_and_native_overlay_share_last_published_strategy_snapshot() -> None:
    class OverlayStub:
        prefs_path = None

        def __init__(self) -> None:
            self.payload: dict[str, str] = {}

        def update_payload(
            self,
            *,
            text: str,
            strategy_card_text: str,
            strategy_card: dict,
            detail: str,
            image_path: str,
        ) -> None:
            self.payload = {
                "text": text,
                "strategy_card_text": strategy_card_text,
                "strategy_card": strategy_card,
                "detail": detail,
                "image_path": image_path,
            }

    plugin = MahjongCoachPlugin.__new__(MahjongCoachPlugin)
    plugin.ctx = SimpleNamespace(plugin_id="mahjong_coach")
    plugin._cfg = MahjongCoachConfig()
    plugin._engine = RoundCoachEngine(plugin._cfg)
    plugin._engine_lock = asyncio.Lock()
    plugin._live_state = LiveSessionState(overlay_enabled=True)
    plugin._live_timing_log = []
    plugin._display_snapshot = {}
    plugin._display_revision = 0
    plugin._overlay = OverlayStub()
    published_decision = {
        "decision_type": "defense_alert",
        "action_required": True,
        "suggestion": "兜牌：打1万，保持听牌并保留和牌路线。",
        "perception": {
            "strategy": {
                "posture": "mawashi",
                "risk_budget": 62.0,
                "top_candidates": [{"tile": "1m", "defense_risk": 58.0, "shanten": 0, "effective_count": 8}],
            }
        },
    }
    published_state = {"defense_posture": "mawashi", "defense_risk_budget": 62.0}
    plugin._update_overlay({"last_decision": published_decision, "round_state": published_state})

    # A quiet diagnostic frame becomes the latest engine result, but it must
    # not make the dashboard diverge from the still-visible native overlay.
    plugin._last_decision = {"decision_type": "coach_checkpoint", "quiet": True, "perception": {}}
    payload = (await plugin.mahjong_coach_status()).unwrap()

    assert payload["last_decision"]["quiet"] is True
    assert payload["display_snapshot"]["last_decision"]["decision_type"] == "defense_alert"
    assert payload["display_snapshot"]["last_decision"]["perception"]["strategy"]["top_candidates"][0]["tile"] == "1m"
    assert payload["overlay_text"] == plugin._overlay.payload["text"]
    assert payload["display_snapshot"]["overlay_text"] == plugin._overlay.payload["text"]
    assert payload["display_snapshot"]["strategy_card_text"] == plugin._overlay.payload["strategy_card_text"]
    assert payload["display_snapshot"]["strategy_card"] == plugin._overlay.payload["strategy_card"]
    assert payload["display_snapshot"]["strategy_card"]["focus_tile"] == "1万"
    assert "本地兜牌 · 守中求和" in payload["overlay_text"]


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


def test_tsumo_window_uses_the_same_win_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = RoundCoachEngine(MahjongCoachConfig())
    engine.state.opening_emitted = True

    def fail_if_called(_path: Path | None) -> FastHandResult:
        raise AssertionError("hand scan should not run for tsumo windows")

    monkeypatch.setattr(engine, "_detect_hand", fail_if_called)

    decision = engine.analyze_frame(observed_buttons=["tsumo"])

    assert decision.decision_type == "win_window"
    assert decision.priority == 100
    assert decision.action_required is True
    assert decision.buttons == ["tsumo"]
    assert decision.reason_codes == ["critical_action_interrupt"]


def test_blue_table_frame_never_latches_a_false_win_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frame_path = tmp_path / "ordinary_blue_table.png"
    image = Image.new("RGB", (1920, 1080), (43, 82, 126))
    draw = ImageDraw.Draw(image)
    draw.line((250, 880, 960, 610, 1670, 880), fill=(32, 61, 98), width=7)
    for column in range(7):
        left = 720 + column * 70
        draw.rounded_rectangle((left, 620, left + 58, 715), radius=5, fill=(188, 196, 193), outline=(45, 53, 57), width=3)
    image.save(frame_path)

    engine = RoundCoachEngine(MahjongCoachConfig())
    engine.state.opening_emitted = True
    monkeypatch.setattr(engine, "_handle_settlement_boundary", lambda _path, _started: None)
    monkeypatch.setattr(
        engine,
        "_detect_and_remember",
        lambda _path: (FastHandResult(reason="test_no_hand"), MeldStateResult(reason="test_no_meld")),
    )

    decisions = [engine.analyze_frame(frame_path) for _ in range(3)]

    assert [decision.decision_type for decision in decisions] == ["observe", "observe", "observe"]
    assert all(decision.buttons == [] for decision in decisions)
    assert all(decision.perception["action"]["detected_buttons"] == [] for decision in decisions)


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


def test_river_fingerprint_change_bypasses_static_hand_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = RoundCoachEngine(MahjongCoachConfig(river_tracking_mode="live"))
    engine.state.opening_emitted = True
    engine.state.update_count = 2
    engine.state.last_hand_tiles = list(HAND)
    river_calls: list[Path | None] = []
    riichi_calls: list[Path | None] = []

    monkeypatch.setattr(engine, "_handle_settlement_boundary", lambda _path, _started: None)
    monkeypatch.setattr(engine, "_resolve_buttons", lambda _path, _buttons: ([], {"source": "test"}))
    monkeypatch.setattr(
        coach_module,
        "quick_frame_fingerprint",
        lambda *_args, **_kwargs: {
            "action_changed": False,
            "hand_changed": False,
            "river_changed": True,
            "river_changes": {"top_opponent": True},
            "hashes": {},
        },
    )
    monkeypatch.setattr(
        engine,
        "_detect_and_remember",
        lambda _path: (
            FastHandResult(ok=True, hand_tiles=list(HAND), confidence=0.95, reason="test_hand"),
            MeldStateResult(reason="no_self_melds"),
        ),
    )
    monkeypatch.setattr(
        engine,
        "_track_river_for_frame",
        lambda path, _reason: river_calls.append(path) or RiverStateResult(ok=True, reason="river_changed"),
    )
    monkeypatch.setattr(
        engine,
        "_detect_riichi_players",
        lambda path: riichi_calls.append(path) or [],
    )

    decision = engine.analyze_frame("frame.png")

    assert decision.decision_type == "observe"
    assert river_calls == [Path("frame.png")]
    assert riichi_calls == [Path("frame.png")]


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


def test_river_tracking_preserves_and_restores_structured_opponent_melds() -> None:
    engine = RoundCoachEngine(MahjongCoachConfig(river_tracking_mode="live"))
    top_pon = {
        "owner": "top_opponent",
        "meld_index": 1,
        "kind": "pon",
        "tiles": ["5z", "5z", "5z"],
        "tile_identity_reliable": True,
        "bbox": [70, 70, 160, 120],
    }
    engine.state.last_opponent_melds = {"top_opponent": [top_pon]}
    engine.state.last_opponent_meld_tiles = ["5z", "5z", "5z"]
    engine.state.river_tracking_initialized = True

    merged = engine._merge_incremental_river(
        RiverStateResult(ok=True, reason="no_new_discards")
    )
    engine._remember_river(merged)
    restored = engine._river_result_from_state("cached_after_occlusion")

    assert merged.opponent_melds == {"top_opponent": [top_pon]}
    assert merged.opponent_meld_tiles == ["5z", "5z", "5z"]
    assert restored.ok is True
    assert restored.opponent_melds == merged.opponent_melds
    assert restored.opponent_meld_tiles == ["5z", "5z", "5z"]


def test_opponent_meld_tiles_participate_in_visible_wall_counts() -> None:
    river = RiverStateResult(
        ok=True,
        discard_piles={"right_opponent": [{"tile": "1m"}]},
        visible_tiles=["1m"],
        opponent_melds={
            "top_opponent": [
                {
                    "owner": "top_opponent",
                    "meld_index": 1,
                    "kind": "pon",
                    "tiles": ["5z", "5z", "5z"],
                    "tile_identity_reliable": True,
                }
            ]
        },
        opponent_meld_tiles=["5z", "5z", "5z"],
    )

    visible_tiles = coach_module._visible_tiles_for_plan(river)
    defense = coach_module._defense_options(
        ["5z", "2m"],
        ["shimocha"],
        river.discard_piles,
        visible_tiles=visible_tiles,
    )

    assert visible_tiles == ["1m", "5z", "5z", "5z"]
    assert defense["fully_visible"] == ["5z"]


def test_called_discard_stays_genbutsu_history_but_counts_only_once() -> None:
    engine = RoundCoachEngine(MahjongCoachConfig(river_tracking_mode="live", tile_recognition_mode="yolo26"))
    engine.state.last_discard_piles = {
        "top_opponent": [
            {
                "tile": "5z",
                "player": "top_opponent",
                "turn_index": 3,
                "slot_id": "discard_top_opponent_03",
                "bbox": [360, 250, 400, 300],
                "confidence": 0.96,
            }
        ]
    }
    engine.state.last_visible_discards = ["5z"]
    engine.state.river_tracking_initialized = True
    pon = {
        "owner": "right_opponent",
        "meld_index": 1,
        "kind": "pon",
        "tiles": ["5z", "5z", "5z"],
        "called_tile_index": 1,
        "tile_identity_reliable": True,
        "bbox": [680, 120, 750, 250],
    }
    snapshot = RiverStateResult(
        ok=True,
        discard_piles={"top_opponent": []},
        opponent_melds={"right_opponent": [pon]},
        opponent_meld_tiles=["5z", "5z", "5z"],
        reason="recognized_yolo26_discards",
    )

    reconciled = engine._reconcile_full_river(snapshot, confirm_new=False)

    historical = reconciled.discard_piles["top_opponent"][0]
    linked_meld = reconciled.opponent_melds["right_opponent"][0]
    assert historical["claimed_into_meld"] is True
    assert historical["claimed_by"] == "right_opponent"
    assert historical["claimed_meld_index"] == 1
    assert reconciled.visible_tiles == ["5z"]
    assert coach_module._visible_tiles_for_plan(reconciled) == ["5z", "5z", "5z"]
    assert reconciled.analysis_hints["physical_visible_discard_count"] == 0
    assert reconciled.analysis_hints["claimed_discard_link_count"] == 1
    assert linked_meld["called_from_owner"] == "top_opponent"

    defense = coach_module._defense_options(
        ["5z"],
        ["toimen"],
        reconciled.discard_piles,
        visible_tiles=coach_module._visible_tiles_for_plan(reconciled),
    )
    assert defense["exact_safe"] == ["5z"]
    assert defense["fully_visible"] == ["5z"]

    engine._remember_river(reconciled)
    refreshed = engine._reconcile_full_river(snapshot, confirm_new=False)
    assert refreshed.discard_piles["top_opponent"][0]["claimed_into_meld"] is True
    assert refreshed.opponent_melds["right_opponent"][0]["claim_discard_linked"] is True
    assert refreshed.analysis_hints["claimed_discard_link_count"] == 0


def test_new_opponent_meld_waits_for_delayed_discard_disappearance() -> None:
    engine = RoundCoachEngine(MahjongCoachConfig(river_tracking_mode="live", tile_recognition_mode="yolo26"))
    discard = {
        "tile": "7p",
        "player": "top_opponent",
        "turn_index": 2,
        "slot_id": "discard_top_opponent_02",
        "bbox": [360, 250, 400, 300],
        "confidence": 0.95,
    }
    engine.state.last_discard_piles = {"top_opponent": [discard]}
    engine.state.last_visible_discards = ["7p"]
    engine.state.river_tracking_initialized = True
    pon = {
        "owner": "right_opponent",
        "meld_index": 1,
        "kind": "pon",
        "tiles": ["7p", "7p", "7p"],
        "called_tile_index": 2,
        "tile_identity_reliable": True,
        "bbox": [680, 120, 750, 250],
    }
    first_snapshot = RiverStateResult(
        ok=True,
        discard_piles={"top_opponent": [dict(discard)]},
        opponent_melds={"right_opponent": [pon]},
        opponent_meld_tiles=["7p", "7p", "7p"],
    )

    first = engine._reconcile_full_river(first_snapshot, confirm_new=False)
    assert "claimed_into_meld" not in first.discard_piles["top_opponent"][0]
    assert first.opponent_melds["right_opponent"][0]["claim_link_pending_scans"] == 2
    engine._remember_river(first)

    second = engine._reconcile_full_river(
        RiverStateResult(
            ok=True,
            discard_piles={"top_opponent": []},
            opponent_melds={"right_opponent": [pon]},
            opponent_meld_tiles=["7p", "7p", "7p"],
        ),
        confirm_new=False,
    )

    assert second.discard_piles["top_opponent"][0]["claimed_into_meld"] is True
    assert second.opponent_melds["right_opponent"][0]["claim_discard_linked"] is True


def test_longstanding_opponent_meld_cannot_claim_a_later_detection_dropout() -> None:
    engine = RoundCoachEngine(MahjongCoachConfig(river_tracking_mode="live", tile_recognition_mode="yolo26"))
    old_pon = {
        "owner": "right_opponent",
        "meld_index": 1,
        "kind": "pon",
        "tiles": ["3m", "3m", "3m"],
        "called_tile_index": 1,
        "tile_identity_reliable": True,
        "bbox": [680, 120, 750, 250],
    }
    engine.state.last_opponent_melds = {"right_opponent": [old_pon]}
    engine.state.last_opponent_meld_tiles = ["3m", "3m", "3m"]
    engine.state.last_discard_piles = {
        "top_opponent": [
            {
                "tile": "3m",
                "player": "top_opponent",
                "turn_index": 4,
                "bbox": [360, 250, 400, 300],
                "confidence": 0.95,
            }
        ]
    }
    engine.state.river_tracking_initialized = True

    reconciled = engine._reconcile_full_river(
        RiverStateResult(
            ok=True,
            discard_piles={"top_opponent": []},
            opponent_melds={"right_opponent": [old_pon]},
            opponent_meld_tiles=["3m", "3m", "3m"],
        ),
        confirm_new=False,
    )

    assert "claimed_into_meld" not in reconciled.discard_piles["top_opponent"][0]
    assert "claim_discard_linked" not in reconciled.opponent_melds["right_opponent"][0]
    assert coach_module._visible_tiles_for_plan(reconciled) == ["3m", "3m", "3m", "3m"]


def test_chi_claim_links_only_the_callers_left_source_player() -> None:
    engine = RoundCoachEngine(MahjongCoachConfig(river_tracking_mode="live", tile_recognition_mode="yolo26"))
    engine.state.last_discard_piles = {
        "left_opponent": [{"tile": "3m", "player": "left_opponent", "turn_index": 2, "confidence": 0.95}],
        "top_opponent": [{"tile": "3m", "player": "top_opponent", "turn_index": 2, "confidence": 0.96}],
    }
    chi = {
        "owner": "left_opponent",
        "meld_index": 1,
        "kind": "chi",
        "tiles": ["3m", "4m", "5m"],
        "called_tile_index": 0,
        "tile_identity_reliable": True,
        "bbox": [40, 560, 100, 720],
    }

    reconciled = engine._reconcile_full_river(
        RiverStateResult(
            ok=True,
            discard_piles={"left_opponent": [], "top_opponent": []},
            opponent_melds={"left_opponent": [chi]},
            opponent_meld_tiles=["3m", "4m", "5m"],
        ),
        confirm_new=False,
    )

    assert "claimed_into_meld" not in reconciled.discard_piles["left_opponent"][0]
    assert reconciled.discard_piles["top_opponent"][0]["claimed_into_meld"] is True
    assert reconciled.opponent_melds["left_opponent"][0]["called_from_owner"] == "top_opponent"


def test_self_call_marks_the_source_discard_after_new_meld_is_observed() -> None:
    engine = RoundCoachEngine(MahjongCoachConfig(river_tracking_mode="live"))
    engine.state.last_discard_piles = {
        "right_opponent": [
            {
                "tile": "5z",
                "player": "right_opponent",
                "turn_index": 2,
                "slot_id": "discard_right_opponent_02",
                "confidence": 0.96,
            }
        ]
    }
    engine._pending_self_call_claim = {
        "tile": "5z",
        "player": "right_opponent",
        "turn_index": 2,
        "slot_id": "discard_right_opponent_02",
        "buttons": ["pon"],
        "observed_at": coach_module.time.monotonic(),
        "previous_open_meld_count": 0,
    }

    engine._remember_melds(
        MeldStateResult(
            ok=True,
            melds=[{"meld_index": 1, "tiles": ["5z", "5z", "5z"]}],
            tiles=["5z", "5z", "5z"],
            open_meld_count=1,
            confidence=0.94,
            analysis_hints={"tile_identity_reliable": True},
        )
    )

    discard = engine.state.last_discard_piles["right_opponent"][0]
    assert discard["claimed_into_meld"] is True
    assert discard["claimed_by"] == "self"
    assert engine.state.last_melds[0]["claim_discard_linked"] is True
    assert engine.state.last_melds[0]["called_from_owner"] == "right_opponent"
    assert engine._pending_self_call_claim == {}
    river = RiverStateResult(
        ok=True,
        discard_piles=engine.state.last_discard_piles,
        visible_tiles=["5z"],
    )
    assert coach_module._visible_tiles_for_plan(river) == []


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


def test_full_river_rescan_appends_immediately_for_call_window() -> None:
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

    reconciled = engine._reconcile_full_river(snapshot, confirm_new=False)

    assert reconciled.visible_tiles == ["5z"]
    assert reconciled.analysis_hints["new_discard_count"] == 1
    assert reconciled.analysis_hints["river_pending_corrections"] == 0


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
    assert decision.coach_state["attack_defense_bias"] == decision.perception["strategy"]["legacy_mode"]


def test_opponent_riichi_detection_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = RoundCoachEngine(
        MahjongCoachConfig(
            opponent_riichi_recognition_enabled=False,
            tile_recognition_mode="yolo26",
        )
    )
    engine.state.riichi_pending = {"right_opponent": 1}
    monkeypatch.setattr(
        engine,
        "_detect_yolo26_table",
        lambda _path: (_ for _ in ()).throw(AssertionError("opponent riichi detector should be opt-in")),
    )

    assert engine._detect_riichi_players(Path("frame.png")) == []
    assert engine.state.riichi_pending == {}


def test_legacy_mode_does_not_infer_riichi_from_ui_counter() -> None:
    engine = RoundCoachEngine(MahjongCoachConfig())
    engine.state.riichi_stick_baseline = 0
    engine.state.last_riichi_stick_count = 1
    engine.state.riichi_pending = {"unknown": 1}

    assert engine._detect_riichi_players(Path("frame.png")) == []
    assert engine.state.riichi_pending == {}


def test_yolo_riichi_seat_requires_two_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = RoundCoachEngine(MahjongCoachConfig(tile_recognition_mode="yolo26"))
    monkeypatch.setattr(
        engine,
        "_detect_yolo26_table",
        lambda _path: SimpleNamespace(
            ok=True,
            river_inference_ok=True,
            riichi_players=["right_opponent"],
        ),
    )

    assert engine._detect_riichi_players(Path("frame.png")) == []
    assert engine._detect_riichi_players(Path("frame.png")) == ["right_opponent"]


def test_specific_riichi_seat_replaces_stale_unknown_state(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = RoundCoachEngine(MahjongCoachConfig(tile_recognition_mode="yolo26"))
    engine.state.riichi_players = ["unknown"]
    engine.state.riichi_pending = {"unknown": 2}
    monkeypatch.setattr(
        engine,
        "_detect_yolo26_table",
        lambda _path: SimpleNamespace(
            ok=True,
            river_inference_ok=True,
            riichi_players=["right_opponent"],
        ),
    )
    first = engine._detect_riichi_players(Path("frame.png"))
    second = engine._detect_riichi_players(Path("frame.png"))

    assert first == []
    assert engine.state.riichi_players == []
    assert second == ["right_opponent"]
    assert "unknown" not in engine.state.riichi_pending


def test_yolo_riichi_detection_never_invents_an_unlocated_second_player(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = RoundCoachEngine(MahjongCoachConfig(tile_recognition_mode="yolo26"))
    engine.state.riichi_players = ["right_opponent"]
    engine.state.riichi_pending = {"unknown": 1}
    monkeypatch.setattr(
        engine,
        "_detect_yolo26_table",
        lambda _path: SimpleNamespace(
            ok=True,
            river_inference_ok=True,
            riichi_players=["right_opponent"],
        ),
    )
    detected = engine._detect_riichi_players(Path("frame.png"))

    assert detected == ["right_opponent"]
    assert "unknown" not in engine.state.riichi_pending


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


def test_honor_risk_evidence_names_exact_tile_copies_and_sources() -> None:
    hand = ["1m", "2m", "3m", "4p", "5p", "6p", "2s", "3s", "4s", "5s", "6s", "7s", "1z", "5z"]
    river = RiverStateResult(
        ok=True,
        discard_piles={
            "right_opponent": [{"tile": "9m", "confidence": 0.96}],
            "left_opponent": [
                {"tile": "1z", "confidence": 0.97},
                {"tile": "6z", "confidence": 0.96},
                {"tile": "7z", "confidence": 0.95},
            ],
        },
        visible_tiles=["9m", "1z", "6z", "7z"],
        confidence=0.96,
        reason="test_river",
    )

    ranking = rank_discard_decisions(hand, MahjongCoachConfig(), ["shimocha"], river)
    east = next(item for item in ranking["candidates"] if item["tile"] == "1z")

    assert east["risk_by_player"]["right_opponent"]["basis"] == "字牌已见2枚"
    assert east["visibility"]["known_count"] == 2
    assert east["visibility"]["unseen_count"] == 2
    assert "手牌东×1" in east["visibility"]["summary"]
    assert "上家牌河东×1" in east["visibility"]["summary"]
    assert "尚有东×2未见" in east["safety_evidence"][0]
    assert "不是把其他字牌合并成2枚" in east["safety_evidence"][0]
    assert east["risk_level"] == "中等"
    assert east["risk_components"]["base_risk"] == 55.0
    assert east["risk_calculation"] == "下家字牌已见2枚基础55 = 55/100"
    assert "不是放铳概率" in east["risk_scale_note"]
    assert "当前可接受上限" in east["risk_budget_explanation"]
    assert "0–9极低" in ranking["risk_scale_legend"]
    assert "字牌已见3/2/0–1枚=38/55/72" in ranking["risk_model_legend"]
    assert "1家立直压力+0" in ranking["risk_budget_calculation"]


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


def test_live_config_defaults_to_a_small_diagnostic_ring() -> None:
    cfg = MahjongCoachConfig.from_payload({})

    assert cfg.live_interval_ms == 400
    assert cfg.live_keep_frames == 20
    assert cfg.live_save_format == "jpg"


@pytest.mark.parametrize(("configured", "expected"), [(0, 0), (-5, 0), (1, 1), (20, 20)])
def test_live_config_accepts_zero_frame_retention(configured: int, expected: int) -> None:
    cfg = MahjongCoachConfig.from_payload({"live": {"keep_frames": configured}})

    assert cfg.live_keep_frames == expected


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
async def test_start_live_explicit_overlay_choice_overrides_stale_disabled_config() -> None:
    plugin = MahjongCoachPlugin.__new__(MahjongCoachPlugin)
    overlay = _FakeOverlay()
    plugin._overlay = overlay
    plugin._cfg = MahjongCoachConfig(live_overlay_enabled=False)
    plugin._engine = RoundCoachEngine(plugin._cfg)
    plugin._live_task = None
    plugin._live_state = LiveSessionState()
    plugin._clear_live_hand_gap = lambda: None
    plugin.logger = SimpleNamespace(info=lambda *_args, **_kwargs: None, warning=lambda *_args, **_kwargs: None)

    async def idle_live_loop(**_kwargs) -> None:
        await asyncio.sleep(60)

    plugin._run_live_loop = idle_live_loop

    result = await plugin._overlay_start_live(overlay=True)

    payload = result.unwrap()
    assert payload["overlay_ready"] is True
    assert plugin._live_state.overlay_enabled is True
    assert overlay.calls == ["start", "strategy"]
    plugin._live_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await plugin._live_task


@pytest.mark.asyncio
async def test_start_live_does_not_claim_success_when_overlay_fails() -> None:
    plugin = MahjongCoachPlugin.__new__(MahjongCoachPlugin)
    overlay = _FailingOverlay("tk failed")
    plugin._overlay = overlay
    plugin._cfg = MahjongCoachConfig()
    plugin._engine = RoundCoachEngine(plugin._cfg)
    plugin._live_task = None
    plugin._live_state = LiveSessionState()
    plugin.logger = SimpleNamespace(info=lambda *_args, **_kwargs: None, warning=lambda *_args, **_kwargs: None)

    result = await plugin._overlay_start_live(overlay=True)

    assert result.is_err()
    assert plugin._live_task is None


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

    result = await plugin._overlay_start_live(
        overlay=True,
        play_style="fast",
        strategy_preset="standard",
        river_tracking_mode="live",
    )

    assert result.unwrap()["status"] == "already_running"
    assert plugin._cfg.play_style == "fast"
    assert plugin._cfg.strategy_preset == "standard"
    assert plugin._cfg.river_tracking_mode == "live"
    assert plugin._engine.config.play_style == "fast"
    assert plugin._engine.config.strategy_preset == "standard"
    assert plugin._engine.config.river_tracking_mode == "live"
    assert plugin._engine.state.play_style == "fast"
    assert plugin._engine.state.strategy_preset == "standard"
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


def test_overlay_names_mawashi_as_defense_while_preserving_win_route() -> None:
    text = overlay_text_from_payload(
        {
            "last_decision": {
                "decision_type": "defense_alert",
                "action_required": True,
                "suggestion": "兜牌：先打1万，保持听牌，继续保留和牌路线。",
                "perception": {"strategy": {"posture": "mawashi"}},
            },
            "round_state": {"defense_posture": "mawashi"},
        }
    )

    assert "本地兜牌 · 守中求和" in text
    assert "继续保留和牌路线" in text
    assert "先满足安全预算，再保留向听与有效牌" in text


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

    def update_payload(
        self,
        *,
        text: str,
        strategy_card_text: str = "",
        strategy_card: dict | None = None,
        detail: str = "",
        image_path: str = "",
    ) -> None:
        self.calls.append(
            f"payload:{bool(text)}:{bool(strategy_card_text)}:{bool(strategy_card)}:{bool(detail)}:{bool(image_path)}"
        )


class _FailingOverlay(_FakeOverlay):
    def __init__(self, error: str) -> None:
        super().__init__()
        self.last_error = error

    def start(self) -> bool:
        self.calls.append("start")
        return False
