from __future__ import annotations

import asyncio
import importlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from plugin.plugins.mahjong_coach import capture
from plugin.plugins.mahjong_coach.__init__ import MahjongCoachPlugin, _live_performance_summary
from plugin.plugins.mahjong_coach.capture import CaptureSession, DefaultCaptureProvider
from plugin.plugins.mahjong_coach.coach import _discard_ranking_text, build_round_plan, rank_discard_decisions
from plugin.plugins.mahjong_coach.models import (
    CapturePreferences,
    FramePacket,
    LiveSessionState,
    MahjongCoachConfig,
    PlayerProfile,
    WindowTargetDescriptor,
)
from plugin.plugins.mahjong_coach.perception.river_state import RiverStateResult
from plugin.plugins.mahjong_coach.player_profile import (
    AmaeKoromoProvider,
    ProfileLookupError,
)
from plugin.plugins.mahjong_coach.preferences import PreferencesStore
from plugin.plugins.mahjong_coach.window_binding import WindowBindingResult
from plugin.plugins.mahjong_coach.yakuman import (
    YakumanEstimateService,
    assess_yakuman_routes,
    is_route_complete,
    monte_carlo_yakuman,
    route_distance,
)


mahjong_plugin_module = importlib.import_module("plugin.plugins.mahjong_coach.__init__")


def test_legacy_fast_style_migrates_to_versioned_aggressive_profile() -> None:
    profile = PlayerProfile.from_payload({}, legacy_play_style="fast")

    assert profile.version == 1
    assert profile.risk_tolerance == "aggressive"
    assert profile.goal_bias == "speed"
    assert profile.call_bias == "open"

    config = MahjongCoachConfig.from_payload({"decision": {"play_style": "fast"}})
    assert config.play_style == "fast"
    assert config.strategy_preset == "simple"
    assert config.player_profile.risk_tolerance == "aggressive"
    assert config.player_profile.goal_bias == "speed"

    standard = MahjongCoachConfig.from_payload({"decision": {"strategy_preset": "standard"}})
    assert standard.strategy_preset == "standard"


def test_preferences_store_never_persists_hwnd_and_can_clear_target(tmp_path: Path) -> None:
    path = tmp_path / "coach_preferences.json"
    path.write_text(
        json.dumps(
            {
                "target": {
                    "title": "雀魂 - Mahjong Soul",
                    "app_name": "Majsoul",
                    "hwnd": 123456,
                }
            }
        ),
        encoding="utf-8",
    )
    store = PreferencesStore(path)

    loaded = store.load()
    assert loaded.target.title == "雀魂 - Mahjong Soul"
    assert "hwnd" not in loaded.target.to_dict()

    cleared = store.update(clear_target=True)
    assert cleared.target == WindowTargetDescriptor()
    assert "hwnd" not in path.read_text(encoding="utf-8")


def test_capture_session_reuses_binding_before_revalidation(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    binding = WindowBindingResult(
        bound=True,
        window_title="雀魂 - Mahjong Soul",
        hwnd=42,
        left=0,
        top=0,
        width=1280,
        height=720,
    )
    monkeypatch.setattr(capture, "bind_window_from_descriptor", lambda *_args: calls.append("bind") or binding)
    monkeypatch.setattr(capture, "refresh_cached_window", lambda current: calls.append("refresh") or current)
    session = CaptureSession(["雀魂"], revalidate_seconds=2.0)

    assert session.locate_window().bound is True
    assert session.locate_window().hwnd == 42
    assert calls == ["bind"]

    session._last_validation -= 3.0
    assert session.locate_window().bound is True
    assert calls == ["bind", "refresh"]


def test_capture_session_rebinds_after_window_size_change(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    original = WindowBindingResult(
        bound=True,
        window_title="雀魂",
        hwnd=42,
        width=1280,
        height=720,
    )
    resized = WindowBindingResult(
        bound=True,
        window_title="雀魂",
        hwnd=42,
        width=1920,
        height=1080,
    )
    monkeypatch.setattr(capture, "bind_window_from_descriptor", lambda *_args: calls.append("bind") or resized)
    monkeypatch.setattr(capture, "refresh_cached_window", lambda _current: calls.append("refresh") or resized)
    session = CaptureSession(["雀魂"], revalidate_seconds=0.25)
    session.binding = original
    session._last_validation = time.monotonic() - 1.0

    result = session.locate_window()

    assert result.width == 1920
    assert calls == ["refresh", "bind"]


def test_capture_memory_frame_does_not_write_a_frame_file(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = DefaultCaptureProvider()
    source_image = Image.new("RGB", (1280, 720), (30, 80, 120))
    monkeypatch.setattr(provider, "_capture_image", lambda _binding: (source_image.copy(), "memory-test"))

    packet = provider.capture_memory_frame(
        binding_result=WindowBindingResult(
            bound=True,
            window_title="雀魂",
            width=1280,
            height=720,
        )
    )

    assert packet.image_path == ""
    assert packet.image is not None
    assert packet.image.size == (1280, 720)
    assert packet.fingerprint
    packet.image.close()
    source_image.close()


@pytest.mark.asyncio
async def test_live_capture_queue_keeps_only_the_newest_frame(tmp_path: Path) -> None:
    plugin = MahjongCoachPlugin.__new__(MahjongCoachPlugin)
    plugin._cfg = MahjongCoachConfig()
    plugin._live_state = LiveSessionState(running=True)
    plugin._live_stop_event = asyncio.Event()
    plugin._preferences = CapturePreferences()

    binding = WindowBindingResult(
        bound=True,
        window_title="雀魂",
        hwnd=42,
        left=0,
        top=0,
        width=8,
        height=8,
    )
    session = SimpleNamespace(
        keywords=["雀魂"],
        target=WindowTargetDescriptor(),
        locate_window=lambda: binding,
        invalidate=lambda _reason: None,
    )
    counter = 0

    def capture_memory_frame(*, binding_result: WindowBindingResult) -> FramePacket:
        nonlocal counter
        counter += 1
        packet = FramePacket(
            timestamp_ms=counter,
            image=Image.new("RGB", (8, 8), (counter, 0, 0)),
            source="test",
        )
        if counter == 2:
            plugin._live_stop_event.set()
        return packet

    provider = SimpleNamespace(capture_memory_frame=capture_memory_frame)
    queue: asyncio.Queue = asyncio.Queue(maxsize=1)

    await plugin._capture_live_frames(
        session=session,
        provider=provider,
        frame_queue=queue,
        interval_ms=200,
        frames_dir=tmp_path,
    )

    packet, _locate_ms, _capture_ms = queue.get_nowait()
    assert packet.timestamp_ms == 2
    assert plugin._live_state.dropped_frames == 1
    assert not list(tmp_path.iterdir())
    packet.image.close()


def test_live_performance_summary_reports_advice_p95_and_drops() -> None:
    rows = [
        {
            "decision": "defense_alert" if index % 2 else "observe",
            "loop_ms": float(index),
        }
        for index in range(1, 21)
    ]

    summary = _live_performance_summary(rows, dropped_frames=3)

    assert summary["sample_count"] == 20
    assert summary["advice_sample_count"] == 10
    assert summary["frame_p95_ms"] == 19.0
    assert summary["advice_p95_ms"] == 19.0
    assert summary["dropped_frames"] == 3


@pytest.mark.asyncio
async def test_direct_live_entry_binds_a_unique_window_without_web_ui(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = MahjongCoachPlugin.__new__(MahjongCoachPlugin)
    plugin._cfg = MahjongCoachConfig()
    plugin._engine = SimpleNamespace(
        config=plugin._cfg,
        state=SimpleNamespace(play_style="riichi"),
    )
    plugin._live_task = None
    plugin._live_state = LiveSessionState()
    plugin._preferences = CapturePreferences()
    plugin.logger = SimpleNamespace(info=lambda *_args, **_kwargs: None, warning=lambda *_args, **_kwargs: None)
    plugin._clear_live_hand_gap = lambda: None
    captured: dict[str, object] = {}

    async def idle_live_loop(**kwargs) -> None:
        captured.update(kwargs)
        await asyncio.sleep(60)

    plugin._run_live_loop = idle_live_loop
    monkeypatch.setattr(
        mahjong_plugin_module,
        "list_window_candidates",
        lambda _keywords: [
            {
                "title": "雀魂 - Mahjong Soul",
                "app_name": "Majsoul",
                "match_keyword": "雀魂",
                "matches_keywords": True,
            }
        ],
    )

    result = await plugin._overlay_start_live(overlay=False)
    await asyncio.sleep(0)

    assert result.unwrap()["status"] == "starting"
    assert captured["target"] == WindowTargetDescriptor(
        title="雀魂 - Mahjong Soul",
        app_name="Majsoul",
        match_keyword="雀魂",
    )
    plugin._live_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await plugin._live_task


def test_amae_koromo_profile_is_cached_and_requires_confirmation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    provider = AmaeKoromoProvider(cache_path=tmp_path / "profile-cache.json")
    calls: list[str] = []

    def request(url: str):
        calls.append(url)
        if "search_player" in url:
            return [{"id": 123, "nickname": "Neko", "level": {"id": 40101, "score": 800}}]
        if "player_extended_stats" in url:
            return {"和牌率": 0.24, "放铳率": 0.12, "立直率": 0.21, "副露率": 0.31, "count": 200}
        return {"nickname": "Neko", "level": {"id": 50101}, "count": 200}

    monkeypatch.setattr(provider, "_request_json", request)

    candidates = provider.search("Neko")
    profile = provider.fetch_profile("123", nickname="Neko")
    call_count = len(calls)
    cached_profile = provider.fetch_profile("123", nickname="Neko")

    assert candidates[0]["account_id"] == "123"
    assert candidates[0]["rank"] == "master"
    assert profile.rank == "saint"
    assert profile.sample_count == 200
    assert profile.win_rate == pytest.approx(0.24)
    assert profile.confirmed is False
    assert cached_profile == profile
    assert len(calls) == call_count


@pytest.mark.asyncio
async def test_external_profile_failure_falls_back_without_reassigning_user() -> None:
    class FailingProvider:
        def search(self, *_args, **_kwargs):
            raise ProfileLookupError("offline")

    existing = PlayerProfile(rank="master", confirmed=True)
    plugin = MahjongCoachPlugin.__new__(MahjongCoachPlugin)
    plugin._profile_provider = FailingProvider()
    plugin._preferences = CapturePreferences(profile=existing)

    result = await plugin.mahjong_coach_search_player("Neko")
    payload = result.unwrap()

    assert payload["status"] == "fallback_manual"
    assert payload["profile"]["rank"] == "master"
    assert plugin._preferences.profile == existing


def test_player_risk_profile_changes_edge_defense_result() -> None:
    hand = ["1m", "2m", "3m", "4p", "5p", "6p", "2s", "3s", "4s", "5s", "6s", "7s", "1z", "1z"]
    river = RiverStateResult(
        ok=True,
        discard_piles={"right_opponent": [{"tile": "9m", "confidence": 0.96}]},
        visible_tiles=["9m"],
        confidence=0.96,
        reason="test",
    )
    conservative = rank_discard_decisions(
        hand,
        MahjongCoachConfig(player_profile=PlayerProfile(risk_tolerance="conservative")),
        ["shimocha"],
        river,
        turn_number=10,
    )
    balanced = rank_discard_decisions(
        hand,
        MahjongCoachConfig(player_profile=PlayerProfile(risk_tolerance="balanced")),
        ["shimocha"],
        river,
        turn_number=10,
    )
    aggressive = rank_discard_decisions(
        hand,
        MahjongCoachConfig(player_profile=PlayerProfile(risk_tolerance="aggressive")),
        ["shimocha"],
        river,
        turn_number=10,
    )
    standard_balanced = rank_discard_decisions(
        hand,
        MahjongCoachConfig(
            strategy_preset="standard",
            player_profile=PlayerProfile(risk_tolerance="balanced"),
        ),
        ["shimocha"],
        river,
        turn_number=10,
    )

    assert conservative["posture"] == "fold"
    assert conservative["simple_policy_active"] is False
    assert balanced["posture"] == "push"
    assert balanced["legacy_mode"] == "attack"
    assert balanced["preserve_win_chance"] is True
    assert balanced["win_potential"] == "strong"
    assert balanced["top_candidates"][0]["within_risk_budget"] is True
    assert balanced["top_candidates"][0]["defense_risk"] <= balanced["risk_budget"]
    assert balanced["top_candidates"][0]["shanten"] == 0
    assert balanced["simple_policy_active"] is True
    assert balanced["strategy_preset"] == "simple"
    assert "简易策略轻防守+12" in balanced["risk_budget_calculation"]
    assert aggressive["posture"] == "push"
    assert aggressive["risk_budget"] > conservative["risk_budget"]
    assert standard_balanced["posture"] == "mawashi"
    assert standard_balanced["simple_policy_active"] is False
    assert balanced["risk_weight"] < standard_balanced["risk_weight"]
    assert balanced["risk_budget"] > standard_balanced["risk_budget"]
    assert len(aggressive["top_candidates"]) == 3
    assert {"safety", "shape_loss", "effective_count", "effective_count_delta"} <= aggressive["top_candidates"][0].keys()


def test_low_value_bad_shape_folds_against_multiple_riichi_players() -> None:
    hand = ["1m", "1m", "3m", "5m", "7m", "9m", "1p", "4p", "7p", "1s", "4s", "7s", "1z", "5z"]
    river = RiverStateResult(
        ok=True,
        discard_piles={
            "right_opponent": [{"tile": "9m", "confidence": 0.96}],
            "left_opponent": [{"tile": "1p", "confidence": 0.96}],
        },
        visible_tiles=["9m", "1p"],
        confidence=0.96,
        reason="test",
    )

    ranking = rank_discard_decisions(
        hand,
        MahjongCoachConfig(),
        ["shimocha", "kamicha"],
        river,
        turn_number=14,
    )

    assert ranking["posture"] == "fold"
    assert ranking["riichi_count"] == 2


def test_yakuman_routes_report_distance_keys_and_blockers() -> None:
    kokushi = ["1m", "9m", "1p", "9p", "1s", "9s", "1z", "2z", "3z", "4z", "5z", "6z", "7z"]
    completed = [*kokushi, "1m"]
    chuuren = ["1m", "1m", "1m", "2m", "3m", "4m", "5m", "5m", "6m", "7m", "8m", "9m", "9m", "9m"]

    routes = {item.route: item for item in assess_yakuman_routes(kokushi)}

    assert route_distance("kokushi", kokushi) == 1
    assert set(routes["kokushi"].key_tiles) == set(kokushi)
    assert is_route_complete("kokushi", completed)
    assert is_route_complete("chuuren", chuuren)
    assert "该路线要求门清" in {
        item.route: item for item in assess_yakuman_routes(kokushi, open_melds=1)
    }["kokushi"].blockers


def test_yakuman_preference_surfaces_close_route_even_with_cleanup_tiles() -> None:
    hand = ["1m", "9m", "1p", "9p", "1s", "9s", "1z", "2z", "3z", "4z", "5z", "6z", "5m"]
    config = MahjongCoachConfig(player_profile=PlayerProfile(goal_bias="yakuman"))

    plan = build_round_plan(hand, config)

    assert plan["direction"].startswith("役满候选：国士无双")
    assert any(item.startswith("役满路线：国士无双") for item in plan["targets"])


def test_seeded_yakuman_estimate_is_bounded_and_reproducible() -> None:
    hand = ["1m", "9m", "1p", "9p", "1s", "9s", "1z", "2z", "3z", "4z", "5z", "6z", "7z"]

    started = time.perf_counter()
    first = monte_carlo_yakuman(hand, max_trials=3, time_budget_ms=1000, seed=19)
    second = monte_carlo_yakuman(hand, max_trials=3, time_budget_ms=1000, seed=19)
    elapsed = time.perf_counter() - started

    assert [item.to_dict() for item in first] == [item.to_dict() for item in second]
    estimated = [item for item in first if item.trials]
    assert len(estimated) == 9
    assert elapsed < 1.0
    for item in estimated:
        assert set(item.tsumo_probability) == {"6", "12", "18"}
        assert all(0.0 <= value <= 1.0 for value in item.tsumo_probability.values())
        assert all(0.0 <= low <= high <= 1.0 for low, high in item.confidence_interval.values())


def test_yakuman_service_returns_immediately_and_finishes_in_background() -> None:
    hand = ["1m", "9m", "1p", "9p", "1s", "9s", "1z", "2z", "3z", "4z", "5z", "6z", "7z"]
    service = YakumanEstimateService(max_trials=20, time_budget_ms=50)
    try:
        started = time.perf_counter()
        initial = service.request(hand)
        call_elapsed = time.perf_counter() - started
        assert initial["status"] in {"running", "ready"}
        assert call_elapsed < 0.2

        deadline = time.monotonic() + 2.0
        result = initial
        while result["status"] != "ready" and time.monotonic() < deadline:
            time.sleep(0.01)
            result = service.request(hand)
        assert result["status"] == "ready"
        assert result["routes"]
    finally:
        service.close()
