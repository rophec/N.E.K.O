from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

import plugin.plugins.mahjong_coach as mahjong_plugin_module
from plugin.plugins.mahjong_coach import MahjongCoachPlugin
from plugin.plugins.mahjong_coach.coach import RoundCoachEngine
from plugin.plugins.mahjong_coach.models import CoachDecision, FramePacket, LiveSessionState, MahjongCoachConfig
from plugin.plugins.mahjong_coach.window_binding import WindowBindingResult


def _plugin_fixture() -> MahjongCoachPlugin:
    plugin = MahjongCoachPlugin.__new__(MahjongCoachPlugin)
    plugin._cfg = MahjongCoachConfig()
    plugin._engine = RoundCoachEngine(plugin._cfg)
    plugin._engine_lock = asyncio.Lock()
    plugin._last_decision = {}
    plugin._live_state = LiveSessionState()
    plugin._live_last_hand_signature = ""
    plugin._live_gap_hand_tiles = []
    plugin._live_gap_candidate_tiles = []
    plugin._live_gap_candidate_frames = 0
    plugin._live_last_checkpoint_at = 0.0
    plugin._live_timing_log = []
    plugin.logger = SimpleNamespace(
        info=lambda *_args, **_kwargs: None,
        warning=lambda *_args, **_kwargs: None,
    )
    return plugin


def test_engine_mutex_is_safe_across_concurrent_host_event_loops() -> None:
    plugin = _plugin_fixture()
    owner_entered = threading.Event()
    release_owner = threading.Event()
    waiter_entered = threading.Event()
    errors: list[BaseException] = []

    async def owner() -> None:
        async with plugin._get_engine_lock():
            owner_entered.set()
            await asyncio.to_thread(release_owner.wait, 5.0)

    async def waiter() -> None:
        async with plugin._get_engine_lock():
            waiter_entered.set()

    def run(coro) -> None:
        try:
            asyncio.run(coro())
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    owner_thread = threading.Thread(target=run, args=(owner,))
    waiter_thread = threading.Thread(target=run, args=(waiter,))
    owner_thread.start()
    assert owner_entered.wait(timeout=2.0)
    waiter_thread.start()
    assert waiter_entered.wait(timeout=0.05) is False

    release_owner.set()
    owner_thread.join(timeout=2.0)
    waiter_thread.join(timeout=2.0)

    assert errors == []
    assert owner_thread.is_alive() is False
    assert waiter_thread.is_alive() is False
    assert waiter_entered.is_set() is True


def test_runtime_resource_history_is_bounded_and_reports_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = _plugin_fixture()
    counter = 0

    def memory_snapshot():
        nonlocal counter
        counter += 1
        return {
            "rss_mb": float(counter),
            "peak_rss_mb": float(counter),
            "private_memory_mb": float(counter),
        }

    monkeypatch.setattr(mahjong_plugin_module, "_process_memory_snapshot", memory_snapshot)
    monkeypatch.setattr(
        mahjong_plugin_module,
        "perception_runtime_stats",
        lambda: {
            "yolo_sessions": 0,
            "tile_classifier_sessions": 0,
            "ocr_sessions": 0,
        },
    )

    for _ in range(400):
        plugin._sample_runtime_resources(phase="stopped", force=True)
    summary = plugin._runtime_resource_summary()

    assert summary["sample_count"] == 360
    assert summary["rss_current_mb"] == 400.0
    assert summary["rss_start_mb"] == 41.0
    assert summary["caches_released_after_stop"] is True


@pytest.mark.asyncio
async def test_manual_reset_waits_for_running_analysis() -> None:
    plugin = _plugin_fixture()
    assert plugin._engine is not None
    started = threading.Event()
    release = threading.Event()
    calls: list[str] = []

    def analyze_frame(*_args, **_kwargs) -> CoachDecision:
        calls.append("analyze:start")
        started.set()
        assert release.wait(timeout=5.0)
        calls.append("analyze:end")
        return CoachDecision(summary="done")

    original_reset = plugin._engine.reset_round

    def reset_round(round_id: str = "default"):
        calls.append("reset")
        return original_reset(round_id)

    plugin._engine.analyze_frame = analyze_frame  # type: ignore[method-assign]
    plugin._engine.reset_round = reset_round  # type: ignore[method-assign]

    analyze_task = asyncio.create_task(plugin.mahjong_coach_analyze_frame())
    for _ in range(100):
        if started.is_set():
            break
        await asyncio.sleep(0.01)
    assert started.is_set()

    reset_task = asyncio.create_task(plugin.mahjong_coach_reset_round("queued-round"))
    await asyncio.sleep(0.05)
    assert calls == ["analyze:start"]

    release.set()
    assert (await analyze_task).is_ok()
    assert (await reset_task).is_ok()
    assert calls == ["analyze:start", "analyze:end", "reset"]


@pytest.mark.asyncio
async def test_cancelled_engine_wait_holds_mutex_until_worker_finishes() -> None:
    plugin = _plugin_fixture()
    started = threading.Event()
    release = threading.Event()
    waiter_acquired = asyncio.Event()

    def blocking_worker() -> None:
        started.set()
        assert release.wait(timeout=5.0)

    async def owner() -> None:
        async with plugin._get_engine_lock():
            await plugin._await_engine_thread(blocking_worker)

    async def waiter() -> None:
        async with plugin._get_engine_lock():
            waiter_acquired.set()

    owner_task = asyncio.create_task(owner())
    for _ in range(100):
        if started.is_set():
            break
        await asyncio.sleep(0.01)
    assert started.is_set()

    owner_task.cancel()
    waiter_task = asyncio.create_task(waiter())
    await asyncio.sleep(0.05)
    assert waiter_acquired.is_set() is False

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await owner_task
    await waiter_task
    assert waiter_acquired.is_set() is True


@pytest.mark.asyncio
async def test_gpu_warmup_is_reused_before_live_analysis(monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = _plugin_fixture()
    calls: list[str] = []

    def warmup(*, inference_provider: str):
        calls.append(inference_provider)
        return {
            "status": "ready",
            "inference_provider": inference_provider,
            "providers": ["DmlExecutionProvider", "CPUExecutionProvider"],
            "elapsed_ms": 42.0,
        }

    monkeypatch.setattr(mahjong_plugin_module, "warmup_yolo26_runtime", warmup)

    first = (await plugin.mahjong_coach_warmup_yolo26("speed")).unwrap()
    second = (await plugin.mahjong_coach_warmup_yolo26("speed")).unwrap()

    assert first["status"] == "ready"
    assert second["status"] == "ready"
    assert calls == ["speed"]
    assert plugin._yolo_warmup_state["providers"][0] == "DmlExecutionProvider"


@pytest.mark.asyncio
async def test_live_new_round_reset_and_reanalysis_are_atomic() -> None:
    plugin = _plugin_fixture()
    assert plugin._engine is not None
    second_started = threading.Event()
    release_second = threading.Event()
    calls: list[str] = []
    analysis_count = 0

    def analyze_frame(*_args, **_kwargs) -> CoachDecision:
        nonlocal analysis_count
        analysis_count += 1
        calls.append(f"analyze:{analysis_count}:start")
        if analysis_count == 2:
            second_started.set()
            assert release_second.wait(timeout=5.0)
        calls.append(f"analyze:{analysis_count}:end")
        return CoachDecision(decision_type="observe", summary=f"analysis {analysis_count}")

    original_reset = plugin._engine.reset_round

    def reset_round(round_id: str = "default"):
        calls.append(f"reset:{round_id}")
        return original_reset(round_id)

    plugin._engine.analyze_frame = analyze_frame  # type: ignore[method-assign]
    plugin._engine.reset_round = reset_round  # type: ignore[method-assign]
    plugin._classify_live_hand_gap = lambda _decision: "new_round"  # type: ignore[method-assign]

    live_task = asyncio.create_task(
        plugin._analyze_live_packet("frame.png", force_checkpoint_by_time=False)
    )
    for _ in range(100):
        if second_started.is_set():
            break
        await asyncio.sleep(0.01)
    assert second_started.is_set()

    manual_reset = asyncio.create_task(plugin.mahjong_coach_reset_round("manual-round"))
    await asyncio.sleep(0.05)
    assert "reset:manual-round" not in calls

    release_second.set()
    decision, _gap_state, _paused, _state = await live_task
    assert "auto_new_round_detected" in decision.reason_codes
    assert (await manual_reset).is_ok()
    assert calls == [
        "analyze:1:start",
        "analyze:1:end",
        "reset:auto-gap-round-1",
        "analyze:2:start",
        "analyze:2:end",
        "reset:manual-round",
    ]


@pytest.mark.asyncio
async def test_live_error_still_prunes_zero_retention(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    plugin = _plugin_fixture()
    plugin._cfg = MahjongCoachConfig(live_keep_frames=0, live_save_format="jpg")
    assert plugin._engine is not None
    plugin._engine.config = plugin._cfg
    plugin._live_state = LiveSessionState(running=True)
    plugin._live_stop_event = asyncio.Event()
    plugin.data_path = lambda name: tmp_path / name  # type: ignore[method-assign]

    class Provider:
        def locate_window(self, _keywords):
            return WindowBindingResult(bound=True, window_title="Mahjong Soul", hwnd=1234)

        def capture_frame(self, *, samples_dir, **_kwargs):
            samples_dir.mkdir(parents=True, exist_ok=True)
            path = samples_dir / "20260727-000000-000001-frame.jpg"
            path.write_bytes(b"frame")
            plugin._live_stop_event.set()
            return FramePacket(timestamp_ms=1, image_path=str(path), source="test")

    def fail_analysis(*_args, **_kwargs):
        raise RuntimeError("analysis failed")

    plugin._engine.analyze_frame = fail_analysis  # type: ignore[method-assign]
    monkeypatch.setattr(mahjong_plugin_module, "DefaultCaptureProvider", Provider)

    await plugin._run_live_loop(keywords=["Mahjong Soul"], interval_ms=200, overlay_enabled=False)

    assert list((tmp_path / "live_frames").glob("*-frame.*")) == []
    assert plugin._live_state.last_frame_path == ""


@pytest.mark.asyncio
async def test_cancelled_live_analysis_prunes_after_worker_finishes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    plugin = _plugin_fixture()
    plugin._cfg = MahjongCoachConfig(live_keep_frames=0, live_save_format="jpg")
    assert plugin._engine is not None
    plugin._engine.config = plugin._cfg
    plugin._live_state = LiveSessionState(running=True)
    plugin._live_stop_event = asyncio.Event()
    plugin.data_path = lambda name: tmp_path / name  # type: ignore[method-assign]
    analysis_started = threading.Event()
    release_analysis = threading.Event()

    class Provider:
        def locate_window(self, _keywords):
            return WindowBindingResult(bound=True, window_title="Mahjong Soul", hwnd=1234)

        def capture_frame(self, *, samples_dir, **_kwargs):
            samples_dir.mkdir(parents=True, exist_ok=True)
            path = samples_dir / "20260727-000000-000002-frame.jpg"
            path.write_bytes(b"frame")
            return FramePacket(timestamp_ms=2, image_path=str(path), source="test")

    def blocked_analysis(*_args, **_kwargs):
        analysis_started.set()
        assert release_analysis.wait(timeout=5.0)
        return CoachDecision()

    plugin._engine.analyze_frame = blocked_analysis  # type: ignore[method-assign]
    monkeypatch.setattr(mahjong_plugin_module, "DefaultCaptureProvider", Provider)
    live_task = asyncio.create_task(
        plugin._run_live_loop(keywords=["Mahjong Soul"], interval_ms=200, overlay_enabled=False)
    )
    for _ in range(100):
        if analysis_started.is_set():
            break
        await asyncio.sleep(0.01)
    assert analysis_started.is_set()

    live_task.cancel()
    await asyncio.sleep(0.05)
    assert list((tmp_path / "live_frames").glob("*-frame.*"))
    release_analysis.set()
    with pytest.raises(asyncio.CancelledError):
        await live_task

    assert list((tmp_path / "live_frames").glob("*-frame.*")) == []
    assert plugin._live_state.last_frame_path == ""
