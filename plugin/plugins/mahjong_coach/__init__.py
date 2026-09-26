from __future__ import annotations

import asyncio
import base64
import contextlib
from collections import Counter
from dataclasses import replace
from io import BytesIO
import json
import math
from pathlib import Path
import sys
import threading
import time
from typing import TYPE_CHECKING, Any
import uuid

from .runtime_bootstrap import prioritize_plugin_vendor_path

# The host may already have added this directory later in sys.path.  Move it
# to the front before importing PIL or any perception/runtime dependency so the
# plugin's accelerated ONNX Runtime cannot be shadowed by the host CPU build.
prioritize_plugin_vendor_path()

from PIL import Image

from plugin.sdk.plugin import Err, NekoPluginBase, Ok, SdkError, lifecycle, neko_plugin, plugin_entry, tr

from .capture import CaptureSession, DefaultCaptureProvider, ensure_windows_dpi_awareness, prune_frames
from .companion_transport import DirectCompanionSubmitResult, submit_companion_direct
from .models import (
    CapturePreferences,
    LiveSessionState,
    MahjongCoachConfig,
    PlayerProfile,
    WindowTargetDescriptor,
    _clean_string_list,
    _valid_inference_provider,
    _valid_live_advice_mode,
    _valid_play_style,
    _valid_strategy_preset,
    _valid_river_tracking_mode,
    _valid_tile_recognition_mode,
)
from .overlay import (
    CoachOverlayController,
    overlay_detail_text_from_payload,
    overlay_strategy_card_from_payload,
    overlay_strategy_card_text_from_payload,
    overlay_text_from_payload,
)
from .player_profile import AmaeKoromoProvider, ProfileLookupError
from .presentation import (
    build_absurd_banter_reply,
    build_neko_companion_cue,
    build_public_payload,
    sanitize_round_history,
)
from .preferences import PreferencesStore
from .tile_labels import normalize_tile
from .window_binding import choose_window_candidate_native, list_window_candidates
from .yakuman import YakumanEstimateService

if TYPE_CHECKING:
    from .coach import RoundCoachEngine
    from .perception.settlement_detector import SettlementFrameResult


# The N.E.K.O host imports the entry module in its parent process to scan
# metadata.  Keep native perception dependencies out of that import path:
# vendored cv2/numpy DLLs cannot be unloaded on Windows and otherwise keep the
# plugin directory locked even after its child process has stopped.
def release_strategy_runtime_caches() -> None:
    from .coach import release_strategy_runtime_caches as release

    release()


def perception_runtime_stats() -> dict[str, object]:
    from .perception import perception_runtime_stats as collect

    return collect()


def release_perception_runtime_caches() -> None:
    from .perception import release_perception_runtime_caches as release

    release()


def source_identity(*args: Any, **kwargs: Any) -> Any:
    from .perception.image_source import source_identity as identify

    return identify(*args, **kwargs)


def detect_settlement_image(*args: Any, **kwargs: Any) -> Any:
    from .perception.settlement_detector import detect_settlement_image as detect

    return detect(*args, **kwargs)


def render_settlement_diagnostic_image(*args: Any, **kwargs: Any) -> Any:
    from .perception.settlement_detector import render_settlement_diagnostic_image as render

    return render(*args, **kwargs)


def detect_table_surface(*args: Any, **kwargs: Any) -> Any:
    from .perception.table_surface import detect_table_surface as detect

    return detect(*args, **kwargs)


def render_yolo26_region_diagnostic_image(*args: Any, **kwargs: Any) -> Any:
    from .perception.yolo26_visible_tiles import render_yolo26_region_diagnostic_image as render

    return render(*args, **kwargs)


def warmup_yolo26_runtime(*args: Any, **kwargs: Any) -> Any:
    from .perception.yolo26_visible_tiles import warmup_yolo26_runtime as warmup

    return warmup(*args, **kwargs)


def reset_yolo26_accelerator_session() -> None:
    from .perception.yolo26_visible_tiles import (
        reset_yolo26_accelerator_session as reset,
    )

    reset()


_NEKO_RARE_JOKE_MIN_EVENTS = 12
_NEKO_RARE_JOKE_MIN_INTERVAL_SECONDS = 15 * 60
_NEKO_AMBIENT_MIN_INTERVAL_SECONDS = 6.0
_NEKO_AMBIENT_SETTLE_SECONDS = 1.0
_NEKO_IDLE_HEARTBEAT_SECONDS = 12.0
_NEKO_AMBIENT_MAX_PER_ROUND = 24
_NEKO_TRANSPORT_RETRY_SECONDS = 3.0
_NEKO_TRANSPORT_PROBE_TIMEOUT_SECONDS = 0.25
_NEKO_HOST_RECEIPT_WAIT_SECONDS = 2.0
_YOLO_WARMUP_RETRY_SECONDS = 60.0
_LIVE_PREVIEW_ACTIVE_INTERVAL_SECONDS = 0.8
_LIVE_PREVIEW_ACTIVE_TTL_SECONDS = 3.0
_LIVE_PREVIEW_IDLE_INTERVAL_SECONDS = 4.0


class _CrossLoopAsyncLock:
    """Small async context lock that is safe across host event-loop threads.

    Plugin entries and the long-running live task can be dispatched by the host
    on different asyncio loops. ``asyncio.Lock`` binds to one of those loops as
    soon as it has a waiter, which can stop capture after a reload or status
    request. A non-blocking ``threading.Lock`` keeps the same critical section
    without binding mutable engine state to one event loop.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def locked(self) -> bool:
        return self._lock.locked()

    async def acquire(self) -> bool:
        while not self._lock.acquire(blocking=False):
            await asyncio.sleep(0.005)
        return True

    def release(self) -> None:
        self._lock.release()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        self.release()


@neko_plugin
class MahjongCoachPlugin(NekoPluginBase):
    def __init__(self, ctx):
        super().__init__(ctx)
        self.file_logger = self.enable_file_logging(log_level="INFO")
        self.logger = self.file_logger
        self._cfg = MahjongCoachConfig()
        self._engine: RoundCoachEngine | None = None
        self._engine_lock = _CrossLoopAsyncLock()
        self._last_decision: dict[str, Any] = {}
        self._display_snapshot: dict[str, Any] = {}
        self._display_revision = 0
        self._neko_companion_last_signature = ""
        self._neko_companion_last_push_at = 0.0
        self._neko_companion_push_count = 0
        self._neko_companion_last_event_kind = ""
        self._neko_companion_last_submit_ms: float | None = None
        self._neko_companion_delivery_stage = "waiting_capture"
        self._neko_companion_delivery_id = ""
        self._neko_companion_last_error = ""
        self._neko_companion_message_plane_received_count = 0
        self._neko_companion_host_received_count = 0
        self._neko_companion_failure_count = 0
        self._neko_companion_next_transport_probe_at = 0.0
        self._neko_companion_receipt_task: asyncio.Task | None = None
        self._neko_companion_events_since_rare_joke = 0
        self._neko_companion_last_rare_joke_at = 0.0
        self._reset_neko_companion_ambient_state()
        self._live_state = LiveSessionState()
        self._live_task: asyncio.Task | None = None
        self._live_stop_event: asyncio.Event | None = None
        self._live_last_hand_signature = ""
        self._live_gap_hand_tiles: list[str] = []
        self._live_gap_candidate_tiles: list[str] = []
        self._live_gap_candidate_frames = 0
        self._live_last_checkpoint_at = 0.0
        self._live_timing_log: list[dict[str, Any]] = []
        self._runtime_resource_samples: list[dict[str, Any]] = []
        self._runtime_last_resource_sample_at = 0.0
        self._live_frame_queue_size = 0
        self._live_last_preview_at = 0.0
        self._live_preview_requested_at = 0.0
        self._live_preview_yolo_snapshot: dict[str, Any] = {}
        self._settlement_candidate_preview: dict[str, Any] = {}
        self._settlement_confirmed_preview: dict[str, Any] = {}
        self._settlement_preview_revision = 0
        self._yolo_warmup_task: asyncio.Task | None = None
        self._yolo_warmup_provider = ""
        self._yolo_warmup_state: dict[str, Any] = {"status": "idle"}
        self._runtime_warmup_pending = False
        self._auto_start_live_pending = False
        self._deferred_auto_start_task: asyncio.Task | None = None
        self._live_last_yakuman_key = ""
        self._loop: asyncio.AbstractEventLoop | None = None
        self._tk_loop: asyncio.AbstractEventLoop | None = None
        self._tk_loop_thread: threading.Thread | None = None
        self._overlay = CoachOverlayController(
            prefs_path=self.data_path("overlay_prefs.json"),
        )
        self._preferences_store = PreferencesStore(self.data_path("coach_preferences.json"))
        self._preferences = self._preferences_store.load()
        self._profile_provider = AmaeKoromoProvider(cache_path=self.data_path("player_profile_cache.json"))
        self._yakuman_service: YakumanEstimateService | None = None

    @lifecycle(id="startup")
    async def startup(self, **_):
        startup_started = time.perf_counter()
        try:
            from .coach import RoundCoachEngine

            dpi_mode = ensure_windows_dpi_awareness()
            self.logger.info("mahjong coach capture dpi awareness={}", dpi_mode)
            raw = await self.config.dump(timeout=5.0)
            self._cfg = MahjongCoachConfig.from_payload(raw if isinstance(raw, dict) else {})
            self._preferences_store = PreferencesStore(self.data_path("coach_preferences.json"))
            self._preferences = self._preferences_store.load()
            self._profile_provider = AmaeKoromoProvider(cache_path=self.data_path("player_profile_cache.json"))
            self._yakuman_service = YakumanEstimateService()
            if self._preferences.profile != PlayerProfile():
                self._cfg = replace(
                    self._cfg,
                    player_profile=self._preferences.profile,
                    play_style=_legacy_style_for_profile(self._preferences.profile),
                )
            self._engine = RoundCoachEngine(
                self._cfg,
                calibration_dir=Path(__file__).resolve().parent / "data" / "calibration" / "profiles",
            )
            self._loop = asyncio.get_running_loop()
            self._overlay = CoachOverlayController(
                prefs_path=self.data_path("overlay_prefs.json"),
                on_start=self._on_overlay_start_sync,
                on_preferences_change=self._on_overlay_preferences_sync,
                on_stop=self._on_overlay_stop_sync,
            )
            # 启动时只注册插件能力，不自动弹出 Tk 浮窗，避免浮窗遮挡牌桌截图。
            # Register plugin capabilities on startup without auto-opening the Tk overlay,
            # so the overlay cannot cover the Mahjong Soul frame being captured.
            # 中文：开发中的工作台必须绕过浏览器缓存，避免重载后仍显示旧版 UI。
            # English: Bypass browser caching so plugin reloads always show the current dashboard.
            self.register_static_ui("static", cache_control="no-store")
            self.set_list_actions(
                [
                    {
                        "id": "open_ui",
                        "kind": "ui",
                        "target": f"/plugin/{self.plugin_id}/ui/",
                        "open_in": "new_tab",
                    },
                    {
                        "id": "status",
                        "kind": "entry",
                        "target": "mahjong_coach_status",
                    },
                    {
                        "id": "show_overlay",
                        "kind": "entry",
                        "target": "mahjong_coach_show_overlay",
                    },
                    {
                        "id": "analyze_frame",
                        "kind": "entry",
                        "target": "mahjong_coach_analyze_frame",
                    },
                    {
                        "id": "start_live",
                        "kind": "entry",
                        "target": "mahjong_coach_start_live",
                    },
                    {
                        "id": "stop_live",
                        "kind": "entry",
                        "target": "mahjong_coach_stop_live",
                    },
                ]
            )
            # N.E.K.O executes lifecycle.startup through a short-lived
            # ``asyncio.run()`` loop.  Tasks created here are cancelled during
            # loop teardown, and a cancelled ``to_thread`` warmup still waits
            # for the native inference call.  That made an apparently
            # non-blocking GPU warmup hold the startup acknowledgement past
            # the host's 10-second deadline.  Defer all long-lived work until
            # the first plugin entry runs on the persistent command loop.
            self._runtime_warmup_pending = bool(
                self._cfg.tile_recognition_mode == "yolo26"
                and self._cfg.inference_provider == "speed"
            )
            self._auto_start_live_pending = bool(
                self._preferences.auto_start_live
                and (
                self._preferences.target.title or self._preferences.target.app_name
                )
            )
            self.logger.info(
                "mahjong coach startup ready elapsed_ms={} warmup_deferred={} auto_start_deferred={}",
                round(_elapsed_ms(startup_started), 1),
                self._runtime_warmup_pending,
                self._auto_start_live_pending,
            )
            return Ok({"status": "ready", "config": self._cfg.to_dict()})
        except Exception as exc:
            self.logger.warning("mahjong coach startup failed: {}", exc)
            return Err(SdkError("failed to start mahjong_coach"))

    def _show_overlay(self, *, strategy: bool = False) -> bool:
        if not self._overlay.start():
            self.logger.warning(
                "mahjong coach overlay failed to start: {}",
                getattr(self._overlay, "last_error", "unknown overlay startup error"),
            )
            return False
        logger = getattr(self, "logger", None)
        if logger is not None:
            logger.info(
                "mahjong coach overlay started backend={} hwnd={} visible={}",
                getattr(self._overlay, "backend", "unknown"),
                getattr(self._overlay, "window_handle", 0),
                getattr(self._overlay, "window_visible", False),
            )
        snapshot = dict(getattr(self, "_display_snapshot", {}) or {})
        live_state = getattr(self, "_live_state", LiveSessionState())
        self._overlay.update_payload(
            text=str(snapshot.get("overlay_text") or "Mahjong Coach"),
            strategy_card_text=str(snapshot.get("strategy_card_text") or snapshot.get("overlay_text") or "Mahjong Coach"),
            strategy_card=snapshot.get("strategy_card") or {},
            detail=str(snapshot.get("overlay_detail") or ""),
            image_path=str(snapshot.get("image_path") or ""),
            image_revision=int(snapshot.get("image_revision") or 0),
            live_state=live_state.to_dict(),
            companion_status=self._neko_companion_status_payload(),
        )
        if strategy:
            self._overlay.show_strategy()
        else:
            self._overlay.show_config()
        return True

    def _ensure_tk_loop(self) -> asyncio.AbstractEventLoop | None:
        if self._tk_loop is not None and self._tk_loop.is_running():
            return self._tk_loop
        if self._tk_loop_thread is not None and self._tk_loop_thread.is_alive():
            return self._tk_loop
        try:
            self._tk_loop = asyncio.new_event_loop()
            ready = threading.Event()

            def _run():
                asyncio.set_event_loop(self._tk_loop)
                ready.set()
                self._tk_loop.run_forever()

            self._tk_loop_thread = threading.Thread(target=_run, daemon=True)
            self._tk_loop_thread.start()
            if not ready.wait(timeout=2.0) or not self._tk_loop.is_running():
                self.logger.warning("tk event loop failed to start")
                return None
            return self._tk_loop
        except Exception as exc:
            self.logger.warning("failed to start tk event loop: {}", exc)
            return None

    def _on_overlay_start_sync(
        self,
        style: str,
        strategy_preset: str = "simple",
        live_advice_mode: str = "companion",
        neko_companion_enabled: bool = True,
        absurd_banter_enabled: bool = False,
    ) -> None:
        self.logger.info(
            "_on_overlay_start_sync called style={} strategy_preset={} live_advice_mode={} neko_companion_enabled={} absurd_banter_enabled={}",
            style,
            strategy_preset,
            live_advice_mode,
            neko_companion_enabled,
            absurd_banter_enabled,
        )
        self._schedule_on_loop(
            self._overlay_start_live(
                play_style=style,
                strategy_preset=strategy_preset,
                live_advice_mode=live_advice_mode,
                neko_companion_enabled=neko_companion_enabled,
                absurd_banter_enabled=absurd_banter_enabled,
                # The native overlay has no resource-mode selector.  Always
                # request the accelerated runtime here so upgraded installs do
                # not keep an old persisted ``memory`` value.  ONNX Runtime
                # falls back to CPU when CUDA / DirectML is unavailable.
                inference_provider="speed",
            ),
            "overlay_start",
        )

    def _on_overlay_preferences_sync(
        self,
        strategy_preset: str,
        live_advice_mode: str,
        neko_companion_enabled: bool,
        absurd_banter_enabled: bool = False,
    ) -> None:
        """Apply in-session overlay choices without restarting capture."""
        self._schedule_on_loop(
            self._overlay_update_preferences(
                strategy_preset=strategy_preset,
                live_advice_mode=live_advice_mode,
                neko_companion_enabled=neko_companion_enabled,
                absurd_banter_enabled=absurd_banter_enabled,
            ),
            "overlay_preferences",
        )

    def _on_overlay_stop_sync(self) -> None:
        self._schedule_on_loop(self._overlay_stop_live(hide_overlay=True), "overlay_stop")

    async def _overlay_update_preferences(
        self,
        *,
        strategy_preset: str,
        live_advice_mode: str,
        neko_companion_enabled: bool,
        absurd_banter_enabled: bool = False,
    ):
        if self._engine is None:
            return Err(SdkError("mahjong coach is not initialized"))
        async with self._get_engine_lock():
            preset_before = self._cfg.strategy_preset
            mode_before = self._cfg.live_advice_mode
            self._apply_runtime_strategy_preset(strategy_preset)
            self._apply_runtime_live_advice_mode(live_advice_mode)
            self._apply_runtime_neko_companion_enabled(neko_companion_enabled)
            self._apply_runtime_absurd_banter_enabled(absurd_banter_enabled)
            if self._cfg.strategy_preset != preset_before:
                self._invalidate_live_plan_for_style_change()
            if self._cfg.live_advice_mode != mode_before:
                self._display_snapshot = {}
            payload = {
                "last_decision": dict(self._last_decision),
                "round_state": self._engine.state.to_dict(),
            }
        if self._live_state.running and payload["last_decision"]:
            self._update_overlay(payload)
        return Ok(
            {
                "strategy_preset": self._cfg.strategy_preset,
                "live_advice_mode": self._cfg.live_advice_mode,
                "neko_companion_enabled": self._cfg.neko_companion_enabled,
                "absurd_banter_enabled": self._cfg.absurd_banter_enabled,
                "running": self._live_state.running,
            }
        )

    def _schedule_on_loop(self, coro, label: str) -> None:
        loop = None
        if self._loop is not None and self._loop.is_running():
            loop = self._loop
        else:
            loop = self._ensure_tk_loop()
        if loop is None:
            self.logger.warning("{} requested but no running event loop available", label)
            return
        try:
            future = asyncio.run_coroutine_threadsafe(coro, loop)
        except Exception as exc:
            self.logger.warning("asyncio.run_coroutine_threadsafe failed for {}: {}", label, exc)
            return

        def _on_done(f):
            exc = f.exception()
            if exc:
                self.logger.warning("{} failed: {}", label, exc)

        future.add_done_callback(_on_done)

    @lifecycle(id="shutdown")
    async def shutdown(self, **_):
        self._runtime_warmup_pending = False
        self._auto_start_live_pending = False
        await self._stop_live_task()
        # 中文：插件重载或后端退出时同步销毁 Tk 窗口，避免遗留失效线程。
        # English: Destroy the Tk window during reload or shutdown to avoid a stale UI thread.
        self._overlay.stop()
        yakuman_service = getattr(self, "_yakuman_service", None)
        if yakuman_service is not None:
            yakuman_service.close()
        self.clear_list_actions()
        return Ok({"status": "stopped"})

    def _get_engine_lock(self) -> _CrossLoopAsyncLock:
        """Return the single mutex guarding the mutable round engine.

        A lazy fallback keeps direct ``__new__`` test fixtures and older host
        reloads compatible without creating a second lock in normal runtime.
        """
        lock = getattr(self, "_engine_lock", None)
        if not isinstance(lock, _CrossLoopAsyncLock):
            lock = _CrossLoopAsyncLock()
            self._engine_lock = lock
        return lock

    async def _await_thread_result(self, func, /, *args, **kwargs) -> tuple[Any, bool]:
        """Wait for a thread to finish and report whether its waiter was cancelled."""
        worker = asyncio.create_task(asyncio.to_thread(func, *args, **kwargs))
        cancelled = False
        while True:
            try:
                result = await asyncio.shield(worker)
                return result, cancelled
            except asyncio.CancelledError:
                cancelled = True
                continue
            except BaseException:
                if cancelled:
                    raise asyncio.CancelledError from None
                raise

    async def _await_engine_thread(self, func, /, *args, **kwargs):
        """Await an engine worker without releasing its mutex on cancellation."""
        result, cancelled = await self._await_thread_result(func, *args, **kwargs)
        if cancelled:
            raise asyncio.CancelledError
        return result

    def _neko_companion_status_payload(self) -> dict[str, Any]:
        cfg = getattr(self, "_cfg", MahjongCoachConfig())
        live_state = getattr(self, "_live_state", LiveSessionState())
        enabled = bool(cfg.neko_companion_enabled)
        live_running = bool(getattr(live_state, "running", False))
        stage = str(getattr(self, "_neko_companion_delivery_stage", "") or "")
        if not enabled:
            stage = "disabled"
        elif not live_running:
            stage = "waiting_capture"
        elif stage in {"", "disabled", "waiting_capture"}:
            stage = "ready"
        return {
            "enabled": enabled,
            "active": bool(enabled and live_running),
            "capture_running": live_running,
            "capture_status": str(getattr(live_state, "status", "stopped") or "stopped"),
            "delivery_stage": stage,
            "delivery_id": str(getattr(self, "_neko_companion_delivery_id", "") or ""),
            "submit_count": int(getattr(self, "_neko_companion_push_count", 0) or 0),
            # Compatibility aliases.  These mean SDK submissions, not spoken replies.
            "push_count": int(getattr(self, "_neko_companion_push_count", 0) or 0),
            "last_submitted_at": float(getattr(self, "_neko_companion_last_push_at", 0.0) or 0.0),
            "last_push_at": float(getattr(self, "_neko_companion_last_push_at", 0.0) or 0.0),
            "message_plane_received_count": int(
                getattr(self, "_neko_companion_message_plane_received_count", 0) or 0
            ),
            "host_received_count": int(
                getattr(self, "_neko_companion_host_received_count", 0) or 0
            ),
            "failure_count": int(getattr(self, "_neko_companion_failure_count", 0) or 0),
            "last_error": str(getattr(self, "_neko_companion_last_error", "") or ""),
            "last_event_kind": str(getattr(self, "_neko_companion_last_event_kind", "") or ""),
            "last_submit_ms": getattr(self, "_neko_companion_last_submit_ms", None),
            "ambient_count": int(getattr(self, "_neko_companion_ambient_count", 0) or 0),
            "ambient_limit": _NEKO_AMBIENT_MAX_PER_ROUND,
            "ordinary_interval_seconds": _NEKO_AMBIENT_MIN_INTERVAL_SECONDS,
            "idle_interval_seconds": _NEKO_IDLE_HEARTBEAT_SECONDS,
            "critical_bypasses_cooldown": True,
            "next_ordinary_in_seconds": _remaining_cooldown_seconds(
                float(getattr(self, "_neko_companion_last_push_at", 0.0) or 0.0),
                _NEKO_AMBIENT_MIN_INTERVAL_SECONDS,
            ),
            "next_idle_in_seconds": _remaining_cooldown_seconds(
                float(getattr(self, "_neko_companion_last_push_at", 0.0) or 0.0),
                _NEKO_IDLE_HEARTBEAT_SECONDS,
            ),
        }

    @plugin_entry(
        id="mahjong_coach_status",
        name=tr("entries.status.name", default="Mahjong Coach Status"),
        description=tr("entries.status.description", default="Inspect current Mahjong Coach round state."),
        input_schema={
            "type": "object",
            "properties": {
                "dashboard_visible": {"type": "boolean", "default": False},
            },
        },
        llm_result_fields=["status", "guidance_mode", "last_update_reason"],
    )
    async def mahjong_coach_status(self, dashboard_visible: bool = False, **_):
        if self._engine is None:
            return Err(SdkError("mahjong coach is not initialized"))
        self._activate_deferred_runtime_work()
        if dashboard_visible:
            # The diagnostic page is actively watching the transformed table.
            # Temporarily raise only the single overwritten preview cadence;
            # normal headless play keeps the lower disk/CPU rate.
            self._live_preview_requested_at = time.monotonic()
        async with self._get_engine_lock():
            state = self._engine.state.to_dict()
            round_history = self._engine.round_history
            last_decision = dict(self._last_decision)
            config = self._cfg.to_dict()
        payload = {"last_decision": last_decision, "round_state": state}
        public_payload = build_public_payload(payload, mode=self._cfg.live_advice_mode)
        display_snapshot = getattr(self, "_display_snapshot", {})
        snapshot_mode = str((display_snapshot.get("presentation") or {}).get("mode") or "")
        snapshot_round_id = str((display_snapshot.get("round_state") or {}).get("round_id") or "")
        current_round_id = str(state.get("round_id") or "")
        if (
            not display_snapshot
            or snapshot_mode != self._cfg.live_advice_mode
            or snapshot_round_id != current_round_id
        ):
            display_snapshot = self._make_display_snapshot(
                payload,
                revision=int(getattr(self, "_display_revision", 0) or 0),
            )
        public_state = public_payload["round_state"]
        public_decision = public_payload["last_decision"]
        public_history = sanitize_round_history(round_history, mode=self._cfg.live_advice_mode)
        timing_log = list(getattr(self, "_live_timing_log", []))
        yakuman_service = getattr(self, "_yakuman_service", None)
        yakuman_runtime = yakuman_service.runtime_stats() if yakuman_service is not None else {}
        self._sample_runtime_resources(
            phase="live" if self._live_state.running else "stopped",
            force=False,
        )
        resource_summary = self._runtime_resource_summary()
        return Ok(
            {
                "status": "ready",
                "config": config,
                "round_state": public_state,
                "last_decision": public_decision,
                "display_snapshot": display_snapshot,
                "round_history": public_history,
                "last_round_archive": public_history[-1] if public_history else {},
                "overlay_text": str(display_snapshot.get("overlay_text") or ""),
                "live": self._live_state.to_dict(),
                "timing_log": timing_log,
                "performance": _live_performance_summary(
                    timing_log,
                    dropped_frames=int(getattr(self._live_state, "dropped_frames", 0) or 0),
                ),
                "yakuman_runtime": yakuman_runtime,
                "runtime_resources": resource_summary,
                "inference_runtime": self._inference_runtime_status(timing_log),
                "yolo_warmup": dict(getattr(self, "_yolo_warmup_state", {"status": "idle"})),
                "neko_companion_sync": self._neko_companion_status_payload(),
                "preferences": getattr(self, "_preferences", CapturePreferences()).to_dict(),
                "ui_path": f"/plugin/{self.plugin_id}/ui/",
                **public_state,
            }
        )

    @plugin_entry(
        id="mahjong_coach_warmup_yolo26",
        name="Warm Up Mahjong Coach YOLO26",
        description="Preload and exercise the selected YOLO26 GPU runtime before live capture.",
        input_schema={
            "type": "object",
            "properties": {
                "inference_provider": {"type": "string", "default": "speed"},
            },
        },
        llm_result_fields=["status", "elapsed_ms", "providers"],
    )
    async def mahjong_coach_warmup_yolo26(
        self,
        inference_provider: str = "speed",
        **_,
    ):
        provider = _valid_inference_provider(inference_provider)
        if provider != "speed":
            return Ok({"status": "skipped", "reason": "gpu_warmup_not_requested"})
        state = await self._ensure_yolo26_warmup(provider)
        return Ok(dict(state))

    @plugin_entry(
        id="mahjong_coach_save_preferences",
        name="Save Mahjong Coach Capture Preferences",
        description="Save optional live auto-start and a stable Mahjong Soul window descriptor.",
        input_schema={
            "type": "object",
            "properties": {
                "auto_start_live": {"type": "boolean"},
                "target_window_title": {"type": "string", "default": ""},
                "target_app_name": {"type": "string", "default": ""},
            },
        },
        llm_result_fields=["status", "preferences"],
    )
    async def mahjong_coach_save_preferences(
        self,
        auto_start_live: bool | None = None,
        target_window_title: str = "",
        target_app_name: str = "",
        **_,
    ):
        target = None
        if str(target_window_title or target_app_name).strip():
            target = WindowTargetDescriptor(
                title=str(target_window_title or "").strip(),
                app_name=str(target_app_name or "").strip(),
            )
        self._preferences = self._preferences_store.update(
            auto_start_live=auto_start_live,
            target=target,
            clear_target=target is None,
        )
        return Ok({"status": "saved", "preferences": self._preferences.to_dict()})

    @plugin_entry(
        id="mahjong_coach_save_profile",
        name="Save Mahjong Coach Player Profile",
        description="Save an explicitly selected manual four-player rank and play style.",
        input_schema={
            "type": "object",
            "properties": {
                "rank": {"type": "string", "default": "unknown"},
                "room": {"type": "string", "default": "unknown"},
                "risk_tolerance": {"type": "string", "default": "balanced"},
                "goal_bias": {"type": "string", "default": "balanced"},
                "call_bias": {"type": "string", "default": "balanced"},
            },
        },
        llm_result_fields=["status", "profile"],
    )
    async def mahjong_coach_save_profile(
        self,
        rank: str = "unknown",
        room: str = "unknown",
        risk_tolerance: str = "balanced",
        goal_bias: str = "balanced",
        call_bias: str = "balanced",
        **_,
    ):
        profile = PlayerProfile.from_payload(
            {
                "rank": rank,
                "room": room,
                "risk_tolerance": risk_tolerance,
                "goal_bias": goal_bias,
                "call_bias": call_bias,
                "source": "manual",
                "confirmed": True,
            }
        )
        profile = replace(profile, confirmed=True)
        self._apply_player_profile(profile)
        return Ok({"status": "saved", "profile": profile.to_dict()})

    @plugin_entry(
        id="mahjong_coach_search_player",
        name="Search Mahjong Soul Player",
        description="Explicitly search the read-only Amae-Koromo four-player index by nickname.",
        input_schema={
            "type": "object",
            "properties": {
                "nickname": {"type": "string"},
                "limit": {"type": "integer", "default": 10},
                "force_refresh": {"type": "boolean", "default": False},
            },
            "required": ["nickname"],
        },
        timeout=8.0,
        llm_result_fields=["status", "candidates"],
    )
    async def mahjong_coach_search_player(
        self,
        nickname: str,
        limit: int = 10,
        force_refresh: bool = False,
        **_,
    ):
        try:
            candidates = await asyncio.to_thread(
                self._profile_provider.search,
                nickname,
                limit=limit,
                force_refresh=bool(force_refresh),
            )
        except ProfileLookupError as exc:
            return Ok({
                "status": "fallback_manual",
                "candidates": [],
                "error": str(exc),
                "profile": self._preferences.profile.to_dict(),
            })
        return Ok({"status": "selection_required", "candidates": candidates})

    @plugin_entry(
        id="mahjong_coach_preview_player_profile",
        name="Preview Mahjong Soul Player Profile",
        description="Fetch a suggested profile for one selected account without applying it.",
        input_schema={
            "type": "object",
            "properties": {
                "account_id": {"type": "string"},
                "nickname": {"type": "string", "default": ""},
                "force_refresh": {"type": "boolean", "default": False},
            },
            "required": ["account_id"],
        },
        timeout=10.0,
        llm_result_fields=["status", "suggested_profile"],
    )
    async def mahjong_coach_preview_player_profile(
        self,
        account_id: str,
        nickname: str = "",
        force_refresh: bool = False,
        **_,
    ):
        try:
            profile = await asyncio.to_thread(
                self._profile_provider.fetch_profile,
                account_id,
                nickname=nickname,
                force_refresh=bool(force_refresh),
            )
        except ProfileLookupError as exc:
            return Ok({
                "status": "fallback_manual",
                "error": str(exc),
                "suggested_profile": {},
            })
        return Ok({"status": "confirmation_required", "suggested_profile": profile.to_dict()})

    @plugin_entry(
        id="mahjong_coach_confirm_player_profile",
        name="Confirm Mahjong Soul Player Profile",
        description="Apply a user-confirmed external account and optional style overrides.",
        input_schema={
            "type": "object",
            "properties": {
                "account_id": {"type": "string"},
                "nickname": {"type": "string", "default": ""},
                "risk_tolerance": {"type": "string", "default": ""},
                "goal_bias": {"type": "string", "default": ""},
                "call_bias": {"type": "string", "default": ""},
                "room": {"type": "string", "default": ""},
            },
            "required": ["account_id"],
        },
        timeout=10.0,
        llm_result_fields=["status", "profile"],
    )
    async def mahjong_coach_confirm_player_profile(
        self,
        account_id: str,
        nickname: str = "",
        risk_tolerance: str = "",
        goal_bias: str = "",
        call_bias: str = "",
        room: str = "",
        **_,
    ):
        try:
            suggested = await asyncio.to_thread(
                self._profile_provider.fetch_profile,
                account_id,
                nickname=nickname,
            )
        except ProfileLookupError as exc:
            return Ok({
                "status": "fallback_manual",
                "error": str(exc),
                "profile": self._preferences.profile.to_dict(),
            })
        merged = suggested.to_dict()
        for key, value in {
            "risk_tolerance": risk_tolerance,
            "goal_bias": goal_bias,
            "call_bias": call_bias,
            "room": room,
        }.items():
            if str(value or "").strip():
                merged[key] = value
        merged["confirmed"] = True
        profile = PlayerProfile.from_payload(merged)
        profile = replace(profile, confirmed=True, source="amae_koromo")
        self._apply_player_profile(profile)
        return Ok({"status": "confirmed", "profile": profile.to_dict()})

    def _apply_player_profile(self, profile: PlayerProfile) -> None:
        self._preferences = self._preferences_store.update(profile=profile)
        self._cfg = replace(
            self._cfg,
            player_profile=profile,
            play_style=_legacy_style_for_profile(profile),
        )
        if self._engine is not None:
            self._engine.config = self._cfg
            self._engine.state.play_style = self._cfg.play_style
            self._invalidate_live_plan_for_style_change()

    @plugin_entry(
        id="mahjong_coach_frame_preview",
        name="Mahjong Coach Frame Preview",
        description="Return a compact preview of the latest captured Mahjong Soul frame.",
        input_schema={
            "type": "object",
            "properties": {"image_path": {"type": "string", "default": ""}},
        },
        llm_result_fields=["status", "image_path", "width", "height"],
    )
    async def mahjong_coach_frame_preview(self, image_path: str = "", **_):
        if self._engine is None:
            return Err(SdkError("mahjong coach is not initialized"))
        raw_path = str(image_path or self._live_state.last_frame_path or "").strip()
        if not raw_path:
            return Ok({"status": "empty", "image_path": "", "data_url": ""})
        frame_path = _resolve_preview_frame_path(raw_path, self.data_path("live_frames"))
        if frame_path is None:
            return Err(SdkError("latest frame is unavailable"))
        try:
            preview = await asyncio.to_thread(_build_frame_preview_payload, frame_path)
        except (OSError, ValueError):
            return Err(SdkError("failed to build frame preview"))
        return Ok({"status": "ready", **preview})

    @plugin_entry(
        id="mahjong_coach_table_region_preview",
        name="Mahjong Coach Warped Table Region Preview",
        description="Return the perspective-corrected table with YOLO ownership regions and detections.",
        input_schema={
            "type": "object",
            "properties": {"image_path": {"type": "string", "default": ""}},
        },
        llm_result_fields=["status", "image_path", "transformed", "width", "height", "detection_count"],
    )
    async def mahjong_coach_table_region_preview(self, image_path: str = "", **_):
        if self._engine is None:
            return Err(SdkError("mahjong coach is not initialized"))
        raw_path = str(image_path or self._live_state.last_frame_path or "").strip()
        if not raw_path:
            return Ok({"status": "empty", "image_path": "", "data_url": "", "transformed": False})
        frame_path = _resolve_preview_frame_path(raw_path, self.data_path("live_frames"))
        if frame_path is None:
            return Err(SdkError("latest frame is unavailable"))

        raw_detections: list[dict[str, Any]] = []
        opponent_melds: dict[str, list[dict[str, Any]]] = {}
        exact_preview: dict[str, Any] = {}
        live_snapshot = getattr(self, "_live_preview_yolo_snapshot", {})
        snapshot_path = str(live_snapshot.get("image_path") or "")
        snapshot_revision = int(live_snapshot.get("revision") or 0)
        snapshot_round_id = str(live_snapshot.get("round_id") or "")
        current_round_id = str(self._engine.state.round_id or "")
        if (
            snapshot_path
            and Path(snapshot_path).resolve() == frame_path
            and snapshot_revision == int(self._live_state.last_frame_revision or 0)
            and snapshot_round_id == current_round_id
        ):
            raw_detections = [
                dict(item)
                for item in live_snapshot.get("raw_detections") or []
                if isinstance(item, dict)
            ]
            opponent_melds = {
                str(owner): [dict(item) for item in items if isinstance(item, dict)]
                for owner, items in (live_snapshot.get("opponent_melds") or {}).items()
                if isinstance(items, list)
            }
            cached_preview = live_snapshot.get("table_region_preview")
            if isinstance(cached_preview, dict):
                exact_preview = dict(cached_preview)
        if exact_preview.get("data_url"):
            return Ok({"status": "ready", **exact_preview})
        async with self._get_engine_lock():
            yolo_path = getattr(self._engine, "_last_yolo26_path", None)
            yolo_result = getattr(self._engine, "_last_yolo26_result", None)
            if (
                not raw_detections
                and isinstance(yolo_path, Path)
                and yolo_path.resolve() == frame_path
                and yolo_result is not None
            ):
                raw_detections = [dict(item) for item in yolo_result.raw_detections if isinstance(item, dict)]
                opponent_melds = {
                    str(owner): [dict(item) for item in items if isinstance(item, dict)]
                    for owner, items in yolo_result.opponent_melds.items()
                }
        try:
            preview = await asyncio.to_thread(
                _build_table_region_preview_payload,
                frame_path,
                raw_detections=raw_detections,
                opponent_melds=opponent_melds,
            )
        except (OSError, ValueError):
            return Err(SdkError("failed to build warped table region preview"))
        return Ok({"status": "ready" if preview.get("data_url") else "unavailable", **preview})

    @plugin_entry(
        id="mahjong_coach_settlement_preview",
        name="Mahjong Coach Settlement Diagnostic Preview",
        description="Return an in-memory annotated preview of settlement detection evidence.",
        input_schema={
            "type": "object",
            "properties": {"image_path": {"type": "string", "default": ""}},
        },
        llm_result_fields=["status", "image_path", "detected", "kind", "confidence"],
    )
    async def mahjong_coach_settlement_preview(self, image_path: str = "", **_):
        if self._engine is None:
            return Err(SdkError("mahjong coach is not initialized"))
        settlement_phase = str(self._engine.state.settlement_phase or "playing")
        frozen = (
            dict(getattr(self, "_settlement_candidate_preview", {}))
            if settlement_phase == "settlement_candidate"
            else dict(getattr(self, "_settlement_confirmed_preview", {}))
        )
        if str(frozen.get("round_id") or "") != str(self._engine.state.round_id or ""):
            frozen = {}
        if frozen:
            return Ok(frozen)
        raw_path = str(image_path or "").strip()
        if not raw_path:
            return Ok({
                "status": "empty",
                "phase": settlement_phase,
                "image_path": "",
                "data_url": "",
                "reason": "no_frozen_settlement_evidence",
            })
        frame_path = _resolve_preview_frame_path(raw_path, self.data_path("live_frames"))
        if frame_path is None:
            return Err(SdkError("latest frame is unavailable"))
        try:
            preview = await asyncio.to_thread(
                _build_settlement_diagnostic_preview_payload,
                frame_path,
                min_confidence=self._cfg.settlement_min_confidence,
            )
        except (OSError, ValueError):
            return Err(SdkError("failed to build settlement diagnostic preview"))
        return Ok({"status": "ready" if preview.get("data_url") else "empty", **preview})

    @plugin_entry(
        id="mahjong_coach_reset_round",
        name=tr("entries.reset_round.name", default="Reset Mahjong Coach Round"),
        description=tr("entries.reset_round.description", default="Reset opening and checkpoint memory for a new hand."),
        input_schema={"type": "object", "properties": {"round_id": {"type": "string", "default": "default"}}},
        llm_result_fields=["round_id"],
    )
    async def mahjong_coach_reset_round(self, round_id: str = "default", **_):
        if self._engine is None:
            return Err(SdkError("mahjong coach is not initialized"))
        async with self._get_engine_lock():
            state = self._engine.reset_round(round_id)
            self._reset_live_round_caches(current_hand_signature="", clear_last_decision=True)
            getattr(self, "_live_timing_log", []).clear()
            history = self._engine.round_history
            state_payload = state.to_dict()
            state_round_id = state.round_id
        public_state = build_public_payload(
            {"last_decision": {}, "round_state": state_payload},
            mode=self._cfg.live_advice_mode,
        )["round_state"]
        public_history = sanitize_round_history(history, mode=self._cfg.live_advice_mode)
        return Ok(
            {
                "status": "reset",
                "round_state": public_state,
                "round_id": state_round_id,
                "round_history": public_history,
                "last_round_archive": public_history[-1] if public_history else {},
            }
        )

    @plugin_entry(
        id="mahjong_coach_analyze_frame",
        name=tr("entries.analyze_frame.name", default="Analyze Mahjong Frame"),
        description=tr(
            "entries.analyze_frame.description",
            default="Analyze one Mahjong Soul screenshot and update the quiet round coach state.",
        ),
        input_schema={
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "default": ""},
                "observed_buttons": {"type": "array", "items": {"type": "string"}, "default": []},
                "self_turn_index": {"type": "integer", "default": 0},
                "force_checkpoint": {"type": "boolean", "default": False},
                "riichi_players": {"type": "array", "items": {"type": "string"}, "default": []},
                "round_wind": {"type": "string", "default": ""},
                "seat_wind": {"type": "string", "default": ""},
                "dora_tiles": {"type": "array", "items": {"type": "string"}, "default": []},
                "play_style": {"type": "string", "default": ""},
                "strategy_preset": {"type": "string", "default": ""},
                "live_advice_mode": {"type": "string", "default": ""},
                "river_tracking_mode": {"type": "string", "default": ""},
                "tile_recognition_mode": {"type": "string", "default": ""},
                "inference_provider": {"type": "string", "default": "speed"},
                "settlement_recognition_enabled": {"type": "boolean"},
                "settlement_min_confidence": {"type": "number"},
                "settlement_confirm_frames": {"type": "integer"},
                "settlement_confirm_max_gap_ms": {"type": "integer"},
            },
        },
        timeout=20.0,
        llm_result_fields=["decision_type", "summary", "detail"],
    )
    async def mahjong_coach_analyze_frame(
        self,
        image_path: str = "",
        observed_buttons: list[str] | None = None,
        self_turn_index: int | None = None,
        force_checkpoint: bool = False,
        riichi_players: list[str] | None = None,
        round_wind: str = "",
        seat_wind: str = "",
        dora_tiles: list[str] | None = None,
        play_style: str = "",
        strategy_preset: str = "",
        live_advice_mode: str = "",
        river_tracking_mode: str = "",
        tile_recognition_mode: str = "",
        inference_provider: str = "speed",
        settlement_recognition_enabled: bool | None = None,
        settlement_min_confidence: float | None = None,
        settlement_confirm_frames: int | None = None,
        settlement_confirm_max_gap_ms: int | None = None,
        **_,
    ):
        if self._engine is None:
            return Err(SdkError("mahjong coach is not initialized"))
        try:
            async with self._get_engine_lock():
                self._apply_runtime_round_context(
                    round_wind=round_wind,
                    seat_wind=seat_wind,
                    dora_tiles=dora_tiles,
                    play_style=play_style,
                )
                self._apply_runtime_strategy_preset(strategy_preset)
                self._apply_runtime_live_advice_mode(live_advice_mode)
                self._apply_runtime_river_tracking_mode(river_tracking_mode)
                self._apply_runtime_tile_recognition_mode(tile_recognition_mode)
                self._apply_runtime_inference_provider(inference_provider)
                self._apply_runtime_settlement_config(
                    enabled=settlement_recognition_enabled,
                    min_confidence=settlement_min_confidence,
                    confirm_frames=settlement_confirm_frames,
                    confirm_max_gap_ms=settlement_confirm_max_gap_ms,
                )
                decision = await self._await_engine_thread(
                    self._engine.analyze_frame,
                    image_path or None,
                    observed_buttons=observed_buttons or [],
                    self_turn_index=self_turn_index if self_turn_index and self_turn_index > 0 else None,
                    force_checkpoint=bool(force_checkpoint),
                    riichi_players=riichi_players or [],
                )
                self._observe_live_round_transition(decision)
                decision = self._enrich_yakuman_decision(decision)
                self._last_decision = decision.to_dict()
                raw_decision_payload = dict(self._last_decision)
                engine_state = self._engine.state.to_dict()
        except Exception as exc:
            self.logger.warning("mahjong coach frame analysis failed: {}", exc)
            return Err(SdkError(str(exc)))
        # The external panel follows every completed recognition transaction.
        # Quiet observations must remain visible to both diagnostics and the
        # companion cadence; the cue builder decides whether speech is due.
        self._update_overlay({"last_decision": raw_decision_payload, "round_state": engine_state})
        decision_payload = build_public_payload(
            {"last_decision": raw_decision_payload, "round_state": engine_state},
            mode=self._cfg.live_advice_mode,
        )["last_decision"]
        self._append_live_timing(
            {
                **_mahjong_timing_from_decision(decision_payload),
                "frame": self._live_state.frame_index,
                "status": "manual_analysis",
                "locate_ms": None,
                "capture_ms": None,
                "analyze_ms": _read_float(decision_payload.get("engine_meta"), "elapsed_ms"),
                "loop_ms": _read_float(decision_payload.get("engine_meta"), "elapsed_ms"),
            }
        )
        return Ok(decision_payload)

    @plugin_entry(
        id="mahjong_coach_start_live",
        name=tr("entries.start_live.name", default="Start Mahjong Coach Live"),
        description=tr(
            "entries.start_live.description",
            default="Start screenshot-only live observation for Mahjong Soul and update the strategy board.",
        ),
        input_schema={
            "type": "object",
            "properties": {
                "keywords": {"type": "array", "items": {"type": "string"}, "default": []},
                "interval_ms": {"type": "integer", "default": 0},
                "overlay": {"type": "boolean", "default": True},
                "round_wind": {"type": "string", "default": ""},
                "seat_wind": {"type": "string", "default": ""},
                "dora_tiles": {"type": "array", "items": {"type": "string"}, "default": []},
                "play_style": {"type": "string", "default": ""},
                "strategy_preset": {"type": "string", "default": ""},
                "live_advice_mode": {"type": "string", "default": ""},
                "neko_companion_enabled": {"type": "boolean"},
                "absurd_banter_enabled": {"type": "boolean"},
                "river_tracking_mode": {"type": "string", "default": ""},
                "tile_recognition_mode": {"type": "string", "default": ""},
                "inference_provider": {"type": "string", "default": "speed"},
                "target_window_title": {"type": "string", "default": ""},
                "auto_start_live": {"type": "boolean"},
                "settlement_recognition_enabled": {"type": "boolean"},
                "settlement_min_confidence": {"type": "number"},
                "settlement_confirm_frames": {"type": "integer"},
                "settlement_confirm_max_gap_ms": {"type": "integer"},
            },
        },
        llm_result_fields=["status", "running"],
    )
    async def mahjong_coach_start_live(
        self,
        keywords: list[str] | None = None,
        interval_ms: int | None = None,
        overlay: bool = True,
        round_wind: str = "",
        seat_wind: str = "",
        dora_tiles: list[str] | None = None,
        play_style: str = "",
        strategy_preset: str = "",
        live_advice_mode: str = "",
        neko_companion_enabled: bool | None = None,
        absurd_banter_enabled: bool | None = None,
        river_tracking_mode: str = "",
        tile_recognition_mode: str = "",
        inference_provider: str = "speed",
        target_window_title: str = "",
        auto_start_live: bool | None = None,
        settlement_recognition_enabled: bool | None = None,
        settlement_min_confidence: float | None = None,
        settlement_confirm_frames: int | None = None,
        settlement_confirm_max_gap_ms: int | None = None,
        **_,
    ):
        return await self._overlay_start_live(
            keywords=keywords,
            interval_ms=interval_ms,
            overlay=overlay,
            round_wind=round_wind,
            seat_wind=seat_wind,
            dora_tiles=dora_tiles,
            play_style=play_style,
            strategy_preset=strategy_preset,
            live_advice_mode=live_advice_mode,
            neko_companion_enabled=neko_companion_enabled,
            absurd_banter_enabled=absurd_banter_enabled,
            river_tracking_mode=river_tracking_mode,
            tile_recognition_mode=tile_recognition_mode,
            inference_provider=inference_provider,
            target_window_title=target_window_title,
            auto_start_live=auto_start_live,
            settlement_recognition_enabled=settlement_recognition_enabled,
            settlement_min_confidence=settlement_min_confidence,
            settlement_confirm_frames=settlement_confirm_frames,
            settlement_confirm_max_gap_ms=settlement_confirm_max_gap_ms,
        )

    async def _overlay_start_live(
        self,
        keywords: list[str] | None = None,
        interval_ms: int | None = None,
        overlay: bool = True,
        round_wind: str = "",
        seat_wind: str = "",
        dora_tiles: list[str] | None = None,
        play_style: str = "",
        strategy_preset: str = "",
        live_advice_mode: str = "",
        neko_companion_enabled: bool | None = None,
        absurd_banter_enabled: bool | None = None,
        river_tracking_mode: str = "",
        tile_recognition_mode: str = "",
        inference_provider: str = "speed",
        target_window_title: str = "",
        auto_start_live: bool | None = None,
        settlement_recognition_enabled: bool | None = None,
        settlement_min_confidence: float | None = None,
        settlement_confirm_frames: int | None = None,
        settlement_confirm_max_gap_ms: int | None = None,
    ):
        self.logger.info(
            "_overlay_start_live called play_style={} strategy_preset={} live_advice_mode={} river_tracking_mode={} tile_recognition_mode={} inference_provider={}",
            play_style,
            strategy_preset,
            live_advice_mode,
            river_tracking_mode,
            tile_recognition_mode,
            inference_provider,
        )
        if self._engine is None:
            self.logger.warning("_overlay_start_live early return: engine is None")
            return Err(SdkError("mahjong coach is not initialized"))
        async with self._get_engine_lock():
            style_before = self._cfg.play_style
            strategy_preset_before = self._cfg.strategy_preset
            advice_mode_before = self._cfg.live_advice_mode
            river_mode_before = self._cfg.river_tracking_mode
            tile_mode_before = self._cfg.tile_recognition_mode
            inference_provider_before = self._cfg.inference_provider
            self._apply_runtime_round_context(
                round_wind=round_wind,
                seat_wind=seat_wind,
                dora_tiles=dora_tiles,
                play_style=play_style,
            )
            self._apply_runtime_strategy_preset(strategy_preset)
            self._apply_runtime_live_advice_mode(live_advice_mode)
            self._apply_runtime_neko_companion_enabled(neko_companion_enabled)
            self._apply_runtime_absurd_banter_enabled(absurd_banter_enabled)
            self._apply_runtime_river_tracking_mode(river_tracking_mode)
            self._apply_runtime_tile_recognition_mode(tile_recognition_mode)
            self._apply_runtime_inference_provider(inference_provider)
            self._apply_runtime_settlement_config(
                enabled=settlement_recognition_enabled,
                min_confidence=settlement_min_confidence,
                confirm_frames=settlement_confirm_frames,
                confirm_max_gap_ms=settlement_confirm_max_gap_ms,
            )
            if self._cfg.river_tracking_mode != river_mode_before:
                self.logger.info(
                    "mahjong coach river tracking mode changed {} -> {}",
                    river_mode_before,
                    self._cfg.river_tracking_mode,
                )
            if play_style and self._cfg.play_style != style_before:
                self._invalidate_live_plan_for_style_change()
            if strategy_preset and self._cfg.strategy_preset != strategy_preset_before:
                self._invalidate_live_plan_for_style_change()
            if live_advice_mode and self._cfg.live_advice_mode != advice_mode_before:
                self._display_snapshot = {}
            if tile_recognition_mode and self._cfg.tile_recognition_mode != tile_mode_before:
                self._invalidate_live_plan_for_style_change()
            inference_provider_changed = bool(
                inference_provider
                and self._cfg.inference_provider != inference_provider_before
            )
            if inference_provider_changed:
                warmup_state = getattr(self, "_yolo_warmup_state", {})
                warmup_matches = (
                    self._yolo_warmup_provider == self._cfg.inference_provider
                    and warmup_state.get("status") in {"warming", "ready"}
                )
                if not warmup_matches:
                    await self._stop_yolo26_warmup()
                    release_perception_runtime_caches()
            already_running = self._live_task is not None and not self._live_task.done()
            if (
                self._cfg.tile_recognition_mode == "yolo26"
                and self._cfg.inference_provider == "speed"
                and (not already_running or inference_provider_changed)
                # Fully initialized plugin instances own this lifecycle field.
                # Lightweight ``__new__`` fixtures and partially restored old
                # hosts must not launch an orphan native-model worker.
                and hasattr(self, "_yolo_warmup_state")
            ):
                self._start_yolo26_warmup("speed")
            if not already_running:
                selected_keywords = _clean_string_list(keywords) or list(self._cfg.live_window_keywords)
                selected_interval = max(200, int(interval_ms or self._cfg.live_interval_ms))
                explicit_title = str(target_window_title or "").strip()
                if not hasattr(self, "_preferences"):
                    self._preferences = CapturePreferences()
                target = self._preferences.target
                if explicit_title:
                    target = WindowTargetDescriptor(title=explicit_title)
                if not target.title and not target.app_name:
                    candidates = await asyncio.to_thread(list_window_candidates, selected_keywords)
                    matching = [item for item in candidates if bool(item.get("matches_keywords"))]
                    if len(matching) == 1:
                        item = matching[0]
                        target = WindowTargetDescriptor(
                            title=str(item.get("title") or ""),
                            app_name=str(item.get("app_name") or ""),
                            match_keyword=str(item.get("match_keyword") or ""),
                        )
                    elif len(matching) > 1:
                        target = await asyncio.to_thread(choose_window_candidate_native, matching)
                        if target is not None:
                            matching = []
                    if len(matching) > 1:
                        if overlay:
                            self._show_overlay(strategy=False)
                            self._overlay.update("Mahjong Coach\n检测到多个雀魂窗口，请先选择目标窗口")
                        return Ok({
                            "status": "selection_required",
                            "running": False,
                            "candidates": matching,
                        })
                preferences_store = getattr(self, "_preferences_store", None)
                if preferences_store is not None:
                    self._preferences = preferences_store.update(
                        auto_start_live=auto_start_live,
                        target=target if target.title or target.app_name else None,
                    )
                else:
                    self._preferences = replace(
                        self._preferences,
                        auto_start_live=(
                            self._preferences.auto_start_live
                            if auto_start_live is None
                            else bool(auto_start_live)
                        ),
                        target=target,
                    )
                # 中文：页面本次明确选择优先于可能过期的持久化设置。
                # English: The explicit page choice wins over a stale persisted setting.
                overlay_enabled = bool(overlay)
                if overlay_enabled and not self._show_overlay(strategy=True):
                    overlay_error = getattr(self._overlay, "last_error", "overlay startup failed")
                    self.logger.warning("mahjong coach live start aborted: {}", overlay_error)
                    return Err(SdkError(f"failed to open mahjong coach overlay: {overlay_error}"))
                self._live_stop_event = asyncio.Event()
                self._live_state = LiveSessionState(
                    running=True,
                    status="starting",
                    started_at=time.time(),
                    updated_at=time.time(),
                    overlay_enabled=overlay_enabled,
                )
                self._neko_companion_delivery_stage = (
                    "ready" if self._cfg.neko_companion_enabled else "disabled"
                )
                self._neko_companion_last_error = ""
                self._live_preview_yolo_snapshot = {}
                self._settlement_candidate_preview = {}
                self._settlement_confirmed_preview = {}
                self._settlement_preview_revision = 0
                self._runtime_resource_samples = []
                self._runtime_last_resource_sample_at = 0.0
                self._live_frame_queue_size = 0
                self._reset_neko_companion_ambient_state()
                self._clear_live_hand_gap()
                self._live_task = asyncio.create_task(
                    self._run_live_loop(
                        keywords=selected_keywords,
                        interval_ms=selected_interval,
                        overlay_enabled=overlay_enabled,
                        target=target,
                    )
                )
            live_payload = self._live_state.to_dict()

        if already_running:
            self.logger.warning("_overlay_start_live early return: live_task already running")
            overlay_ready = False
            if overlay:
                overlay_ready = self._show_overlay(strategy=True)
                if not overlay_ready:
                    overlay_error = getattr(self._overlay, "last_error", "overlay startup failed")
                    return Err(SdkError(f"failed to reopen mahjong coach overlay: {overlay_error}"))
            return Ok({
                "status": "already_running",
                "running": True,
                "overlay_ready": overlay_ready,
                "live": live_payload,
            })
        return Ok({
            "status": "starting",
            "running": True,
            "overlay_ready": bool(overlay_enabled),
            "live": live_payload,
        })

    def _apply_runtime_round_context(
        self,
        *,
        round_wind: str = "",
        seat_wind: str = "",
        dora_tiles: list[str] | None = None,
        play_style: str = "",
    ) -> None:
        dora_list = None if dora_tiles is None else _clean_string_list(dora_tiles)
        style = _valid_play_style(play_style) if play_style else ""
        updated = replace(
            self._cfg,
            round_wind=str(round_wind or self._cfg.round_wind or "").strip(),
            seat_wind=str(seat_wind or self._cfg.seat_wind or "").strip(),
            dora_tiles=dora_list if dora_list is not None else list(self._cfg.dora_tiles),
            play_style=style or self._cfg.play_style,
        )
        self._cfg = updated
        if self._engine is not None:
            self._engine.config = updated
            self._engine.state.play_style = updated.play_style

    def _apply_runtime_river_tracking_mode(self, river_tracking_mode: str = "") -> None:
        if not river_tracking_mode:
            return
        mode = _valid_river_tracking_mode(river_tracking_mode)
        updated = replace(self._cfg, river_tracking_mode=mode)
        self._cfg = updated
        if self._engine is not None:
            self._engine.config = updated

    def _apply_runtime_strategy_preset(self, strategy_preset: str = "") -> None:
        if not strategy_preset:
            return
        preset = _valid_strategy_preset(strategy_preset)
        updated = replace(self._cfg, strategy_preset=preset)
        self._cfg = updated
        if self._engine is not None:
            self._engine.config = updated
            self._engine.state.strategy_preset = preset

    def _apply_runtime_live_advice_mode(self, live_advice_mode: str = "") -> None:
        if not live_advice_mode:
            return
        mode = _valid_live_advice_mode(live_advice_mode)
        updated = replace(self._cfg, live_advice_mode=mode)
        self._cfg = updated
        if self._engine is not None:
            self._engine.config = updated

    def _apply_runtime_neko_companion_enabled(self, enabled: bool | None = None) -> None:
        if enabled is None:
            return
        normalized = bool(enabled)
        changed = normalized != self._cfg.neko_companion_enabled
        updated = replace(self._cfg, neko_companion_enabled=normalized)
        self._cfg = updated
        if changed:
            self._neko_companion_last_signature = ""
            self._reset_neko_companion_ambient_state()
            self._neko_companion_delivery_stage = (
                "ready"
                if normalized and bool(getattr(self._live_state, "running", False))
                else "waiting_capture" if normalized else "disabled"
            )
            self._neko_companion_last_error = ""
        if self._engine is not None:
            self._engine.config = updated

    def _apply_runtime_absurd_banter_enabled(self, enabled: bool | None = None) -> None:
        if enabled is None:
            return
        normalized = bool(enabled)
        changed = normalized != self._cfg.absurd_banter_enabled
        updated = replace(self._cfg, absurd_banter_enabled=normalized)
        self._cfg = updated
        if changed:
            # Force the next eligible companion event to be rebuilt under the
            # new dialogue policy instead of being hidden by the old signature.
            self._neko_companion_last_signature = ""
        if self._engine is not None:
            self._engine.config = updated

    def _apply_runtime_inference_provider(self, inference_provider: str = "") -> None:
        if not inference_provider:
            return
        provider = _valid_inference_provider(inference_provider)
        updated = replace(self._cfg, inference_provider=provider)
        self._cfg = updated
        if self._engine is not None:
            self._engine.config = updated

    def _start_yolo26_warmup(self, provider: str = "speed") -> asyncio.Task | None:
        normalized = _valid_inference_provider(provider)
        if normalized != "speed":
            return None
        task = getattr(self, "_yolo_warmup_task", None)
        if task is not None and not task.done() and self._yolo_warmup_provider == normalized:
            return task
        state = getattr(self, "_yolo_warmup_state", {})
        if state.get("status") == "ready" and self._yolo_warmup_provider == normalized:
            return task
        degraded_retry_due = bool(
            state.get("status") in {"failed", "unavailable", "fallback"}
            and self._yolo_warmup_provider == normalized
            and time.time() - float(state.get("completed_at") or 0.0) >= _YOLO_WARMUP_RETRY_SECONDS
        )
        if (
            state.get("status") in {"failed", "unavailable", "fallback"}
            and self._yolo_warmup_provider == normalized
            and not degraded_retry_due
        ):
            return task
        if degraded_retry_due:
            # The cached ``speed`` session may be a CPU fallback.  Retry only
            # after the cooldown so a missing GPU provider cannot rebuild the
            # 82 MB model on every captured frame.
            reset_yolo26_accelerator_session()
        self._yolo_warmup_provider = normalized
        self._yolo_warmup_state = {
            "status": "warming",
            "inference_provider": normalized,
            "started_at": time.time(),
        }
        task = asyncio.create_task(self._run_yolo26_warmup(normalized))
        self._yolo_warmup_task = task
        return task

    def _activate_deferred_runtime_work(self) -> None:
        """Launch startup-deferred work on the persistent plugin-entry loop."""

        if bool(getattr(self, "_runtime_warmup_pending", False)):
            self._runtime_warmup_pending = False
            if (
                getattr(self, "_engine", None) is not None
                and self._cfg.tile_recognition_mode == "yolo26"
                and self._cfg.inference_provider == "speed"
            ):
                self._start_yolo26_warmup("speed")

        if not bool(getattr(self, "_auto_start_live_pending", False)):
            return
        self._auto_start_live_pending = False
        task = asyncio.create_task(
            self._overlay_start_live(overlay=True, inference_provider="speed")
        )
        self._deferred_auto_start_task = task

        def _record_auto_start_result(done: asyncio.Task) -> None:
            try:
                result = done.result()
            except asyncio.CancelledError:
                return
            except Exception as exc:
                self.logger.warning(
                    "mahjong coach deferred auto-start failed error_type={}",
                    type(exc).__name__,
                )
                return
            if isinstance(result, Err):
                self.logger.warning("mahjong coach deferred auto-start rejected: {}", result.error)

        task.add_done_callback(_record_auto_start_result)

    async def _run_yolo26_warmup(self, provider: str) -> dict[str, Any]:
        try:
            result, cancelled = await self._await_thread_result(
                warmup_yolo26_runtime,
                inference_provider=provider,
            )
            if cancelled:
                raise asyncio.CancelledError
            self._yolo_warmup_state = {
                **dict(result),
                "completed_at": time.time(),
            }
            self.logger.info(
                "mahjong coach inference warmup configured={} status={} accelerated={} actual_providers={} available_providers={} fallback_reason={} elapsed_ms={}",
                provider,
                self._yolo_warmup_state.get("status"),
                self._yolo_warmup_state.get("accelerated"),
                self._yolo_warmup_state.get("providers") or [],
                self._yolo_warmup_state.get("available_providers") or [],
                self._yolo_warmup_state.get("fallback_reason") or self._yolo_warmup_state.get("reason") or "",
                self._yolo_warmup_state.get("elapsed_ms"),
            )
        except asyncio.CancelledError:
            self._yolo_warmup_state = {"status": "idle"}
            raise
        except Exception as exc:
            self._yolo_warmup_state = {
                "status": "failed",
                "reason": type(exc).__name__,
                "inference_provider": provider,
                "completed_at": time.time(),
            }
        return dict(self._yolo_warmup_state)

    async def _ensure_yolo26_warmup(self, provider: str = "speed") -> dict[str, Any]:
        task = self._start_yolo26_warmup(provider)
        if task is not None and not task.done():
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.shield(task)
        return dict(getattr(self, "_yolo_warmup_state", {"status": "idle"}))

    async def _stop_yolo26_warmup(self) -> None:
        task = getattr(self, "_yolo_warmup_task", None)
        self._yolo_warmup_task = None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._yolo_warmup_provider = ""
        self._yolo_warmup_state = {"status": "idle"}

    def _apply_runtime_tile_recognition_mode(self, tile_recognition_mode: str = "") -> None:
        if not tile_recognition_mode:
            return
        mode = _valid_tile_recognition_mode(tile_recognition_mode)
        updated = replace(self._cfg, tile_recognition_mode=mode)
        self._cfg = updated
        if self._engine is not None:
            self._engine.config = updated

    def _apply_runtime_settlement_config(
        self,
        *,
        enabled: bool | None = None,
        min_confidence: float | None = None,
        confirm_frames: int | None = None,
        confirm_max_gap_ms: int | None = None,
    ) -> None:
        # 中文：网页面板只覆盖明确提交的结算参数，未提交项继续使用插件配置。
        # English: Override only explicitly submitted settlement settings.
        updated = replace(
            self._cfg,
            settlement_recognition_enabled=(
                self._cfg.settlement_recognition_enabled if enabled is None else bool(enabled)
            ),
            settlement_min_confidence=(
                self._cfg.settlement_min_confidence
                if min_confidence is None
                else max(0.0, min(1.0, float(min_confidence)))
            ),
            settlement_confirm_frames=(
                self._cfg.settlement_confirm_frames
                if confirm_frames is None
                else max(1, min(8, int(confirm_frames)))
            ),
            settlement_confirm_max_gap_ms=(
                self._cfg.settlement_confirm_max_gap_ms
                if confirm_max_gap_ms is None
                else max(200, min(10_000, int(confirm_max_gap_ms)))
            ),
        )
        if updated == self._cfg:
            return
        self._cfg = updated
        if self._engine is not None:
            self._engine.config = updated
        self.logger.info(
            "mahjong coach settlement config enabled={} confidence={} frames={} max_gap_ms={}",
            updated.settlement_recognition_enabled,
            updated.settlement_min_confidence,
            updated.settlement_confirm_frames,
            updated.settlement_confirm_max_gap_ms,
        )

    def _invalidate_live_plan_for_style_change(self) -> None:
        if self._engine is None:
            return
        self._engine.state.opening_emitted = False
        self._engine.state.opening_plan = ""
        self._engine.state.current_plan = ""
        self._engine.state.local_direction = ""
        self._engine.state.local_plan = ""
        self._engine.state.local_detail = ""
        self._engine.state.target_shapes = []
        self._engine.state.caution_points = []
        self._engine.state.last_update_reason = "style_changed"
        self._live_last_checkpoint_at = 0.0

    @plugin_entry(
        id="mahjong_coach_stop_live",
        name=tr("entries.stop_live.name", default="Stop Mahjong Coach Live"),
        description=tr("entries.stop_live.description", default="Stop live screenshot observation and overlay updates."),
        input_schema={"type": "object", "properties": {}},
        llm_result_fields=["status", "running"],
    )
    async def mahjong_coach_stop_live(self, **_):
        return await self._overlay_stop_live(hide_overlay=True)

    @plugin_entry(
        id="mahjong_coach_show_overlay",
        name=tr("entries.show_overlay.name", default="Show Mahjong Coach Overlay"),
        description=tr("entries.show_overlay.description", default="Reopen the Mahjong Coach overlay after it has been closed."),
        input_schema={"type": "object", "properties": {}},
        llm_result_fields=["status", "running"],
    )
    async def mahjong_coach_show_overlay(self, **_):
        running = self._live_task is not None and not self._live_task.done()
        # Opening the external panel is always an explicit strategy-selection
        # action. Background analysis may already be running, but it must not
        # skip the user's choice and jump straight into the result card.
        if not self._show_overlay(strategy=False):
            return Err(SdkError("failed to show mahjong coach overlay"))
        # Start model/session creation while the user is choosing a play style.
        # By the time capture starts, the first actionable frame can use the
        # hot CUDA / DirectML session instead of paying the cold-start cost.
        if (
            getattr(self, "_engine", None) is not None
            and self._cfg.tile_recognition_mode == "yolo26"
        ):
            self._start_yolo26_warmup("speed")
        return Ok({"status": "overlay_open", "running": running, "live": self._live_state.to_dict()})

    async def _overlay_stop_live(self, hide_overlay: bool = False):
        await self._stop_live_task()
        if hide_overlay:
            self._overlay.stop()
        return Ok({"status": self._live_state.status, "running": self._live_state.running, "live": self._live_state.to_dict()})

    @plugin_entry(
        id="mahjong_coach_window_candidates",
        name=tr("entries.window_candidates.name", default="List Mahjong Window Candidates"),
        description=tr("entries.window_candidates.description", default="List visible windows considered by Mahjong Coach live capture."),
        input_schema={"type": "object", "properties": {"keywords": {"type": "array", "items": {"type": "string"}, "default": []}}},
        llm_result_fields=["candidates"],
    )
    async def mahjong_coach_window_candidates(self, keywords: list[str] | None = None, **_):
        selected_keywords = _clean_string_list(keywords) or list(self._cfg.live_window_keywords)
        candidates = await asyncio.to_thread(list_window_candidates, selected_keywords)
        return Ok({"keywords": selected_keywords, "candidates": candidates})

    @plugin_entry(
        id="mahjong_coach_extract_hand_crops",
        name=tr("entries.extract_crops.name", default="Extract Hand Crops for Training"),
        description=tr(
            "entries.extract_crops.description",
            default="Extract hand tile crops from saved live frames for classifier training.",
        ),
        input_schema={
            "type": "object",
            "properties": {
                "output_dir": {"type": "string", "default": ""},
                "classify": {"type": "boolean", "default": True},
            },
        },
        timeout=60.0,
        llm_result_fields=["total", "per_label"],
    )
    async def mahjong_coach_extract_hand_crops(
        self,
        output_dir: str = "",
        classify: bool = True,
        **_,
    ):
        from .perception.calibration import resolve_calibration_profile
        from .perception.hand_layout import build_hand_layout
        from .perception.roi import collect_region_metrics
        from .perception.tile_classifier_dispatch import classify_hand_tile
        from .perception.tile_templates import is_probably_occupied_hand_slot
        from .tile_labels import normalize_tile

        frames_dir = self.data_path("live_frames")
        out = Path(output_dir) if output_dir else self.data_path("hand_crops")
        calibration_dir = Path(__file__).parent / "data" / "calibration" / "profiles"

        images = sorted(p for p in frames_dir.glob("*-frame.*") if p.is_file())
        if not images:
            return Err(SdkError(f"No frames found in {frames_dir}"))

        counts: dict[str, int] = {}
        min_confidence = 0.10
        IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

        for image_path in images:
            from PIL import Image
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
            w, h = image.size
            calibration = resolve_calibration_profile(w, h, calibration_dir=calibration_dir)
            template_payload = calibration.hand_tile_templates
            layout = build_hand_layout(w, h, calibration=calibration)

            for slot in layout["hand"][:14]:
                metrics = collect_region_metrics(image, slot.box, sample_step=6)
                occupied = is_probably_occupied_hand_slot({
                    "slot_mean_luma": metrics.get("mean_luma"),
                    "slot_bright_ratio": metrics.get("bright_ratio"),
                    "slot_dark_ratio": metrics.get("dark_ratio"),
                    "slot_stddev": metrics.get("stddev"),
                })
                if not occupied:
                    continue
                crop = image.crop((slot.box.left, slot.box.top, slot.box.right, slot.box.bottom))

                if classify and template_payload:
                    match = classify_hand_tile(crop, template_payload)
                    if match and match.confidence >= min_confidence:
                        label = match.tile
                    else:
                        label = "unclassified"
                else:
                    label = "unclassified"

                label_dir = out / label
                label_dir.mkdir(parents=True, exist_ok=True)
                crop.save(label_dir / f"{image_path.stem}_{slot.slot_id}.png")
                counts[label] = counts.get(label, 0) + 1

        total = sum(counts.values())
        self.logger.info("Extracted {} hand crops: {}", total, counts)
        return Ok({"total": total, "per_label": counts, "output_dir": str(out)})

    async def _run_live_loop(
        self,
        *,
        keywords: list[str],
        interval_ms: int,
        overlay_enabled: bool,
        target: WindowTargetDescriptor | None = None,
    ) -> None:
        assert self._engine is not None
        provider = DefaultCaptureProvider()
        session = CaptureSession(keywords, target=target)
        frames_dir = self.data_path("live_frames")
        frame_queue: asyncio.Queue[tuple[Any, float, float]] = asyncio.Queue(maxsize=1)
        producer = asyncio.create_task(
            self._capture_live_frames(
                session=session,
                provider=provider,
                frame_queue=frame_queue,
                interval_ms=interval_ms,
                frames_dir=frames_dir,
            )
        )
        if overlay_enabled:
            self._overlay.start()
            self._overlay.update("Mahjong Coach\n等待雀魂窗口")
        try:
            while self._live_stop_event is not None and not self._live_stop_event.is_set():
                loop_started = time.monotonic()
                packet = None
                try:
                    try:
                        packet, locate_ms, capture_ms = await asyncio.wait_for(frame_queue.get(), timeout=1.0)
                    except asyncio.TimeoutError:
                        if producer.done():
                            producer.result()
                        continue
                    self._live_state.last_capture_source = packet.source
                    self._live_state.last_window_title = packet.window_title
                    analyze_started = time.perf_counter()
                    decision, gap_state, paused_decisions, engine_state = await self._analyze_live_packet(
                        packet.image if packet.image is not None else packet.image_path,
                        force_checkpoint_by_time=True,
                    )
                    analyze_ms = _elapsed_ms(analyze_started)
                    decision_ready_at_ms = int(time.time() * 1000)
                    companion_pushed = False
                    companion_ms: float | None = None
                    companion_ready_at_ms = 0
                    self._live_state.running = True
                    if decision.decision_type in paused_decisions:
                        self._live_state.status = decision.decision_type
                    else:
                        self._live_state.status = (
                            gap_state
                            if gap_state in {"view_obstructed", "verifying_new_round"}
                            else "observing"
                        )
                    self._live_state.frame_index += 1
                    self._live_state.updated_at = time.time()
                    self._live_state.last_error = {
                        "view_obstructed": "牌桌被菜单或其他窗口遮挡；已暂停刷新并保留当前对局。",
                        "settlement_candidate": "检测到结算候选；正在用下一帧复核。",
                        "round_settlement": "小局结算已确认；上一局状态已冻结。",
                        "awaiting_next_round": "结算已结束；等待稳定的新手牌。",
                    }.get(self._live_state.status, "")
                    if self._live_state.status == "waiting_for_game":
                        self._live_state.last_error = "尚未确认正式牌桌；识别与策略分析已暂停。"
                    payload = {"last_decision": dict(self._last_decision), "round_state": engine_state}
                    snapshot = self._update_overlay(payload)
                    # Evaluate every recognized frame, including quiet observe
                    # frames. Previously this was guarded by ``decision.quiet``
                    # and therefore made the ambient/idle cadence unreachable
                    # during exactly the uneventful stretches it was built for.
                    companion_started = time.perf_counter()
                    companion_pushed = await self._push_neko_companion_if_needed(snapshot)
                    companion_ms = _elapsed_ms(companion_started)
                    if companion_pushed:
                        companion_ready_at_ms = int(time.time() * 1000)
                    await self._capture_settlement_evidence(packet, decision)
                    # Diagnostic JPEG persistence is deliberately after overlay
                    # and companion submission so a critical riichi/call event
                    # never waits for disk encoding.
                    if packet.image is not None and self._should_persist_preview(decision):
                        preview_path = frames_dir / "last-preview.jpg"
                        await asyncio.to_thread(provider.persist_packet, packet, preview_path)
                        self._live_state.last_frame_path = str(preview_path)
                        self._live_state.last_frame_revision += 1
                        self._remember_live_preview_yolo_snapshot(
                            preview_path,
                            image_source=packet.image,
                        )
                        self._live_last_preview_at = time.monotonic()
                        # Publish a second lightweight revision after the JPEG
                        # has actually been replaced. Text was already visible,
                        # while the detail pane now receives the matching image.
                        self._update_overlay(payload)
                    self._log_live_timing(
                        decision=dict(self._last_decision),
                        locate_ms=locate_ms,
                        capture_ms=capture_ms,
                        analyze_ms=analyze_ms,
                        loop_ms=_elapsed_ms(loop_started),
                        captured_at_ms=int(packet.timestamp_ms or 0),
                        decision_ready_at_ms=decision_ready_at_ms,
                        companion_pushed=companion_pushed,
                        companion_ms=companion_ms,
                        companion_ready_at_ms=companion_ready_at_ms,
                    )
                    self._live_frame_queue_size = frame_queue.qsize()
                    self._sample_runtime_resources(phase="live", force=False)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self._live_state.status = "error"
                    self._live_state.last_error = repr(exc)
                    self._live_state.updated_at = time.time()
                    if packet is not None and packet.image is not None:
                        error_path = self.data_path("live_frames", f"error-{int(time.time())}.jpg")
                        with contextlib.suppress(Exception):
                            await asyncio.to_thread(provider.persist_packet, packet, error_path)
                            self._live_state.last_frame_path = str(error_path)
                            self._live_state.last_frame_revision += 1
                            self._live_preview_yolo_snapshot = {}
                    self._append_live_timing(
                        {
                            "frame": self._live_state.frame_index,
                            "status": "error",
                            "decision": "error",
                            "source": type(exc).__name__,
                            "locate_ms": None,
                            "capture_ms": None,
                            "analyze_ms": None,
                            "engine_total_ms": None,
                            "hand_ms": None,
                            "meld_ms": None,
                            "action_ms": None,
                            "river_ms": None,
                            "settlement_ms": None,
                            "strategy_ms": None,
                            "loop_ms": _elapsed_ms(loop_started),
                        }
                    )
                    async with self._get_engine_lock():
                        engine_state = self._engine.state.to_dict()
                    self._update_overlay({"last_decision": {"summary": "实战观察错误", "suggestion": repr(exc)}, "round_state": engine_state})
                finally:
                    if packet is not None and packet.image is not None:
                        with contextlib.suppress(Exception):
                            packet.image.close()
        finally:
            producer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await producer
            while not frame_queue.empty():
                packet, _locate_ms, _capture_ms = frame_queue.get_nowait()
                if packet.image is not None:
                    with contextlib.suppress(Exception):
                        packet.image.close()
            await self._prune_live_frames(frames_dir)
            self._live_state.running = False
            self._live_state.status = "stopped"
            self._live_state.updated_at = time.time()

    async def _capture_live_frames(
        self,
        *,
        session: CaptureSession,
        provider: DefaultCaptureProvider,
        frame_queue: asyncio.Queue,
        interval_ms: int,
        frames_dir: Path,
    ) -> None:
        while self._live_stop_event is not None and not self._live_stop_event.is_set():
            loop_started = time.monotonic()
            try:
                locate_started = time.perf_counter()
                capture_memory = getattr(provider, "capture_memory_frame", None)
                if callable(capture_memory):
                    binding = await asyncio.to_thread(session.locate_window)
                else:
                    binding = await asyncio.to_thread(provider.locate_window, session.keywords)
                locate_ms = _elapsed_ms(locate_started)
                self._live_state.last_binding = binding.to_dict()
                if not binding.bound:
                    self._live_state.status = "waiting_for_window"
                    self._live_state.last_error = binding.error or "window_not_found"
                    self._live_state.updated_at = time.time()
                    self._update_overlay({
                        "last_decision": {"summary": "等待雀魂窗口", "suggestion": self._live_state.last_error},
                        "round_state": self._engine.state.to_dict() if self._engine is not None else {},
                    })
                    await self._sleep_live(loop_started, max(750, interval_ms))
                    continue
                preferences = getattr(self, "_preferences", CapturePreferences())
                preferences_store = getattr(self, "_preferences_store", None)
                if session.target != preferences.target and preferences_store is not None:
                    self._preferences = preferences_store.update(target=session.target)
                capture_started = time.perf_counter()
                if callable(capture_memory):
                    packet, cancelled = await self._await_thread_result(
                        capture_memory,
                        binding_result=binding,
                    )
                else:
                    packet, cancelled = await self._await_thread_result(
                        provider.capture_frame,
                        samples_dir=frames_dir,
                        binding_result=binding,
                        save_format=self._cfg.live_save_format,
                    )
                capture_ms = _elapsed_ms(capture_started)
                if cancelled:
                    raise asyncio.CancelledError
                if frame_queue.full():
                    stale, _old_locate, _old_capture = frame_queue.get_nowait()
                    if stale.image is not None:
                        with contextlib.suppress(Exception):
                            stale.image.close()
                    self._live_state.dropped_frames += 1
                frame_queue.put_nowait((packet, locate_ms, capture_ms))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                session.invalidate(type(exc).__name__)
                self._live_state.status = "waiting_for_window"
                self._live_state.last_error = repr(exc)
                self._live_state.updated_at = time.time()
            await self._sleep_live(loop_started, interval_ms)

    def _should_persist_preview(self, decision: Any) -> bool:
        if bool(getattr(decision, "action_required", False)):
            return True
        now = time.monotonic()
        requested_at = float(getattr(self, "_live_preview_requested_at", 0.0) or 0.0)
        dashboard_active = requested_at > 0 and (now - requested_at) <= _LIVE_PREVIEW_ACTIVE_TTL_SECONDS
        interval = (
            _LIVE_PREVIEW_ACTIVE_INTERVAL_SECONDS
            if dashboard_active
            else max(
                _LIVE_PREVIEW_IDLE_INTERVAL_SECONDS,
                float(self._cfg.live_checkpoint_interval_seconds),
            )
        )
        return (now - self._live_last_preview_at) >= interval

    def _remember_live_preview_yolo_snapshot(
        self,
        frame_path: Path,
        *,
        image_source: Any | None = None,
    ) -> None:
        """Pin the exact YOLO warped input and detections to one dashboard revision."""

        self._live_preview_yolo_snapshot = {}
        if self._engine is None or self._cfg.tile_recognition_mode != "yolo26":
            return
        result = getattr(self._engine, "_last_yolo26_result", None)
        if result is None:
            return
        engine_state = getattr(self._engine, "state", None)
        table_region_preview: dict[str, Any] = {}
        scene = getattr(self._engine, "_last_game_scene_result", None)
        table_surface = getattr(scene, "table_surface", None)
        warped_image = getattr(table_surface, "warped_image", None)
        current_identity = source_identity(image_source) if image_source is not None else None
        scene_identity = getattr(self._engine, "_last_game_scene_identity", None)
        yolo_identity = getattr(self._engine, "_last_yolo26_identity", None)
        scene_matches_current = current_identity is None or current_identity == scene_identity
        yolo_matches_current = current_identity is None or current_identity == yolo_identity
        if (
            scene_matches_current
            and bool(getattr(table_surface, "ok", False))
            and warped_image is not None
        ):
            try:
                table_region_preview = _build_warped_table_region_preview_payload(
                    warped_image,
                    image_path=frame_path,
                    raw_detections=result.raw_detections,
                    opponent_melds=result.opponent_melds,
                    reason=str(getattr(table_surface, "reason", "") or ""),
                    table_surface_method=str(getattr(table_surface, "method", "") or ""),
                    evidence_source=(
                        "engine_inference_warp"
                        if yolo_matches_current
                        else "engine_current_warp_reused_detections"
                    ),
                )
            except (OSError, ValueError):
                table_region_preview = {}
        self._live_preview_yolo_snapshot = {
            "image_path": str(frame_path.resolve()),
            "revision": int(self._live_state.last_frame_revision or 0),
            "round_id": str(getattr(engine_state, "round_id", "") or ""),
            "raw_detections": [
                dict(item) for item in result.raw_detections if isinstance(item, dict)
            ],
            "opponent_melds": {
                str(owner): [dict(item) for item in items if isinstance(item, dict)]
                for owner, items in result.opponent_melds.items()
            },
            "table_region_preview": table_region_preview,
        }

    async def _capture_settlement_evidence(self, packet: Any, decision: Any) -> None:
        """Freeze only real settlement candidate/confirmation frames for diagnostics."""

        decision_type = str(getattr(decision, "decision_type", "") or "")
        if decision_type not in {"settlement_candidate", "round_settlement"}:
            if (
                self._engine is not None
                and str(self._engine.state.settlement_phase or "playing") == "playing"
            ):
                self._settlement_candidate_preview = {}
            return

        perception = getattr(decision, "perception", {})
        settlement = perception.get("settlement") if isinstance(perception, dict) else {}
        if not isinstance(settlement, dict) or not bool(settlement.get("detected")):
            return
        image = getattr(packet, "image", None)
        if image is None:
            raw_path = str(getattr(packet, "image_path", "") or "").strip()
            if not raw_path:
                return
            try:
                with Image.open(raw_path) as source:
                    image = source.convert("RGB")
            except (OSError, ValueError):
                return
        else:
            image = image.copy()

        from .perception.settlement_detector import SettlementFrameResult

        result = SettlementFrameResult(
            detected=True,
            kind=str(settlement.get("kind") or "unknown"),
            confidence=float(settlement.get("confidence") or 0.0),
            reason=str(settlement.get("reason") or "settlement_evidence"),
            evidence=[str(item) for item in (settlement.get("evidence") or [])],
            metrics=dict(settlement.get("metrics") or {}),
            elapsed_ms=float(settlement.get("elapsed_ms") or 0.0),
        )
        phase = "candidate" if decision_type == "settlement_candidate" else "confirmed"
        self._settlement_preview_revision += 1
        try:
            preview = await asyncio.to_thread(
                _build_settlement_diagnostic_preview_from_image,
                image,
                result=result,
                image_path=str(getattr(packet, "image_path", "") or "memory://live-frame"),
                min_confidence=self._cfg.settlement_min_confidence,
            )
        finally:
            image.close()
        preview.update(
            {
                "status": "ready",
                "phase": phase,
                "round_id": str(self._engine.state.round_id if self._engine is not None else ""),
                "captured_at": float(getattr(packet, "timestamp_ms", 0) or 0) / 1000.0,
                "revision": self._settlement_preview_revision,
            }
        )
        if phase == "candidate":
            self._settlement_candidate_preview = preview
        else:
            self._settlement_confirmed_preview = preview
            self._settlement_candidate_preview = {}

    async def _prune_live_frames(self, frames_dir: Path) -> None:
        """Enforce retention after every loop outcome, including cancellation."""
        _unused, cancelled = await self._await_thread_result(
            prune_frames,
            frames_dir,
            keep=self._cfg.live_keep_frames,
        )
        raw_path = str(self._live_state.last_frame_path or "").strip()
        if raw_path and not Path(raw_path).is_file():
            self._live_state.last_frame_path = ""
        if cancelled:
            raise asyncio.CancelledError

    async def _analyze_live_packet(
        self,
        image_source: Any,
        *,
        force_checkpoint_by_time: bool,
    ) -> tuple[Any, str, set[str], dict[str, Any]]:
        """Run one complete live engine transaction under the shared mutex."""
        assert self._engine is not None
        if self._cfg.tile_recognition_mode == "yolo26" and self._cfg.inference_provider == "speed":
            await self._ensure_yolo26_warmup("speed")
        async with self._get_engine_lock():
            force_checkpoint = self._checkpoint_due_by_time() if force_checkpoint_by_time else False
            decision = await self._await_engine_thread(
                self._engine.analyze_frame,
                None,
                image=image_source,
                self_turn_index=self._live_state.observed_hand_changes or None,
                force_checkpoint=force_checkpoint,
                require_game_scene=True,
            )
            settlement_decisions = {
                "settlement_candidate",
                "round_settlement",
                "awaiting_next_round",
            }
            paused_decisions = {*settlement_decisions, "waiting_for_game"}
            round_transition = self._observe_live_round_transition(decision)
            gap_state = (
                "none"
                if round_transition or decision.decision_type in paused_decisions
                else self._classify_live_hand_gap(decision)
            )
            if gap_state == "new_round":
                previous_round_id = self._engine.state.round_id
                self._engine.reset_round(f"auto-gap-round-{self._live_state.frame_index + 1}")
                decision = await self._await_engine_thread(
                    self._engine.analyze_frame,
                    None,
                    image=image_source,
                    self_turn_index=None,
                    force_checkpoint=False,
                    require_game_scene=True,
                )
                self._engine.state.last_update_reason = "auto_new_round_detected"
                decision = replace(
                    decision,
                    reason_codes=[*decision.reason_codes, "auto_new_round_detected"],
                    coach_state=self._engine.state.to_dict(),
                    engine_meta={
                        **decision.engine_meta,
                        "round_transition": "hand_gap_replacement",
                        "previous_round_id": previous_round_id,
                    },
                )
                round_transition = self._observe_live_round_transition(decision)

            decision = self._enrich_yakuman_decision(decision)
            self._last_decision = decision.to_dict()
            if not round_transition and decision.decision_type not in paused_decisions:
                self._observe_live_hand_change()
            if (
                gap_state not in {"view_obstructed", "verifying_new_round"}
                and decision.decision_type not in paused_decisions
                and decision.decision_type in {
                    "opening_plan",
                    "coach_checkpoint",
                    "defense_alert",
                }
            ):
                self._live_last_checkpoint_at = time.time()
            return decision, gap_state, paused_decisions, self._engine.state.to_dict()

    def _enrich_yakuman_decision(self, decision: Any) -> Any:
        service = getattr(self, "_yakuman_service", None)
        if service is None or self._engine is None:
            return decision
        hand_tiles = [
            normalize_tile(tile)
            for tile in (getattr(decision, "hand_tiles", None) or self._engine.state.last_hand_tiles)
            if normalize_tile(tile)
        ]
        if len(hand_tiles) not in {13, 14}:
            return decision
        visible_tiles = [
            *self._engine.state.last_visible_discards,
            *self._engine.state.last_meld_tiles,
            *self._engine.state.last_opponent_meld_tiles,
        ]
        payload = service.request(
            hand_tiles,
            visible_tiles=visible_tiles,
            open_melds=self._engine.state.last_open_meld_count,
            run_background=(
                str(getattr(getattr(self._cfg, "player_profile", None), "goal_bias", "balanced"))
                == "yakuman"
            ),
        )
        perception = dict(getattr(decision, "perception", {}) or {})
        perception["yakuman"] = payload
        engine_meta = dict(getattr(decision, "engine_meta", {}) or {})
        engine_meta["yakuman_status"] = payload.get("status")
        ready_key = str(payload.get("key") or "") if payload.get("status") == "ready" else ""
        newly_ready = bool(ready_key and ready_key != getattr(self, "_live_last_yakuman_key", ""))
        if newly_ready:
            self._live_last_yakuman_key = ready_key
        return replace(
            decision,
            perception=perception,
            engine_meta=engine_meta,
            quiet=False if newly_ready else bool(getattr(decision, "quiet", False)),
        )

    def _log_live_timing(
        self,
        *,
        decision: dict[str, Any],
        locate_ms: float,
        capture_ms: float,
        analyze_ms: float,
        loop_ms: float,
        captured_at_ms: int = 0,
        decision_ready_at_ms: int = 0,
        companion_pushed: bool = False,
        companion_ms: float | None = None,
        companion_ready_at_ms: int = 0,
    ) -> None:
        timing = _mahjong_timing_from_decision(decision)
        timing.update(
            {
                "frame": self._live_state.frame_index,
                "status": self._live_state.status,
                "locate_ms": round(float(locate_ms), 1),
                "capture_ms": round(float(capture_ms), 1),
                "analyze_ms": round(float(analyze_ms), 1),
                "loop_ms": round(float(loop_ms), 1),
                "capture_to_decision_ms": (
                    max(0, int(decision_ready_at_ms) - int(captured_at_ms))
                    if captured_at_ms > 0 and decision_ready_at_ms > 0
                    else None
                ),
                "companion_pushed": bool(companion_pushed),
                "companion_ms": round(float(companion_ms), 1) if companion_ms is not None else None,
                "capture_to_companion_ms": (
                    max(0, int(companion_ready_at_ms) - int(captured_at_ms))
                    if companion_pushed and captured_at_ms > 0 and companion_ready_at_ms > 0
                    else None
                ),
            }
        )
        entry = self._append_live_timing(timing)
        self.logger.info("mahjong coach timing {}", entry)
        self.logger.info(
            "mahjong coach river tracking mode={} reason={} ok={} tiles={} new={} corrected={} pending={} full_rescan={} elapsed_ms={}",
            entry.get("river_mode"),
            entry.get("river_reason"),
            entry.get("river_ok"),
            entry.get("river_tile_count"),
            entry.get("river_new_discard_count"),
            entry.get("river_corrected_count"),
            entry.get("river_pending_corrections"),
            entry.get("river_full_rescan"),
            entry.get("river_ms"),
        )
        if entry.get("settlement_phase") or entry.get("settlement_kind"):
            self.logger.info(
                "mahjong coach settlement phase={} kind={} confidence={} archive_id={} evidence={} elapsed_ms={}",
                entry.get("settlement_phase"),
                entry.get("settlement_kind"),
                entry.get("settlement_confidence"),
                entry.get("round_archive_id"),
                entry.get("settlement_evidence"),
                entry.get("settlement_ms"),
            )

    def _append_live_timing(self, timing: dict[str, Any]) -> dict[str, Any]:
        # 保存最近的插件内运行日志，供 Web 面板实时查看。
        # Keep recent in-plugin runtime logs for the web panel.
        if not hasattr(self, "_live_timing_log"):
            self._live_timing_log = []
        entry = dict(timing)
        runtime = self._inference_runtime_snapshot()
        entry.setdefault("inference_provider", runtime["configured_mode"])
        entry.setdefault("inference_backend", runtime["backend"])
        entry.setdefault("inference_accelerated", runtime["accelerated"])
        entry.setdefault("inference_actual_providers", runtime["actual_providers"])
        entry.setdefault("inference_available_providers", runtime["available_providers"])
        entry.setdefault("inference_fallback_reason", runtime["fallback_reason"])
        entry.setdefault("inference_runtime_version", runtime["runtime_version"])
        entry.setdefault("inference_runtime_origin", runtime["runtime_origin"])
        entry["timestamp_ms"] = int(time.time() * 1000)
        self._live_timing_log.append(entry)
        del self._live_timing_log[:-80]
        return entry

    def _sample_runtime_resources(self, *, phase: str, force: bool) -> dict[str, Any]:
        now = time.monotonic()
        last_at = float(getattr(self, "_runtime_last_resource_sample_at", 0.0) or 0.0)
        samples = getattr(self, "_runtime_resource_samples", None)
        if not isinstance(samples, list):
            samples = []
            self._runtime_resource_samples = samples
        if not force and samples and now - last_at < 10.0:
            return dict(samples[-1])

        engine = getattr(self, "_engine", None)
        config = getattr(self, "_cfg", None)
        engine_stats = engine.runtime_stats() if engine is not None else {}
        perception_stats = perception_runtime_stats()
        memory_stats = _process_memory_snapshot()
        sample = {
            "timestamp_ms": int(time.time() * 1000),
            "monotonic_s": round(now, 3),
            "phase": str(phase),
            **memory_stats,
            **perception_stats,
            "engine": engine_stats,
            "timing_log_entries": len(getattr(self, "_live_timing_log", [])),
            "frame_queue_size": max(0, int(getattr(self, "_live_frame_queue_size", 0) or 0)),
            "live_task_active": bool(
                getattr(self, "_live_task", None) is not None
                and not self._live_task.done()
            ),
            "warmup_task_active": bool(
                getattr(self, "_yolo_warmup_task", None) is not None
                and not self._yolo_warmup_task.done()
            ),
            "inference_provider": str(getattr(config, "inference_provider", "memory")),
            # ONNX Runtime does not expose portable per-process VRAM figures for
            # DirectML. Keep the field explicit instead of presenting system GPU
            # allocation as plugin-owned memory.
            "gpu_memory_mb": None,
            "gpu_memory_note": "per-process DirectML/CUDA VRAM unavailable from portable ONNX Runtime APIs",
        }
        samples.append(sample)
        del samples[:-360]
        self._runtime_last_resource_sample_at = now
        return dict(sample)

    def _runtime_resource_summary(self) -> dict[str, Any]:
        samples = list(getattr(self, "_runtime_resource_samples", []))
        if not samples:
            return {"sample_count": 0}
        latest = dict(samples[-1])
        rss_values = [
            float(item["rss_mb"])
            for item in samples
            if isinstance(item.get("rss_mb"), (int, float))
        ]
        elapsed_seconds = max(
            0.0,
            float(samples[-1].get("monotonic_s") or 0.0)
            - float(samples[0].get("monotonic_s") or 0.0),
        )
        rss_delta = round(rss_values[-1] - rss_values[0], 2) if len(rss_values) >= 2 else 0.0
        growth_per_hour = (
            round(rss_delta * 3600.0 / elapsed_seconds, 2)
            if elapsed_seconds >= 60.0
            else None
        )
        engine_stats = latest.get("engine") if isinstance(latest.get("engine"), dict) else {}
        caches_released = bool(
            latest.get("phase") == "stopped"
            and int(latest.get("yolo_sessions") or 0) == 0
            and int(latest.get("tile_classifier_sessions") or 0) == 0
            and int(latest.get("ocr_sessions") or 0) == 0
            and not engine_stats.get("yolo_frame_cached")
            and not engine_stats.get("game_scene_frame_cached")
            and not latest.get("live_task_active")
            and not latest.get("warmup_task_active")
        )
        return {
            "sample_count": len(samples),
            "sample_interval_seconds": 10,
            "retention_samples": 360,
            "observed_seconds": round(elapsed_seconds, 1),
            "rss_start_mb": rss_values[0] if rss_values else None,
            "rss_current_mb": rss_values[-1] if rss_values else None,
            "rss_peak_mb": max(rss_values) if rss_values else None,
            "rss_delta_mb": rss_delta,
            "rss_growth_mb_per_hour": growth_per_hour,
            "caches_released_after_stop": caches_released,
            "latest": latest,
        }

    def _inference_runtime_snapshot(self) -> dict[str, Any]:
        stats = perception_runtime_stats()
        yolo_runtime = stats.get("yolo_runtime")
        runtime = dict(yolo_runtime) if isinstance(yolo_runtime, dict) else {}
        warmup = dict(getattr(self, "_yolo_warmup_state", {"status": "idle"}))
        providers = [str(item) for item in (runtime.get("actual_providers") or [])]
        if not providers and warmup.get("status") == "ready":
            providers = [str(item) for item in (warmup.get("providers") or [])]
        if "CUDAExecutionProvider" in providers:
            backend = "NVIDIA CUDA"
        elif "DmlExecutionProvider" in providers:
            backend = "DirectML"
        elif "CPUExecutionProvider" in providers:
            backend = "CPU"
        else:
            backend = "未加载"
        configured = str(getattr(self._cfg, "inference_provider", "speed") or "speed")
        fallback_reason = str(
            runtime.get("fallback_reason")
            or warmup.get("fallback_reason")
            or warmup.get("reason")
            or ""
        )
        return {
            "configured_mode": configured,
            "backend": backend,
            "actual_providers": providers,
            "available_providers": list(runtime.get("available_providers") or warmup.get("available_providers") or []),
            "session_loaded": bool(runtime.get("session_loaded")),
            "accelerated": backend in {"NVIDIA CUDA", "DirectML"},
            "fallback_reason": fallback_reason,
            "runtime_version": str(runtime.get("runtime_version") or ""),
            "runtime_origin": str(runtime.get("runtime_origin") or ""),
            "warmup_status": str(warmup.get("status") or "idle"),
            "warmup_elapsed_ms": warmup.get("elapsed_ms"),
        }

    def _inference_runtime_status(self, timing_log: list[dict[str, Any]]) -> dict[str, Any]:
        snapshot = self._inference_runtime_snapshot()
        latest = next(
            (
                item
                for item in reversed(timing_log)
                if isinstance(item, dict) and item.get("timestamp_ms")
            ),
            {},
        )
        updated_ms = int(latest.get("timestamp_ms") or 0)
        return {
            **snapshot,
            "configured_label": "极速" if snapshot["configured_mode"] == "speed" else "低内存",
            "last_recognition_at": updated_ms / 1000.0 if updated_ms else 0.0,
            "last_recognition_age_seconds": (
                round(max(0.0, time.time() - updated_ms / 1000.0), 1) if updated_ms else None
            ),
            "last_recognition_ms": latest.get("analyze_ms"),
            "last_decision": str(latest.get("decision") or ""),
        }

    async def _stop_live_task(self) -> None:
        await self._stop_yolo26_warmup()
        if self._live_stop_event is not None:
            self._live_stop_event.set()
        task = self._live_task
        self._live_task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                current_loop = asyncio.get_running_loop()
                if task.get_loop() is current_loop:
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
            except RuntimeError:
                pass
        if task is None or task.done():
            engine = getattr(self, "_engine", None)
            if engine is not None:
                async with self._get_engine_lock():
                    engine.release_transient_resources()
            release_strategy_runtime_caches()
            await asyncio.to_thread(release_perception_runtime_caches)
            self._live_preview_yolo_snapshot = {}
        self._live_frame_queue_size = 0
        self._live_state.running = False
        self._live_state.status = "stopped"
        self._live_state.updated_at = time.time()
        self._neko_companion_delivery_stage = (
            "waiting_capture" if self._cfg.neko_companion_enabled else "disabled"
        )
        self._sample_runtime_resources(phase="stopped", force=True)

    async def _sleep_live(self, loop_started: float, sleep_ms: int) -> None:
        elapsed_ms = (time.monotonic() - loop_started) * 1000.0
        remaining = max(0.05, (float(sleep_ms) - elapsed_ms) / 1000.0)
        if self._live_stop_event is None:
            await asyncio.sleep(remaining)
            return
        try:
            await asyncio.wait_for(self._live_stop_event.wait(), timeout=remaining)
        except asyncio.TimeoutError:
            pass

    def _observe_live_hand_change(self) -> None:
        if self._engine is None:
            return
        signature = str(self._engine.state.last_hand_signature or "")
        if not signature or signature == self._live_last_hand_signature:
            return
        self._live_last_hand_signature = signature
        self._live_state.observed_hand_changes += 1

    def _observe_live_round_transition(self, decision: Any) -> bool:
        # 中文：引擎确认新局后同步清空直播层巡目，避免把上一局的手牌变化次数带过去。
        # English: Reset live turn counters when the engine confirms a new round.
        reason_codes = {str(item) for item in (getattr(decision, "reason_codes", []) or [])}
        if "auto_new_round_detected" not in reason_codes or self._engine is None:
            return False
        self._reset_live_round_caches(
            current_hand_signature=str(self._engine.state.last_hand_signature or ""),
            clear_last_decision=True,
        )
        self.logger.info(
            "mahjong coach auto new round round_id={} river_before={} river_now={}",
            self._engine.state.round_id,
            getattr(decision, "engine_meta", {}).get("previous_river_count"),
            getattr(decision, "engine_meta", {}).get("current_river_count"),
        )
        return True

    def _reset_live_round_caches(
        self,
        *,
        current_hand_signature: str,
        clear_last_decision: bool,
    ) -> None:
        """Drop every presentation/dedupe cache that is scoped to one hand."""

        if clear_last_decision:
            self._last_decision = {}
        self._display_snapshot = {}
        self._live_preview_yolo_snapshot = {}
        self._settlement_candidate_preview = {}
        self._settlement_confirmed_preview = {}
        self._settlement_preview_revision = int(
            getattr(self, "_settlement_preview_revision", 0) or 0
        ) + 1
        self._live_last_yakuman_key = ""
        self._neko_companion_last_signature = ""
        self._reset_neko_companion_ambient_state()
        self._live_last_hand_signature = str(current_hand_signature or "")
        self._live_last_checkpoint_at = 0.0
        self._live_last_preview_at = 0.0
        if hasattr(self, "_live_state"):
            self._live_state.observed_hand_changes = 0
            self._live_state.last_frame_path = ""
            self._live_state.last_frame_revision += 1
        self._clear_live_hand_gap()

    def _clear_live_hand_gap(self) -> None:
        # 中文：只清除遮挡判定的临时证据，不触碰本局策略、牌河和立直状态。
        # English: Clear temporary obstruction evidence without touching round state.
        if hasattr(self, "_live_state"):
            self._live_state.missing_hand_frames = 0
        self._live_gap_hand_tiles = []
        self._live_gap_candidate_tiles = []
        self._live_gap_candidate_frames = 0

    def _classify_live_hand_gap(self, decision: Any) -> str:
        """Classify a missing-hand gap without destroying the current round."""
        if self._engine is None:
            return "none"
        if str(getattr(decision, "decision_type", "") or "") in {
            "settlement_candidate",
            "round_settlement",
            "awaiting_next_round",
        }:
            return "none"

        perception = getattr(decision, "perception", {}) or {}
        hand_meta = perception.get("hand", {}) if isinstance(perception, dict) else {}
        hand_meta = hand_meta if isinstance(hand_meta, dict) else {}
        hand_reason = str(hand_meta.get("reason") or "").strip()
        reason_codes = [str(item) for item in (getattr(decision, "reason_codes", []) or [])]
        current_tiles = _normalized_tiles(getattr(decision, "hand_tiles", []) or hand_meta.get("hand_tiles", []))
        gap_frames = int(getattr(self._live_state, "missing_hand_frames", 0) or 0)

        if current_tiles:
            if gap_frames <= 0:
                self._clear_live_hand_gap()
                return "none"

            previous_tiles = _normalized_tiles(
                getattr(self, "_live_gap_hand_tiles", []) or self._engine.state.last_hand_tiles
            )
            shared_previous = _shared_live_hand_tiles(previous_tiles, current_tiles)
            resume_threshold = max(1, min(len(previous_tiles), len(current_tiles)) - 1)
            if previous_tiles and shared_previous >= resume_threshold:
                was_obstructed = gap_frames >= 4
                self._clear_live_hand_gap()
                if was_obstructed:
                    self.logger.info(
                        "mahjong coach view resumed missing_frames={} shared_tiles={}",
                        gap_frames,
                        shared_previous,
                    )
                return "resumed"

            if 12 <= len(current_tiles) <= 14:
                if str(getattr(self._engine.config, "tile_recognition_mode", "yolo26")) == "yolo26":
                    # 中文：YOLO 模式必须同时拿到牌河归零证据；只换了手牌不能证明是新局。
                    # English: YOLO mode requires river-reset evidence; a changed hand alone is insufficient.
                    if self._engine.has_pending_new_round_confirmation():
                        return "verifying_new_round"
                    self._clear_live_hand_gap()
                    return "resumed"

                candidate_tiles = _normalized_tiles(getattr(self, "_live_gap_candidate_tiles", []))
                if candidate_tiles and _shared_live_hand_tiles(candidate_tiles, current_tiles) >= 10:
                    self._live_gap_candidate_frames = int(
                        getattr(self, "_live_gap_candidate_frames", 0) or 0
                    ) + 1
                else:
                    self._live_gap_candidate_tiles = list(current_tiles)
                    self._live_gap_candidate_frames = 1

                if self._live_gap_candidate_frames < 2:
                    # 中文：下一帧必须绕过指纹缓存，再独立确认一次新手牌。
                    # English: Bypass the fingerprint cache for an independent confirmation frame.
                    self._engine.request_full_rescan()
                    return "verifying_new_round"

                self.logger.info(
                    "mahjong coach new round confirmed after obstruction missing_frames={} shared_tiles={}",
                    gap_frames,
                    shared_previous,
                )
                self._clear_live_hand_gap()
                return "new_round"

            return "view_obstructed" if gap_frames >= 4 else "none"

        if getattr(decision, "action_required", False):
            return "view_obstructed" if gap_frames >= 4 else "none"
        if not self._engine.state.opening_emitted:
            self._clear_live_hand_gap()
            return "none"

        fingerprint_match = hand_reason == "fingerprint_match" or "hand_fingerprint_match" in reason_codes
        if fingerprint_match and int(getattr(self, "_live_gap_candidate_frames", 0) or 0) > 0:
            self._engine.request_full_rescan()
            return "verifying_new_round"

        hand_failed = bool(hand_reason and hand_reason != "fingerprint_match") or any(
            code.startswith("hand_") and code != "hand_fingerprint_match" for code in reason_codes
        )
        if not hand_failed and not (fingerprint_match and gap_frames > 0):
            return "view_obstructed" if gap_frames >= 4 else "none"

        if gap_frames <= 0:
            self._live_gap_hand_tiles = _normalized_tiles(self._engine.state.last_hand_tiles)
        self._live_gap_candidate_tiles = []
        self._live_gap_candidate_frames = 0
        self._live_state.missing_hand_frames += 1
        if self._live_state.missing_hand_frames == 4:
            self.logger.info(
                "mahjong coach view obstructed missing_frames={} round_id={}",
                self._live_state.missing_hand_frames,
                self._engine.state.round_id,
            )
        return "view_obstructed" if self._live_state.missing_hand_frames >= 4 else "none"

    def _checkpoint_due_by_time(self) -> bool:
        if self._engine is None or not self._engine.state.last_hand_tiles:
            return False
        if self._live_last_checkpoint_at <= 0:
            return False
        return (time.time() - self._live_last_checkpoint_at) >= self._cfg.live_checkpoint_interval_seconds

    async def _read_companion_bus_records(
        self,
        namespace_name: str,
        **kwargs: Any,
    ) -> list[Any] | None:
        """Read message-plane records; ``None`` means no SDK context in a unit stub."""

        if not hasattr(self, "ctx"):
            return None
        try:
            bus = self.bus
            namespace = getattr(bus, namespace_name, None) if bus is not None else None
            getter = getattr(namespace, "get", None)
            if not callable(getter):
                raise RuntimeError(f"message-plane {namespace_name} bus is unavailable")
            result = getter(**kwargs)
            if asyncio.iscoroutine(result):
                result = await result
            if isinstance(result, Err):
                raise RuntimeError(str(result.error))
            value = result.value if isinstance(result, Ok) else result
            if value is None:
                return []
            return list(value)
        except Exception as exc:
            raise RuntimeError(f"message-plane {namespace_name} query failed") from exc

    async def _probe_companion_message_plane(self, *, now: float) -> bool | None:
        plugin_id = self.plugin_id if hasattr(self, "ctx") else "mahjong_coach"
        try:
            records = await self._read_companion_bus_records(
                "messages",
                plugin_id=plugin_id,
                source=plugin_id,
                max_count=1,
                since_ts=max(0.0, now - 1.0),
                timeout=_NEKO_TRANSPORT_PROBE_TIMEOUT_SECONDS,
            )
        except RuntimeError:
            return False
        return None if records is None else True

    @staticmethod
    def _record_delivery_id(record: Any) -> str:
        metadata = getattr(record, "metadata", None)
        if not isinstance(metadata, dict):
            raw = getattr(record, "raw", None)
            metadata = raw.get("metadata") if isinstance(raw, dict) else None
        return str(metadata.get("delivery_id") or "") if isinstance(metadata, dict) else ""

    async def _confirm_companion_message_ingest(
        self,
        delivery_id: str,
        *,
        submitted_at: float,
    ) -> bool | None:
        if not hasattr(self, "ctx"):
            return None
        for delay in (0.0, 0.03, 0.07):
            if delay:
                await asyncio.sleep(delay)
            try:
                records = await self._read_companion_bus_records(
                    "messages",
                    plugin_id=self.plugin_id,
                    source=self.plugin_id,
                    max_count=12,
                    since_ts=max(0.0, submitted_at - 1.0),
                    timeout=_NEKO_TRANSPORT_PROBE_TIMEOUT_SECONDS,
                )
            except RuntimeError:
                return False
            if records is not None and any(
                self._record_delivery_id(record) == delivery_id for record in records
            ):
                return True
        return False

    async def _watch_companion_host_receipt(
        self,
        delivery_id: str,
        *,
        submitted_at: float,
    ) -> None:
        deadline = time.monotonic() + _NEKO_HOST_RECEIPT_WAIT_SECONDS
        while time.monotonic() < deadline:
            await asyncio.sleep(0.1)
            try:
                records = await self._read_companion_bus_records(
                    "events",
                    plugin_id=self.plugin_id,
                    max_count=20,
                    filter={"type": "plugin_delivery_receipt"},
                    since_ts=max(0.0, submitted_at - 1.0),
                    timeout=_NEKO_TRANSPORT_PROBE_TIMEOUT_SECONDS,
                )
            except RuntimeError:
                break
            for record in records or []:
                if self._record_delivery_id(record) != delivery_id:
                    continue
                metadata = getattr(record, "metadata", {})
                if not isinstance(metadata, dict):
                    metadata = {}
                if delivery_id != str(getattr(self, "_neko_companion_delivery_id", "") or ""):
                    return
                stage = str(metadata.get("stage") or "host_received")
                if stage == "host_rejected":
                    self._neko_companion_delivery_stage = "host_rejected"
                    self._neko_companion_last_error = str(
                        metadata.get("reason") or "宿主未接受本次搭话"
                    )
                    self._neko_companion_failure_count = int(
                        getattr(self, "_neko_companion_failure_count", 0) or 0
                    ) + 1
                else:
                    self._neko_companion_delivery_stage = "host_received"
                    self._neko_companion_last_error = ""
                    self._neko_companion_host_received_count = int(
                        getattr(self, "_neko_companion_host_received_count", 0) or 0
                    ) + 1
                return
        if delivery_id == str(getattr(self, "_neko_companion_delivery_id", "") or ""):
            self._neko_companion_delivery_stage = "host_unconfirmed"

    async def _submit_companion_direct_fallback(
        self,
        cue: str,
        *,
        delivery_id: str,
        event_kind: str,
        priority: int,
        metadata: dict[str, Any],
        direct_reply: bool = False,
    ) -> DirectCompanionSubmitResult:
        """Bypass a dead message-plane while keeping the host event contract."""

        return await asyncio.to_thread(
            submit_companion_direct,
            cue,
            plugin_id=self.plugin_id if hasattr(self, "ctx") else "mahjong_coach",
            delivery_id=delivery_id,
            event_kind=event_kind,
            priority=priority,
            metadata=metadata,
            direct_reply=direct_reply,
        )

    async def _push_neko_companion_if_needed(self, snapshot: dict[str, Any]) -> bool:
        """Send a hidden, non-prescriptive event cue to the active N.E.K.O chat."""
        if not self._cfg.neko_companion_enabled:
            return False
        events_since_rare = int(
            getattr(self, "_neko_companion_events_since_rare_joke", 0) or 0
        )
        last_rare_at = float(
            getattr(self, "_neko_companion_last_rare_joke_at", 0.0) or 0.0
        )
        now = time.time()
        rare_joke_due = bool(
            events_since_rare >= (_NEKO_RARE_JOKE_MIN_EVENTS - 1)
            and (
                last_rare_at <= 0
                or (now - last_rare_at) >= _NEKO_RARE_JOKE_MIN_INTERVAL_SECONDS
            )
        )
        absurd_banter_enabled = bool(
            self._cfg.neko_companion_enabled and self._cfg.absurd_banter_enabled
        )
        absurd_variant = int(getattr(self, "_neko_companion_push_count", 0) or 0)
        activity_signature = self._observe_neko_companion_activity(snapshot, now=now)
        opponent_meld_event = self._observe_neko_companion_opponent_meld(snapshot)
        cue, signature, event_kind = build_neko_companion_cue(
            snapshot,
            include_rare_joke=rare_joke_due,
            opponent_meld_event=opponent_meld_event,
            absurd_banter_enabled=absurd_banter_enabled,
            absurd_variant=absurd_variant,
        )
        if not cue:
            ambient_event = self._select_neko_companion_ambient_event(
                snapshot,
                activity_signature=activity_signature,
                now=now,
            )
            if not ambient_event:
                return False
            cue_activity_signature = activity_signature
            if ambient_event in {"idle_heartbeat", "casual_chat"}:
                # Keep a quiet but continuing conversation alive even when the
                # recognized table facts stay unchanged for several intervals.
                # The cadence counter makes each heartbeat independently
                # deduplicatable without inventing a fake table event.
                cadence_label = "casual" if ambient_event == "casual_chat" else "idle"
                cue_activity_signature = (
                    f"{activity_signature}|{cadence_label}:"
                    f"{int(getattr(self, '_neko_companion_ambient_count', 0) or 0)}"
                )
            cue, signature, event_kind = build_neko_companion_cue(
                snapshot,
                include_rare_joke=rare_joke_due,
                ambient_event=ambient_event,
                ambient_signature=cue_activity_signature,
                absurd_banter_enabled=absurd_banter_enabled,
                absurd_variant=absurd_variant,
            )
        if not cue or not signature:
            return False
        if signature == str(getattr(self, "_neko_companion_last_signature", "") or ""):
            return False
        if now < float(getattr(self, "_neko_companion_next_transport_probe_at", 0.0) or 0.0):
            return False
        delivery_id = str(uuid.uuid4())
        priority = (
            2
            if event_kind in {"ambient_observation", "idle_heartbeat", "casual_chat"}
            else 4
        )
        delivery_metadata = {
            "context_type": "mahjong_companion",
            "event_kind": event_kind,
            "non_prescriptive": True,
            "reply_max_sentences": 2,
            "dialogue_policy_owner": "plugin",
            "delivery_id": delivery_id,
        }
        absurd_direct_reply = (
            build_absurd_banter_reply(
                snapshot,
                event_kind=event_kind,
                absurd_variant=absurd_variant,
                opponent_meld_event=opponent_meld_event,
            )
            if absurd_banter_enabled and not rare_joke_due
            else ""
        )
        if absurd_direct_reply:
            direct_started = time.perf_counter()
            direct_result = await self._submit_companion_direct_fallback(
                absurd_direct_reply,
                delivery_id=delivery_id,
                event_kind=event_kind,
                priority=priority,
                metadata=delivery_metadata,
                direct_reply=True,
            )
            if not direct_result.submitted:
                self._neko_companion_delivery_stage = "direct_reply_unavailable"
                self._neko_companion_last_error = (
                    "逆天模式固定话术未送达"
                    f"（{direct_result.reason or 'unknown'}）"
                )
                self._neko_companion_failure_count = int(
                    getattr(self, "_neko_companion_failure_count", 0) or 0
                ) + 1
                return False
            self._neko_companion_last_submit_ms = _elapsed_ms(direct_started)
            self._neko_companion_delivery_stage = "direct_reply_submitted"
            self._neko_companion_delivery_id = delivery_id
            self._neko_companion_last_error = ""
            self._neko_companion_last_signature = signature
            self._neko_companion_last_push_at = now
            self._neko_companion_last_event_kind = event_kind
            self._neko_companion_push_count = int(
                getattr(self, "_neko_companion_push_count", 0) or 0
            ) + 1
            self._finalize_neko_companion_cadence(
                event_kind=event_kind,
                activity_signature=activity_signature,
                rare_joke_due=rare_joke_due,
                events_since_rare=events_since_rare,
                now=now,
            )
            return True
        transport_ready = await self._probe_companion_message_plane(now=now)
        if transport_ready is False:
            direct_started = time.perf_counter()
            direct_result = await self._submit_companion_direct_fallback(
                cue,
                delivery_id=delivery_id,
                event_kind=event_kind,
                priority=priority,
                metadata=delivery_metadata,
            )
            if direct_result.submitted:
                self._neko_companion_last_submit_ms = _elapsed_ms(direct_started)
                self._neko_companion_delivery_stage = "direct_host_submitted"
                self._neko_companion_delivery_id = delivery_id
                self._neko_companion_last_error = "消息平面不可用，已通过兼容通道直接提交"
                self._neko_companion_next_transport_probe_at = 0.0
                self._neko_companion_last_signature = signature
                self._neko_companion_last_push_at = now
                self._neko_companion_last_event_kind = event_kind
                self._neko_companion_push_count = int(
                    getattr(self, "_neko_companion_push_count", 0) or 0
                ) + 1
                self._finalize_neko_companion_cadence(
                    event_kind=event_kind,
                    activity_signature=activity_signature,
                    rare_joke_due=rare_joke_due,
                    events_since_rare=events_since_rare,
                    now=now,
                )
                return True
            self._neko_companion_delivery_stage = "message_plane_unavailable"
            self._neko_companion_last_error = (
                f"消息平面未启动，兼容通道也不可用（{direct_result.reason or 'unknown'}）"
            )
            self._neko_companion_next_transport_probe_at = now + _NEKO_TRANSPORT_RETRY_SECONDS
            self._neko_companion_failure_count = int(
                getattr(self, "_neko_companion_failure_count", 0) or 0
            ) + 1
            return False
        submit_started = time.perf_counter()
        try:
            result = self.push_message(
                source=self.plugin_id if hasattr(self, "ctx") else "mahjong_coach",
                visibility=[],
                ai_behavior="respond",
                parts=[{"type": "text", "text": cue}],
                priority=priority,
                coalesce_key=(
                    f"{self.plugin_id}:mahjong_companion"
                    if hasattr(self, "ctx")
                    else "mahjong_coach:mahjong_companion"
                ),
                metadata=delivery_metadata,
            )
            if asyncio.iscoroutine(result):
                # Await submission so critical-event latency is measurable and
                # repeated ambient cues cannot leave untracked background tasks.
                await asyncio.wait_for(result, timeout=0.75)
        except Exception as exc:
            logger = getattr(self, "logger", None)
            if logger is not None:
                logger.warning(
                    "mahjong companion cue push failed event={} error_type={}",
                    event_kind,
                    type(exc).__name__,
                )
            return False
        if isinstance(result, dict) and result.get("submitted") is False:
            direct_result = await self._submit_companion_direct_fallback(
                cue,
                delivery_id=delivery_id,
                event_kind=event_kind,
                priority=priority,
                metadata=delivery_metadata,
            )
            if not direct_result.submitted:
                self._neko_companion_delivery_stage = "message_plane_unavailable"
                self._neko_companion_last_error = (
                    "SDK 拒绝提交，兼容通道也不可用"
                    f"（{direct_result.reason or result.get('reason') or 'unknown'}）"
                )
                self._neko_companion_next_transport_probe_at = now + _NEKO_TRANSPORT_RETRY_SECONDS
                self._neko_companion_failure_count = int(
                    getattr(self, "_neko_companion_failure_count", 0) or 0
                ) + 1
                return False
            self._neko_companion_last_submit_ms = _elapsed_ms(submit_started)
            self._neko_companion_delivery_stage = "direct_host_submitted"
            self._neko_companion_delivery_id = delivery_id
            self._neko_companion_last_error = "SDK 通道拒绝提交，已改走兼容通道"
            self._neko_companion_next_transport_probe_at = 0.0
            self._neko_companion_last_signature = signature
            self._neko_companion_last_push_at = now
            self._neko_companion_last_event_kind = event_kind
            self._neko_companion_push_count = int(
                getattr(self, "_neko_companion_push_count", 0) or 0
            ) + 1
            self._finalize_neko_companion_cadence(
                event_kind=event_kind,
                activity_signature=activity_signature,
                rare_joke_due=rare_joke_due,
                events_since_rare=events_since_rare,
                now=now,
            )
            return True
        self._neko_companion_last_submit_ms = _elapsed_ms(submit_started)
        self._neko_companion_delivery_stage = "submitted"
        self._neko_companion_delivery_id = delivery_id
        self._neko_companion_last_error = ""
        self._neko_companion_next_transport_probe_at = 0.0
        self._neko_companion_last_signature = signature
        self._neko_companion_last_push_at = now
        self._neko_companion_last_event_kind = event_kind
        self._neko_companion_push_count = int(
            getattr(self, "_neko_companion_push_count", 0) or 0
        ) + 1
        ingested = await self._confirm_companion_message_ingest(
            delivery_id,
            submitted_at=now,
        )
        if ingested is True:
            self._neko_companion_delivery_stage = "message_plane_received"
            self._neko_companion_message_plane_received_count = int(
                getattr(self, "_neko_companion_message_plane_received_count", 0) or 0
            ) + 1
            receipt_task = asyncio.create_task(
                self._watch_companion_host_receipt(delivery_id, submitted_at=now)
            )
            self._neko_companion_receipt_task = receipt_task
        elif ingested is False and transport_ready is not None:
            self._neko_companion_delivery_stage = "submitted_unconfirmed"
            self._neko_companion_last_error = "SDK 已提交，但消息平面尚未确认接收"
        self._finalize_neko_companion_cadence(
            event_kind=event_kind,
            activity_signature=activity_signature,
            rare_joke_due=rare_joke_due,
            events_since_rare=events_since_rare,
            now=now,
        )
        return True

    def _finalize_neko_companion_cadence(
        self,
        *,
        event_kind: str,
        activity_signature: str,
        rare_joke_due: bool,
        events_since_rare: int,
        now: float,
    ) -> None:
        if event_kind in {"ambient_observation", "idle_heartbeat", "casual_chat"}:
            self._neko_companion_ambient_count = int(
                getattr(self, "_neko_companion_ambient_count", 0) or 0
            ) + 1
            self._neko_companion_pending_activity_signature = ""
            if event_kind in {"idle_heartbeat", "casual_chat"}:
                self._neko_companion_last_idle_signature = activity_signature
        elif event_kind == "opponent_meld":
            self._neko_companion_pending_opponent_meld_event = {}
        elif (
            activity_signature
            and activity_signature
            == str(getattr(self, "_neko_companion_pending_activity_signature", "") or "")
        ):
            self._neko_companion_pending_activity_signature = ""
        if rare_joke_due:
            self._neko_companion_events_since_rare_joke = 0
            self._neko_companion_last_rare_joke_at = now
        else:
            self._neko_companion_events_since_rare_joke = events_since_rare + 1

    def _reset_neko_companion_ambient_state(self) -> None:
        self._neko_companion_round_id = ""
        self._neko_companion_activity_signature = ""
        self._neko_companion_activity_changed_at = 0.0
        self._neko_companion_pending_activity_signature = ""
        self._neko_companion_last_idle_signature = ""
        self._neko_companion_ambient_count = 0
        self._neko_companion_opponent_melds_initialized = False
        self._neko_companion_seen_opponent_meld_counts: dict[str, dict[str, int]] = {}
        self._neko_companion_pending_opponent_meld_event: dict[str, Any] = {}

    def _observe_neko_companion_activity(self, snapshot: dict[str, Any], *, now: float) -> str:
        state = snapshot.get("round_state") if isinstance(snapshot.get("round_state"), dict) else {}
        round_id = str(state.get("round_id") or "")
        tracked_round = str(getattr(self, "_neko_companion_round_id", "") or "")
        if round_id != tracked_round:
            self._reset_neko_companion_ambient_state()
            self._neko_companion_round_id = round_id
        if not bool(state.get("opening_emitted")):
            return ""
        if str(state.get("settlement_phase") or "playing") != "playing":
            return ""
        signature = _neko_companion_activity_signature(state)
        previous = str(getattr(self, "_neko_companion_activity_signature", "") or "")
        if signature and signature != previous:
            self._neko_companion_activity_signature = signature
            self._neko_companion_activity_changed_at = now
            self._neko_companion_pending_activity_signature = signature
            self._neko_companion_last_idle_signature = ""
        return signature

    def _observe_neko_companion_opponent_meld(
        self,
        snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        """Return only newly confirmed opponent calls within the current round.

        Opponent meld recognition is already stabilized by the coach engine.
        This tracker is monotonic within a round so a temporary visual dropout
        cannot make the same pon/chi/kan look new a second time.
        """

        state = snapshot.get("round_state") if isinstance(snapshot.get("round_state"), dict) else {}
        current_melds = _normalized_opponent_melds(state)
        current_counts = _opponent_meld_counts(current_melds)
        pending_event = getattr(self, "_neko_companion_pending_opponent_meld_event", {})
        pending_event = dict(pending_event) if isinstance(pending_event, dict) else {}
        if not bool(getattr(self, "_neko_companion_opponent_melds_initialized", False)):
            self._neko_companion_opponent_melds_initialized = True
            self._neko_companion_seen_opponent_meld_counts = current_counts
            return {}

        seen_counts = getattr(self, "_neko_companion_seen_opponent_meld_counts", {})
        seen_counts = seen_counts if isinstance(seen_counts, dict) else {}
        new_melds: list[dict[str, Any]] = []
        for player in ("left_opponent", "top_opponent", "right_opponent"):
            observed_occurrences: Counter[str] = Counter()
            player_seen = seen_counts.get(player)
            player_seen = player_seen if isinstance(player_seen, dict) else {}
            for meld in current_melds.get(player, []):
                identity = _opponent_meld_identity(meld)
                observed_occurrences[identity] += 1
                if observed_occurrences[identity] > int(player_seen.get(identity) or 0):
                    new_melds.append({"owner": player, **meld})

        merged_counts: dict[str, dict[str, int]] = {}
        for player in ("left_opponent", "top_opponent", "right_opponent"):
            previous = seen_counts.get(player)
            previous = previous if isinstance(previous, dict) else {}
            observed = current_counts.get(player, {})
            merged_counts[player] = {
                identity: max(int(previous.get(identity) or 0), int(observed.get(identity) or 0))
                for identity in set(previous) | set(observed)
            }
        self._neko_companion_seen_opponent_meld_counts = merged_counts
        if new_melds:
            pending_melds = [
                item
                for item in (pending_event.get("new_melds") or [])
                if isinstance(item, dict)
            ]
            known = {
                f"{str(item.get('owner') or '')}:{_opponent_meld_identity(item)}"
                for item in pending_melds
            }
            for meld in new_melds:
                identity = f"{str(meld.get('owner') or '')}:{_opponent_meld_identity(meld)}"
                if identity not in known:
                    pending_melds.append(meld)
                    known.add(identity)
            pending_event = {
                "new_melds": pending_melds,
                "current_melds": current_melds,
            }
            self._neko_companion_pending_opponent_meld_event = pending_event
        elif pending_event and any(current_melds.values()):
            pending_event["current_melds"] = current_melds
            self._neko_companion_pending_opponent_meld_event = pending_event
        return pending_event

    def _select_neko_companion_ambient_event(
        self,
        snapshot: dict[str, Any],
        *,
        activity_signature: str,
        now: float,
    ) -> str:
        presentation = (
            snapshot.get("presentation")
            if isinstance(snapshot.get("presentation"), dict)
            else {}
        )
        state = snapshot.get("round_state") if isinstance(snapshot.get("round_state"), dict) else {}
        if str(presentation.get("source_decision_type") or "") != "observe":
            return ""
        if not activity_signature or not bool(state.get("opening_emitted")):
            return ""
        if str(state.get("settlement_phase") or "playing") != "playing":
            return ""
        if int(getattr(self, "_neko_companion_ambient_count", 0) or 0) >= _NEKO_AMBIENT_MAX_PER_ROUND:
            return ""

        last_push_at = float(getattr(self, "_neko_companion_last_push_at", 0.0) or 0.0)
        changed_at = float(getattr(self, "_neko_companion_activity_changed_at", 0.0) or 0.0)
        pending = str(getattr(self, "_neko_companion_pending_activity_signature", "") or "")
        since_push = now - last_push_at if last_push_at > 0 else float("inf")
        since_change = now - changed_at if changed_at > 0 else float("inf")
        riichi_players = [str(player) for player in (state.get("riichi_players") or []) if str(player)]
        if not riichi_players:
            quiet_elapsed = since_push if last_push_at > 0 else since_change
            if (
                quiet_elapsed >= _NEKO_IDLE_HEARTBEAT_SECONDS
                and since_change >= _NEKO_AMBIENT_SETTLE_SECONDS
            ):
                return "casual_chat"
            return ""
        if (
            pending == activity_signature
            and since_push >= _NEKO_AMBIENT_MIN_INTERVAL_SECONDS
            and since_change >= _NEKO_AMBIENT_SETTLE_SECONDS
        ):
            return "ambient_observation"

        if (
            not pending
            and since_push >= _NEKO_IDLE_HEARTBEAT_SECONDS
            and since_change >= _NEKO_IDLE_HEARTBEAT_SECONDS
        ):
            return "idle_heartbeat"
        return ""

    def _update_overlay(self, payload: dict[str, Any]) -> dict[str, Any]:
        snapshot = self._publish_display_snapshot(payload)
        if not self._live_state.overlay_enabled:
            return snapshot
        self._overlay.update_payload(
            text=str(snapshot.get("overlay_text") or ""),
            strategy_card_text=str(snapshot.get("strategy_card_text") or snapshot.get("overlay_text") or ""),
            strategy_card=snapshot.get("strategy_card") or {},
            detail=str(snapshot.get("overlay_detail") or ""),
            image_path=str(snapshot.get("image_path") or ""),
            image_revision=int(snapshot.get("image_revision") or 0),
            live_state=snapshot.get("live") or self._live_state.to_dict(),
            companion_status=self._neko_companion_status_payload(),
        )
        return snapshot

    def _make_display_snapshot(self, payload: dict[str, Any], *, revision: int) -> dict[str, Any]:
        """Build the single published view consumed by both dashboard and overlay."""
        overlay_payload = build_public_payload(payload, mode=self._cfg.live_advice_mode)
        overlay_payload.setdefault("live", self._live_state.to_dict())
        prefs_path = getattr(getattr(self, "_overlay", None), "prefs_path", None)
        compact_text = overlay_text_from_payload(overlay_payload, prefs_path=prefs_path)
        strategy_card = overlay_strategy_card_from_payload(overlay_payload)
        return {
            "revision": int(revision),
            "published_at": time.time(),
            "last_decision": overlay_payload.get("last_decision") or {},
            "round_state": overlay_payload.get("round_state") or {},
            "presentation": overlay_payload.get("presentation") or {},
            "live": overlay_payload.get("live") or {},
            "overlay_text": compact_text,
            "strategy_card_text": overlay_strategy_card_text_from_payload(
                overlay_payload,
                prefs_path=prefs_path,
                compact_text=compact_text,
                strategy_card=strategy_card,
            ),
            "strategy_card": strategy_card,
            "overlay_detail": overlay_detail_text_from_payload(overlay_payload),
            "image_path": str(self._live_state.last_frame_path or ""),
            "image_revision": int(self._live_state.last_frame_revision or 0),
        }

    def _publish_display_snapshot(self, payload: dict[str, Any]) -> dict[str, Any]:
        revision = int(getattr(self, "_display_revision", 0) or 0) + 1
        self._display_revision = revision
        snapshot = self._make_display_snapshot(payload, revision=revision)
        self._display_snapshot = snapshot
        return snapshot


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 1)


def _remaining_cooldown_seconds(last_at: float, interval: float) -> float:
    if last_at <= 0:
        return 0.0
    return round(max(0.0, float(interval) - (time.time() - float(last_at))), 1)


def _process_memory_snapshot() -> dict[str, float | None]:
    if sys.platform.startswith("win"):
        try:
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCountersEx(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCountersEx()
            counters.cb = ctypes.sizeof(counters)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(ProcessMemoryCountersEx),
                wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            handle = kernel32.GetCurrentProcess()
            if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                raise ctypes.WinError(ctypes.get_last_error())
            scale = 1024.0 * 1024.0
            return {
                "rss_mb": round(counters.WorkingSetSize / scale, 2),
                "peak_rss_mb": round(counters.PeakWorkingSetSize / scale, 2),
                "private_memory_mb": round(counters.PrivateUsage / scale, 2),
            }
        except Exception:
            return {"rss_mb": None, "peak_rss_mb": None, "private_memory_mb": None}
    try:
        import resource

        peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform != "darwin":
            peak /= 1024.0
        else:
            peak /= 1024.0 * 1024.0
        return {
            "rss_mb": round(peak, 2),
            "peak_rss_mb": round(peak, 2),
            "private_memory_mb": None,
        }
    except Exception:
        return {"rss_mb": None, "peak_rss_mb": None, "private_memory_mb": None}


def _normalized_opponent_melds(state: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    raw_melds = state.get("last_opponent_melds")
    raw_melds = raw_melds if isinstance(raw_melds, dict) else {}
    normalized: dict[str, list[dict[str, Any]]] = {}
    for player in ("left_opponent", "top_opponent", "right_opponent"):
        groups: list[dict[str, Any]] = []
        for raw_group in raw_melds.get(player) or []:
            if not isinstance(raw_group, dict):
                continue
            tiles = [
                tile
                for value in (raw_group.get("tiles") or [])
                if (tile := normalize_tile(value))
            ]
            if not tiles:
                continue
            groups.append(
                {
                    "kind": str(raw_group.get("kind") or "unknown").lower(),
                    "tiles": tiles,
                }
            )
        normalized[player] = groups
    return normalized


def _opponent_meld_identity(meld: dict[str, Any]) -> str:
    return json.dumps(
        {
            "kind": str(meld.get("kind") or "unknown").lower(),
            "tiles": sorted(
                tile
                for value in (meld.get("tiles") or [])
                if (tile := normalize_tile(value))
            ),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _opponent_meld_counts(
    melds: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, int]]:
    return {
        player: dict(Counter(_opponent_meld_identity(meld) for meld in groups))
        for player, groups in melds.items()
    }


def _neko_companion_activity_signature(state: dict[str, Any]) -> str:
    """Summarize stable table facts without retaining frame geometry or confidence noise."""

    discard_piles = state.get("last_discard_piles")
    discard_summary: dict[str, list[str]] = {}
    if isinstance(discard_piles, dict):
        for player, items in discard_piles.items():
            if not isinstance(items, list):
                continue
            tiles: list[str] = []
            for item in items:
                raw_tile = item.get("tile") if isinstance(item, dict) else item
                tiles.append(normalize_tile(raw_tile) or "?")
            discard_summary[str(player)] = tiles

    opponent_melds = state.get("last_opponent_melds")
    opponent_meld_summary: dict[str, list[dict[str, Any]]] = {}
    if isinstance(opponent_melds, dict):
        for player, melds in opponent_melds.items():
            if not isinstance(melds, list):
                continue
            groups: list[dict[str, Any]] = []
            for meld in melds:
                raw_tiles = meld.get("tiles") if isinstance(meld, dict) else []
                groups.append(
                    {
                        "kind": str(meld.get("kind") or "unknown")
                        if isinstance(meld, dict)
                        else "unknown",
                        "tiles": sorted(
                            tile
                            for value in (raw_tiles or [])
                            if (tile := normalize_tile(value))
                        ),
                    }
                )
            opponent_meld_summary[str(player)] = groups

    payload = {
        "round": str(state.get("round_id") or ""),
        "hand": sorted(
            tile
            for value in (state.get("last_hand_tiles") or [])
            if (tile := normalize_tile(value))
        ),
        "self_melds": sorted(
            tile
            for value in (state.get("last_meld_tiles") or [])
            if (tile := normalize_tile(value))
        ),
        "opponent_melds": opponent_meld_summary,
        "discards": discard_summary,
        "riichi": sorted(str(value) for value in (state.get("riichi_players") or []) if str(value)),
        "scores": state.get("player_scores") if isinstance(state.get("player_scores"), dict) else {},
        "ranks": state.get("player_ranks") if isinstance(state.get("player_ranks"), dict) else {},
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _normalized_tiles(tiles: Any) -> list[str]:
    return [normalized for tile in (tiles or []) if (normalized := normalize_tile(tile))]


def _legacy_style_for_profile(profile: PlayerProfile) -> str:
    if (
        profile.risk_tolerance == "aggressive"
        or profile.goal_bias == "speed"
        or profile.call_bias == "open"
    ):
        return "fast"
    return "riichi"


def _shared_live_hand_tiles(left: Any, right: Any) -> int:
    left_counts = Counter(_normalized_tiles(left))
    right_counts = Counter(_normalized_tiles(right))
    return sum((left_counts & right_counts).values())


def _build_frame_preview_payload(image_path: Path) -> dict[str, Any]:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        image.thumbnail((960, 540), Image.Resampling.LANCZOS)
        output = BytesIO()
        image.save(output, format="JPEG", quality=74, optimize=True)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return {
        "image_path": str(image_path),
        "data_url": f"data:image/jpeg;base64,{encoded}",
        "width": image.width,
        "height": image.height,
    }


def _resolve_preview_frame_path(raw_path: str, frames_dir: Path) -> Path | None:
    """Resolve one explicit live frame without allowing paths outside plugin data."""

    try:
        frame_path = Path(str(raw_path or "").strip()).resolve()
        allowed_dir = frames_dir.resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    if not frame_path.is_relative_to(allowed_dir) or not frame_path.is_file():
        return None
    return frame_path


def _build_table_region_preview_payload(
    image_path: Path,
    *,
    raw_detections: list[dict[str, Any]] | None = None,
    opponent_melds: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Fallback reconstruction for manually supplied frames without live evidence."""

    with Image.open(image_path) as source:
        full_image = source.convert("RGB")
    try:
        table_surface = detect_table_surface(full_image)
    finally:
        full_image.close()
    if not table_surface.ok or table_surface.warped_image is None:
        return {
            "image_path": str(image_path),
            "data_url": "",
            "transformed": False,
            "reason": table_surface.reason or "table_surface_unavailable",
            "width": 0,
            "height": 0,
            "detection_count": 0,
        }

    try:
        return _build_warped_table_region_preview_payload(
            table_surface.warped_image,
            image_path=image_path,
            raw_detections=raw_detections,
            opponent_melds=opponent_melds,
            reason=table_surface.reason,
            table_surface_method=table_surface.method,
            evidence_source="reconstructed_from_saved_frame",
        )
    finally:
        table_surface.warped_image.close()


def _build_warped_table_region_preview_payload(
    warped_image: Image.Image,
    *,
    image_path: Path,
    raw_detections: list[dict[str, Any]] | None = None,
    opponent_melds: dict[str, list[dict[str, Any]]] | None = None,
    reason: str = "",
    table_surface_method: str = "",
    evidence_source: str,
) -> dict[str, Any]:
    """Encode one exact warped-table image without locating or warping it again."""

    warped_detections = [
        dict(item)
        for item in raw_detections or []
        if isinstance(item, dict)
        and str(item.get("coordinate_space") or "warped_table") == "warped_table"
    ]
    preview = render_yolo26_region_diagnostic_image(
        warped_image,
        raw_detections=warped_detections,
        opponent_melds=opponent_melds,
    )
    try:
        output = BytesIO()
        preview.save(output, format="JPEG", quality=84, optimize=True)
        width, height = preview.size
    finally:
        preview.close()
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return {
        "image_path": str(image_path),
        "data_url": f"data:image/jpeg;base64,{encoded}",
        "transformed": True,
        "input_space": "warped_table",
        "evidence_source": evidence_source,
        "reason": reason,
        "width": width,
        "height": height,
        "detection_count": len(warped_detections),
        "opponent_meld_count": sum(len(items) for items in (opponent_melds or {}).values()),
        "table_surface_method": table_surface_method,
    }


def _build_settlement_diagnostic_preview_payload(
    image_path: Path,
    *,
    min_confidence: float = 0.72,
) -> dict[str, Any]:
    # 中文：诊断图只在内存中生成，不在 live_frames 旁边写入额外文件。
    # English: Build the diagnostic preview in memory without creating sidecar files.
    with Image.open(image_path) as source:
        full_image = source.convert("RGB")
    try:
        result = detect_settlement_image(full_image, min_confidence=min_confidence)
        if not result.detected:
            return {
                "image_path": str(image_path),
                "data_url": "",
                "width": 0,
                "height": 0,
                "phase": "none",
                **result.to_dict(),
            }
        return _build_settlement_diagnostic_preview_from_image(
            full_image,
            result=result,
            image_path=str(image_path),
            min_confidence=min_confidence,
        )
    finally:
        full_image.close()


def _build_settlement_diagnostic_preview_from_image(
    image: Image.Image,
    *,
    result: SettlementFrameResult,
    image_path: str,
    min_confidence: float = 0.72,
) -> dict[str, Any]:
    preview = image.copy()
    try:
        preview.thumbnail((960, 540), Image.Resampling.LANCZOS)
        annotated = render_settlement_diagnostic_image(
            preview,
            result=result,
            min_confidence=min_confidence,
        )
        try:
            output = BytesIO()
            annotated.save(output, format="JPEG", quality=82, optimize=True)
            width, height = annotated.size
        finally:
            annotated.close()
    finally:
        preview.close()
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return {
        "image_path": image_path,
        "data_url": f"data:image/jpeg;base64,{encoded}",
        "width": width,
        "height": height,
        **result.to_dict(),
    }


def _live_performance_summary(
    rows: list[dict[str, Any]],
    *,
    dropped_frames: int = 0,
) -> dict[str, Any]:
    advice_types = {
        "opening_plan",
        "coach_checkpoint",
        "call_window",
        "riichi_window",
        "defense_alert",
    }
    all_values = sorted(
        float(item["loop_ms"])
        for item in rows
        if isinstance(item, dict) and isinstance(item.get("loop_ms"), (int, float))
    )
    advice_values = sorted(
        float(item["loop_ms"])
        for item in rows
        if isinstance(item, dict)
        and str(item.get("decision") or "") in advice_types
        and isinstance(item.get("loop_ms"), (int, float))
    )
    return {
        "sample_count": len(all_values),
        "advice_sample_count": len(advice_values),
        "frame_p50_ms": _nearest_rank_percentile(all_values, 0.50),
        "frame_p95_ms": _nearest_rank_percentile(all_values, 0.95),
        "advice_p95_ms": _nearest_rank_percentile(advice_values, 0.95),
        "dropped_frames": max(0, int(dropped_frames)),
    }


def _nearest_rank_percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    index = max(0, min(len(values) - 1, math.ceil(len(values) * quantile) - 1))
    return round(float(values[index]), 1)


def _read_float(payload: Any, key: str) -> float | None:
    if not isinstance(payload, dict) or key not in payload:
        return None
    try:
        return round(float(payload.get(key) or 0.0), 1)
    except (TypeError, ValueError):
        return None


def _mahjong_timing_from_decision(decision: dict[str, Any]) -> dict[str, Any]:
    perception = decision.get("perception") if isinstance(decision.get("perception"), dict) else {}
    engine_meta = decision.get("engine_meta") if isinstance(decision.get("engine_meta"), dict) else {}
    engine_timings = engine_meta.get("timings_ms") if isinstance(engine_meta.get("timings_ms"), dict) else {}
    riichi_meta = engine_meta.get("riichi_detection") if isinstance(engine_meta.get("riichi_detection"), dict) else {}

    def _step_ms(name: str) -> float | None:
        payload = perception.get(name) if isinstance(perception.get(name), dict) else {}
        if "elapsed_ms" not in payload:
            return None
        try:
            return round(float(payload.get("elapsed_ms") or 0.0), 1)
        except (TypeError, ValueError):
            return None

    river = perception.get("river") if isinstance(perception.get("river"), dict) else {}
    river_hints = river.get("analysis_hints") if isinstance(river.get("analysis_hints"), dict) else {}
    settlement = perception.get("settlement") if isinstance(perception.get("settlement"), dict) else {}
    river_visible = river.get("visible_tiles") if isinstance(river.get("visible_tiles"), list) else []
    river_piles = river.get("discard_piles") if isinstance(river.get("discard_piles"), dict) else {}
    river_tile_count = len(river_visible) or sum(len(items) for items in river_piles.values() if isinstance(items, list))
    decision_type = str(decision.get("decision_type") or "")

    return {
        "decision": decision_type,
        "source": str(engine_meta.get("source") or ""),
        "tile_mode": str(engine_meta.get("tile_recognition_mode") or "yolo26"),
        "river_mode": str(engine_meta.get("river_tracking_mode") or "checkpoint"),
        "river_reason": str(river.get("reason") or ""),
        "fallback_reason": _first_fallback_reason(perception),
        "river_ok": bool(river.get("ok")),
        "river_tile_count": int(river_tile_count),
        "river_new_discard_count": int(river_hints.get("new_discard_count") or 0),
        "river_corrected_count": int(river_hints.get("river_corrected_count") or 0),
        "river_pending_corrections": int(river_hints.get("river_pending_corrections") or 0),
        "river_full_rescan": bool(river_hints.get("river_full_rescan")),
        "engine_total_ms": round(float(engine_meta.get("elapsed_ms") or engine_timings.get("total") or 0.0), 1),
        "hand_ms": _step_ms("hand"),
        "meld_ms": _step_ms("meld"),
        "action_ms": _step_ms("action"),
        "river_ms": _step_ms("river"),
        "settlement_ms": _step_ms("settlement"),
        "settlement_phase": str(settlement.get("phase") or engine_meta.get("settlement_phase") or ""),
        "settlement_kind": str(settlement.get("kind") or engine_meta.get("settlement_kind") or ""),
        "settlement_confidence": round(float(settlement.get("confidence") or 0.0), 4),
        "settlement_evidence": list(settlement.get("evidence") or []),
        "round_archive_id": str(settlement.get("round_archive_id") or ""),
        "strategy_ms": round(float(engine_timings.get("strategy") or 0.0), 1) if "strategy" in engine_timings else None,
        "riichi_event": decision_type in {"riichi_window", "defense_alert"} and bool(
            decision_type == "riichi_window" or riichi_meta.get("confirmed_players")
        ),
        "riichi_detection_ms": _read_float(engine_timings, "riichi_detection"),
        "riichi_detection_mode": str(riichi_meta.get("mode") or ""),
        "riichi_immediate_players": list(riichi_meta.get("immediate_players") or []),
        "riichi_pending_players": dict(riichi_meta.get("pending_players") or {}),
    }


def _first_fallback_reason(perception: dict[str, Any]) -> str:
    for name in ("hand", "meld", "river"):
        payload = perception.get(name) if isinstance(perception.get(name), dict) else {}
        hints = payload.get("analysis_hints") if isinstance(payload.get("analysis_hints"), dict) else {}
        reason = str(hints.get("fallback_reason") or "").strip()
        if reason:
            return reason
    return ""
