from __future__ import annotations

import asyncio
import base64
import copy
import contextlib
from collections import Counter
from dataclasses import replace
from io import BytesIO
import math
from pathlib import Path
import threading
import time
from typing import Any

from PIL import Image

from plugin.sdk.plugin import Err, NekoPluginBase, Ok, SdkError, lifecycle, neko_plugin, plugin_entry, tr

from .capture import CaptureSession, DefaultCaptureProvider, prune_frames
from .coach import RoundCoachEngine
from .models import (
    CapturePreferences,
    LiveSessionState,
    MahjongCoachConfig,
    PlayerProfile,
    WindowTargetDescriptor,
    _clean_string_list,
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
from .preferences import PreferencesStore
from .perception.settlement_detector import (
    detect_settlement_image,
    render_settlement_diagnostic_image,
)
from .perception.table_surface import detect_table_surface
from .perception.yolo26_visible_tiles import render_yolo26_region_diagnostic_image
from .tile_labels import normalize_tile
from .window_binding import choose_window_candidate_native, list_window_candidates
from .yakuman import YakumanEstimateService


@neko_plugin
class MahjongCoachPlugin(NekoPluginBase):
    def __init__(self, ctx):
        super().__init__(ctx)
        self.file_logger = self.enable_file_logging(log_level="INFO")
        self.logger = self.file_logger
        self._cfg = MahjongCoachConfig()
        self._engine: RoundCoachEngine | None = None
        self._engine_lock = asyncio.Lock()
        self._last_decision: dict[str, Any] = {}
        self._display_snapshot: dict[str, Any] = {}
        self._display_revision = 0
        self._live_state = LiveSessionState()
        self._live_task: asyncio.Task | None = None
        self._live_stop_event: asyncio.Event | None = None
        self._live_last_hand_signature = ""
        self._live_gap_hand_tiles: list[str] = []
        self._live_gap_candidate_tiles: list[str] = []
        self._live_gap_candidate_frames = 0
        self._live_last_checkpoint_at = 0.0
        self._live_timing_log: list[dict[str, Any]] = []
        self._live_last_preview_at = 0.0
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
        try:
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
            if self._preferences.auto_start_live and (
                self._preferences.target.title or self._preferences.target.app_name
            ):
                asyncio.create_task(self._overlay_start_live(overlay=True))
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

    def _on_overlay_start_sync(self, style: str, strategy_preset: str = "simple") -> None:
        self.logger.info(
            "_on_overlay_start_sync called style={} strategy_preset={}",
            style,
            strategy_preset,
        )
        self._schedule_on_loop(
            self._overlay_start_live(play_style=style, strategy_preset=strategy_preset),
            "overlay_start",
        )

    def _on_overlay_stop_sync(self) -> None:
        self._schedule_on_loop(self._overlay_stop_live(hide_overlay=True), "overlay_stop")

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
        await self._stop_live_task()
        # 中文：插件重载或后端退出时同步销毁 Tk 窗口，避免遗留失效线程。
        # English: Destroy the Tk window during reload or shutdown to avoid a stale UI thread.
        self._overlay.stop()
        yakuman_service = getattr(self, "_yakuman_service", None)
        if yakuman_service is not None:
            yakuman_service.close()
        self.clear_list_actions()
        return Ok({"status": "stopped"})

    def _get_engine_lock(self) -> asyncio.Lock:
        """Return the single mutex guarding the mutable round engine.

        A lazy fallback keeps direct ``__new__`` test fixtures and older host
        reloads compatible without creating a second lock in normal runtime.
        """
        lock = getattr(self, "_engine_lock", None)
        if lock is None:
            lock = asyncio.Lock()
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

    @plugin_entry(
        id="mahjong_coach_status",
        name=tr("entries.status.name", default="Mahjong Coach Status"),
        description=tr("entries.status.description", default="Inspect current Mahjong Coach round state."),
        input_schema={"type": "object", "properties": {}},
        llm_result_fields=["status", "current_plan", "last_update_reason"],
    )
    async def mahjong_coach_status(self, **_):
        if self._engine is None:
            return Err(SdkError("mahjong coach is not initialized"))
        async with self._get_engine_lock():
            state = self._engine.state.to_dict()
            round_history = self._engine.round_history
            last_decision = dict(self._last_decision)
            config = self._cfg.to_dict()
        payload = {"last_decision": last_decision, "round_state": state}
        display_snapshot = copy.deepcopy(getattr(self, "_display_snapshot", {}))
        if not display_snapshot:
            display_snapshot = self._make_display_snapshot(payload, revision=0)
        timing_log = list(getattr(self, "_live_timing_log", []))
        return Ok(
            {
                "status": "ready",
                "config": config,
                "round_state": state,
                "last_decision": last_decision,
                "display_snapshot": display_snapshot,
                "round_history": round_history,
                "last_round_archive": round_history[-1] if round_history else {},
                "overlay_text": str(display_snapshot.get("overlay_text") or ""),
                "live": self._live_state.to_dict(),
                "timing_log": timing_log,
                "performance": _live_performance_summary(
                    timing_log,
                    dropped_frames=int(getattr(self._live_state, "dropped_frames", 0) or 0),
                ),
                "preferences": getattr(self, "_preferences", CapturePreferences()).to_dict(),
                "ui_path": f"/plugin/{self.plugin_id}/ui/",
                **state,
            }
        )

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
        async with self._get_engine_lock():
            yolo_path = getattr(self._engine, "_last_yolo26_path", None)
            yolo_result = getattr(self._engine, "_last_yolo26_result", None)
            if isinstance(yolo_path, Path) and yolo_path.resolve() == frame_path and yolo_result is not None:
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
        raw_path = str(image_path or self._live_state.last_frame_path or "").strip()
        if not raw_path:
            return Ok({"status": "empty", "image_path": "", "data_url": ""})
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
        return Ok({"status": "ready", **preview})

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
            self._last_decision = {}
            self._display_snapshot = {}
            self._live_last_hand_signature = ""
            self._live_state.observed_hand_changes = 0
            self._clear_live_hand_gap()
            self._live_last_checkpoint_at = 0.0
            getattr(self, "_live_timing_log", []).clear()
            history = self._engine.round_history
            state_payload = state.to_dict()
            state_round_id = state.round_id
        return Ok(
            {
                "status": "reset",
                "round_state": state_payload,
                "round_id": state_round_id,
                "round_history": history,
                "last_round_archive": history[-1] if history else {},
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
                "river_tracking_mode": {"type": "string", "default": ""},
                "tile_recognition_mode": {"type": "string", "default": ""},
                "settlement_recognition_enabled": {"type": "boolean"},
                "settlement_min_confidence": {"type": "number"},
                "settlement_confirm_frames": {"type": "integer"},
                "settlement_confirm_max_gap_ms": {"type": "integer"},
            },
        },
        timeout=20.0,
        llm_result_fields=["decision_type", "summary", "suggestion"],
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
        river_tracking_mode: str = "",
        tile_recognition_mode: str = "",
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
                self._apply_runtime_river_tracking_mode(river_tracking_mode)
                self._apply_runtime_tile_recognition_mode(tile_recognition_mode)
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
                decision = self._enrich_yakuman_decision(decision)
                self._last_decision = decision.to_dict()
                decision_payload = dict(self._last_decision)
                engine_state = self._engine.state.to_dict()
        except Exception as exc:
            self.logger.warning("mahjong coach frame analysis failed: {}", exc)
            return Err(SdkError(str(exc)))
        if not getattr(decision, "quiet", False):
            self._update_overlay({"last_decision": decision_payload, "round_state": engine_state})
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
                "river_tracking_mode": {"type": "string", "default": ""},
                "tile_recognition_mode": {"type": "string", "default": ""},
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
        river_tracking_mode: str = "",
        tile_recognition_mode: str = "",
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
            river_tracking_mode=river_tracking_mode,
            tile_recognition_mode=tile_recognition_mode,
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
        river_tracking_mode: str = "",
        tile_recognition_mode: str = "",
        target_window_title: str = "",
        auto_start_live: bool | None = None,
        settlement_recognition_enabled: bool | None = None,
        settlement_min_confidence: float | None = None,
        settlement_confirm_frames: int | None = None,
        settlement_confirm_max_gap_ms: int | None = None,
    ):
        self.logger.info("_overlay_start_live called play_style={} strategy_preset={} river_tracking_mode={} tile_recognition_mode={}", play_style, strategy_preset, river_tracking_mode, tile_recognition_mode)
        if self._engine is None:
            self.logger.warning("_overlay_start_live early return: engine is None")
            return Err(SdkError("mahjong coach is not initialized"))
        async with self._get_engine_lock():
            style_before = self._cfg.play_style
            strategy_preset_before = self._cfg.strategy_preset
            river_mode_before = self._cfg.river_tracking_mode
            tile_mode_before = self._cfg.tile_recognition_mode
            self._apply_runtime_round_context(
                round_wind=round_wind,
                seat_wind=seat_wind,
                dora_tiles=dora_tiles,
                play_style=play_style,
            )
            self._apply_runtime_strategy_preset(strategy_preset)
            self._apply_runtime_river_tracking_mode(river_tracking_mode)
            self._apply_runtime_tile_recognition_mode(tile_recognition_mode)
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
            if tile_recognition_mode and self._cfg.tile_recognition_mode != tile_mode_before:
                self._invalidate_live_plan_for_style_change()
            already_running = self._live_task is not None and not self._live_task.done()
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
        if not self._show_overlay(strategy=running):
            return Err(SdkError("failed to show mahjong coach overlay"))
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
                    if packet.image is not None and self._should_persist_preview(decision):
                        preview_path = frames_dir / "last-preview.jpg"
                        await asyncio.to_thread(provider.persist_packet, packet, preview_path)
                        self._live_state.last_frame_path = str(preview_path)
                        self._live_last_preview_at = time.monotonic()
                    if not getattr(decision, "quiet", False):
                        payload = {"last_decision": dict(self._last_decision), "round_state": engine_state}
                        self._update_overlay(payload)
                    self._log_live_timing(
                        decision=dict(self._last_decision),
                        locate_ms=locate_ms,
                        capture_ms=capture_ms,
                        analyze_ms=analyze_ms,
                        loop_ms=_elapsed_ms(loop_started),
                    )
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
        if bool(getattr(decision, "action_required", False)) or not bool(getattr(decision, "quiet", False)):
            return True
        return (time.monotonic() - self._live_last_preview_at) >= max(
            4.0,
            float(self._cfg.live_checkpoint_interval_seconds),
        )

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
        entry["timestamp_ms"] = int(time.time() * 1000)
        self._live_timing_log.append(entry)
        del self._live_timing_log[:-80]
        return entry

    async def _stop_live_task(self) -> None:
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
        self._live_state.running = False
        self._live_state.status = "stopped"
        self._live_state.updated_at = time.time()

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
        self._live_last_hand_signature = str(self._engine.state.last_hand_signature or "")
        self._live_last_checkpoint_at = 0.0
        self._live_state.observed_hand_changes = 0
        self._clear_live_hand_gap()
        self.logger.info(
            "mahjong coach auto new round round_id={} river_before={} river_now={}",
            self._engine.state.round_id,
            getattr(decision, "engine_meta", {}).get("previous_river_count"),
            getattr(decision, "engine_meta", {}).get("current_river_count"),
        )
        return True

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
                if str(getattr(self._engine.config, "tile_recognition_mode", "legacy")) == "yolo26":
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

    def _update_overlay(self, payload: dict[str, Any]) -> None:
        snapshot = self._publish_display_snapshot(payload)
        if not self._live_state.overlay_enabled:
            return
        self._overlay.update_payload(
            text=str(snapshot.get("overlay_text") or ""),
            strategy_card_text=str(snapshot.get("strategy_card_text") or snapshot.get("overlay_text") or ""),
            strategy_card=copy.deepcopy(snapshot.get("strategy_card") or {}),
            detail=str(snapshot.get("overlay_detail") or ""),
            image_path=str(snapshot.get("image_path") or ""),
        )

    def _make_display_snapshot(self, payload: dict[str, Any], *, revision: int) -> dict[str, Any]:
        """Build the single published view consumed by both dashboard and overlay."""
        overlay_payload = copy.deepcopy(dict(payload))
        overlay_payload.setdefault("live", self._live_state.to_dict())
        prefs_path = getattr(getattr(self, "_overlay", None), "prefs_path", None)
        return {
            "revision": int(revision),
            "published_at": time.time(),
            "last_decision": copy.deepcopy(overlay_payload.get("last_decision") or {}),
            "round_state": copy.deepcopy(overlay_payload.get("round_state") or {}),
            "live": copy.deepcopy(overlay_payload.get("live") or {}),
            "overlay_text": overlay_text_from_payload(overlay_payload, prefs_path=prefs_path),
            "strategy_card_text": overlay_strategy_card_text_from_payload(overlay_payload, prefs_path=prefs_path),
            "strategy_card": overlay_strategy_card_from_payload(overlay_payload),
            "overlay_detail": overlay_detail_text_from_payload(overlay_payload),
            "image_path": str(self._live_state.last_frame_path or ""),
        }

    def _publish_display_snapshot(self, payload: dict[str, Any]) -> dict[str, Any]:
        revision = int(getattr(self, "_display_revision", 0) or 0) + 1
        self._display_revision = revision
        snapshot = self._make_display_snapshot(payload, revision=revision)
        self._display_snapshot = snapshot
        return snapshot


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 1)


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
    # 中文：面板分区图严格来自透视变换结果；变换失败时不拿原图冒充。
    # English: The region panel must use the perspective-corrected table and never masquerade the raw frame as warped.
    with Image.open(image_path) as source:
        full_image = source.convert("RGB")
        table_surface = detect_table_surface(full_image)
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

    warped_detections = [
        dict(item)
        for item in raw_detections or []
        if isinstance(item, dict)
        and str(item.get("coordinate_space") or "warped_table") == "warped_table"
    ]
    preview = render_yolo26_region_diagnostic_image(
        table_surface.warped_image,
        raw_detections=warped_detections,
        opponent_melds=opponent_melds,
    )
    output = BytesIO()
    preview.save(output, format="JPEG", quality=84, optimize=True)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return {
        "image_path": str(image_path),
        "data_url": f"data:image/jpeg;base64,{encoded}",
        "transformed": True,
        "input_space": "warped_table",
        "reason": table_surface.reason,
        "width": preview.width,
        "height": preview.height,
        "detection_count": len(warped_detections),
        "opponent_meld_count": sum(len(items) for items in (opponent_melds or {}).values()),
        "table_surface_method": table_surface.method,
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
        result = detect_settlement_image(full_image, min_confidence=min_confidence)
        preview = full_image.copy()
        preview.thumbnail((960, 540), Image.Resampling.LANCZOS)
        preview = render_settlement_diagnostic_image(
            preview,
            result=result,
            min_confidence=min_confidence,
        )
        output = BytesIO()
        preview.save(output, format="JPEG", quality=82, optimize=True)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return {
        "image_path": str(image_path),
        "data_url": f"data:image/jpeg;base64,{encoded}",
        "width": preview.width,
        "height": preview.height,
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

    return {
        "decision": str(decision.get("decision_type") or ""),
        "source": str(engine_meta.get("source") or ""),
        "tile_mode": str(engine_meta.get("tile_recognition_mode") or "legacy"),
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
    }


def _first_fallback_reason(perception: dict[str, Any]) -> str:
    for name in ("hand", "meld", "river"):
        payload = perception.get(name) if isinstance(perception.get(name), dict) else {}
        hints = payload.get("analysis_hints") if isinstance(payload.get("analysis_hints"), dict) else {}
        reason = str(hints.get("fallback_reason") or "").strip()
        if reason:
            return reason
    return ""
