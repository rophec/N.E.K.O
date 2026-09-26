from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from plugin.plugins.mahjong_coach.overlay import (
    CoachOverlayController,
    _load_prefs,
    _save_prefs,
    overlay_text_from_payload,
)


def test_overlay_preferences_use_injected_plugin_data_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unrelated_local_app_data = tmp_path / "unrelated-local-app-data"
    monkeypatch.setenv("LOCALAPPDATA", str(unrelated_local_app_data))
    prefs_path = tmp_path / "custom-plugin-root" / "mahjong_coach" / "overlay_prefs.json"
    controller = CoachOverlayController(prefs_path=prefs_path)

    _save_prefs(controller.prefs_path, 520, 180, 22, "beginner")
    stored = json.loads(prefs_path.read_text(encoding="utf-8"))

    assert controller.prefs_path == prefs_path.resolve()
    assert stored["display_mode"] == "beginner"
    assert _load_prefs(controller.prefs_path)["font_size"] == 22
    assert not unrelated_local_app_data.exists()


def test_overlay_text_without_injected_path_has_no_filesystem_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    text = overlay_text_from_payload(
        {
            "decision_type": "waiting_for_game",
            "summary": "waiting",
            "round_state": {},
        }
    )

    assert isinstance(text, str)
    assert not (tmp_path / "overlay_prefs.json").exists()


def test_overlay_start_waits_for_real_readiness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = CoachOverlayController(prefs_path=tmp_path / "overlay.json")
    release = threading.Event()

    def fake_run() -> None:
        controller._ready_event.set()
        release.wait(timeout=1.0)

    monkeypatch.setattr(controller, "_run", fake_run)

    assert controller.start(timeout=0.5) is True
    assert controller._thread is not None and controller._thread.is_alive()

    release.set()
    controller._thread.join(timeout=1.0)


def test_overlay_start_reports_thread_initialization_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = CoachOverlayController(prefs_path=tmp_path / "overlay.json")

    def fake_run() -> None:
        controller.last_error = "tk failed"
        controller._ready_event.set()

    monkeypatch.setattr(controller, "_run", fake_run)

    assert controller.start(timeout=0.5) is False
    assert controller.last_error == "tk failed"


def test_overlay_start_uses_win32_fallback_when_frozen_host_omits_tk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = CoachOverlayController(prefs_path=tmp_path / "overlay.json")
    release = threading.Event()

    def missing_tk() -> None:
        raise SystemExit("Nuitka: Need to use '--enable-plugin=tk-inter' option during compilation")

    def fake_win32() -> None:
        controller.backend = "win32"
        controller.window_handle = 42
        controller.window_visible = True
        controller._ready_event.set()
        release.wait(timeout=1.0)

    monkeypatch.setattr(controller, "_run", missing_tk)
    monkeypatch.setattr(controller, "_run_win32", fake_win32)

    assert controller.start(timeout=0.5) is True
    assert controller.backend == "win32"
    assert controller.window_visible is True

    release.set()
    controller._thread.join(timeout=1.0)


def test_overlay_start_reports_win32_fallback_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = CoachOverlayController(prefs_path=tmp_path / "overlay.json")

    def missing_tk() -> None:
        raise SystemExit("Nuitka: Need to use '--enable-plugin=tk-inter' option during compilation")

    def broken_win32() -> None:
        raise TypeError("native startup failed")

    monkeypatch.setattr(controller, "_run", missing_tk)
    monkeypatch.setattr(controller, "_run_win32", broken_win32)

    assert controller.start(timeout=0.5) is False
    assert controller.last_error == "Win32 overlay TypeError: native startup failed"
