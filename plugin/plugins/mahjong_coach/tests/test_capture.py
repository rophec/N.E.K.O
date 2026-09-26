from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from plugin.plugins.mahjong_coach import capture, window_binding
from plugin.plugins.mahjong_coach.capture import CaptureContext, DefaultCaptureProvider, prune_frames
from plugin.plugins.mahjong_coach.perception import browser_canvas
from plugin.plugins.mahjong_coach.perception.browser_canvas import BrowserCanvasResult
from plugin.plugins.mahjong_coach.window_binding import WindowBindingResult


def test_entry_metadata_import_does_not_load_vendored_native_perception_modules() -> None:
    script = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path('plugin/plugins/mahjong_coach/vendor').resolve()))
import plugin.plugins.mahjong_coach
blocked = [name for name in ('pyautogui', 'pyscreeze', 'cv2', 'numpy', 'onnxruntime') if name in sys.modules]
if blocked:
    raise SystemExit('eager native imports: ' + ', '.join(blocked))
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[4],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


def _context(tmp_path: Path) -> CaptureContext:
    return CaptureContext(
        file_path=tmp_path / "frame.png",
        binding_result=WindowBindingResult(
            bound=True,
            window_title="雀魂 - Mahjong Soul",
            hwnd=1234,
            left=10,
            top=20,
            width=1280,
            height=720,
        ),
    )


def test_windows_dpi_awareness_prefers_per_monitor_v2(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, object]] = []

    class FakeUser32:
        @staticmethod
        def SetProcessDpiAwarenessContext(value):
            calls.append(("v2", value.value))
            return 1

        @staticmethod
        def SetProcessDPIAware():
            calls.append(("system", None))
            return 1

    fake_windll = SimpleNamespace(
        user32=FakeUser32(),
        shcore=SimpleNamespace(
            SetProcessDpiAwareness=lambda value: calls.append(("per-monitor", value)) or 0,
        ),
    )
    monkeypatch.setattr(capture.platform, "system", lambda: "Windows")
    monkeypatch.setattr(capture.ctypes, "windll", fake_windll, raising=False)
    monkeypatch.setattr(capture, "_windows_dpi_awareness_initialized", False)
    monkeypatch.setattr(capture, "_windows_dpi_awareness_mode", "uninitialized")

    assert capture.ensure_windows_dpi_awareness() == "per-monitor-v2"
    assert capture.ensure_windows_dpi_awareness() == "per-monitor-v2"
    assert calls == [("v2", 18446744073709551612)]


def test_windows_dpi_awareness_falls_back_without_importing_pyautogui(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []
    fake_windll = SimpleNamespace(
        user32=SimpleNamespace(
            SetProcessDpiAwarenessContext=lambda _value: calls.append(("v2", None)) or 0,
            SetProcessDPIAware=lambda: calls.append(("system", None)) or 1,
        ),
        shcore=SimpleNamespace(
            SetProcessDpiAwareness=lambda value: calls.append(("per-monitor", value)) or 1,
        ),
    )
    monkeypatch.setattr(capture.platform, "system", lambda: "Windows")
    monkeypatch.setattr(capture.ctypes, "windll", fake_windll, raising=False)
    monkeypatch.setattr(capture, "_windows_dpi_awareness_initialized", False)
    monkeypatch.setattr(capture, "_windows_dpi_awareness_mode", "uninitialized")
    monkeypatch.setattr(capture, "_pyautogui_import_attempted", False)

    assert capture.ensure_windows_dpi_awareness() == "system"
    assert calls == [("v2", None), ("per-monitor", 2), ("system", None)]
    assert capture._pyautogui_import_attempted is False


def test_windows_capture_prefers_print_window_before_other_backends(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    provider = DefaultCaptureProvider()
    calls: list[str] = []
    monkeypatch.setattr(capture.platform, "system", lambda: "Windows")
    monkeypatch.setattr(capture, "ImageGrab", object())
    monkeypatch.setattr(capture, "pyautogui", object())
    monkeypatch.setattr(provider, "_save_with_print_window", lambda *_args: calls.append("print-window") or "print-window")
    monkeypatch.setattr(provider, "_save_with_imagegrab_window", lambda *_args: calls.append("imagegrab-window"))
    monkeypatch.setattr(provider, "_save_with_pyautogui", lambda *_args: calls.append("pyautogui"))

    source = provider._save_screenshot(_context(tmp_path))

    assert source == "print-window"
    assert calls == ["print-window"]


def test_windows_capture_falls_back_from_both_hwnd_backends(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    provider = DefaultCaptureProvider()
    calls: list[str] = []
    monkeypatch.setattr(capture.platform, "system", lambda: "Windows")
    monkeypatch.setattr(capture, "ImageGrab", object())
    monkeypatch.setattr(capture, "pyautogui", object())

    def fail(name: str):
        calls.append(name)
        raise RuntimeError(name)

    monkeypatch.setattr(provider, "_save_with_print_window", lambda *_args: fail("print-window"))
    monkeypatch.setattr(provider, "_save_with_imagegrab_window", lambda *_args: fail("imagegrab-window"))
    monkeypatch.setattr(
        provider,
        "_save_with_pyautogui",
        lambda _path, _region: calls.append("pyautogui-region") or "pyautogui-region",
    )

    source = provider._save_screenshot(_context(tmp_path))

    assert source == "pyautogui-region"
    assert calls == ["print-window", "imagegrab-window", "pyautogui-region"]


def test_failed_print_window_uses_imagegrab_window_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    provider = DefaultCaptureProvider()
    calls: list[str] = []
    monkeypatch.setattr(capture.platform, "system", lambda: "Windows")
    monkeypatch.setattr(capture, "ImageGrab", object())
    monkeypatch.setattr(provider, "_save_with_print_window", lambda *_args: (_ for _ in ()).throw(RuntimeError("failed")))
    monkeypatch.setattr(provider, "_save_with_imagegrab_window", lambda *_args: calls.append("imagegrab-window") or "imagegrab-window")

    source = provider._save_screenshot(_context(tmp_path))

    assert source == "imagegrab-window"
    assert calls == ["imagegrab-window"]


def test_windows_bgrx_pixels_are_decoded_without_swapping_red_and_blue() -> None:
    provider = DefaultCaptureProvider()

    image = provider._image_from_windows_bgrx(bytes((10, 20, 240, 0)), (1, 1))

    assert image.getpixel((0, 0)) == (240, 20, 10)


def test_window_capture_validation_rejects_uniform_but_accepts_table_like_image() -> None:
    provider = DefaultCaptureProvider()
    with pytest.raises(RuntimeError, match="blank or uniform"):
        provider._validate_window_capture(Image.new("RGB", (1280, 720), (0, 0, 0)))

    image = Image.new("RGB", (1280, 720), (20, 55, 85))
    ImageDraw.Draw(image).rectangle((300, 200, 900, 600), fill=(225, 220, 205))
    provider._validate_window_capture(image)


def test_browser_canvas_discovery_uses_bitmap_edges_and_scene_validation() -> None:
    frame = Image.new("RGB", (1200, 800), (9, 20, 31))
    draw = ImageDraw.Draw(frame)
    expected_box = (88, 190, 1112, 766)
    draw.rectangle((88, 190, 1111, 765), fill=(22, 71, 112))
    draw.rectangle((500, 390, 700, 570), fill=(30, 40, 45))

    def detect(candidate: Image.Image) -> SimpleNamespace:
        matched = candidate.size == (1024, 576) and candidate.getpixel((0, 0)) == (22, 71, 112)
        return SimpleNamespace(
            detected=matched,
            confidence=0.97 if matched else 0.0,
            table_surface=None,
        )

    result = browser_canvas.find_browser_game_canvas(frame, scene_detector=detect)

    assert result is not None
    assert result.box == expected_box
    assert result.confidence == pytest.approx(0.97)
    assert result.normalized_box == pytest.approx((88 / 1200, 190 / 800, 1112 / 1200, 766 / 800))


@pytest.mark.parametrize(
    "title",
    [
        "雀魂麻将 - Microsoft Edge",
        "Mahjong Soul - Google Chrome",
        "雀魂 - Mozilla Firefox",
    ],
)
def test_browser_window_detection_accepts_supported_browser_titles(title: str) -> None:
    assert browser_canvas.is_browser_window(title)


def test_memory_capture_auto_crops_browser_once_then_reuses_cached_box(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = DefaultCaptureProvider()
    full = Image.new("RGB", (1200, 800), (10, 20, 30))
    calls: list[tuple[int, int]] = []
    result = BrowserCanvasResult(
        box=(88, 190, 1112, 766),
        normalized_box=(88 / 1200, 190 / 800, 1112 / 1200, 766 / 800),
        confidence=0.97,
    )
    monkeypatch.setattr(
        provider,
        "_capture_image",
        lambda _binding: (full.copy(), "print-window-fullcontent"),
    )
    monkeypatch.setattr(
        browser_canvas,
        "find_browser_game_canvas",
        lambda image: calls.append(image.size) or result,
    )
    binding = WindowBindingResult(
        bound=True,
        window_title="雀魂麻将 - Microsoft Edge",
        hwnd=4321,
        width=800,
        height=533,
    )

    first = provider.capture_memory_frame(binding_result=binding)
    second = provider.capture_memory_frame(binding_result=binding)

    assert first.image is not None and first.image.size == (1024, 576)
    assert second.image is not None and second.image.size == (1024, 576)
    assert first.source == "print-window-fullcontent+browser-canvas-auto"
    assert calls == [(1200, 800)]
    first.image.close()
    second.image.close()
    full.close()


def test_matching_window_lookup_never_activates_window(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    window = SimpleNamespace(
        title="雀魂 - Mahjong Soul",
        isMinimized=False,
        isActive=False,
        left=10,
        top=20,
        width=1280,
        height=720,
        _hWnd=1234,
        activate=lambda: calls.append("activate"),
        restore=lambda: calls.append("restore"),
    )
    monkeypatch.setattr(window_binding, "_all_windows_windows", lambda: [window])

    result = window_binding._find_matching_window_windows(["雀魂", "Mahjong Soul"])

    assert result is not None and result.bound is True
    assert result.hwnd == 1234
    assert calls == []


@pytest.mark.parametrize(("keep", "remaining"), [(0, []), (1, [24]), (20, list(range(5, 25)))])
def test_prune_frames_enforces_zero_and_positive_retention(
    tmp_path: Path,
    keep: int,
    remaining: list[int],
) -> None:
    for index in range(25):
        (tmp_path / f"20260727-000000-{index:06d}-frame.jpg").write_bytes(b"frame")

    prune_frames(tmp_path, keep=keep)

    actual = [int(path.stem.split("-")[-2]) for path in sorted(tmp_path.glob("*-frame.jpg"))]
    assert actual == remaining
