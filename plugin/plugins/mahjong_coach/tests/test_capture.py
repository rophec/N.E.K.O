from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from plugin.plugins.mahjong_coach import capture, window_binding
from plugin.plugins.mahjong_coach.capture import CaptureContext, DefaultCaptureProvider, prune_frames
from plugin.plugins.mahjong_coach.window_binding import WindowBindingResult


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
