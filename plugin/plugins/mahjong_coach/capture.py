from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import platform
import shutil
import subprocess
import time
from typing import Any

from .models import FramePacket, WindowTargetDescriptor
from .window_binding import (
    WindowBindingResult,
    bind_window_from_descriptor,
    bind_window_from_keywords,
    refresh_cached_window,
)

try:
    import pyautogui
except Exception:
    pyautogui = None

try:
    from PIL import Image, ImageGrab  # type: ignore[import-not-found]
except Exception:
    Image = None
    ImageGrab = None


@dataclass
class CaptureContext:
    file_path: Path
    binding_result: WindowBindingResult


class CaptureSession:
    """Keep one target window bound and only enumerate windows when required."""

    def __init__(
        self,
        keywords: list[str],
        *,
        target: WindowTargetDescriptor | None = None,
        revalidate_seconds: float = 2.0,
    ) -> None:
        self.keywords = [str(item).strip() for item in keywords if str(item).strip()]
        self.target = target or WindowTargetDescriptor()
        self.revalidate_seconds = max(0.25, float(revalidate_seconds))
        self.binding = WindowBindingResult(bound=False, source="capture-session")
        self._last_validation = 0.0

    def locate_window(self, *, force: bool = False) -> WindowBindingResult:
        now = time.monotonic()
        if self.binding.bound and not force:
            if now - self._last_validation < self.revalidate_seconds:
                return self.binding
            previous = self.binding
            refreshed = refresh_cached_window(previous)
            self._last_validation = now
            descriptor_changed = refreshed.bound and (
                refreshed.window_title != previous.window_title
                or refreshed.width != previous.width
                or refreshed.height != previous.height
            )
            if refreshed.bound and not descriptor_changed:
                self.binding = refreshed
                return refreshed
        self.binding = bind_window_from_descriptor(self.keywords, self.target)
        self._last_validation = now
        if self.binding.bound:
            self.target = WindowTargetDescriptor(
                title=self.binding.window_title,
                app_name=self.binding.app_name,
                match_keyword=self.binding.match_keyword,
            )
        return self.binding

    def invalidate(self, reason: str = "capture_failed") -> None:
        self.binding = WindowBindingResult(bound=False, source="capture-session", error=reason)
        self._last_validation = 0.0


class DefaultCaptureProvider:
    def locate_window(self, keywords: list[str]) -> WindowBindingResult:
        return bind_window_from_keywords(keywords)

    def capture_frame(self, *, samples_dir: Path, binding_result: WindowBindingResult, save_format: str = "png") -> FramePacket:
        samples_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc)
        safe_format = save_format if save_format in {"png", "jpg", "jpeg"} else "png"
        file_path = samples_dir / f"{timestamp.strftime('%Y%m%d-%H%M%S-%f')}-frame.{safe_format}"
        source = self._save_screenshot(CaptureContext(file_path=file_path, binding_result=binding_result))
        return FramePacket(
            timestamp_ms=int(timestamp.timestamp() * 1000),
            image_path=str(file_path),
            window_title=binding_result.window_title or binding_result.app_name,
            width=int(binding_result.width or 0),
            height=int(binding_result.height or 0),
            source=source,
        )

    def capture_memory_frame(self, *, binding_result: WindowBindingResult) -> FramePacket:
        timestamp = datetime.now(timezone.utc)
        image, source = self._capture_image(binding_result)
        image = image.convert("RGB")
        setattr(image, "_neko_frame_id", int(timestamp.timestamp() * 1_000_000))
        digest_image = image.resize((32, 18), Image.Resampling.BILINEAR)
        fingerprint = hashlib.blake2s(digest_image.tobytes(), digest_size=12).hexdigest()
        return FramePacket(
            timestamp_ms=int(timestamp.timestamp() * 1000),
            window_title=binding_result.window_title or binding_result.app_name,
            width=int(image.width),
            height=int(image.height),
            source=source,
            image=image,
            fingerprint=fingerprint,
        )

    def persist_packet(self, packet: FramePacket, file_path: Path, *, save_format: str = "jpg") -> str:
        if packet.image is None:
            return packet.image_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        self._persist_image(packet.image, file_path)
        packet.image_path = str(file_path)
        return packet.image_path

    def _capture_image(self, binding_result: WindowBindingResult) -> tuple[Any, str]:
        region = self._resolve_capture_region(binding_result)
        errors: list[str] = []
        system = platform.system().lower()
        if system == "windows" and binding_result.hwnd:
            try:
                return self._grab_with_print_window(int(binding_result.hwnd))
            except Exception as exc:
                errors.append(f"print-window: {exc}")
        if system == "windows" and ImageGrab is not None and binding_result.hwnd:
            try:
                image = ImageGrab.grab(window=int(binding_result.hwnd))
                self._validate_window_capture(image)
                return image, "imagegrab-window"
            except Exception as exc:
                errors.append(f"imagegrab-window: {exc}")
        if region is not None and pyautogui is not None:
            try:
                return pyautogui.screenshot(region=region), "pyautogui-region"
            except Exception as exc:
                errors.append(f"pyautogui-region: {exc}")
        if region is not None and ImageGrab is not None:
            try:
                left, top, width, height = region
                return ImageGrab.grab(bbox=(left, top, left + width, top + height)), "imagegrab-region"
            except Exception as exc:
                errors.append(f"imagegrab-region: {exc}")
        if pyautogui is not None:
            try:
                return pyautogui.screenshot(), "pyautogui"
            except Exception as exc:
                errors.append(f"pyautogui: {exc}")
        if ImageGrab is not None:
            try:
                return ImageGrab.grab(), "imagegrab"
            except Exception as exc:
                errors.append(f"imagegrab: {exc}")
        raise RuntimeError("no in-memory screenshot backend succeeded: " + "; ".join(errors))

    def _save_screenshot(self, context: CaptureContext) -> str:
        region = self._resolve_capture_region(context.binding_result)
        errors: list[str] = []
        system = platform.system().lower()

        # PrintWindow asks the target window to render itself, so an overlapping
        # window is not copied into the Mahjong Soul frame. Pillow's
        # ImageGrab(window=...) can still behave like a screen crop on Windows,
        # so it is only a compatibility fallback here.
        if system == "windows" and context.binding_result.hwnd:
            try:
                return self._save_with_print_window(context.file_path, int(context.binding_result.hwnd))
            except Exception as exc:
                errors.append(f"print-window: {exc}")

        if system == "windows" and ImageGrab is not None and context.binding_result.hwnd:
            try:
                return self._save_with_imagegrab_window(context.file_path, int(context.binding_result.hwnd))
            except Exception as exc:
                errors.append(f"imagegrab-window: {exc}")

        if region is not None and pyautogui is not None:
            try:
                return self._save_with_pyautogui(context.file_path, region)
            except Exception as exc:
                errors.append(f"pyautogui-region: {exc}")

        if region is not None and ImageGrab is not None:
            try:
                return self._save_with_imagegrab(context.file_path, region)
            except Exception as exc:
                errors.append(f"imagegrab-region: {exc}")

        if pyautogui is not None:
            try:
                return self._save_with_pyautogui(context.file_path, region)
            except Exception as exc:
                errors.append(f"pyautogui: {exc}")

        if ImageGrab is not None:
            try:
                return self._save_with_imagegrab(context.file_path, region)
            except Exception as exc:
                errors.append(f"imagegrab: {exc}")

        if system == "darwin" and shutil.which("screencapture"):
            try:
                return self._save_with_screencapture(context.file_path, region)
            except Exception as exc:
                errors.append(f"screencapture: {exc}")
        if system == "linux":
            try:
                return self._save_with_linux_tools(context.file_path, region)
            except Exception as exc:
                errors.append(f"linux-tools: {exc}")
        if errors:
            raise RuntimeError("no screenshot backend succeeded: " + "; ".join(errors))
        raise RuntimeError("no screenshot backend available")

    def _resolve_capture_region(self, binding_result: WindowBindingResult) -> tuple[int, int, int, int] | None:
        if not binding_result.bound or not binding_result.has_bounds():
            return None
        assert binding_result.left is not None
        assert binding_result.top is not None
        assert binding_result.width is not None
        assert binding_result.height is not None
        return (int(binding_result.left), int(binding_result.top), int(binding_result.width), int(binding_result.height))

    def _save_with_print_window(self, file_path: Path, hwnd: int) -> str:
        image, source = self._grab_with_print_window(hwnd)
        self._persist_image(image, file_path)
        return source

    def _grab_with_print_window(self, hwnd: int) -> tuple[Any, str]:
        import ctypes
        import ctypes.wintypes

        if Image is None:
            raise RuntimeError("PIL Image unavailable")

        user32 = ctypes.windll.user32
        gdi32 = ctypes.windll.gdi32

        rect = ctypes.wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        w = rect.right - rect.left
        h = rect.bottom - rect.top
        if w <= 0 or h <= 0:
            raise RuntimeError(f"invalid window size {w}x{h}")

        hdc = user32.GetWindowDC(hwnd)
        if not hdc:
            raise RuntimeError(f"GetWindowDC returned null for hwnd={hwnd}")
        hdc_mem = gdi32.CreateCompatibleDC(hdc)
        hbm = gdi32.CreateCompatibleBitmap(hdc, w, h)
        gdi32.SelectObject(hdc_mem, hbm)

        try:
            PW_RENDERFULLCONTENT = 2
            used_fullcontent = bool(user32.PrintWindow(hwnd, hdc_mem, PW_RENDERFULLCONTENT))
            if not used_fullcontent:
                if not user32.PrintWindow(hwnd, hdc_mem, 0):
                    raise RuntimeError("PrintWindow failed")

            class BITMAPINFOHEADER(ctypes.Structure):
                _fields_ = [
                    ("biSize", ctypes.wintypes.DWORD),
                    ("biWidth", ctypes.wintypes.LONG),
                    ("biHeight", ctypes.wintypes.LONG),
                    ("biPlanes", ctypes.wintypes.WORD),
                    ("biBitCount", ctypes.wintypes.WORD),
                    ("biCompression", ctypes.wintypes.DWORD),
                    ("biSizeImage", ctypes.wintypes.DWORD),
                    ("biXPelsPerMeter", ctypes.wintypes.LONG),
                    ("biYPelsPerMeter", ctypes.wintypes.LONG),
                    ("biClrUsed", ctypes.wintypes.DWORD),
                    ("biClrImportant", ctypes.wintypes.DWORD),
                ]

            bmi = BITMAPINFOHEADER()
            bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
            bmi.biWidth = w
            bmi.biHeight = -h
            bmi.biPlanes = 1
            bmi.biBitCount = 32
            bmi.biCompression = 0

            buf = ctypes.create_string_buffer(w * h * 4)
            gdi32.GetDIBits(hdc_mem, hbm, 0, h, buf, ctypes.byref(bmi), 0)
        finally:
            gdi32.DeleteObject(hbm)
            gdi32.DeleteDC(hdc_mem)
            user32.ReleaseDC(hwnd, hdc)

        image = self._image_from_windows_bgrx(buf.raw, (w, h))
        self._validate_window_capture(image)
        return image, "print-window-fullcontent" if used_fullcontent else "print-window"

    def _image_from_windows_bgrx(self, pixels: bytes, size: tuple[int, int]) -> Any:
        if Image is None:
            raise RuntimeError("PIL Image unavailable")
        return Image.frombytes("RGB", size, pixels, "raw", "BGRX")

    def _save_with_pyautogui(self, file_path: Path, region: tuple[int, int, int, int] | None) -> str:
        if pyautogui is None:
            raise RuntimeError("pyautogui unavailable")
        if region is not None:
            try:
                image = pyautogui.screenshot(region=region)
                source = "pyautogui-region"
            except Exception:
                image = pyautogui.screenshot()
                source = "pyautogui-fullscreen-fallback"
        else:
            image = pyautogui.screenshot()
            source = "pyautogui"
        self._persist_image(image, file_path)
        return source

    def _save_with_imagegrab(self, file_path: Path, region: tuple[int, int, int, int] | None) -> str:
        if ImageGrab is None:
            raise RuntimeError("ImageGrab unavailable")
        if region is not None:
            left, top, width, height = region
            try:
                image = ImageGrab.grab(bbox=(left, top, left + width, top + height))
                source = "imagegrab-region"
            except Exception:
                image = ImageGrab.grab()
                source = "imagegrab-fullscreen-fallback"
        else:
            image = ImageGrab.grab()
            source = "imagegrab"
        self._persist_image(image, file_path)
        return source

    def _save_with_imagegrab_window(self, file_path: Path, hwnd: int) -> str:
        if ImageGrab is None:
            raise RuntimeError("ImageGrab unavailable")
        image = ImageGrab.grab(window=hwnd)
        self._validate_window_capture(image)
        self._persist_image(image, file_path)
        return "imagegrab-window"

    def _validate_window_capture(self, image: Any) -> None:
        width, height = getattr(image, "size", (0, 0))
        if int(width or 0) < 64 or int(height or 0) < 64:
            raise RuntimeError(f"window capture returned invalid size {width}x{height}")
        sample = image.convert("RGB").resize((64, 36), Image.Resampling.BILINEAR)
        extrema = sample.getextrema()
        if not extrema or max(int(high) - int(low) for low, high in extrema) <= 3:
            raise RuntimeError("window capture returned a blank or uniform image")

    def _save_with_screencapture(self, file_path: Path, region: tuple[int, int, int, int] | None) -> str:
        command = ["screencapture", "-x"]
        if region is not None:
            left, top, width, height = region
            try:
                subprocess.run(command + ["-R", f"{left},{top},{width},{height}", str(file_path)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return "screencapture-region"
            except Exception:
                pass
        subprocess.run(command + [str(file_path)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return "screencapture"

    def _save_with_linux_tools(self, file_path: Path, region: tuple[int, int, int, int] | None) -> str:
        if shutil.which("grim"):
            if region is not None:
                left, top, width, height = region
                try:
                    subprocess.run(["grim", "-g", f"{left},{top} {width}x{height}", str(file_path)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return "grim-region"
                except Exception:
                    pass
            subprocess.run(["grim", str(file_path)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return "grim"
        if shutil.which("gnome-screenshot"):
            subprocess.run(["gnome-screenshot", "-f", str(file_path)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return "gnome-screenshot"
        raise RuntimeError("no supported linux screenshot tool found")

    def _persist_image(self, image: Any, file_path: Path) -> None:
        suffix = file_path.suffix.lower()
        if suffix in {".jpg", ".jpeg"} and getattr(image, "mode", "") not in {"RGB", "L"}:
            image = image.convert("RGB")
        if suffix == ".png":
            image.save(file_path, compress_level=1)
        elif suffix in {".jpg", ".jpeg"}:
            image.save(file_path, quality=88)
        else:
            image.save(file_path)


def prune_frames(frames_dir: Path, *, keep: int) -> None:
    if not frames_dir.exists():
        return
    frames = sorted(path for path in frames_dir.glob("*-frame.*") if path.is_file())
    extra = len(frames) - max(0, int(keep))
    if extra <= 0:
        return
    for path in frames[:extra]:
        try:
            path.unlink()
        except OSError:
            pass
