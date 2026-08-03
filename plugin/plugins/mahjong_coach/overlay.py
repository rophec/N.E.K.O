from __future__ import annotations

import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable


# N.E.K.O design system (light mode)
_BG = "#ffffff"
_CARD = "#f5f5f5"
_BORDER = "#d0d0d0"
_BORDER_FOCUS = "#2a7bc4"
_ACCENT = "#44b7fe"
_BTN_PRIMARY = "#2a7bc4"
_BTN_HOVER = "#3590d9"
_SUCCESS = "#16a34a"
_WARNING = "#d97706"
_DANGER = "#dc2626"
_PURPLE = "#7c3aed"
_TEXT = "#1e1e1e"
_TEXT_MUTED = "#666666"
_FONT = "Segoe UI"
_FONT_SIZE = 18
_HEADER_H = 3

# Badge colors
_BADGE_LOCAL_BG = "#2a7bc4"
_BADGE_LOCAL_FG = "#ffffff"

# Action badge overrides
_ACTION_WIN = ("和牌", _SUCCESS, "#ffffff")
_ACTION_RIICHI = ("立直", _PURPLE, "#ffffff")
_ACTION_CALL = ("鸣牌", _WARNING, "#ffffff")
_ACTION_PUSH = ("推进", _SUCCESS, "#ffffff")
_ACTION_MAWASHI = ("兜牌", _WARNING, "#ffffff")
_ACTION_DEFENSE = ("防守", _DANGER, "#ffffff")
_ACTION_GENERIC = ("操作", _BTN_PRIMARY, "#ffffff")

# Resize bounds
_MIN_WIDTH = 280
_MIN_HEIGHT = 80
_MAX_WIDTH = 1000
_MAX_HEIGHT = 840

# Default size
_DEFAULT_WIDTH = 420
_DEFAULT_HEIGHT = 140

# Config mode (two buttons) compact size
_CONFIG_WIDTH = 640
_CONFIG_HEIGHT = 335
_DETAIL_WIDTH = 1160
_DETAIL_HEIGHT = 870
_DETAIL_IMAGE_MAX_WIDTH = 540
_DETAIL_IMAGE_MAX_HEIGHT = 740
_SWITCH_STYLE_LABEL = "打法/显示"
_CONFIG_STYLE_HINT = "先选策略力度，再选择打法开始或应用；预测中点“打法/显示”可随时调整"

def _load_prefs(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            w = int(data.get("width", _DEFAULT_WIDTH))
            h = int(data.get("height", _DEFAULT_HEIGHT))
            fs = int(data.get("font_size", _FONT_SIZE))
            display_mode = _normalize_display_mode(data.get("display_mode"))
            panel_mode = _normalize_panel_mode(data.get("panel_mode"))
            strategy_preset = _normalize_strategy_preset(data.get("strategy_preset"))
            return {
                "width": max(_MIN_WIDTH, min(_MAX_WIDTH, w)),
                "height": max(_MIN_HEIGHT, min(_MAX_HEIGHT, h)),
                "font_size": max(16, min(32, fs)),
                "display_mode": display_mode,
                "panel_mode": panel_mode,
                "strategy_preset": strategy_preset,
            }
    except Exception:
        pass
    return {
        "width": _DEFAULT_WIDTH,
        "height": _DEFAULT_HEIGHT,
        "font_size": _FONT_SIZE,
        "display_mode": "compact",
        "panel_mode": "compact",
        "strategy_preset": "simple",
    }


def _save_prefs(
    path: Path,
    width: int,
    height: int,
    font_size: int,
    display_mode: str | None = None,
    panel_mode: str | None = None,
    strategy_preset: str | None = None,
) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        existing_mode = "compact"
        existing_panel_mode = "compact"
        existing_strategy_preset = "simple"
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                existing_mode = _normalize_display_mode(existing.get("display_mode"))
                existing_panel_mode = _normalize_panel_mode(existing.get("panel_mode"))
                existing_strategy_preset = _normalize_strategy_preset(existing.get("strategy_preset"))
        except Exception:
            pass
        path.write_text(
            json.dumps(
                {
                    "width": width,
                    "height": height,
                    "font_size": font_size,
                    "display_mode": _normalize_display_mode(display_mode or existing_mode),
                    "panel_mode": _normalize_panel_mode(panel_mode or existing_panel_mode),
                    "strategy_preset": _normalize_strategy_preset(strategy_preset or existing_strategy_preset),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass


def _normalize_display_mode(value: Any) -> str:
    mode = str(value or "").strip().lower()
    return "beginner" if mode in {"beginner", "newbie", "helper", "learn"} else "compact"


def _normalize_panel_mode(value: Any) -> str:
    mode = str(value or "").strip().lower()
    return "full" if mode in {"full", "card", "strategy", "complete"} else "compact"


def _normalize_strategy_preset(value: Any) -> str:
    return "standard" if str(value or "").strip().lower() in {"standard", "full", "complete"} else "simple"


def _overlay_panel_text(panel_mode: str, compact_text: str, strategy_card_text: str) -> str:
    if _normalize_panel_mode(panel_mode) == "full" and str(strategy_card_text or "").strip():
        return str(strategy_card_text).strip()
    return str(compact_text or "").strip() or "Mahjong Coach"


def _overlay_risk_level(value: float) -> str:
    risk = max(0.0, min(100.0, float(value)))
    if risk <= 9:
        return "极低"
    if risk <= 39:
        return "较低"
    if risk <= 59:
        return "中等"
    if risk <= 79:
        return "高"
    return "很高"


def _win32_topbar_action(mode: str, x: int, y: int, width: int) -> str:
    if y > 38:
        return ""
    if x >= width - 34:
        return "close"
    if mode != "strategy":
        return ""
    if x >= width - 138:
        return "switch_style"
    if x >= width - 196:
        return "detail"
    return ""


class CoachOverlayController:
    def __init__(
        self,
        *,
        prefs_path: Path,
        on_start: Callable[[str, str], None] | None = None,
        on_stop: Callable[[], None] | None = None,
    ) -> None:
        self._queue: queue.Queue[Any] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._ready_event = threading.Event()
        self._stop_requested = threading.Event()
        self._startup_error = ""
        self.last_error = ""
        self.backend = ""
        self.window_handle = 0
        self.window_visible = False
        self._prefs_path = prefs_path.expanduser().resolve()
        self._on_start = on_start
        self._on_stop = on_stop

    @property
    def prefs_path(self) -> Path:
        return self._prefs_path

    def start(self, timeout: float = 5.0) -> bool:
        """启动浮窗线程并等待窗口真正建立。 / Start the overlay thread and wait for its window."""
        if self._thread is not None and self._thread.is_alive():
            if not self._ready_event.wait(timeout=max(0.0, float(timeout))):
                self.last_error = self.last_error or f"overlay startup timed out ({self.backend or 'unknown backend'})"
                return False
            return True

        self._discard_pending_commands()
        self.last_error = ""
        self._startup_error = ""
        self.backend = ""
        self.window_handle = 0
        self.window_visible = False
        self._ready_event.clear()
        self._stop_requested.clear()
        self._thread = threading.Thread(target=self._run_guarded, name="MahjongCoachOverlay", daemon=True)
        self._thread.start()
        if not self._ready_event.wait(timeout=max(0.0, float(timeout))):
            self.last_error = self.last_error or f"overlay startup timed out ({self.backend or 'unknown backend'})"
            return False
        if self._startup_error or self._thread is None or not self._thread.is_alive():
            self.last_error = self.last_error or "overlay thread exited during startup"
            return False
        return True

    def _discard_pending_commands(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def _run_guarded(self) -> None:
        """Preserve fatal thread startup errors that threading would otherwise hide."""
        try:
            self._run()
        except BaseException as exc:
            detail = str(exc).strip() or repr(exc)
            if os.name == "nt" and "--enable-plugin=tk-inter" in detail:
                # 中文：Steam 冻结版未打包 Tk 时，自动切换到无需 Tk 的原生 Win32 浮窗。
                # English: Use the native Win32 overlay when the frozen Steam host omitted Tk.
                self.backend = "win32:starting"
                self.last_error = ""
                self._startup_error = ""
                try:
                    self._run_win32()
                    return
                except BaseException as native_exc:
                    native_detail = str(native_exc).strip() or repr(native_exc)
                    self.last_error = f"Win32 overlay {type(native_exc).__name__}: {native_detail}"
                    self._startup_error = self.last_error
                    self._ready_event.set()
                    return
            self.last_error = f"{type(exc).__name__}: {detail}"
            self._startup_error = self.last_error
            self._ready_event.set()

    def _run_win32(self) -> None:
        """Run the Windows-native overlay used by frozen hosts without Tk."""
        import win32api  # type: ignore[import-not-found]
        import win32con  # type: ignore[import-not-found]
        import win32gui  # type: ignore[import-not-found]

        self.backend = "win32:imports-ready"
        prefs = _load_prefs(self._prefs_path)
        state: dict[str, Any] = {
            "main": 0,
            "detail": 0,
            "mode": "config",
            "text": "Mahjong Coach",
            "strategy_card_text": "Mahjong Coach",
            "strategy_card": {},
            "detail_text": "等待识别详情",
            "image_path": "",
            "detail_image": None,
            "closed": False,
            "started": False,
            "width": prefs["width"],
            "height": prefs["height"],
            "font_size": prefs["font_size"],
            "display_mode": prefs["display_mode"],
            "panel_mode": prefs["panel_mode"],
            "strategy_preset": prefs["strategy_preset"],
        }
        class_name = f"NekoMahjongCoachOverlay_{id(self):x}"
        instance = win32api.GetModuleHandle(None)
        white_brush = win32gui.CreateSolidBrush(win32api.RGB(255, 255, 255))
        card_brush = win32gui.CreateSolidBrush(win32api.RGB(245, 245, 245))
        accent_brush = win32gui.CreateSolidBrush(win32api.RGB(68, 183, 254))
        border_brush = win32gui.CreateSolidBrush(win32api.RGB(208, 208, 208))
        success_brush = win32gui.CreateSolidBrush(win32api.RGB(22, 163, 74))
        warning_brush = win32gui.CreateSolidBrush(win32api.RGB(217, 119, 6))
        danger_brush = win32gui.CreateSolidBrush(win32api.RGB(220, 38, 38))
        primary_light_brush = win32gui.CreateSolidBrush(win32api.RGB(234, 246, 255))
        safe_light_brush = win32gui.CreateSolidBrush(win32api.RGB(250, 250, 250))
        danger_light_brush = win32gui.CreateSolidBrush(win32api.RGB(255, 241, 242))
        def _create_font(height: int, weight: int) -> int:
            spec = win32gui.LOGFONT()
            spec.lfFaceName = _FONT
            spec.lfHeight = -height
            spec.lfWeight = weight
            return win32gui.CreateFontIndirect(spec)

        font = _create_font(int(state["font_size"]), 400)
        small_font = _create_font(14, 600)
        headline_font = _create_font(26, 700)
        candidate_font = _create_font(19, 700)
        self.backend = "win32:gdi-ready"

        def _window_rect(hwnd: int) -> tuple[int, int, int, int]:
            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
            return left, top, max(1, right - left), max(1, bottom - top)

        def _draw_text(
            hdc: int,
            value: str,
            rect: tuple[int, int, int, int],
            *,
            color: int,
            small: bool = False,
            custom_font: int = 0,
        ) -> None:
            previous = win32gui.SelectObject(hdc, custom_font or (small_font if small else font))
            win32gui.SetBkMode(hdc, win32con.TRANSPARENT)
            win32gui.SetTextColor(hdc, color)
            win32gui.DrawText(
                hdc,
                str(value or ""),
                -1,
                rect,
                win32con.DT_LEFT | win32con.DT_TOP | win32con.DT_WORDBREAK | win32con.DT_NOPREFIX,
            )
            win32gui.SelectObject(hdc, previous)

        def _paint_strategy_card(hdc: int, width: int, height: int, card: dict[str, Any]) -> None:
            posture_value = str(card.get("posture_value") or "")
            posture_brush = {
                "push": success_brush,
                "mawashi": warning_brush,
                "fold": danger_brush,
            }.get(posture_value, accent_brush)
            win32gui.FillRect(hdc, (12, 42, width - 12, height - 12), white_brush)
            win32gui.FrameRect(hdc, (12, 42, width - 12, height - 12), border_brush)

            win32gui.FillRect(hdc, (24, 58, 88, 86), posture_brush)
            _draw_text(hdc, str(card.get("posture") or "策略"), (36, 65, 88, 84), color=win32api.RGB(255, 255, 255), small=True)
            _draw_text(
                hdc,
                f"重点分析  {str(card.get('focus_tile') or '')}",
                (104, 52, width - 245, 88),
                color=win32api.RGB(30, 30, 30),
                custom_font=headline_font,
            )
            _draw_text(
                hdc,
                f"{str(card.get('posture_summary') or '')} · {str(card.get('shape_summary') or '')}",
                (105, 92, width - 245, 116),
                color=win32api.RGB(70, 70, 70),
                small=True,
            )

            risk_ok = str(card.get("risk_status") or "") != "超预算"
            risk_left = max(410, width - 230)
            win32gui.FillRect(hdc, (risk_left, 54, width - 26, 113), primary_light_brush if risk_ok else danger_light_brush)
            win32gui.FrameRect(hdc, (risk_left, 54, width - 26, 113), accent_brush if risk_ok else danger_brush)
            _draw_text(hdc, str(card.get("risk_summary") or ""), (risk_left + 12, 65, width - 34, 87), color=win32api.RGB(30, 30, 30), small=True)
            _draw_text(
                hdc,
                "✓ 预算内" if risk_ok else "! 超预算",
                (risk_left + 12, 89, width - 34, 108),
                color=win32api.RGB(22, 130, 65) if risk_ok else win32api.RGB(220, 38, 38),
                small=True,
            )
            _draw_text(
                hdc,
                str(card.get("explanation") or ""),
                (24, 126, width - 24, 150),
                color=win32api.RGB(80, 80, 80),
                small=True,
            )
            _draw_text(
                hdc,
                f"场况：{str(card.get('match_context_summary') or '尚未稳定识别四家点数')} ",
                (24, 151, width - 24, 177),
                color=win32api.RGB(42, 123, 196),
                small=True,
            )
            _draw_text(
                hdc,
                str(card.get("match_context_explanation") or ""),
                (24, 176, width - 24, 202),
                color=win32api.RGB(70, 70, 70),
                small=True,
            )
            _draw_text(
                hdc,
                str(card.get("decision_process") or ""),
                (24, 203, width - 24, 225),
                color=win32api.RGB(42, 123, 196),
                small=True,
            )
            _draw_text(
                hdc,
                f"风险刻度：{str(card.get('risk_scale_note') or '')}；{str(card.get('risk_scale_legend') or '')}",
                (24, 226, width - 24, 248),
                color=win32api.RGB(70, 70, 70),
                small=True,
            )
            _draw_text(
                hdc,
                str(card.get("risk_model_legend") or ""),
                (24, 249, width - 24, 279),
                color=win32api.RGB(70, 70, 70),
                small=True,
            )
            _draw_text(
                hdc,
                f"本局允许线：{str(card.get('risk_budget_calculation') or '')}",
                (24, 280, width - 24, 302),
                color=win32api.RGB(42, 123, 196),
                small=True,
            )

            candidates = [item for item in (card.get("candidates") or [])[:3] if isinstance(item, dict)]
            if not candidates:
                return
            gap = 8
            row_left = 24
            row_right = width - 24
            card_width = max(120, (row_right - row_left - gap * 2) // 3)
            card_top = 310
            card_bottom = height - 22
            for index, candidate in enumerate(candidates):
                left = row_left + index * (card_width + gap)
                right = row_right if index == 2 else left + card_width
                tone = str(candidate.get("tone") or "safe")
                background = primary_light_brush if tone == "primary" else (danger_light_brush if tone == "danger" else safe_light_brush)
                outline = accent_brush if tone == "primary" else (danger_brush if tone == "danger" else border_brush)
                win32gui.FillRect(hdc, (left, card_top, right, card_bottom), background)
                win32gui.FrameRect(hdc, (left, card_top, right, card_bottom), outline)
                rank_label = f"方案{'ABC'[index]}  {'排序基准' if tone == 'primary' else '对照方案'}"
                _draw_text(hdc, rank_label, (left + 12, card_top + 10, right - 8, card_top + 30), color=win32api.RGB(42, 123, 196) if tone == "primary" else win32api.RGB(90, 90, 90), small=True)
                _draw_text(hdc, f"候选牌 {candidate.get('tile', '?')}", (left + 12, card_top + 34, right - 8, card_top + 64), color=win32api.RGB(30, 30, 30), custom_font=candidate_font)
                risk_color = win32api.RGB(220, 38, 38) if tone == "danger" else win32api.RGB(30, 30, 30)
                _draw_text(hdc, f"{candidate.get('risk_summary', '')} · {candidate.get('risk_delta_label', candidate.get('risk_status', ''))}", (left + 12, card_top + 68, right - 8, card_top + 90), color=risk_color, small=True)
                _draw_text(hdc, str(candidate.get("shape_summary") or ""), (left + 12, card_top + 92, right - 8, card_top + 114), color=win32api.RGB(50, 50, 50), small=True)
                win32gui.FillRect(hdc, (left + 12, card_top + 122, right - 12, card_top + 123), outline)
                _draw_text(hdc, "为什么", (left + 12, card_top + 132, right - 8, card_top + 152), color=win32api.RGB(42, 123, 196), small=True)
                _draw_text(hdc, str(candidate.get("safety_reason") or ""), (left + 12, card_top + 156, right - 10, card_top + 222), color=win32api.RGB(70, 70, 70), small=True)
                _draw_text(hdc, "风险怎么算", (left + 12, card_top + 226, right - 8, card_top + 246), color=win32api.RGB(42, 123, 196), small=True)
                _draw_text(hdc, str(candidate.get("risk_reason") or ""), (left + 12, card_top + 250, right - 10, card_top + 330), color=win32api.RGB(70, 70, 70), small=True)
                _draw_text(hdc, str(candidate.get("shape_reason") or ""), (left + 12, card_top + 334, right - 10, card_top + 382), color=win32api.RGB(70, 70, 70), small=True)
                verdict_top = card_bottom - 45
                _draw_text(
                    hdc,
                    str(candidate.get("comparison_reason") or ""),
                    (left + 12, card_top + 386, right - 10, verdict_top - 10),
                    color=win32api.RGB(70, 70, 70),
                    small=True,
                )
                win32gui.FillRect(hdc, (left + 8, verdict_top, right - 8, card_bottom - 8), primary_light_brush if tone == "primary" else (danger_light_brush if tone == "danger" else white_brush))
                _draw_text(
                    hdc,
                    str(candidate.get("tradeoff") or ""),
                    (left + 14, verdict_top + 8, right - 14, card_bottom - 10),
                    color=win32api.RGB(42, 123, 196) if tone == "primary" else (win32api.RGB(220, 38, 38) if tone == "danger" else win32api.RGB(80, 80, 80)),
                    small=True,
                )

        def _draw_top_buttons(hdc: int, width: int, *, detail: bool = False) -> None:
            if detail or state["mode"] == "config":
                _draw_text(hdc, "×", (width - 24, 8, width, 30), color=win32api.RGB(102, 102, 102), small=True)
                return
            win32gui.FillRect(hdc, (width - 194, 5, width - 140, 35), card_brush)
            win32gui.FrameRect(hdc, (width - 194, 5, width - 140, 35), border_brush)
            _draw_text(hdc, "详情", (width - 183, 12, width - 140, 33), color=win32api.RGB(30, 30, 30), small=True)
            win32gui.FillRect(hdc, (width - 136, 5, width - 38, 35), accent_brush)
            _draw_text(hdc, _SWITCH_STYLE_LABEL, (width - 119, 12, width - 38, 33), color=win32api.RGB(255, 255, 255), small=True)
            _draw_text(hdc, "×", (width - 24, 8, width, 30), color=win32api.RGB(102, 102, 102), small=True)

        def _paint_main(hwnd: int, hdc: int, rect: tuple[int, int, int, int]) -> None:
            width, height = rect[2], rect[3]
            win32gui.FillRect(hdc, (0, 0, width, height), white_brush)
            win32gui.FillRect(hdc, (0, 0, width, 3), accent_brush)
            win32gui.FrameRect(hdc, (0, 0, width, height), border_brush)
            _draw_top_buttons(hdc, width)
            if state["mode"] == "config":
                _draw_text(hdc, "打牌风格", (18, 14, width - 100, 40), color=win32api.RGB(30, 30, 30), small=True)
                mid = width // 2
                win32gui.FillRect(hdc, (16, 44, mid - 6, 92), accent_brush)
                win32gui.FillRect(hdc, (mid + 6, 44, width - 16, 92), accent_brush)
                _draw_text(hdc, "立直（门清憋大牌）", (30, 60, mid - 14, 88), color=win32api.RGB(255, 255, 255), small=True)
                _draw_text(hdc, "快攻（积极副露）", (mid + 22, 60, width - 24, 88), color=win32api.RGB(255, 255, 255), small=True)
                _draw_text(hdc, "策略力度", (18, 108, 110, 138), color=win32api.RGB(102, 102, 102), small=True)
                simple_brush = accent_brush if state["strategy_preset"] == "simple" else card_brush
                standard_brush = accent_brush if state["strategy_preset"] == "standard" else card_brush
                win32gui.FillRect(hdc, (mid - 174, 104, mid - 6, 142), simple_brush)
                win32gui.FillRect(hdc, (mid + 6, 104, mid + 174, 142), standard_brush)
                _draw_text(hdc, "简易策略 · 轻防守", (mid - 151, 115, mid - 10, 138), color=win32api.RGB(255, 255, 255) if state["strategy_preset"] == "simple" else win32api.RGB(30, 30, 30), small=True)
                _draw_text(hdc, "完整攻守 · 全风险", (mid + 28, 115, mid + 168, 138), color=win32api.RGB(255, 255, 255) if state["strategy_preset"] == "standard" else win32api.RGB(30, 30, 30), small=True)
                _draw_text(hdc, "文案模式", (18, 158, 110, 188), color=win32api.RGB(102, 102, 102), small=True)
                compact_brush = accent_brush if state["display_mode"] == "compact" else card_brush
                beginner_brush = accent_brush if state["display_mode"] == "beginner" else card_brush
                win32gui.FillRect(hdc, (mid - 126, 154, mid - 10, 190), compact_brush)
                win32gui.FillRect(hdc, (mid + 10, 154, mid + 126, 190), beginner_brush)
                _draw_text(hdc, "简洁", (mid - 88, 165, mid - 10, 188), color=win32api.RGB(255, 255, 255) if state["display_mode"] == "compact" else win32api.RGB(30, 30, 30), small=True)
                _draw_text(hdc, "新手", (mid + 48, 165, mid + 126, 188), color=win32api.RGB(255, 255, 255) if state["display_mode"] == "beginner" else win32api.RGB(30, 30, 30), small=True)
                _draw_text(hdc, "展示内容", (18, 208, 110, 238), color=win32api.RGB(102, 102, 102), small=True)
                compact_panel_brush = accent_brush if state["panel_mode"] == "compact" else card_brush
                full_panel_brush = accent_brush if state["panel_mode"] == "full" else card_brush
                win32gui.FillRect(hdc, (mid - 174, 202, mid - 6, 244), compact_panel_brush)
                win32gui.FillRect(hdc, (mid + 6, 202, mid + 174, 244), full_panel_brush)
                _draw_text(hdc, "简洁主建议", (mid - 136, 215, mid - 10, 240), color=win32api.RGB(255, 255, 255) if state["panel_mode"] == "compact" else win32api.RGB(30, 30, 30), small=True)
                _draw_text(hdc, "详细策略推理", (mid + 38, 215, mid + 168, 240), color=win32api.RGB(255, 255, 255) if state["panel_mode"] == "full" else win32api.RGB(30, 30, 30), small=True)
                _draw_text(hdc, _CONFIG_STYLE_HINT, (80, 275, width - 60, 310), color=win32api.RGB(102, 102, 102), small=True)
                return
            win32gui.FillRect(hdc, (12, 14, 62, 36), accent_brush)
            _draw_text(hdc, "本地", (22, 18, 62, 36), color=win32api.RGB(255, 255, 255), small=True)
            if state["panel_mode"] == "full" and isinstance(state.get("strategy_card"), dict) and state["strategy_card"].get("candidates"):
                _paint_strategy_card(hdc, width, height, state["strategy_card"])
                return
            win32gui.FillRect(hdc, (12, 42, width - 12, height - 12), card_brush)
            panel_text = _overlay_panel_text(str(state["panel_mode"]), str(state["text"]), str(state["strategy_card_text"]))
            _draw_text(hdc, panel_text, (24, 54, width - 24, height - 20), color=win32api.RGB(30, 30, 30))

        def _paint_detail(hwnd: int, hdc: int, rect: tuple[int, int, int, int]) -> None:
            width, height = rect[2], rect[3]
            win32gui.FillRect(hdc, (0, 0, width, height), white_brush)
            win32gui.FillRect(hdc, (0, 0, width, 3), accent_brush)
            win32gui.FrameRect(hdc, (0, 0, width, height), border_brush)
            _draw_top_buttons(hdc, width, detail=True)
            _draw_text(hdc, "识别与策略详情", (18, 14, width - 50, 40), color=win32api.RGB(30, 30, 30), small=True)
            split = max(420, width // 2)
            win32gui.FillRect(hdc, (12, 44, split - 6, height - 12), card_brush)
            _draw_text(hdc, str(state["detail_text"]), (24, 56, split - 18, height - 24), color=win32api.RGB(30, 30, 30), small=True)
            image = state.get("detail_image")
            if image is not None:
                try:
                    from PIL import ImageWin  # type: ignore[import-not-found]

                    target_width = max(1, width - split - 30)
                    target_height = max(1, height - 72)
                    copy = image.copy()
                    copy.thumbnail((target_width, target_height))
                    left = split + 12 + max(0, (target_width - copy.width) // 2)
                    top = 52 + max(0, (target_height - copy.height) // 2)
                    ImageWin.Dib(copy).draw(hdc, (left, top, left + copy.width, top + copy.height))
                    return
                except Exception:
                    pass
            _draw_text(hdc, "等待截图\n" + str(state["image_path"] or ""), (split + 24, 64, width - 24, height - 24), color=win32api.RGB(102, 102, 102), small=True)

        def _resize_main(mode: str) -> None:
            hwnd = int(state["main"] or 0)
            if not hwnd or not win32gui.IsWindow(hwnd):
                return
            state["mode"] = mode
            width = _CONFIG_WIDTH if mode == "config" else int(state["width"])
            height = _CONFIG_HEIGHT if mode == "config" else int(state["height"])
            if mode == "strategy" and state["panel_mode"] == "full":
                width = max(1000, width)
                height = max(840, height)
            screen_width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
            screen_height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
            x, y = _overlay_geometry(screen_width, screen_height, width, height)
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, x, y, width, height, win32con.SWP_SHOWWINDOW)
            win32gui.InvalidateRect(hwnd, None, True)

        def _load_detail_image(path_text: str) -> None:
            state["detail_image"] = None
            path = Path(path_text) if path_text else None
            if path is None or not path.exists():
                return
            try:
                from PIL import Image  # type: ignore[import-not-found]

                with Image.open(path) as opened:
                    state["detail_image"] = opened.convert("RGB").copy()
            except Exception:
                state["detail_image"] = None

        def _show_detail() -> None:
            hwnd = int(state["detail"] or 0)
            if hwnd and win32gui.IsWindow(hwnd):
                win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
                win32gui.SetForegroundWindow(hwnd)
                return
            main_x, main_y, _, main_h = _window_rect(int(state["main"]))
            screen_width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
            screen_height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
            width = min(_DETAIL_WIDTH, max(900, screen_width - 80))
            height = min(_DETAIL_HEIGHT, max(820, screen_height - 100))
            x = min(max(0, main_x + 24), max(0, screen_width - width - 8))
            y = min(max(0, main_y + main_h + 8), max(0, screen_height - height - 40))
            detail_hwnd = win32gui.CreateWindowEx(
                win32con.WS_EX_TOPMOST | win32con.WS_EX_TOOLWINDOW,
                class_name,
                "Mahjong Coach Details",
                win32con.WS_POPUP | win32con.WS_THICKFRAME,
                x,
                y,
                width,
                height,
                0,
                0,
                instance,
                None,
            )
            state["detail"] = detail_hwnd
            win32gui.ShowWindow(detail_hwnd, win32con.SW_SHOW)
            win32gui.UpdateWindow(detail_hwnd)

        def _close_main() -> None:
            try:
                if self._on_stop is not None:
                    self._on_stop()
                    return
            except Exception as exc:
                self.last_error = f"close click failed: {exc}"
            self._queue.put(None)

        def _wnd_proc(hwnd: int, message: int, wparam: int, lparam: int) -> int:
            if message == win32con.WM_PAINT:
                hdc, paint = win32gui.BeginPaint(hwnd)
                try:
                    left, top, right, bottom = win32gui.GetClientRect(hwnd)
                    rect = (left, top, right, bottom)
                    if hwnd == state["detail"]:
                        _paint_detail(hwnd, hdc, rect)
                    else:
                        _paint_main(hwnd, hdc, rect)
                finally:
                    win32gui.EndPaint(hwnd, paint)
                return 0
            if message == win32con.WM_LBUTTONDOWN:
                x = win32api.LOWORD(lparam)
                y = win32api.HIWORD(lparam)
                _, _, width, _ = _window_rect(hwnd)
                if hwnd == state["detail"]:
                    if y <= 34 and x >= width - 36:
                        win32gui.DestroyWindow(hwnd)
                    else:
                        win32gui.ReleaseCapture()
                        win32gui.SendMessage(hwnd, win32con.WM_NCLBUTTONDOWN, win32con.HTCAPTION, 0)
                    return 0
                topbar_action = _win32_topbar_action(str(state["mode"]), x, y, width)
                if topbar_action == "close":
                    _close_main()
                    return 0
                if topbar_action == "switch_style":
                    _resize_main("config")
                    return 0
                if topbar_action == "detail":
                    _show_detail()
                    return 0
                if state["mode"] == "config":
                    midpoint = width // 2
                    if 44 <= y <= 96:
                        try:
                            if self._on_start is not None:
                                self._on_start("riichi" if x < midpoint else "fast", str(state["strategy_preset"]))
                        except Exception as exc:
                            self.last_error = f"button click failed: {exc}"
                        state["started"] = True
                        _resize_main("strategy")
                        return 0
                    if 100 <= y <= 146:
                        state["strategy_preset"] = "simple" if x < midpoint else "standard"
                        _save_prefs(self._prefs_path, int(state["width"]), int(state["height"]), int(state["font_size"]), str(state["display_mode"]), str(state["panel_mode"]), str(state["strategy_preset"]))
                        win32gui.InvalidateRect(hwnd, None, True)
                        return 0
                    if 150 <= y <= 194:
                        state["display_mode"] = "compact" if x < midpoint else "beginner"
                        _save_prefs(self._prefs_path, int(state["width"]), int(state["height"]), int(state["font_size"]), str(state["display_mode"]), str(state["panel_mode"]))
                        win32gui.InvalidateRect(hwnd, None, True)
                        return 0
                    if 198 <= y <= 250:
                        state["panel_mode"] = "compact" if x < midpoint else "full"
                        _save_prefs(self._prefs_path, int(state["width"]), int(state["height"]), int(state["font_size"]), str(state["display_mode"]), str(state["panel_mode"]))
                        if state["started"]:
                            _resize_main("strategy")
                        win32gui.InvalidateRect(hwnd, None, True)
                        return 0
                win32gui.ReleaseCapture()
                win32gui.SendMessage(hwnd, win32con.WM_NCLBUTTONDOWN, win32con.HTCAPTION, 0)
                return 0
            if message == win32con.WM_CLOSE:
                if hwnd == state["detail"]:
                    win32gui.DestroyWindow(hwnd)
                else:
                    _close_main()
                return 0
            if message == win32con.WM_DESTROY:
                if hwnd == state["detail"]:
                    state["detail"] = 0
                elif hwnd == state["main"]:
                    state["closed"] = True
                    self.window_handle = 0
                    self.window_visible = False
                    detail_hwnd = int(state["detail"] or 0)
                    if detail_hwnd and win32gui.IsWindow(detail_hwnd):
                        win32gui.DestroyWindow(detail_hwnd)
                    win32gui.PostQuitMessage(0)
                return 0
            return win32gui.DefWindowProc(hwnd, message, wparam, lparam)

        window_class = win32gui.WNDCLASS()
        window_class.hInstance = instance
        window_class.lpszClassName = class_name
        window_class.style = win32con.CS_HREDRAW | win32con.CS_VREDRAW
        window_class.hCursor = win32gui.LoadCursor(0, win32con.IDC_ARROW)
        window_class.hbrBackground = white_brush
        window_class.lpfnWndProc = _wnd_proc
        try:
            win32gui.RegisterClass(window_class)
        except win32gui.error as exc:
            if int(getattr(exc, "winerror", 0) or (exc.args[0] if exc.args else 0)) != 1410:
                raise
            try:
                win32gui.UnregisterClass(class_name, instance)
            except win32gui.error:
                pass
            win32gui.RegisterClass(window_class)
        self.backend = "win32:class-ready"

        screen_width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
        screen_height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
        x, y = _overlay_geometry(screen_width, screen_height, _CONFIG_WIDTH, _CONFIG_HEIGHT)
        main_hwnd = win32gui.CreateWindowEx(
            win32con.WS_EX_TOPMOST | win32con.WS_EX_TOOLWINDOW,
            class_name,
            "Mahjong Coach Overlay",
            win32con.WS_POPUP | win32con.WS_THICKFRAME,
            x,
            y,
            _CONFIG_WIDTH,
            _CONFIG_HEIGHT,
            0,
            0,
            instance,
            None,
        )
        self.backend = "win32:window-created"
        state["main"] = main_hwnd
        win32gui.ShowWindow(main_hwnd, win32con.SW_SHOW)
        win32gui.UpdateWindow(main_hwnd)
        self.backend = "win32"
        self.window_handle = int(main_hwnd)
        self.window_visible = bool(win32gui.IsWindowVisible(main_hwnd))
        self._ready_event.set()

        try:
            while not state["closed"]:
                if win32gui.PumpWaitingMessages():
                    break
                changed = False
                while True:
                    try:
                        item = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if item is None:
                        _, _, width, height = _window_rect(main_hwnd)
                        if state["mode"] == "strategy":
                            state["width"], state["height"] = width, height
                        _save_prefs(self._prefs_path, int(state["width"]), int(state["height"]), int(state["font_size"]), str(state["display_mode"]), str(state["panel_mode"]))
                        if win32gui.IsWindow(main_hwnd):
                            win32gui.DestroyWindow(main_hwnd)
                        changed = True
                        break
                    if isinstance(item, dict):
                        command = item.get("cmd")
                        if command == "show_config":
                            _resize_main("config")
                        elif command == "show_strategy":
                            _resize_main("strategy")
                        elif command == "show_detail":
                            _show_detail()
                        elif command == "update_payload":
                            state["text"] = str(item.get("text") or "Mahjong Coach")
                            state["strategy_card_text"] = str(item.get("strategy_card_text") or state["text"])
                            state["strategy_card"] = dict(item.get("strategy_card") or {})
                            state["detail_text"] = str(item.get("detail") or "等待识别详情")
                            state["image_path"] = str(item.get("image_path") or "")
                            _load_detail_image(str(state["image_path"]))
                        changed = True
                        continue
                    state["text"] = str(item or "Mahjong Coach")
                    changed = True
                if changed:
                    if win32gui.IsWindow(main_hwnd):
                        win32gui.InvalidateRect(main_hwnd, None, True)
                    detail_hwnd = int(state["detail"] or 0)
                    if detail_hwnd and win32gui.IsWindow(detail_hwnd):
                        win32gui.InvalidateRect(detail_hwnd, None, True)
                time.sleep(0.015)
        finally:
            try:
                win32gui.UnregisterClass(class_name, instance)
            except win32gui.error:
                pass
            for handle in (
                font,
                small_font,
                headline_font,
                candidate_font,
                white_brush,
                card_brush,
                accent_brush,
                border_brush,
                success_brush,
                warning_brush,
                danger_brush,
                primary_light_brush,
                safe_light_brush,
                danger_light_brush,
            ):
                try:
                    win32gui.DeleteObject(handle)
                except Exception:
                    pass

    def update(self, text: str) -> None:
        if self._thread is None or not self._thread.is_alive():
            return
        self._queue.put(str(text or "").strip() or "Mahjong Coach")

    def update_payload(
        self,
        *,
        text: str,
        strategy_card_text: str = "",
        strategy_card: dict[str, Any] | None = None,
        detail: str = "",
        image_path: str = "",
    ) -> None:
        if self._thread is None or not self._thread.is_alive():
            return
        self._queue.put({
            "cmd": "update_payload",
            "text": text,
            "strategy_card_text": strategy_card_text,
            "strategy_card": dict(strategy_card or {}),
            "detail": detail,
            "image_path": image_path,
        })

    def show_config(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            return
        self._queue.put({"cmd": "show_config"})

    def show_strategy(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            return
        self._queue.put({"cmd": "show_strategy"})

    def show_detail(self) -> None:
        """从 Tk 线程打开详情面板。 / Open the detail panel on the Tk thread."""
        if self._thread is None or not self._thread.is_alive():
            return
        self._queue.put({"cmd": "show_detail"})

    def stop(self) -> None:
        thread = self._thread
        if thread is None:
            return
        if thread.is_alive():
            self._stop_requested.set()
            self._queue.put(None)
            thread.join(timeout=2.0)
        if thread.is_alive():
            self.last_error = self.last_error or "overlay thread did not stop"
            return
        self._thread = None
        self.window_handle = 0
        self.window_visible = False
        self._ready_event.clear()
        self._discard_pending_commands()

    def _run(self) -> None:
        root = None
        fatal_exit = False
        try:
            import tkinter as tk
        except Exception as exc:
            detail = str(exc).strip() or repr(exc)
            self.last_error = f"tkinter unavailable ({type(exc).__name__}): {detail}"
            self._startup_error = self.last_error
            self._ready_event.set()
            return

        try:
            root = tk.Tk()
            self.backend = "tk"
            root.title("Mahjong Coach Overlay")
            root.overrideredirect(True)
            root.attributes("-topmost", True)
            root.configure(bg=_BG)

            # Border layer
            shell = tk.Frame(root, bg=_BORDER, padx=1, pady=1)
            shell.pack(fill="both", expand=True)

            inner = tk.Frame(shell, bg=_BG)
            inner.pack(fill="both", expand=True)

            # Header accent line
            header = tk.Canvas(inner, height=_HEADER_H, bg=_BG, highlightthickness=0, bd=0)
            header.pack(fill="x", side="top")

            def _draw_accent(_event: tk.Event | None = None) -> None:
                header.delete("all")
                w = header.winfo_width()
                if w < 2:
                    return
                r1, g1, b1 = 0x44, 0xB7, 0xFE
                r2, g2, b2 = 0x63, 0x66, 0xF1
                steps = min(w, 120)
                for i in range(steps):
                    ratio = i / max(steps - 1, 1)
                    r = int(r1 + (r2 - r1) * ratio)
                    g = int(g1 + (g2 - g1) * ratio)
                    b = int(b1 + (b2 - b1) * ratio)
                    x0 = int(i * w / steps)
                    x1 = int((i + 1) * w / steps)
                    header.create_rectangle(x0, 0, x1, _HEADER_H, fill=f"#{r:02x}{g:02x}{b:02x}", outline="")

            header.bind("<Configure>", _draw_accent)

            # --- close button (top-right) ---
            close_btn = tk.Label(
                inner, text="✕", bg=_BG, fg=_TEXT_MUTED,
                font=(_FONT, 10), cursor="hand2",
            )
            close_btn.place(relx=1.0, x=-8, y=4, anchor="ne")

            config_btn = tk.Label(
                inner, text=_SWITCH_STYLE_LABEL, bg=_BTN_PRIMARY, fg="#ffffff",
                font=(_FONT, 10, "bold"), cursor="hand2", padx=10, pady=3,
            )
            config_btn.place(relx=1.0, x=-30, y=5, anchor="ne")

            detail_btn = tk.Label(
                inner, text="详情", bg=_CARD, fg=_TEXT,
                font=(_FONT, 10, "bold"), cursor="hand2", padx=8, pady=3,
            )
            detail_btn.place(relx=1.0, x=-126, y=5, anchor="ne")

            def _on_close_click(_event: tk.Event) -> None:
                try:
                    if self._on_stop is not None:
                        self._on_stop()
                except Exception as exc:
                    self.last_error = f"close click failed: {exc}"

            close_btn.bind("<Button-1>", _on_close_click)

            def _on_config_click(_event: tk.Event) -> None:
                try:
                    show_config_mode()
                except Exception as exc:
                    self.last_error = f"config click failed: {exc}"

            config_btn.bind("<Button-1>", _on_config_click)

            detail_state = {
                "window": None,
                "text_widget": None,
                "image_label": None,
                "image_status": None,
                "image_photo": None,
                "text": "等待识别详情",
                "image_path": "",
            }

            def _update_detail_text_widget() -> None:
                widget = detail_state.get("text_widget")
                if widget is None or not widget.winfo_exists():
                    return
                widget.config(state="normal")
                widget.delete("1.0", "end")
                widget.insert("1.0", str(detail_state.get("text") or "等待识别详情"))
                widget.config(state="disabled")

            def _update_detail_image() -> None:
                """刷新右侧截图依据预览。 / Refresh the right-side source screenshot preview."""
                image_label = detail_state.get("image_label")
                image_status = detail_state.get("image_status")
                if image_label is None or not image_label.winfo_exists():
                    return
                path_text = str(detail_state.get("image_path") or "").strip()
                if not path_text:
                    detail_state["image_photo"] = None
                    image_label.config(image="", text="等待截图", fg=_TEXT_MUTED)
                    if image_status is not None and image_status.winfo_exists():
                        image_status.config(text="右侧会显示本次识别使用的截图")
                    return
                path = Path(path_text)
                if not path.exists():
                    detail_state["image_photo"] = None
                    image_label.config(image="", text=f"截图文件不存在\n{path_text}", fg=_DANGER)
                    if image_status is not None and image_status.winfo_exists():
                        image_status.config(text=path_text)
                    return
                try:
                    from PIL import Image, ImageTk  # type: ignore[import-not-found]

                    with Image.open(path) as opened:
                        image = opened.convert("RGB")
                    image.thumbnail((_DETAIL_IMAGE_MAX_WIDTH, _DETAIL_IMAGE_MAX_HEIGHT), Image.LANCZOS)
                    photo = ImageTk.PhotoImage(image)
                    # 保留 PhotoImage 引用，否则 Tk 会回收图片。 / Keep a PhotoImage reference or Tk will garbage-collect it.
                    detail_state["image_photo"] = photo
                    image_label.config(image=photo, text="")
                    if image_status is not None and image_status.winfo_exists():
                        image_status.config(text=path.name)
                except Exception as exc:
                    detail_state["image_photo"] = None
                    image_label.config(image="", text=f"截图无法显示\n{exc}", fg=_DANGER)
                    if image_status is not None and image_status.winfo_exists():
                        image_status.config(text=path_text)

            def show_detail_panel() -> None:
                existing = detail_state.get("window")
                if existing is not None and existing.winfo_exists():
                    existing.lift()
                    return
                panel = tk.Toplevel(root)
                panel.title("Mahjong Coach Details")
                panel.attributes("-topmost", True)
                panel.configure(bg=_BG)
                screen_w = panel.winfo_screenwidth()
                screen_h = panel.winfo_screenheight()
                panel_x = min(max(0, root.winfo_x() + 24), max(0, screen_w - _DETAIL_WIDTH - 8))
                panel_y = min(max(0, root.winfo_y() + root.winfo_height() + 8), max(0, screen_h - _DETAIL_HEIGHT - 40))
                panel.geometry(f"{_DETAIL_WIDTH}x{_DETAIL_HEIGHT}+{panel_x}+{panel_y}")
                panel.minsize(900, 560)
                shell_detail = tk.Frame(panel, bg=_BORDER, padx=1, pady=1)
                shell_detail.pack(fill="both", expand=True)
                inner_detail = tk.Frame(shell_detail, bg=_BG, padx=12, pady=10)
                inner_detail.pack(fill="both", expand=True)
                head = tk.Frame(inner_detail, bg=_BG)
                head.pack(fill="x", pady=(0, 8))
                tk.Label(head, text="识别与策略详情", bg=_BG, fg=_TEXT, font=(_FONT, 12, "bold")).pack(side="left")
                close_detail = tk.Label(head, text="✕", bg=_BG, fg=_TEXT_MUTED, font=(_FONT, 10), cursor="hand2")
                close_detail.pack(side="right")

                detail_body = tk.Frame(inner_detail, bg=_BG)
                detail_body.pack(fill="both", expand=True)
                detail_body.columnconfigure(0, weight=1)
                detail_body.columnconfigure(1, weight=0, minsize=_DETAIL_IMAGE_MAX_WIDTH + 24)
                detail_body.rowconfigure(0, weight=1)

                text_shell = tk.Frame(detail_body, bg=_CARD, padx=10, pady=8)
                text_shell.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
                scrollbar = tk.Scrollbar(text_shell)
                scrollbar.pack(side="right", fill="y")
                text_widget = tk.Text(
                    text_shell,
                    bg=_CARD,
                    fg=_TEXT,
                    font=(_FONT, 12),
                    wrap="word",
                    relief="flat",
                    bd=0,
                    highlightthickness=0,
                    yscrollcommand=scrollbar.set,
                )
                text_widget.pack(side="left", fill="both", expand=True)
                scrollbar.config(command=text_widget.yview)

                image_shell = tk.Frame(detail_body, bg=_CARD, width=_DETAIL_IMAGE_MAX_WIDTH + 24, padx=10, pady=8)
                image_shell.grid(row=0, column=1, sticky="nsew")
                image_shell.pack_propagate(False)
                image_status = tk.Label(
                    image_shell,
                    text="右侧会显示本次识别使用的截图",
                    bg=_CARD,
                    fg=_TEXT_MUTED,
                    anchor="w",
                    justify="left",
                    font=(_FONT, 10),
                    wraplength=_DETAIL_IMAGE_MAX_WIDTH,
                )
                image_status.pack(fill="x", pady=(0, 8))
                image_label = tk.Label(
                    image_shell,
                    text="等待截图",
                    bg="#e8e8e8",
                    fg=_TEXT_MUTED,
                    anchor="center",
                    justify="center",
                    font=(_FONT, 11),
                    width=1,
                    height=1,
                )
                image_label.pack(fill="both", expand=True)
                close_detail.bind("<Button-1>", lambda _e: panel.destroy())
                detail_state["window"] = panel
                detail_state["text_widget"] = text_widget
                detail_state["image_label"] = image_label
                detail_state["image_status"] = image_status
                _update_detail_text_widget()
                _update_detail_image()

            def _on_detail_click(_event: tk.Event) -> None:
                try:
                    show_detail_panel()
                except Exception as exc:
                    self.last_error = f"detail click failed: {exc}"

            detail_btn.bind("<Button-1>", _on_detail_click)

            # Content + grip row
            body = tk.Frame(inner, bg=_BG)
            body.pack(fill="both", expand=True)
            close_btn.lift()
            config_btn.lift()
            detail_btn.lift()

            content = tk.Frame(body, bg=_BG, padx=10, pady=6)
            content.pack(side="top", fill="both", expand=True)

            # Resize grip
            grip = tk.Label(
                body, text="⌟", bg=_BG, fg=_BORDER_FOCUS,
                font=(_FONT, 14), anchor="se", padx=8, pady=2,
            )
            grip.pack(side="bottom", fill="x")

            # --- layout state ---
            prefs = _load_prefs(self._prefs_path)
            layout = {
                "width": prefs["width"],
                "height": prefs["height"],
                "font_size": prefs["font_size"],
                "display_mode": prefs["display_mode"],
                "panel_mode": prefs["panel_mode"],
                "strategy_preset": prefs["strategy_preset"],
            }
            drag_state = {"offset_x": 0, "offset_y": 0, "x": 0, "y": 0, "manual": False}

            # ---------- Config mode UI ----------
            config_frame = tk.Frame(content, bg=_BG)
            session_state = {"started": False}

            btn_row = tk.Frame(config_frame, bg=_BG)
            btn_row.pack(pady=(16, 8))

            def _make_btn(parent, text, style):
                btn = tk.Label(
                    parent, text=text,
                    bg=_BTN_PRIMARY, fg="#ffffff",
                    font=(_FONT, 13, "bold"),
                    padx=20, pady=10, cursor="hand2",
                )
                btn.pack(side="left", padx=6)

                def _on_enter(_e):
                    btn.config(bg=_BTN_HOVER)

                def _on_leave(_e):
                    btn.config(bg=_BTN_PRIMARY)

                def _on_click(_e):
                    try:
                        if self._on_start is not None:
                            self._on_start(style, layout["strategy_preset"])
                    except Exception as exc:
                        self.last_error = f"button click failed: {exc}"
                    session_state["started"] = True
                    show_strategy_mode()

                btn.bind("<Enter>", _on_enter)
                btn.bind("<Leave>", _on_leave)
                btn.bind("<Button-1>", _on_click)
                return btn

            _make_btn(btn_row, "立直（门清憋大牌）", "riichi")
            _make_btn(btn_row, "快攻（积极副露）", "fast")

            tk.Label(
                config_frame,
                text="策略力度",
                bg=_BG,
                fg=_TEXT_MUTED,
                font=(_FONT, 10, "bold"),
            ).pack(pady=(0, 2))
            strategy_preset_row = tk.Frame(config_frame, bg=_BG)
            strategy_preset_row.pack(pady=(0, 8))

            def _refresh_strategy_preset_buttons() -> None:
                for child in strategy_preset_row.winfo_children():
                    selected = getattr(child, "_overlay_strategy_preset", "") == layout["strategy_preset"]
                    child.config(bg=_BTN_PRIMARY if selected else _CARD, fg="#ffffff" if selected else _TEXT)

            def _make_strategy_preset_btn(parent, text, preset):
                btn = tk.Label(
                    parent,
                    text=text,
                    bg=_CARD,
                    fg=_TEXT,
                    font=(_FONT, 11, "bold"),
                    padx=18,
                    pady=6,
                    cursor="hand2",
                )
                btn._overlay_strategy_preset = preset
                btn.pack(side="left", padx=6)

                def _on_click(_e):
                    layout["strategy_preset"] = preset
                    _save_prefs(
                        self._prefs_path,
                        layout["width"],
                        layout["height"],
                        layout["font_size"],
                        layout["display_mode"],
                        layout["panel_mode"],
                        layout["strategy_preset"],
                    )
                    _refresh_strategy_preset_buttons()

                btn.bind("<Button-1>", _on_click)
                return btn

            _make_strategy_preset_btn(strategy_preset_row, "简易策略 · 轻防守", "simple")
            _make_strategy_preset_btn(strategy_preset_row, "完整攻守 · 全风险", "standard")
            _refresh_strategy_preset_buttons()

            mode_row = tk.Frame(config_frame, bg=_BG)
            mode_row.pack(pady=(0, 10))

            def _refresh_mode_buttons() -> None:
                for child in mode_row.winfo_children():
                    selected = getattr(child, "_overlay_mode", "") == layout["display_mode"]
                    child.config(bg=_BTN_PRIMARY if selected else _CARD, fg="#ffffff" if selected else _TEXT)

            def _make_mode_btn(parent, text, mode):
                btn = tk.Label(
                    parent, text=text,
                    bg=_CARD, fg=_TEXT,
                    font=(_FONT, 11, "bold"),
                    padx=18, pady=6, cursor="hand2",
                )
                btn._overlay_mode = mode
                btn.pack(side="left", padx=6)

                def _on_click(_e):
                    layout["display_mode"] = mode
                    _save_prefs(
                        self._prefs_path,
                        layout["width"],
                        layout["height"],
                        layout["font_size"],
                        layout["display_mode"],
                        layout["panel_mode"],
                    )
                    _refresh_mode_buttons()

                btn.bind("<Button-1>", _on_click)
                return btn

            _make_mode_btn(mode_row, "简洁", "compact")
            _make_mode_btn(mode_row, "新手", "beginner")
            _refresh_mode_buttons()

            tk.Label(
                config_frame,
                text="展示内容",
                bg=_BG,
                fg=_TEXT_MUTED,
                font=(_FONT, 10, "bold"),
            ).pack(pady=(0, 2))
            panel_mode_row = tk.Frame(config_frame, bg=_BG)
            panel_mode_row.pack(pady=(0, 8))

            def _refresh_panel_mode_buttons() -> None:
                for child in panel_mode_row.winfo_children():
                    selected = getattr(child, "_overlay_panel_mode", "") == layout["panel_mode"]
                    child.config(bg=_BTN_PRIMARY if selected else _CARD, fg="#ffffff" if selected else _TEXT)

            def _make_panel_mode_btn(parent, text, mode):
                btn = tk.Label(
                    parent,
                    text=text,
                    bg=_CARD,
                    fg=_TEXT,
                    font=(_FONT, 11, "bold"),
                    padx=18,
                    pady=6,
                    cursor="hand2",
                )
                btn._overlay_panel_mode = mode
                btn.pack(side="left", padx=6)

                def _on_click(_e):
                    layout["panel_mode"] = mode
                    _save_prefs(
                        self._prefs_path,
                        layout["width"],
                        layout["height"],
                        layout["font_size"],
                        layout["display_mode"],
                        layout["panel_mode"],
                    )
                    _refresh_panel_mode_buttons()
                    if session_state["started"]:
                        show_strategy_mode()

                btn.bind("<Button-1>", _on_click)
                return btn

            _make_panel_mode_btn(panel_mode_row, "简洁主建议", "compact")
            _make_panel_mode_btn(panel_mode_row, "详细策略推理", "full")
            _refresh_panel_mode_buttons()

            tk.Label(
                config_frame,
                text=_CONFIG_STYLE_HINT,
                bg=_BG,
                fg=_TEXT_MUTED,
                font=(_FONT, 10),
            ).pack(pady=(2, 8))

            # ---------- Strategy mode UI ----------
            strategy_frame = tk.Frame(content, bg=_BG)

            def build_panel(parent, badge_text: str, badge_bg: str, badge_fg: str) -> tuple[tk.Frame, tk.Label, tk.Label]:
                panel = tk.Frame(parent, bg=_CARD, padx=10, pady=8)
                panel.pack_propagate(False)

                badge_frame = tk.Frame(panel, bg=_CARD)
                badge_frame.pack(fill="x", pady=(0, 4))

                badge = tk.Label(
                    badge_frame, text=badge_text,
                    bg=badge_bg, fg=badge_fg,
                    font=(_FONT, 9, "bold"),
                    anchor="w", padx=8, pady=1,
                )
                badge.pack(side="left")

                label = tk.Label(
                    panel, text="",
                    bg=_CARD, fg=_TEXT,
                    font=(_FONT, layout["font_size"]),
                    justify="left", anchor="nw",
                    wraplength=240,
                )
                label.pack(fill="both", expand=True)

                return panel, badge, label

            local_panel, local_badge, local_label = build_panel(strategy_frame, "本地", _BADGE_LOCAL_BG, _BADGE_LOCAL_FG)
            local_panel.pack(side="left", fill="both", expand=True)

            full_panel = tk.Frame(strategy_frame, bg=_BG, padx=10, pady=8)
            full_panel.pack_propagate(False)
            full_header = tk.Frame(full_panel, bg=_BG)
            full_header.pack(fill="x")
            full_posture = tk.Label(
                full_header,
                text="策略",
                bg=_WARNING,
                fg="#ffffff",
                font=(_FONT, 10, "bold"),
                padx=10,
                pady=4,
            )
            full_posture.pack(side="left", padx=(0, 12))
            full_action = tk.Label(full_header, text="等待策略", bg=_BG, fg=_TEXT, font=(_FONT, 22, "bold"), anchor="w")
            full_action.pack(side="left")
            full_risk = tk.Label(
                full_header,
                text="风险待评估",
                bg="#eaf6ff",
                fg=_TEXT,
                font=(_FONT, 10, "bold"),
                padx=10,
                pady=5,
            )
            full_risk.pack(side="right")
            full_summary = tk.Label(full_panel, text="", bg=_BG, fg=_TEXT_MUTED, font=(_FONT, 11, "bold"), anchor="w")
            full_summary.pack(fill="x", pady=(5, 0))
            full_explanation = tk.Label(full_panel, text="", bg=_BG, fg=_TEXT_MUTED, font=(_FONT, 10), anchor="w")
            full_explanation.pack(fill="x", pady=(3, 2))
            full_match_context = tk.Label(full_panel, text="", bg=_BG, fg=_BORDER_FOCUS, font=(_FONT, 9, "bold"), anchor="w", justify="left")
            full_match_context.pack(fill="x", pady=(0, 2))
            full_match_explanation = tk.Label(full_panel, text="", bg=_BG, fg=_TEXT_MUTED, font=(_FONT, 9), anchor="w", justify="left")
            full_match_explanation.pack(fill="x", pady=(0, 3))
            full_process = tk.Label(full_panel, text="", bg=_BG, fg=_BORDER_FOCUS, font=(_FONT, 10, "bold"), anchor="w")
            full_process.pack(fill="x", pady=(0, 3))
            full_risk_scale = tk.Label(full_panel, text="", bg=_BG, fg=_TEXT_MUTED, font=(_FONT, 9), anchor="w", justify="left")
            full_risk_scale.pack(fill="x", pady=(0, 2))
            full_risk_model = tk.Label(full_panel, text="", bg=_BG, fg=_TEXT_MUTED, font=(_FONT, 9), anchor="w", justify="left")
            full_risk_model.pack(fill="x", pady=(0, 2))
            full_risk_budget = tk.Label(full_panel, text="", bg=_BG, fg=_BORDER_FOCUS, font=(_FONT, 9, "bold"), anchor="w", justify="left")
            full_risk_budget.pack(fill="x", pady=(0, 8))
            full_candidates_row = tk.Frame(full_panel, bg=_BG)
            full_candidates_row.pack(fill="both", expand=True)
            full_candidate_widgets: list[dict[str, Any]] = []
            for _index in range(3):
                candidate_frame = tk.Frame(full_candidates_row, bg=_CARD, padx=10, pady=7, highlightthickness=1, highlightbackground=_BORDER)
                candidate_frame.pack(side="left", fill="both", expand=True, padx=(0 if _index == 0 else 4, 0 if _index == 2 else 4))
                rank_label = tk.Label(candidate_frame, text="", bg=_CARD, fg=_TEXT_MUTED, font=(_FONT, 9, "bold"), anchor="w")
                rank_label.pack(fill="x")
                tile_label = tk.Label(candidate_frame, text="", bg=_CARD, fg=_TEXT, font=(_FONT, 17, "bold"), anchor="w")
                tile_label.pack(fill="x", pady=(3, 4))
                risk_label = tk.Label(candidate_frame, text="", bg=_CARD, fg=_TEXT, font=(_FONT, 10, "bold"), anchor="w")
                risk_label.pack(fill="x")
                shape_label = tk.Label(candidate_frame, text="", bg=_CARD, fg=_TEXT, font=(_FONT, 10), anchor="w")
                shape_label.pack(fill="x", pady=(3, 0))
                safety_label = tk.Label(candidate_frame, text="", bg=_CARD, fg=_TEXT_MUTED, font=(_FONT, 9), anchor="w", wraplength=160)
                safety_label.pack(fill="x", pady=(3, 0))
                tk.Frame(candidate_frame, bg=_BORDER, height=1).pack(fill="x", pady=(7, 5))
                why_label = tk.Label(candidate_frame, text="为什么", bg=_CARD, fg=_BORDER_FOCUS, font=(_FONT, 9, "bold"), anchor="w")
                why_label.pack(fill="x")
                safety_reason_label = tk.Label(candidate_frame, text="", bg=_CARD, fg=_TEXT_MUTED, font=(_FONT, 9), anchor="nw", justify="left", wraplength=250)
                safety_reason_label.pack(fill="x", pady=(4, 0))
                risk_heading_label = tk.Label(candidate_frame, text="风险怎么算", bg=_CARD, fg=_BORDER_FOCUS, font=(_FONT, 9, "bold"), anchor="w")
                risk_heading_label.pack(fill="x", pady=(4, 0))
                risk_reason_label = tk.Label(candidate_frame, text="", bg=_CARD, fg=_TEXT_MUTED, font=(_FONT, 9), anchor="nw", justify="left", wraplength=250)
                risk_reason_label.pack(fill="x", pady=(2, 0))
                shape_reason_label = tk.Label(candidate_frame, text="", bg=_CARD, fg=_TEXT_MUTED, font=(_FONT, 9), anchor="nw", justify="left", wraplength=250)
                shape_reason_label.pack(fill="x", pady=(4, 0))
                comparison_label = tk.Label(candidate_frame, text="", bg=_CARD, fg=_TEXT_MUTED, font=(_FONT, 9), anchor="nw", justify="left", wraplength=250)
                comparison_label.pack(fill="x", pady=(4, 0))
                verdict_label = tk.Label(candidate_frame, text="", bg="#ffffff", fg=_TEXT, font=(_FONT, 9, "bold"), anchor="w", justify="left", wraplength=250, padx=7, pady=5)
                verdict_label.pack(side="bottom", fill="x", pady=(6, 0))
                full_candidate_widgets.append(
                    {
                        "frame": candidate_frame,
                        "labels": (
                            rank_label,
                            tile_label,
                            risk_label,
                            shape_label,
                            safety_label,
                            why_label,
                            safety_reason_label,
                            risk_heading_label,
                            risk_reason_label,
                            shape_reason_label,
                            comparison_label,
                        ),
                        "rank": rank_label,
                        "tile": tile_label,
                        "risk": risk_label,
                        "shape": shape_label,
                        "safety": safety_label,
                        "safety_reason": safety_reason_label,
                        "risk_reason": risk_reason_label,
                        "shape_reason": shape_reason_label,
                        "comparison": comparison_label,
                        "verdict": verdict_label,
                    }
                )

            strategy_card_state: dict[str, Any] = {}

            def _render_full_strategy_card(card: dict[str, Any]) -> None:
                strategy_card_state.clear()
                strategy_card_state.update(card)
                posture_value = str(card.get("posture_value") or "")
                posture_bg = {"push": _SUCCESS, "mawashi": _WARNING, "fold": _DANGER}.get(posture_value, _BTN_PRIMARY)
                full_posture.config(text=str(card.get("posture") or "策略"), bg=posture_bg)
                full_action.config(text=f"重点分析  {str(card.get('focus_tile') or '等待牌面')}")
                full_summary.config(text=f"{str(card.get('posture_summary') or '')} · {str(card.get('shape_summary') or '')}")
                risk_ok = str(card.get("risk_status") or "") != "超预算"
                full_risk.config(
                    text=f"{str(card.get('risk_summary') or '')}  {'✓ 预算内' if risk_ok else '! 超预算'}",
                    bg="#eaf6ff" if risk_ok else "#fff1f2",
                    fg=_SUCCESS if risk_ok else _DANGER,
                )
                full_explanation.config(text=str(card.get("explanation") or ""))
                full_match_context.config(
                    text=f"场况：{str(card.get('match_context_summary') or '尚未稳定识别四家点数')}"
                )
                full_match_explanation.config(text=str(card.get("match_context_explanation") or ""))
                full_process.config(text=str(card.get("decision_process") or ""))
                full_risk_scale.config(
                    text=f"风险刻度：{str(card.get('risk_scale_note') or '')}；{str(card.get('risk_scale_legend') or '')}"
                )
                full_risk_model.config(text=str(card.get("risk_model_legend") or ""))
                full_risk_budget.config(text=f"本局允许线：{str(card.get('risk_budget_calculation') or '')}")
                candidates = [item for item in (card.get("candidates") or [])[:3] if isinstance(item, dict)]
                for index, widgets in enumerate(full_candidate_widgets):
                    candidate = candidates[index] if index < len(candidates) else {}
                    tone = str(candidate.get("tone") or "safe")
                    bg = "#eaf6ff" if tone == "primary" else ("#fff1f2" if tone == "danger" else _CARD)
                    border = _BORDER_FOCUS if tone == "primary" else (_DANGER if tone == "danger" else _BORDER)
                    widgets["frame"].config(bg=bg, highlightbackground=border)
                    for label in widgets["labels"]:
                        label.config(bg=bg)
                    widgets["rank"].config(text=f"方案{'ABC'[index]}  {'排序基准' if tone == 'primary' else '对照方案'}")
                    widgets["tile"].config(text=f"候选牌 {candidate.get('tile', '?')}")
                    widgets["risk"].config(
                        text=f"{candidate.get('risk_summary', '')} · {candidate.get('risk_delta_label', candidate.get('risk_status', ''))}",
                        fg=_DANGER if tone == "danger" else _TEXT,
                    )
                    widgets["shape"].config(text=str(candidate.get("shape_summary") or ""))
                    widgets["safety"].config(text=str(candidate.get("safety") or ""))
                    widgets["safety_reason"].config(text=str(candidate.get("safety_reason") or ""))
                    widgets["risk_reason"].config(text=str(candidate.get("risk_reason") or ""))
                    widgets["shape_reason"].config(text=str(candidate.get("shape_reason") or ""))
                    widgets["comparison"].config(text=str(candidate.get("comparison_reason") or ""))
                    widgets["verdict"].config(
                        text=str(candidate.get("tradeoff") or ""),
                        bg="#eaf6ff" if tone == "primary" else ("#fff1f2" if tone == "danger" else "#ffffff"),
                        fg=_BORDER_FOCUS if tone == "primary" else (_DANGER if tone == "danger" else _TEXT_MUTED),
                    )

            def _show_active_strategy_panel() -> None:
                local_panel.pack_forget()
                full_panel.pack_forget()
                if layout["panel_mode"] == "full" and strategy_card_state.get("candidates"):
                    full_panel.pack(fill="both", expand=True)
                else:
                    local_panel.pack(side="left", fill="both", expand=True)

            def _recompute_panels() -> None:
                w = layout["width"]
                h = layout["height"]
                padding = 30
                panel_h = max(60, h - 40)
                panel_w = max(80, w - padding)
                wrap = max(40, panel_w - 24)
                local_panel.config(width=panel_w, height=panel_h)
                full_panel.config(width=panel_w, height=panel_h)
                local_label.config(wraplength=wrap)

            def _set_geometry() -> None:
                w, h = layout["width"], layout["height"]
                if drag_state["manual"]:
                    x, y = drag_state["x"], drag_state["y"]
                else:
                    x, y = _overlay_geometry(root.winfo_screenwidth(), root.winfo_screenheight(), w, h)
                    drag_state["x"] = x
                    drag_state["y"] = y
                root.geometry(f"{w}x{h}+{x}+{y}")

            # --- drag to move ---
            def start_drag(event: tk.Event) -> None:
                drag_state["offset_x"] = int(event.x_root) - root.winfo_x()
                drag_state["offset_y"] = int(event.y_root) - root.winfo_y()

            def drag_window(event: tk.Event) -> None:
                drag_state["x"] = int(event.x_root) - drag_state["offset_x"]
                drag_state["y"] = int(event.y_root) - drag_state["offset_y"]
                drag_state["manual"] = True
                root.geometry(f"+{drag_state['x']}+{drag_state['y']}")

            drag_widgets = [
                shell,
                inner,
                header,
                content,
                config_frame,
                strategy_frame,
                local_panel,
                local_badge,
                local_label,
                full_panel,
                full_header,
                full_action,
                full_summary,
                full_explanation,
                full_match_context,
                full_match_explanation,
                full_process,
                full_risk_scale,
                full_risk_model,
                full_risk_budget,
            ]
            for widget in drag_widgets:
                widget.bind("<ButtonPress-1>", start_drag)
                widget.bind("<B1-Motion>", drag_window)

            # --- resize grip ---
            resize_state = {"sx": 0, "sy": 0, "sw": 0, "sh": 0}

            def start_resize(event: tk.Event) -> None:
                resize_state["sx"] = event.x_root
                resize_state["sy"] = event.y_root
                resize_state["sw"] = layout["width"]
                resize_state["sh"] = layout["height"]

            def do_resize(event: tk.Event) -> None:
                dw = event.x_root - resize_state["sx"]
                dh = event.y_root - resize_state["sy"]
                layout["width"] = max(_MIN_WIDTH, min(_MAX_WIDTH, resize_state["sw"] + dw))
                layout["height"] = max(_MIN_HEIGHT, min(_MAX_HEIGHT, resize_state["sh"] + dh))
                _recompute_panels()
                _set_geometry()

            def end_resize(_event: tk.Event) -> None:
                _save_prefs(
                    self._prefs_path,
                    layout["width"],
                    layout["height"],
                    layout["font_size"],
                    layout["display_mode"],
                    layout["panel_mode"],
                )

            grip.bind("<ButtonPress-1>", start_resize)
            grip.bind("<B1-Motion>", do_resize)
            grip.bind("<ButtonRelease-1>", end_resize)

            # --- scroll to adjust font size ---
            def scroll_font(event: tk.Event) -> None:
                delta = 1 if int(event.delta) > 0 else -1
                new_fs = max(16, min(32, layout["font_size"] + delta))
                if new_fs != layout["font_size"]:
                    layout["font_size"] = new_fs
                    local_label.config(font=(_FONT, new_fs))
                    _recompute_panels()
                    _save_prefs(
                        self._prefs_path,
                        layout["width"],
                        layout["height"],
                        layout["font_size"],
                        layout["display_mode"],
                        layout["panel_mode"],
                    )

            root.bind_all("<MouseWheel>", scroll_font)

            # --- action badge styling ---
            def _style_action_label(badge_label: tk.Label, text: str) -> None:
                for prefix, cfg in [
                    ("本地和牌", _ACTION_WIN),
                    ("本地立直", _ACTION_RIICHI),
                    ("本地鸣牌", _ACTION_CALL),
                    ("本地推进", _ACTION_PUSH),
                    ("本地兜牌", _ACTION_MAWASHI),
                    ("本地防守", _ACTION_DEFENSE),
                    ("操作窗口", _ACTION_GENERIC),
                ]:
                    if text.startswith(prefix):
                        label_text, bg, fg = cfg
                        badge_label.config(text=label_text, bg=bg, fg=fg)
                        return
                badge_label.config(text="本地", bg=_BADGE_LOCAL_BG, fg=_BADGE_LOCAL_FG)

            # --- apply text ---
            panel_text_state = {"compact": "Mahjong Coach", "full": "Mahjong Coach"}

            def apply_text(text: str | None = None, strategy_card_text: str | None = None) -> None:
                if text is not None:
                    panel_text_state["compact"] = str(text or "").strip() or "Mahjong Coach"
                if strategy_card_text is not None:
                    panel_text_state["full"] = str(strategy_card_text or "").strip() or panel_text_state["compact"]
                value = _overlay_panel_text(
                    layout["panel_mode"],
                    panel_text_state["compact"],
                    panel_text_state["full"],
                )
                first_line = value.split("\n")[0]
                _style_action_label(local_badge, first_line)
                local_label.config(text=value)
                _recompute_panels()
                if strategy_frame.winfo_ismapped():
                    _set_geometry()

            def apply_payload(
                text: str,
                strategy_card_text: str,
                strategy_card: dict[str, Any],
                detail: str,
                image_path: str = "",
            ) -> None:
                detail_state["text"] = str(detail or "").strip() or "等待识别详情"
                detail_state["image_path"] = str(image_path or "").strip()
                _update_detail_text_widget()
                _update_detail_image()
                _render_full_strategy_card(strategy_card)
                apply_text(text, strategy_card_text)
                if strategy_frame.winfo_ismapped():
                    _show_active_strategy_panel()

            apply_text("Mahjong Coach")

            # --- mode switching ---
            def show_config_mode() -> None:
                strategy_frame.pack_forget()
                config_frame.pack(fill="both", expand=True)
                config_btn.place_forget()
                detail_btn.place_forget()
                saved_w, saved_h = layout["width"], layout["height"]
                layout["width"] = _CONFIG_WIDTH
                layout["height"] = _CONFIG_HEIGHT
                _set_geometry()
                layout["width"], layout["height"] = saved_w, saved_h

            def show_strategy_mode() -> None:
                session_state["started"] = True
                config_frame.pack_forget()
                strategy_frame.pack(fill="both", expand=True)
                config_btn.place(relx=1.0, x=-30, y=5, anchor="ne")
                config_btn.lift()
                detail_btn.place(relx=1.0, x=-126, y=5, anchor="ne")
                detail_btn.lift()
                if layout["panel_mode"] == "full":
                    layout["width"] = max(1000, layout["width"])
                    layout["height"] = max(840, layout["height"])
                _show_active_strategy_panel()
                apply_text()
                _recompute_panels()
                _set_geometry()

            show_config_mode()

            # --- pump ---
            def pump() -> None:
                while True:
                    try:
                        item = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if item is None:
                        _save_prefs(
                            self._prefs_path,
                            layout["width"],
                            layout["height"],
                            layout["font_size"],
                            layout["display_mode"],
                            layout["panel_mode"],
                        )
                        root.destroy()
                        return
                    if isinstance(item, dict):
                        cmd = item.get("cmd")
                        if cmd == "show_config":
                            show_config_mode()
                        elif cmd == "show_strategy":
                            show_strategy_mode()
                        elif cmd == "show_detail":
                            show_detail_panel()
                        elif cmd == "update_payload":
                            apply_payload(
                                str(item.get("text") or ""),
                                str(item.get("strategy_card_text") or item.get("text") or ""),
                                dict(item.get("strategy_card") or {}),
                                str(item.get("detail") or ""),
                                str(item.get("image_path") or ""),
                            )
                        continue
                    apply_text(item)
                root.after(120, pump)

            root.after(120, pump)
            root.update_idletasks()
            root.deiconify()
            root.lift()
            root.update()
            self._ready_event.set()
            # 中文：冻结后的服务进程里 Tk mainloop 可能立即返回，因此由 Tk 线程主动泵送事件。
            # English: A frozen host may return from Tk mainloop immediately, so pump events on the Tk thread.
            while True:
                try:
                    if not root.winfo_exists():
                        break
                    root.update()
                except tk.TclError:
                    break
                time.sleep(0.015)
            if not self._stop_requested.is_set() and not self.last_error:
                self.last_error = "overlay window event loop exited unexpectedly"
        except Exception as exc:
            detail = str(exc).strip() or repr(exc)
            self.last_error = f"{type(exc).__name__}: {detail}"
            if not self._ready_event.is_set():
                self._startup_error = self.last_error
        except BaseException:
            fatal_exit = True
            raise
        finally:
            if root is not None:
                try:
                    if root.winfo_exists():
                        root.destroy()
                except Exception:
                    pass
            if not fatal_exit:
                self._ready_event.set()


# ---------- text formatting (unchanged logic) ----------

def overlay_text_from_payload(payload: dict[str, Any], *, prefs_path: Path | None = None) -> str:
    display_mode = _overlay_display_mode(payload, prefs_path=prefs_path)
    decision = payload.get("last_decision") if isinstance(payload.get("last_decision"), dict) else payload
    state = payload.get("round_state") if isinstance(payload.get("round_state"), dict) else payload.get("coach_state")
    if not isinstance(state, dict):
        state = {}
    decision_type = str(decision.get("decision_type") or "")
    if decision.get("action_required"):
        action_text = _action_overlay_text(decision_type, decision, display_mode=display_mode)
        perception = decision.get("perception") if isinstance(decision.get("perception"), dict) else {}
        strategy = perception.get("strategy") if isinstance(perception.get("strategy"), dict) else {}
        match_line = _compact_overlay_match_line(strategy, state)
        return f"{action_text}\n{match_line}" if match_line else action_text
    if decision_type == "settlement_candidate":
        return _format_overlay("检测到结算候选", "正在用下一帧复核", "上一局数据暂不刷新")
    if decision_type == "round_settlement":
        settlement = decision.get("perception", {}).get("settlement", {})
        kind = _settlement_kind_label(str(settlement.get("kind") or "unknown"))
        return _format_overlay(f"{kind}已确认", "上一局数据已冻结", "等待结算画面结束")
    if decision_type == "awaiting_next_round":
        return _format_overlay("等待下一局", "上一局数据仍保留", "新手牌稳定两帧后自动重开")
    if decision_type == "round_idle":
        return _format_overlay("等待下一局", "上一局已结束", "新手牌出现后自动重开")
    has_plan = state.get("local_direction") or state.get("local_plan") or state.get("current_plan") or state.get("opening_plan")
    if not has_plan and decision.get("decision_type") == "observe":
        return _waiting_hand_overlay(decision)
    riichi_players = [str(p).strip() for p in (state.get("riichi_players") or []) if str(p).strip()]
    defense_posture = str(state.get("defense_posture") or "")
    defense_labels = {
        "push": "本地推进",
        "mawashi": "本地兜牌",
        "fold": "本地防守",
    }
    label = defense_labels.get(defense_posture, "本地防守") if riichi_players else "本地"
    local_direction = str(state.get("local_direction") or "").strip()
    local_plan = str(state.get("local_plan") or state.get("current_plan") or state.get("opening_plan") or decision.get("suggestion") or "").strip()
    local_block = _strategy_overlay_block(
        label,
        local_direction,
        local_plan,
        _string_items(state.get("target_shapes")),
        _string_items(state.get("caution_points")),
        display_mode=display_mode,
    )
    if riichi_players:
        pressure_hint = (
            "有人立直：先守风险预算，再保留和牌路线"
            if defense_posture == "mawashi"
            else "有人立直，注意安全牌"
        )
        local_block = f"{pressure_hint}\n{local_block}"
    return local_block


def _compact_overlay_match_line(strategy: dict[str, Any], state: dict[str, Any]) -> str:
    context = strategy.get("table_context") if isinstance(strategy.get("table_context"), dict) else {}
    scores = context.get("scores") if isinstance(context.get("scores"), dict) else state.get("player_scores")
    ranks = context.get("ranks") if isinstance(context.get("ranks"), dict) else state.get("player_ranks")
    if not isinstance(scores, dict) or "self" not in scores:
        return ""
    ranks = ranks if isinstance(ranks, dict) else {}
    self_score = int(scores["self"])
    self_rank = int(context.get("self_rank") or ranks.get("self") or 0)
    higher = sorted(int(score) for score in scores.values() if int(score) > self_score)
    lower = sorted((int(score) for score in scores.values() if int(score) < self_score), reverse=True)
    gap_text = ""
    if higher:
        gap_text = f"｜距上一位{higher[0] - self_score:,}"
    elif lower:
        gap_text = f"｜领先下一位{self_score - lower[0]:,}"
    honba = context.get("honba_count", state.get("honba_count"))
    sticks = context.get("riichi_stick_count", state.get("table_riichi_stick_count"))
    return (
        f"场况：自己{self_score:,}（{self_rank or '?'}位）{gap_text}｜"
        f"本场{int(honba or 0)}｜供托{int(sticks or 0)}"
    )


def overlay_strategy_card_text_from_payload(payload: dict[str, Any], *, prefs_path: Path | None = None) -> str:
    """Render the full strategy card from the same decision snapshot as the dashboard."""
    compact = overlay_text_from_payload(payload, prefs_path=prefs_path)
    card = overlay_strategy_card_from_payload(payload)
    if not card:
        return compact
    lines = [
        f"{card['posture']} · {card['posture_summary']}",
        f"重点分析　{card['focus_tile']}",
        f"{card['shape_summary']}　　{card['risk_summary']}（{card['risk_status']}）",
        str(card["explanation"]),
        f"场况：{card['match_context_summary']}" if card.get("match_context_summary") else "",
        str(card.get("match_context_explanation") or ""),
        f"风险刻度：{card['risk_scale_note']}；{card['risk_scale_legend']}",
        str(card["risk_model_legend"]),
        f"本局允许线：{card['risk_budget_calculation']}",
        "方案比较（仅展示风险与牌型权衡）",
    ]
    for index, candidate in enumerate(card["candidates"]):
        lines.append(
            f"方案{'ABC'[index]}　候选牌{candidate['tile']}　{candidate['risk_summary']}（{candidate['risk_delta_label']}）　"
            f"{candidate['shape_summary']}"
        )
        lines.append(f"　安全：{candidate['safety_reason']}")
        lines.append(f"　风险：{candidate['risk_reason']}")
        lines.append(f"　成型：{candidate['shape_reason']}")
        lines.append(f"　比较：{candidate['comparison_reason']}")
        lines.append(f"　权衡：{candidate['tradeoff']}")
    return "\n".join(line for line in lines if line)


def overlay_strategy_card_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Build a renderer-neutral strategy card for both native overlay backends."""
    decision = payload.get("last_decision") if isinstance(payload.get("last_decision"), dict) else payload
    state = payload.get("round_state") if isinstance(payload.get("round_state"), dict) else payload.get("coach_state")
    if not isinstance(state, dict):
        state = {}
    perception = decision.get("perception") if isinstance(decision.get("perception"), dict) else {}
    strategy = perception.get("strategy") if isinstance(perception.get("strategy"), dict) else {}
    candidates = [
        item
        for item in (strategy.get("top_candidates") or strategy.get("candidates") or [])[:3]
        if isinstance(item, dict)
    ]
    if not candidates:
        return {}

    posture_value = str(strategy.get("posture") or state.get("defense_posture") or "")
    posture = _defense_posture_label(posture_value)
    budget = float(strategy.get("risk_budget") or state.get("defense_risk_budget") or 0.0)
    risk_scale_note = str(
        strategy.get("risk_scale_note")
        or "0–100规则型相对危险指数，不是放铳概率；数值越高越危险"
    )
    risk_scale_legend = str(
        strategy.get("risk_scale_legend")
        or "0–9极低｜10–39较低｜40–59中等｜60–79高｜80–100很高"
    )
    risk_model_legend = str(
        strategy.get("risk_model_legend")
        or "基准：现物0、全见5、筋30、壁42、字牌已见3/2/0–1枚=38/55/72、无筋幺九58、无筋2/8为72、无筋中张84、座位未知90；宝牌+14、宝牌周边+8，多家立直每多一家+4，总分封顶100"
    )
    risk_budget_calculation = str(
        strategy.get("risk_budget_calculation")
        or f"当前牌型与玩家风格计算出的可接受上限 = {budget:.0f}"
    )
    match_context_summary, match_context_explanation = _overlay_match_context(
        strategy,
        state,
    )
    potential = {
        "strong": "较强",
        "live": "仍有路线",
        "weak": "偏弱",
    }.get(str(strategy.get("win_potential") or ""), "待评估")
    best = candidates[0]
    best_risk = float(best.get("defense_risk") or 0.0)
    best_eligible = bool(best.get("safety_eligible", best_risk <= max(10.0, budget)))
    posture_summary = {
        "push": "攻中带守",
        "mawashi": "守中求和",
        "fold": "安全优先",
    }.get(posture_value, "攻守判断")
    explanation = {
        "push": "牌型收益较高，风险仍在当前允许线内。",
        "mawashi": "存在兼顾安全与成型的候选，和牌路线仍然保留。",
        "fold": "当前候选的牌型收益不足以覆盖高风险。",
    }.get(posture_value, "根据风险与牌型综合排序。")
    rendered_candidates: list[dict[str, Any]] = []
    for index, item in enumerate(candidates, start=1):
        risk = float(item.get("defense_risk") or 0.0)
        shanten = int(item.get("shanten", 8))
        shanten_text = "听牌" if shanten <= 0 else f"{shanten}向听"
        eligible = bool(item.get("safety_eligible", risk <= max(10.0, budget)))
        budget_delta = budget - risk
        effective_tiles = [
            _overlay_tile_label(str(tile))
            for tile in (item.get("effective_tiles") or [])[:6]
            if str(tile).strip()
        ]
        effective_text = f"；主要进张：{'、'.join(effective_tiles)}" if effective_tiles else ""
        shape_loss = int(item.get("shape_loss") or 0)
        safety_evidence = list(dict.fromkeys(
            str(detail).strip()
            for detail in (item.get("safety_evidence") or [])
            if str(detail).strip()
        ))
        visibility = item.get("visibility") if isinstance(item.get("visibility"), dict) else {}
        if not safety_evidence and str(visibility.get("summary") or "").strip():
            safety_evidence.append(str(visibility["summary"]).strip() + "。")
        evidence_text = "；".join(detail.rstrip("。；") for detail in safety_evidence)
        if evidence_text:
            evidence_text += "。"
        risk_level = str(item.get("risk_level") or _overlay_risk_level(risk))
        candidate_risk_scale_note = str(
            item.get("risk_scale_note")
            or risk_scale_note
        )
        risk_calculation = str(
            item.get("risk_calculation")
            or f"{str(item.get('safety') or '牌面风险')}对应模型得分 = {risk:.0f}/100"
        )
        risk_budget_explanation = str(item.get("risk_budget_explanation") or (
            f"{risk:.0f} ≤ 当前可接受上限{budget:.0f}，低{budget - risk:.0f}，可进入候选。"
            if budget_delta >= 0
            else f"{risk:.0f} > 当前可接受上限{budget:.0f}，超{abs(budget_delta):.0f}，位于普通候选区间外。"
        ))
        rendered_candidates.append(
            {
                "rank": index,
                "tile": _overlay_tile_label(str(item.get("tile") or "?")),
                "safety": str(item.get("safety") or "安全度未知"),
                "risk": risk,
                "risk_summary": f"风险{risk_level} {risk:.0f}/100",
                "risk_level": risk_level,
                "risk_status": "预算内" if eligible else "超预算",
                "risk_delta_label": (
                    f"低上限{budget_delta:.0f}"
                    if budget_delta >= 0
                    else f"超上限{abs(budget_delta):.0f}"
                ),
                "risk_scale_note": candidate_risk_scale_note,
                "risk_calculation": risk_calculation,
                "risk_budget_explanation": risk_budget_explanation,
                "risk_reason": f"{risk_calculation}。{risk_budget_explanation}",
                "shape_summary": f"{shanten_text} · 有效{int(item.get('effective_count') or 0)}张",
                "effective_count": int(item.get("effective_count") or 0),
                "shape_loss": shape_loss,
                "tone": "primary" if index == 1 else ("safe" if eligible else "danger"),
                "safety_reason": (
                    f"依据：{str(item.get('safety') or '安全度未知')}。"
                    f"{evidence_text}"
                ),
                "safety_evidence": safety_evidence,
                "visibility": visibility,
                "risk_by_player": dict(item.get("risk_by_player") or {}),
                "shape_reason": (
                    f"切后{shanten_text}，有效牌{int(item.get('effective_count') or 0)}张，"
                    f"牌型损失{shape_loss}{effective_text}。"
                ),
                "comparison_reason": "",
                "tradeoff": "",
            }
        )
    if posture_value == "mawashi" and rendered_candidates:
        best_candidate = rendered_candidates[0]
        safe_alt = next(
            (item for item in rendered_candidates[1:] if item["risk_status"] == "预算内"),
            None,
        )
        danger_alt = next(
            (item for item in rendered_candidates[1:] if item["risk_status"] == "超预算"),
            None,
        )
        tradeoffs: list[str] = []
        if safe_alt is not None:
            effective_gain = best_candidate["effective_count"] - safe_alt["effective_count"]
            if effective_gain > 0:
                tradeoffs.append(
                    f"方案A的{best_candidate['tile']}与{safe_alt['tile']}都在预算内；前者多保留{effective_gain}张有效牌"
                )
        if danger_alt is not None:
            effective_gain = danger_alt["effective_count"] - best_candidate["effective_count"]
            faster_text = f"多{effective_gain}张有效牌" if effective_gain > 0 else "牌效接近"
            tradeoffs.append(
                f"{danger_alt['tile']}虽{faster_text}，但风险{danger_alt['risk']:.0f}超预算"
            )
        if tradeoffs:
            explanation = "。".join(tradeoffs) + "。"
    if rendered_candidates:
        best_candidate = rendered_candidates[0]
        for candidate in rendered_candidates:
            risk_diff = candidate["risk"] - best_candidate["risk"]
            effective_diff = candidate["effective_count"] - best_candidate["effective_count"]
            if candidate["rank"] == 1:
                comparisons: list[str] = []
                safe_alt = next(
                    (item for item in rendered_candidates[1:] if item["risk_status"] == "预算内"),
                    None,
                )
                danger_alt = next(
                    (item for item in rendered_candidates[1:] if item["risk_status"] == "超预算"),
                    None,
                )
                if safe_alt is not None:
                    gain = candidate["effective_count"] - safe_alt["effective_count"]
                    comparisons.append(
                        f"相比{safe_alt['tile']}，风险高{candidate['risk'] - safe_alt['risk']:.0f}，"
                        f"但多保留{max(0, gain)}张有效牌"
                    )
                if danger_alt is not None:
                    comparisons.append(
                        f"相比{danger_alt['tile']}，少{max(0, danger_alt['effective_count'] - candidate['effective_count'])}张有效牌，"
                        f"但风险低{danger_alt['risk'] - candidate['risk']:.0f}并回到预算内"
                    )
                candidate["comparison_reason"] = "；".join(comparisons) + ("。" if comparisons else "")
                candidate["tradeoff"] = "排序基准：预算内且保留的和牌机会最多"
            else:
                risk_text = (
                    f"风险比方案A高{risk_diff:.0f}"
                    if risk_diff > 0
                    else f"风险比方案A低{abs(risk_diff):.0f}"
                )
                effective_text = (
                    f"有效牌比方案A多{effective_diff}张"
                    if effective_diff > 0
                    else f"有效牌比方案A少{abs(effective_diff)}张"
                )
                candidate["comparison_reason"] = f"{risk_text}；{effective_text}。"
                if candidate["risk_status"] == "超预算":
                    candidate["tradeoff"] = "牌效收益尚不足以覆盖超预算风险"
                elif candidate["risk"] < best_candidate["risk"] and candidate["effective_count"] < best_candidate["effective_count"]:
                    candidate["tradeoff"] = "风险更低，同时牺牲较多牌效"
                else:
                    candidate["tradeoff"] = "风险与牌效的综合排序低于方案A"
    best_shanten = int(best.get("shanten", 8))
    best_shanten_text = "听牌" if best_shanten <= 0 else f"{best_shanten}向听"
    return {
        "posture": posture,
        "posture_value": posture_value,
        "posture_summary": posture_summary,
        "potential": potential,
        "focus_tile": rendered_candidates[0]["tile"],
        "shape_summary": f"保持{best_shanten_text} · 有效{int(best.get('effective_count') or 0)}张",
        "risk": best_risk,
        "risk_budget": budget,
        "risk_summary": f"{str(best.get('risk_level') or _overlay_risk_level(best_risk))} {best_risk:.0f}/100 · 上限 {budget:.0f}",
        "risk_status": "预算内" if best_eligible else "超预算",
        "risk_scale_note": risk_scale_note,
        "risk_scale_legend": risk_scale_legend,
        "risk_model_legend": risk_model_legend,
        "risk_budget_calculation": risk_budget_calculation,
        "explanation": explanation,
        "match_context_summary": match_context_summary,
        "match_context_explanation": match_context_explanation,
        "decision_process": "分析顺序：先检查风险预算 → 再比较向听与有效牌 → 最后比较打点与巡目；本卡只解释权衡，不直接给出操作指令",
        "candidates": rendered_candidates,
    }


def _overlay_match_context(
    strategy: dict[str, Any],
    state: dict[str, Any],
) -> tuple[str, str]:
    context = strategy.get("table_context") if isinstance(strategy.get("table_context"), dict) else {}
    scores = context.get("scores") if isinstance(context.get("scores"), dict) else state.get("player_scores")
    ranks = context.get("ranks") if isinstance(context.get("ranks"), dict) else state.get("player_ranks")
    if not isinstance(scores, dict) or len(scores) != 4:
        return "", ""
    ranks = ranks if isinstance(ranks, dict) else {}
    labels = {
        "self": "自己",
        "left_opponent": "上家",
        "top_opponent": "对家",
        "right_opponent": "下家",
    }
    order = ("self", "left_opponent", "top_opponent", "right_opponent")
    score_parts: list[str] = []
    for player in order:
        if player not in scores:
            continue
        rank = int(ranks.get(player) or 0)
        tied = sum(1 for value in scores.values() if int(value) == int(scores[player])) > 1
        rank_text = f"并列{rank}位" if tied and rank else (f"{rank}位" if rank else "顺位待定")
        score_parts.append(f"{labels[player]} {int(scores[player]):,}（{rank_text}）")
    honba = context.get("honba_count", state.get("honba_count"))
    sticks = context.get("riichi_stick_count", state.get("table_riichi_stick_count"))
    stake_text = f"本场{int(honba or 0)}｜供托{int(sticks or 0)}"
    summary = "｜".join([*score_parts, stake_text])

    self_rank = int(context.get("self_rank") or ranks.get("self") or 0)
    gap_above = int(context.get("gap_above") or 0)
    lead_below = int(context.get("lead_below") or 0)
    budget_model = strategy.get("risk_budget_model") if isinstance(strategy.get("risk_budget_model"), dict) else {}
    placement_adjustment = float(budget_model.get("placement_adjustment") or 0.0)
    reward_bonus = int(
        budget_model.get("table_reward_bonus")
        or context.get("win_reward_bonus")
        or 0
    )
    reasons: list[str] = []
    if self_rank == 4 and gap_above:
        reasons.append(f"自己第4位，距第3位{gap_above:,}点，追分让可接受风险上限+{placement_adjustment:.0f}")
    elif self_rank in {1, 2} and lead_below and placement_adjustment < 0:
        reasons.append(f"自己第{self_rank}位，领先下一位{lead_below:,}点，顺位保护让可接受风险上限{placement_adjustment:.0f}")
    elif self_rank:
        reasons.append(f"当前第{self_rank}位，分差未触发额外攻守修正")
    if reward_bonus:
        reasons.append(
            f"本场{int(honba or 0)}带来{int(honba or 0) * 300:,}点、供托{int(sticks or 0)}带来{int(sticks or 0) * 1000:,}点，合计{reward_bonus:,}点计入和牌收益"
        )
    else:
        reasons.append("本场与供托均为0，和牌收益不追加桌面奖励")
    return summary, "；".join(reasons) + "。"


def _overlay_tile_label(tile: str) -> str:
    value = str(tile or "").strip().lower()
    if len(value) != 2:
        return value or "?"
    rank, suit = value[0], value[1]
    if suit in {"m", "p", "s"} and rank.isdigit():
        number = "5" if rank == "0" else rank
        suffix = {"m": "万", "p": "筒", "s": "索"}[suit]
        return f"{'赤' if rank == '0' else ''}{number}{suffix}"
    if suit == "z" and rank in "1234567":
        return {"1": "东", "2": "南", "3": "西", "4": "北", "5": "白", "6": "发", "7": "中"}[rank]
    return value


def overlay_detail_text_from_payload(payload: dict[str, Any]) -> str:
    decision = payload.get("last_decision") if isinstance(payload.get("last_decision"), dict) else payload
    state = payload.get("round_state") if isinstance(payload.get("round_state"), dict) else payload.get("coach_state")
    if not isinstance(state, dict):
        state = {}
    live = payload.get("live") if isinstance(payload.get("live"), dict) else {}
    perception = decision.get("perception") if isinstance(decision.get("perception"), dict) else {}
    hand = perception.get("hand") if isinstance(perception.get("hand"), dict) else {}
    meld = perception.get("meld") if isinstance(perception.get("meld"), dict) else {}
    action = perception.get("action") if isinstance(perception.get("action"), dict) else {}
    river = perception.get("river") if isinstance(perception.get("river"), dict) else {}
    strategy = perception.get("strategy") if isinstance(perception.get("strategy"), dict) else {}
    yakuman = perception.get("yakuman") if isinstance(perception.get("yakuman"), dict) else {}
    settlement = perception.get("settlement") if isinstance(perception.get("settlement"), dict) else {}
    targets = _string_items(state.get("target_shapes"))
    cautions = _string_items(state.get("caution_points"))
    direction = _direction_text(str(state.get("local_direction") or ""), str(state.get("local_plan") or state.get("current_plan") or ""), targets)
    keep = _keep_text(targets, str(state.get("local_plan") or state.get("current_plan") or ""))
    call = _call_text(cautions)
    discard = _discard_text(cautions, str(state.get("local_plan") or state.get("current_plan") or ""))
    efficiency = _efficiency_text(cautions)
    yaku = _yaku_text(targets)
    progress = _overlay_progress_text(decision, state)
    if progress.startswith("流程："):
        progress = progress[len("流程："):]
    frame_path = str(live.get("last_frame_path") or "").strip()
    window_title = str(live.get("last_window_title") or live.get("last_binding", {}).get("window_title") or "").strip()
    capture_source = str(live.get("last_capture_source") or "").strip()
    hand_tiles = [str(tile) for tile in (hand.get("hand_tiles") or decision.get("hand_tiles") or state.get("last_hand_tiles") or []) if str(tile).strip()]
    buttons = [str(button) for button in (decision.get("buttons") or []) if str(button).strip()]
    open_melds = int(meld.get("open_meld_count") or state.get("last_open_meld_count") or 0)
    river_reason = str(river.get("reason") or "未扫描").strip()
    parts = [
        f"截图依据：{frame_path or '等待截图'}",
        f"窗口：{window_title or '未绑定'}；来源：{capture_source or 'unknown'}",
        f"识别流程：{progress}",
        "识别逻辑：capture.py/capture_frame() → coach.py/analyze_frame()",
        (
            "结算逻辑：perception/settlement_detector.py/detect_settlement_path()；"
            f"阶段={str(settlement.get('phase') or state.get('settlement_phase') or 'playing')}；"
            f"类型={_settlement_kind_label(str(settlement.get('kind') or state.get('settlement_kind') or 'none'))}；"
            f"置信度={float(settlement.get('confidence') or state.get('settlement_confidence') or 0.0):.0%}"
        ),
        f"手牌逻辑：perception/fast_hand_path.py/detect_fast_hand_path()；结果={str(hand.get('reason') or '等待')}",
        f"副露逻辑：perception/meld_state.py/detect_meld_state_path()；结果={str(meld.get('reason') or '等待')}",
        f"按钮逻辑：perception/action_detector.py/detect_action_buttons_fast()；结果={str(action.get('source') or '等待')}",
        f"牌河逻辑：perception/river_state.py/detect_river_state_path()；结果={river_reason}",
        f"识别结果：手牌{len(hand_tiles)}张 { _brief_items('、'.join(hand_tiles), max_items=6) or '无' }",
        f"副露：{open_melds}组；来源={str(meld.get('reason') or '无')}",
        f"按钮：{('、'.join(buttons) if buttons else str(action.get('source') or '未发现关键按钮'))}",
        f"牌河/立直：{river_reason}",
        f"策略来源：{str(state.get('last_update_reason') or decision.get('decision_type') or 'unknown')}",
        (
            f"攻守姿态：{_defense_posture_label(str(strategy.get('posture') or state.get('defense_posture') or ''))}；"
            f"风险预算={float(strategy.get('risk_budget') or state.get('defense_risk_budget') or 0.0):.0f}"
        ),
        "策略功能：coach.py / build_round_plan()",
        f"目标写法：取 local_direction / 主线 = {direction}",
    ]
    if yaku:
        parts.append(f"役：来自目标形状里的役牌/已成役 = {yaku}")
    if keep:
        parts.append(f"先留这些：来自“保留”目标或策略摘要 = {keep}")
    if discard:
        parts.append(f"优先考虑打：来自牌效/路线/优先清理 = {discard}")
    if call:
        parts.append(f"吃碰杠规则：来自风险点里的“鸣牌” = {call}")
    if efficiency:
        parts.append(f"牌效依据：{efficiency}")
    defense_candidates = [
        item
        for item in (strategy.get("top_candidates") or strategy.get("candidates") or [])[:3]
        if isinstance(item, dict)
    ]
    for index, item in enumerate(defense_candidates, start=1):
        parts.append(
            f"防守候选#{index}：打{str(item.get('tile') or '?')}；"
            f"{str(item.get('safety') or '安全度未知')}；"
            f"危险={float(item.get('defense_risk') or 0.0):.0f}；"
            f"向听={int(item.get('shanten') or 0)}；"
            f"牌型损失={int(item.get('shape_loss') or 0)}；"
            f"有效牌={int(item.get('effective_count') or 0)}张"
            f"（变化{int(item.get('effective_count_delta') or 0):+d}）"
        )
    yakuman_routes = [item for item in (yakuman.get("routes") or []) if isinstance(item, dict)]
    if yakuman_routes:
        top = yakuman_routes[0]
        tsumo = top.get("tsumo_probability") if isinstance(top.get("tsumo_probability"), dict) else {}
        probability = float(tsumo.get("18") or 0.0)
        status = "估算完成" if yakuman.get("status") == "ready" else "后台估算中"
        parts.append(
            f"役满潜力：{str(top.get('label') or top.get('route') or '未知')}，"
            f"距离约{int(top.get('distance') or 0)}张；18巡自摸估算={probability:.3%}（{status}，不含对手放铳）"
        )
    if not any((keep, call, discard, efficiency, yaku)):
        parts.append("策略细节：等待稳定手牌后生成。")
    return _format_overlay(*parts)


def _settlement_kind_label(kind: str) -> str:
    return {
        "win": "和牌结算",
        "exhaustive_draw": "荒牌流局",
        "abortive_draw": "途中流局",
        "unknown": "小局结算",
        "none": "未检测",
    }.get(kind, "小局结算")


def _overlay_display_mode(payload: dict[str, Any], *, prefs_path: Path | None = None) -> str:
    explicit = payload.get("overlay_display_mode") or payload.get("display_mode")
    if explicit:
        return _normalize_display_mode(explicit)
    if prefs_path is not None:
        return _normalize_display_mode(_load_prefs(prefs_path).get("display_mode"))
    return "compact"


def _action_overlay_text(decision_type: str, decision: dict[str, Any], *, display_mode: str = "compact") -> str:
    if decision_type == "win_window":
        return _format_overlay("本地和牌", "先确认荣和 / 自摸", "这个优先级最高")
    if decision_type == "riichi_window":
        suggestion = str(decision.get("suggestion") or "").strip()
        return _format_overlay("本地立直", _riichi_action_line(suggestion), _riichi_reason_line(suggestion))
    if decision_type == "call_window":
        suggestion = str(decision.get("suggestion") or "").strip()
        if display_mode != "beginner":
            return _format_overlay("本地鸣牌", _clean_line(suggestion))
        return _format_overlay("本地鸣牌", _call_action_line(suggestion), _call_reason_line(suggestion))
    if decision_type == "defense_alert":
        perception = decision.get("perception") if isinstance(decision.get("perception"), dict) else {}
        strategy = perception.get("strategy") if isinstance(perception.get("strategy"), dict) else {}
        posture_value = str(strategy.get("posture") or "")
        posture = _defense_posture_label(posture_value)
        label = {
            "push": "本地推进 · 攻中带守",
            "mawashi": "本地兜牌 · 守中求和",
            "fold": "本地防守 · 全退",
        }.get(posture_value, f"本地防守 · {posture}")
        footer = {
            "push": "风险可控，继续推进和牌",
            "mawashi": "先满足安全预算，再保留向听与有效牌",
            "fold": "和牌收益不足以覆盖当前风险",
        }.get(posture_value, "安全度相近时优先保留牌型")
        return _format_overlay(label, str(decision.get("suggestion") or "有人立直，先防守"), footer)
    return _format_overlay("操作窗口", str(decision.get("suggestion") or decision.get("detail") or "先处理当前按钮"), "")


def _call_text(cautions: list[str]) -> str:
    for item in cautions:
        if item.startswith("鸣牌："):
            return _brief_items(_plain_text(item[3:]), max_items=4)
    return ""


def _defense_posture_label(value: str) -> str:
    return {"push": "推进", "mawashi": "兜牌", "fold": "全退"}.get(value, "观察")


def _strategy_overlay_block(
    label: str,
    direction: str,
    plan: str,
    targets: list[str],
    cautions: list[str],
    *,
    display_mode: str = "compact",
) -> str:
    direction = _direction_text(direction, plan, targets)
    beginner = display_mode == "beginner"
    lines = [f"目标：{_newbie_direction_text(direction)}" if beginner else f"方向：{direction}"]
    yaku = _yaku_text(targets)
    if yaku:
        lines.append(f"役：{yaku}")
    keep = _keep_text(targets, plan)
    if keep:
        lines.append(f"先留这些：{keep}" if beginner else f"留：{keep}")
    call = _call_text(cautions)
    if call:
        lines.append(f"吃碰杠规则：{_newbie_call_policy_text(call)}" if beginner else f"开：{call}")
    return _format_overlay(label, *lines)


def _direction_text(direction: str, plan: str, targets: list[str]) -> str:
    if direction:
        return _plain_text(direction)
    for item in targets:
        if item.startswith("主线："):
            return _clean_line(item)
    return _clean_line(plan) if plan else "继续观察"


def _newbie_direction_text(value: str) -> str:
    text = _plain_text(value)
    if not text:
        return "继续观察"
    replacements = {
        "牌效推进": "先让手牌更快听牌",
        "副露加速": "可以用吃碰加快速度",
        "副露进听优先": "吃碰后优先接近听牌",
        "副露一向听": "已经接近听牌",
        "副露听牌/收束": "目标是尽快听牌/和牌",
        "役牌速攻": "围绕役牌快速和牌",
        "断幺九/平和": "做断幺/平和这类速度型和牌",
        "七对子": "收对子，尽量不要吃碰",
        "清字牌": "先清理孤立字牌",
    }
    for old, new in replacements.items():
        if old in text:
            return new
    return text


def _newbie_call_policy_text(value: str) -> str:
    text = _plain_text(value)
    if not text:
        return "没有明显好处就先跳过"
    text = text.replace("鸣牌", "吃碰杠")
    text = text.replace("开", "点吃/碰/杠")
    text = text.replace("直接听牌", "马上能听牌")
    text = text.replace("进听", "更接近听牌")
    text = text.replace("主线", "当前目标")
    return text


def _keep_text(targets: list[str], plan: str) -> str:
    keep = _first_prefixed_value(targets, "保留：")
    if not keep:
        keep = _extract_after(plan, ("保留",), stop_markers=("，先", "；", "。"))
    return _brief_items(_plain_text(keep), max_items=3)


def _yaku_text(targets: list[str]) -> str:
    values: list[str] = []
    for prefix in ("已成役：", "可成役：", "役牌对子："):
        text = _first_prefixed_value(targets, prefix)
        if text:
            values.append(_plain_text(text))
    return _brief_items("、".join(values), max_items=2)


def _discard_text(cautions: list[str], plan: str) -> str:
    discard = _first_prefixed_value(cautions, "优先清理：")
    if not discard:
        for prefix in ("副露收束：", "路线选择：", "下一步：", "副露牌效：", "牌效："):
            value = _first_prefixed_value(cautions, prefix)
            if not value:
                continue
            discard = _extract_after(value, ("主线打", "优先看打", "打"), stop_markers=("后", "；", "，", "。"))
            if discard:
                break
    if not discard:
        discard = _extract_after(plan, ("先打", "先清", "打："), stop_markers=("，", "；", "。"))
    return _brief_items(_plain_text(discard), max_items=3)


def _efficiency_text(cautions: list[str]) -> str:
    for item in cautions:
        text = _plain_text(str(item or "").strip())
        if "牌效" not in text or "当前" not in text:
            continue
        for prefix in ("副露牌效：", "牌效："):
            if text.startswith(prefix):
                text = text[len(prefix) :]
                break
        if text.startswith("估算") and "，" in text:
            text = text.split("，", 1)[1]
        for marker in ("，已扣", "（"):
            if marker in text:
                text = text.split(marker, 1)[0]
        for stop in ("。", "\n"):
            if stop in text:
                text = text.split(stop, 1)[0]
        return text
    return ""


def _overlay_progress_text(decision: dict[str, Any], state: dict[str, Any]) -> str:
    perception = decision.get("perception") if isinstance(decision.get("perception"), dict) else {}
    hand = perception.get("hand") if isinstance(perception.get("hand"), dict) else {}
    meld = perception.get("meld") if isinstance(perception.get("meld"), dict) else {}
    action = perception.get("action") if isinstance(perception.get("action"), dict) else {}
    river = perception.get("river") if isinstance(perception.get("river"), dict) else {}
    settlement = perception.get("settlement") if isinstance(perception.get("settlement"), dict) else {}
    hand_reason = str(hand.get("reason") or "").strip()

    capture = "截图✓" if decision.get("engine_meta") or decision.get("decision_type") else "截图待"
    settlement_phase = str(settlement.get("phase") or state.get("settlement_phase") or "playing")
    if settlement_phase == "settlement_candidate":
        settlement_step = "结算复核"
    elif settlement_phase == "settlement_latched":
        settlement_step = "结算✓"
    elif settlement_phase == "awaiting_next_round":
        settlement_step = "等新局"
    else:
        settlement_step = "结算-"
    if hand_reason == "missing_hand_tile_templates":
        calibration = "校准!"
    elif hand_reason in {"image_missing", "image_path_missing"}:
        calibration = "校准待"
    else:
        calibration = "校准✓" if hand_reason or hand.get("ok") else "校准待"

    hand_count = len(hand.get("hand_tiles") or decision.get("hand_tiles") or state.get("last_hand_tiles") or [])
    accepted = sum(1 for item in (hand.get("raw_detections") or []) if isinstance(item, dict) and item.get("accepted"))
    if hand.get("ok") or hand_count:
        hand_step = f"手牌✓{hand_count or accepted}"
    elif hand_reason == "unstable_hand_count" and accepted:
        hand_step = f"手牌识{accepted}"
    elif hand_reason and hand_reason != "fingerprint_match":
        hand_step = "手牌!"
    else:
        hand_step = "手牌待"

    open_melds = int(meld.get("open_meld_count") or state.get("last_open_meld_count") or 0)
    if meld.get("ok") or open_melds:
        inferred = hand_reason.startswith("inferred_open_") and not meld.get("ok")
        meld_step = f"副露✓{open_melds}{'推' if inferred else ''}"
    elif meld.get("reason") in {"no_self_melds", "closed_hand_count_no_melds"}:
        meld_step = "副露-"
    elif meld.get("reason"):
        meld_step = "副露待"
    else:
        meld_step = "副露待"

    if decision.get("buttons"):
        action_step = "按钮✓"
    elif action.get("source") == "opening_hand_scan":
        action_step = "按钮-"
    elif action.get("source"):
        action_step = "按钮✓"
    else:
        action_step = "按钮待"

    if river.get("ok") or state.get("last_discard_piles"):
        river_step = "牌河✓"
    elif river.get("reason") in {"opening_skips_river_scan", "river_scan_not_due"}:
        river_step = "牌河-"
    else:
        river_step = "牌河待"

    has_plan = state.get("current_plan") or state.get("opening_plan") or state.get("local_plan") or decision.get("suggestion")
    strategy = "策略✓" if has_plan else "策略待"
    return (
        f"流程：{capture} {settlement_step} {calibration} {hand_step} "
        f"{meld_step} {action_step} {river_step} {strategy}"
    )


def _string_items(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _clean_line(value: str) -> str:
    text = " ".join(str(value or "").split()).strip()
    keep_separator = text.startswith("路线选择：")
    for prefix in ("主线：", "保留：", "对子：", "路线选择："):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    if not keep_separator:
        for separator in ("，", "；", ";", "。"):
            if separator in text:
                text = text.split(separator, 1)[0]
    return _plain_text(text)


def _format_overlay(*parts: str) -> str:
    lines = [_plain_text(" ".join(str(value or "").split())) for value in parts]
    return "\n".join(line for line in lines if line).strip() or "Mahjong Coach"


def _overlay_geometry(screen_width: int, screen_height: int, width: int, height: int) -> tuple[int, int]:
    x = max(0, int((screen_width - width) / 2))
    bottom_gap = max(160, int(screen_height * 0.19))
    y = max(28, screen_height - height - bottom_gap)
    return x, y


def _plain_text(value: str) -> str:
    text = str(value or "").strip()
    replacements = {
        "筒子占比很高": "筒子多",
        "万子占比很高": "万子多",
        "索子占比很高": "索子多",
        "保留同色块": "保留同色",
        "同色块": "同色",
        "做搭子": "找顺子",
        "先清": "先打",
        "不硬染": "别强做清一色",
        "进听": "听牌",
        "加速主线": "明显加速",
        "役牌碰/听牌/明显加速才开，其余跳过": "默认跳过，能听牌再开",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _brief_items(value: str, *, max_items: int) -> str:
    text = str(value or "").strip(" ，、")
    if not text:
        return ""
    parts = [item.strip() for item in text.replace("，", "、").replace(",", "、").split("、") if item.strip()]
    if len(parts) <= max_items:
        return "、".join(parts) if parts else text
    return f"{'、'.join(parts[:max_items])}（共{len(parts)}张）"


def _call_action_line(value: str) -> str:
    text = _plain_text(str(value or "").strip())
    if not text:
        return "建议跳过"
    for sep in ("。", "；", " 当前主线："):
        if sep in text:
            text = text.split(sep, 1)[0]
    text = _clean_line(text)
    if "跳过" in text:
        return "建议跳过"
    if "碰到可以开" in text or "对子可以碰" in text or "可以碰" in text:
        return "建议碰"
    if "能进听" in text or "明显加速" in text or "能推进" in text:
        return "看情况：能马上听牌/加速就点"
    if "吃碰" in text:
        return "看情况：只点有用的吃/碰/杠"
    return _brief_items(_newbie_call_policy_text(text), max_items=3) or "建议跳过"


def _call_reason_line(value: str) -> str:
    text = _plain_text(str(value or "").strip())
    if not text:
        return "只有能听牌或明显变快才点"
    if "跳过" in text:
        return "没有明确好处时，不要随便碰/吃"
    if "役牌" in text:
        return "役牌碰了通常有役，更容易和"
    if "断幺" in text or "中张" in text:
        return "速度手可以吃碰中张，但别破坏好形"
    if "染手" in text or "同色" in text:
        return "只吃碰同色或役牌，别偏离目标"
    return "只有能听牌、进听，或明显变快才点"


def _riichi_action_line(value: str) -> str:
    text = _plain_text(str(value or "").strip())
    if not text:
        return "先确认听牌和打点"
    for prefix in ("推荐立直", "可以立直", "谨慎立直", "可立直"):
        if text.startswith(prefix):
            discard = _extract_after(text, ("打",), stop_markers=("听", "，", "；", "。"))
            waits = _extract_after(text, ("听",), stop_markers=("，", "；", "。"))
            if discard and waits:
                return f"{prefix}：打{discard}听{_brief_items(waits, max_items=3)}"
            return prefix
    return _clean_line(text)


def _riichi_reason_line(value: str) -> str:
    text = _plain_text(str(value or "").strip())
    for marker in ("好形/枚数够", "待牌尚可", "愚形或枚数少"):
        if marker in text:
            return marker
    if "未拿到稳定手牌" in text:
        return "手牌识别不稳，谨慎确认"
    return "不确定就先别立"


def _first_prefixed_value(values: list[str], prefix: str) -> str:
    for item in values:
        if item.startswith(prefix):
            return item[len(prefix) :].strip()
    return ""


def _extract_after(text: str, keywords: tuple[str, ...], *, stop_markers: tuple[str, ...]) -> str:
    value = str(text or "")
    start = -1
    keyword_len = 0
    for keyword in keywords:
        index = value.find(keyword)
        if index >= 0 and (start < 0 or index < start):
            start = index
            keyword_len = len(keyword)
    if start < 0:
        return ""
    tail = value[start + keyword_len :].strip(" ：:")
    stop_positions = [tail.find(marker) for marker in stop_markers if tail.find(marker) >= 0]
    if stop_positions:
        tail = tail[: min(stop_positions)]
    return tail.strip()


def _waiting_hand_overlay(decision: dict[str, Any]) -> str:
    reason_codes = [str(item) for item in (decision.get("reason_codes") or []) if str(item).strip()]
    hand_reason = ""
    for code in reason_codes:
        if code.startswith("hand_"):
            hand_reason = code[len("hand_"):]
            break
    perception = decision.get("perception") if isinstance(decision.get("perception"), dict) else {}
    hand_perception = perception.get("hand") if isinstance(perception.get("hand"), dict) else {}
    if not hand_reason:
        hand_reason = str(hand_perception.get("reason") or "")
    accepted = sum(1 for item in (hand_perception.get("raw_detections") or []) if item.get("accepted"))
    occupied = sum(1 for item in (hand_perception.get("raw_detections") or []) if item.get("occupied"))

    if hand_reason == "missing_hand_tile_templates":
        return _format_overlay("等待手牌", "截图分辨率暂未匹配校准", "优先使用 16:9 游戏画面")
    if hand_reason in {"image_path_missing", "image_missing"}:
        return _format_overlay("等待手牌", "截图获取失败")
    if accepted > 0:
        return _format_overlay("等待手牌", f"已识别{accepted}张，继续确认稳定手牌", "副露后少张也会跟踪")
    if occupied > 0:
        return _format_overlay("等待手牌", f"检测到{occupied}个牌位，识别中", "保持牌桌无遮挡")
    return _format_overlay("等待手牌", "未检测到手牌区域", "确保雀魂窗口可见")
