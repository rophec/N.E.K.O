from __future__ import annotations

import json
import os
import queue
import threading
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
_ACTION_DEFENSE = ("防守", _DANGER, "#ffffff")
_ACTION_GENERIC = ("操作", _BTN_PRIMARY, "#ffffff")

# Resize bounds
_MIN_WIDTH = 280
_MIN_HEIGHT = 80
_MAX_WIDTH = 1000
_MAX_HEIGHT = 360

# Default size
_DEFAULT_WIDTH = 420
_DEFAULT_HEIGHT = 140

# Config mode (two buttons) compact size
_CONFIG_WIDTH = 640
_CONFIG_HEIGHT = 170
_DETAIL_WIDTH = 1160
_DETAIL_HEIGHT = 870
_DETAIL_IMAGE_MAX_WIDTH = 540
_DETAIL_IMAGE_MAX_HEIGHT = 740

# Prefs file
_PREFS_FILENAME = "overlay_prefs.json"


def _prefs_path() -> Path:
    base = os.environ.get("LOCALAPPDATA", "")
    if base:
        return Path(base) / "N.E.K.O" / "plugins" / "mahjong_coach" / "data" / _PREFS_FILENAME
    return Path(_PREFS_FILENAME)


def _load_prefs() -> dict[str, Any]:
    try:
        data = json.loads(_prefs_path().read_text(encoding="utf-8"))
        if isinstance(data, dict):
            w = int(data.get("width", _DEFAULT_WIDTH))
            h = int(data.get("height", _DEFAULT_HEIGHT))
            fs = int(data.get("font_size", _FONT_SIZE))
            display_mode = _normalize_display_mode(data.get("display_mode"))
            return {
                "width": max(_MIN_WIDTH, min(_MAX_WIDTH, w)),
                "height": max(_MIN_HEIGHT, min(_MAX_HEIGHT, h)),
                "font_size": max(16, min(32, fs)),
                "display_mode": display_mode,
            }
    except Exception:
        pass
    return {"width": _DEFAULT_WIDTH, "height": _DEFAULT_HEIGHT, "font_size": _FONT_SIZE, "display_mode": "compact"}


def _save_prefs(width: int, height: int, font_size: int, display_mode: str | None = None) -> None:
    try:
        path = _prefs_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        existing_mode = "compact"
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                existing_mode = _normalize_display_mode(existing.get("display_mode"))
        except Exception:
            pass
        path.write_text(
            json.dumps(
                {
                    "width": width,
                    "height": height,
                    "font_size": font_size,
                    "display_mode": _normalize_display_mode(display_mode or existing_mode),
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


class CoachOverlayController:
    def __init__(
        self,
        on_start: Callable[[str], None] | None = None,
        on_stop: Callable[[], None] | None = None,
    ) -> None:
        self._queue: queue.Queue[Any] = queue.Queue()
        self._thread: threading.Thread | None = None
        self.last_error = ""
        self._on_start = on_start
        self._on_stop = on_stop

    def start(self) -> bool:
        if self._thread is not None and self._thread.is_alive():
            return True
        self.last_error = ""
        self._thread = threading.Thread(target=self._run, name="MahjongCoachOverlay", daemon=True)
        self._thread.start()
        return True

    def update(self, text: str) -> None:
        if self._thread is None or not self._thread.is_alive():
            return
        self._queue.put(str(text or "").strip() or "Mahjong Coach")

    def update_payload(self, *, text: str, detail: str = "", image_path: str = "") -> None:
        if self._thread is None or not self._thread.is_alive():
            return
        self._queue.put({"cmd": "update_payload", "text": text, "detail": detail, "image_path": image_path})

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
        if self._thread is None:
            return
        self._queue.put(None)
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        try:
            import tkinter as tk
        except Exception as exc:
            self.last_error = f"tkinter unavailable: {exc}"
            return

        try:
            root = tk.Tk()
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
                inner, text="↩", bg=_BG, fg=_TEXT_MUTED,
                font=(_FONT, 11, "bold"), cursor="hand2",
            )
            config_btn.place(relx=1.0, x=-32, y=4, anchor="ne")

            detail_btn = tk.Label(
                inner, text="?", bg=_BG, fg=_TEXT_MUTED,
                font=(_FONT, 11, "bold"), cursor="hand2",
            )
            detail_btn.place(relx=1.0, x=-58, y=4, anchor="ne")

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
            prefs = _load_prefs()
            layout = {
                "width": prefs["width"],
                "height": prefs["height"],
                "font_size": prefs["font_size"],
                "display_mode": prefs["display_mode"],
            }
            drag_state = {"offset_x": 0, "offset_y": 0, "x": 0, "y": 0, "manual": False}

            # ---------- Config mode UI ----------
            config_frame = tk.Frame(content, bg=_BG)

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
                            self._on_start(style)
                    except Exception as exc:
                        self.last_error = f"button click failed: {exc}"

                btn.bind("<Enter>", _on_enter)
                btn.bind("<Leave>", _on_leave)
                btn.bind("<Button-1>", _on_click)
                return btn

            _make_btn(btn_row, "立直（门清憋大牌）", "riichi")
            _make_btn(btn_row, "快攻（积极副露）", "fast")

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
                    _save_prefs(layout["width"], layout["height"], layout["font_size"], layout["display_mode"])
                    _refresh_mode_buttons()

                btn.bind("<Button-1>", _on_click)
                return btn

            _make_mode_btn(mode_row, "简洁", "compact")
            _make_mode_btn(mode_row, "新手", "beginner")
            _refresh_mode_buttons()

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

            def _recompute_panels() -> None:
                w = layout["width"]
                h = layout["height"]
                padding = 30
                panel_h = max(60, h - 40)
                panel_w = max(80, w - padding)
                wrap = max(40, panel_w - 24)
                local_panel.config(width=panel_w, height=panel_h)
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

            drag_widgets = [shell, inner, header, content, config_frame, strategy_frame, local_panel, local_badge, local_label]
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
                _save_prefs(layout["width"], layout["height"], layout["font_size"], layout["display_mode"])

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
                    _save_prefs(layout["width"], layout["height"], layout["font_size"], layout["display_mode"])

            root.bind_all("<MouseWheel>", scroll_font)

            # --- action badge styling ---
            def _style_action_label(badge_label: tk.Label, text: str) -> None:
                for prefix, cfg in [
                    ("本地和牌", _ACTION_WIN),
                    ("本地立直", _ACTION_RIICHI),
                    ("本地鸣牌", _ACTION_CALL),
                    ("本地防守", _ACTION_DEFENSE),
                    ("操作窗口", _ACTION_GENERIC),
                ]:
                    if text.startswith(prefix):
                        label_text, bg, fg = cfg
                        badge_label.config(text=label_text, bg=bg, fg=fg)
                        return
                badge_label.config(text="本地", bg=_BADGE_LOCAL_BG, fg=_BADGE_LOCAL_FG)

            # --- apply text ---
            def apply_text(text: str) -> None:
                value = str(text or "").strip() or "Mahjong Coach"
                first_line = value.split("\n")[0]
                _style_action_label(local_badge, first_line)
                local_label.config(text=value)
                _recompute_panels()
                _set_geometry()

            def apply_payload(text: str, detail: str, image_path: str = "") -> None:
                detail_state["text"] = str(detail or "").strip() or "等待识别详情"
                detail_state["image_path"] = str(image_path or "").strip()
                _update_detail_text_widget()
                _update_detail_image()
                apply_text(text)

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
                config_frame.pack_forget()
                strategy_frame.pack(fill="both", expand=True)
                config_btn.place(relx=1.0, x=-32, y=4, anchor="ne")
                config_btn.lift()
                detail_btn.place(relx=1.0, x=-58, y=4, anchor="ne")
                detail_btn.lift()
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
                        _save_prefs(layout["width"], layout["height"], layout["font_size"], layout["display_mode"])
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
                                str(item.get("detail") or ""),
                                str(item.get("image_path") or ""),
                            )
                        continue
                    apply_text(item)
                root.after(120, pump)

            root.after(120, pump)
            root.mainloop()
        except Exception as exc:
            self.last_error = str(exc)


# ---------- text formatting (unchanged logic) ----------

def overlay_text_from_payload(payload: dict[str, Any]) -> str:
    display_mode = _overlay_display_mode(payload)
    decision = payload.get("last_decision") if isinstance(payload.get("last_decision"), dict) else payload
    state = payload.get("round_state") if isinstance(payload.get("round_state"), dict) else payload.get("coach_state")
    if not isinstance(state, dict):
        state = {}
    decision_type = str(decision.get("decision_type") or "")
    if decision.get("action_required"):
        return _action_overlay_text(decision_type, decision, display_mode=display_mode)
    if decision_type == "round_idle":
        return _format_overlay("等待下一局", "上一局已结束", "新手牌出现后自动重开")
    has_plan = state.get("local_direction") or state.get("local_plan") or state.get("current_plan") or state.get("opening_plan")
    if not has_plan and decision.get("decision_type") == "observe":
        return _waiting_hand_overlay(decision)
    riichi_players = [str(p).strip() for p in (state.get("riichi_players") or []) if str(p).strip()]
    label = "本地防守" if riichi_players else "本地"
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
        local_block = f"有人立直，注意安全牌\n{local_block}"
    return local_block


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
        f"手牌逻辑：perception/fast_hand_path.py/detect_fast_hand_path()；结果={str(hand.get('reason') or '等待')}",
        f"副露逻辑：perception/meld_state.py/detect_meld_state_path()；结果={str(meld.get('reason') or '等待')}",
        f"按钮逻辑：perception/action_detector.py/detect_action_buttons_fast()；结果={str(action.get('source') or '等待')}",
        f"牌河逻辑：perception/river_state.py/detect_river_state_path()；结果={river_reason}",
        f"识别结果：手牌{len(hand_tiles)}张 { _brief_items('、'.join(hand_tiles), max_items=6) or '无' }",
        f"副露：{open_melds}组；来源={str(meld.get('reason') or '无')}",
        f"按钮：{('、'.join(buttons) if buttons else str(action.get('source') or '未发现关键按钮'))}",
        f"牌河/立直：{river_reason}",
        f"策略来源：{str(state.get('last_update_reason') or decision.get('decision_type') or 'unknown')}",
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
    if not any((keep, call, discard, efficiency, yaku)):
        parts.append("策略细节：等待稳定手牌后生成。")
    return _format_overlay(*parts)


def _overlay_display_mode(payload: dict[str, Any]) -> str:
    explicit = payload.get("overlay_display_mode") or payload.get("display_mode")
    if explicit:
        return _normalize_display_mode(explicit)
    return _normalize_display_mode(_load_prefs().get("display_mode"))


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
        return _format_overlay("本地防守", str(decision.get("suggestion") or "有人立直，先防守"), "先看现物和安全牌")
    return _format_overlay("操作窗口", str(decision.get("suggestion") or decision.get("detail") or "先处理当前按钮"), "")


def _call_text(cautions: list[str]) -> str:
    for item in cautions:
        if item.startswith("鸣牌："):
            return _brief_items(_plain_text(item[3:]), max_items=4)
    return ""


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
    hand_reason = str(hand.get("reason") or "").strip()

    capture = "截图✓" if decision.get("engine_meta") or decision.get("decision_type") else "截图待"
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
    return f"流程：{capture} {calibration} {hand_step} {meld_step} {action_step} {river_step} {strategy}"


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
