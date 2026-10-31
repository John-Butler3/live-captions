"""Floating always-on-top caption window (Tkinter, no extra dependencies).

The window has a fixed size: new lines scroll in at the bottom, the window
never grows or jumps. Drag anywhere to move it, drag the ◢ corner to resize.
"""

from __future__ import annotations

import queue
import tkinter as tk
import tkinter.font as tkfont
from dataclasses import dataclass
from typing import Callable, Optional

from .engine import Caption
from .settings import Settings

MIN_WIDTH, MIN_HEIGHT = 300, 60
AUTO_LINES = 4  # height used when no explicit height is set


@dataclass
class OverlayConfig:
    width: int = 900
    height: int = 0                  # pixels; 0 = room for AUTO_LINES lines at the font size
    font_family: str = "Helvetica"
    font_size: int = 22
    opacity: float = 0.85
    history: int = 3                 # finalised lines kept on screen
    show_original: bool = True
    show_translation: bool = True
    position: str = "bottom"         # bottom | top | center (used when pos_x/pos_y unset)
    pos_x: int = -1                  # remembered window position, -1 = use `position`
    pos_y: int = -1
    margin: int = 60
    fg: str = "#ffffff"
    fg_dim: str = "#b8b8b8"
    fg_partial: str = "#ffe680"
    bg: str = "#101010"


class CaptionOverlay:
    """Borderless, draggable, semi-transparent window that renders captions.

    Thread-safe: call ``push(caption)`` / ``status(text)`` from any thread; the
    Tk main loop drains the queue.
    """

    def __init__(self, config: OverlayConfig, on_close: Optional[Callable[[], None]] = None,
                 on_settings: Optional[Callable[[], None]] = None,
                 on_geometry: Optional[Callable[[int, int, int, int], None]] = None):
        self.cfg = config
        self.on_close = on_close or (lambda: None)
        self.on_settings = on_settings or (lambda: None)
        self.on_geometry = on_geometry or (lambda x, y, w, h: None)
        self._queue: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self._finals: list[Caption] = []
        self._partial: Optional[Caption] = None
        self._last_final_id = 0
        self._status = "loading model…"
        self._status_sticky = False

        self.root = tk.Tk()
        self.root.title("Live Captions")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        try:
            self.root.attributes("-alpha", config.opacity)
        except tk.TclError:
            pass
        self.root.configure(bg=config.bg)

        self._font = tkfont.Font(family=config.font_family, size=config.font_size)
        self._font_small = tkfont.Font(family=config.font_family, size=max(9, config.font_size - 10))
        self._font_partial = tkfont.Font(family=config.font_family, size=config.font_size, slant="italic")

        self.frame = tk.Frame(self.root, bg=config.bg, padx=16, pady=10)
        self.frame.pack(fill="both", expand=True)
        self.text = tk.Text(
            self.frame, bg=config.bg, fg=config.fg, bd=0, highlightthickness=0,
            wrap="word", font=self._font, cursor="fleur", state="disabled",
            width=1, height=1, padx=0, pady=0, spacing1=2, spacing3=6,
        )
        self.text.pack(fill="both", expand=True)
        self.text.tag_configure("orig", foreground=config.fg_dim if config.show_translation else config.fg)
        self.text.tag_configure("trans", foreground=config.fg)
        self.text.tag_configure("partial", foreground=config.fg_partial, font=self._font_partial)
        self.text.tag_configure("status", foreground="#808080", font=self._font_small)

        self._grip = tk.Label(self.root, text="◢", bg=config.bg, fg="#555555", cursor="sizing")
        self._grip.place(relx=1.0, rely=1.0, anchor="se")
        self._gear = tk.Label(self.root, text="⚙", bg=config.bg, fg="#777777", cursor="hand2",
                              font=tkfont.Font(family=config.font_family, size=13))
        self._gear.place(relx=1.0, rely=0.0, anchor="ne", x=-4, y=2)
        self._gear.bind("<ButtonPress-1>", lambda e: (self.on_settings(), "break")[1])
        self._gear.bind("<Enter>", lambda e: self._gear.configure(fg="#ffffff"))
        self._gear.bind("<Leave>", lambda e: self._gear.configure(fg="#777777"))

        self._bind_mouse()
        self._menu = self._build_menu()
        self.root.bind("<Escape>", lambda e: self.close())
        self.root.bind("<plus>", lambda e: self.change_font(2))
        self.root.bind("<equal>", lambda e: self.change_font(2))
        self.root.bind("<minus>", lambda e: self.change_font(-2))
        self.root.bind("<KP_Add>", lambda e: self.change_font(2))
        self.root.bind("<KP_Subtract>", lambda e: self.change_font(-2))
        self.root.bind("<Button-3>", self._popup)
        self.root.bind("<Button-2>", self._popup)  # macOS secondary click
        self.root.bind("<Control-Button-1>", self._popup)
        self.root.bind("<Double-Button-1>", lambda e: self.on_settings())
        self.root.bind("<comma>", lambda e: self.on_settings())

        self._place_initial()
        self._render()
        self.root.after(50, self._drain)

    # -- public API (any thread) ---------------------------------------------

    def push(self, caption: Caption) -> None:
        self._queue.put(("caption", caption))

    def status(self, text: str) -> None:
        self._queue.put(("status", text))

    def run(self) -> None:
        self.root.mainloop()

    def close(self) -> None:
        try:
            self.on_close()
        finally:
            try:
                self.root.destroy()
            except tk.TclError:
                pass

    @property
    def geometry(self) -> tuple[int, int, int, int]:
        return self._x, self._y, self._w, self._h

    # -- internals (Tk thread) -----------------------------------------------

    def _drain(self) -> None:
        changed = False
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == "caption":
                    cap: Caption = payload  # type: ignore[assignment]
                    if cap.final:
                        self._last_final_id = max(self._last_final_id, cap.id)
                        for i, old in enumerate(self._finals):
                            if old.id == cap.id:
                                self._finals[i] = cap  # translation arrived for a shown line
                                break
                        else:
                            self._finals.append(cap)
                            del self._finals[: -self.cfg.history]
                        if self._partial is not None and self._partial.id <= cap.id:
                            self._partial = None
                    elif cap.id > self._last_final_id:
                        cur = self._partial
                        if not cap.translation:
                            self._partial = cap  # fresh transcription always wins
                        elif cur is not None and cur.id == cap.id and cur.text == cap.text:
                            self._partial = cap  # translation for the text currently shown
                        # else: translation of an outdated partial; ignore
                    if not self._status_sticky:
                        self._status = ""
                elif kind == "status":
                    text = str(payload)
                    self._status_sticky = text.startswith("!")
                    self._status = text.lstrip("!")
                changed = True
        except queue.Empty:
            pass
        if changed:
            self._render()
        self.root.after(50, self._drain)

    def _render(self) -> None:
        t = self.text
        t.configure(state="normal")
        t.delete("1.0", "end")
        lines = 0
        for cap in self._finals:
            lines += self._insert_caption(cap, partial=False)
        if self._partial is not None:
            lines += self._insert_caption(self._partial, partial=True)
        if self._status:
            t.insert("end", self._status + "\n", "status")
            lines += 1
        if lines == 0:
            t.insert("end", "Listening…\n", "status")
        # Trim trailing newline so the box doesn't show an empty last line.
        if t.get("end-2c", "end-1c") == "\n":
            t.delete("end-2c", "end-1c")
        t.configure(state="disabled")
        t.see("end")  # newest line always visible at the bottom

    def _insert_caption(self, cap: Caption, partial: bool) -> int:
        n = 0
        show_orig = self.cfg.show_original or not cap.translation
        show_trans = self.cfg.show_translation and cap.translation
        if cap.text and show_orig:
            tag = "partial" if partial else "orig"
            self.text.insert("end", cap.text + "\n", tag)
            n += 1
        if show_trans:
            tag = "partial" if (partial and not (cap.text and show_orig)) else "trans"
            self.text.insert("end", cap.translation + "\n", tag)
            n += 1
        return n

    # -- geometry ---------------------------------------------------------------

    def _auto_height(self) -> int:
        line = self._font.metrics("linespace") + 8  # spacing1 + spacing3
        return AUTO_LINES * line + 2 * 10 + 4

    def _place_initial(self) -> None:
        self.root.update_idletasks()
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self._w = max(MIN_WIDTH, min(self.cfg.width, sw - 40))
        self._h = max(MIN_HEIGHT, self.cfg.height or self._auto_height())
        if self.cfg.pos_x >= 0 and self.cfg.pos_y >= 0 and self.cfg.pos_x < sw - 50 and self.cfg.pos_y < sh - 30:
            self._x, self._y = self.cfg.pos_x, self.cfg.pos_y
        else:
            self._x = (sw - self._w) // 2
            if self.cfg.position == "top":
                self._y = self.cfg.margin
            elif self.cfg.position == "center":
                self._y = (sh - self._h) // 2
            else:
                self._y = sh - self.cfg.margin - self._h
        self._apply_geometry()
        self.root.update_idletasks()
        try:  # macOS needs alpha re-applied once the window exists
            self.root.attributes("-alpha", self.cfg.opacity)
        except tk.TclError:
            pass
        self.root.focus_force()

    def _apply_geometry(self) -> None:
        self.root.geometry(f"{self._w}x{self._h}+{self._x}+{self._y}")

    def _geometry_changed(self) -> None:
        self.cfg.pos_x, self.cfg.pos_y = self._x, self._y
        self.cfg.width, self.cfg.height = self._w, self._h
        self.on_geometry(self._x, self._y, self._w, self._h)

    # -- mouse: drag to move, grip to resize ----------------------------------

    def _bind_mouse(self) -> None:
        for widget in (self.root, self.frame, self.text):
            widget.bind("<ButtonPress-1>", self._drag_start)
            widget.bind("<B1-Motion>", self._drag_move)
            widget.bind("<ButtonRelease-1>", self._drag_end)
        self._grip.bind("<ButtonPress-1>", self._resize_start)
        self._grip.bind("<B1-Motion>", self._resize_move)
        self._grip.bind("<ButtonRelease-1>", self._resize_end)

    def _drag_start(self, event):
        self._drag = (event.x_root - self._x, event.y_root - self._y)
        self._moved = False
        self.root.focus_force()  # so the +/-/Esc keys work after a click
        return "break"           # stop the Text widget from selecting text

    def _drag_move(self, event):
        dx, dy = self._drag
        self._x, self._y = event.x_root - dx, event.y_root - dy
        self._moved = True
        self._apply_geometry()
        return "break"

    def _drag_end(self, event):
        if getattr(self, "_moved", False):
            self._moved = False
            self._geometry_changed()
        return "break"

    def _resize_start(self, event):
        self._resize = (event.x_root, event.y_root, self._w, self._h)
        return "break"

    def _resize_move(self, event):
        x0, y0, w0, h0 = self._resize
        self._w = max(MIN_WIDTH, w0 + (event.x_root - x0))
        self._h = max(MIN_HEIGHT, h0 + (event.y_root - y0))
        self._apply_geometry()
        self._render()
        return "break"

    def _resize_end(self, event):
        self._geometry_changed()
        return "break"

    # -- menu / settings ------------------------------------------------------

    def _build_menu(self) -> tk.Menu:
        m = tk.Menu(self.root, tearoff=0)
        m.add_command(label="Settings…  (double-click)", command=lambda: self.on_settings())
        m.add_separator()
        m.add_command(label="Bigger text  (+)", command=lambda: self.change_font(2))
        m.add_command(label="Smaller text  (-)", command=lambda: self.change_font(-2))
        m.add_separator()
        self._var_orig = tk.BooleanVar(value=self.cfg.show_original)
        self._var_trans = tk.BooleanVar(value=self.cfg.show_translation)
        m.add_checkbutton(label="Show original", variable=self._var_orig, command=self._toggle_lines)
        m.add_checkbutton(label="Show translation", variable=self._var_trans, command=self._toggle_lines)
        m.add_separator()
        op = tk.Menu(m, tearoff=0)
        for pct in (100, 90, 80, 70, 60, 50):
            op.add_command(label=f"{pct}%", command=lambda p=pct: self.set_opacity(p / 100))
        m.add_cascade(label="Opacity", menu=op)
        m.add_command(label="Fit height to text size", command=self.fit_height)
        m.add_command(label="Clear", command=self.clear)
        m.add_separator()
        m.add_command(label="Quit  (Esc)", command=self.close)
        return m

    def _popup(self, event) -> None:
        try:
            self._menu.tk_popup(event.x_root, event.y_root)
        finally:
            self._menu.grab_release()

    def change_font(self, delta: int) -> None:
        size = max(10, min(72, self._font.cget("size") + delta))
        self._set_font_size(size)
        self._render()

    def _set_font_size(self, size: int) -> None:
        self._font.configure(size=size)
        self._font_partial.configure(size=size)
        self._font_small.configure(size=max(9, size - 10))
        self.cfg.font_size = size

    def fit_height(self) -> None:
        """Resize the window to AUTO_LINES lines at the current font size."""
        self._h = self._auto_height()
        self._apply_geometry()
        self._render()
        self._geometry_changed()

    def set_opacity(self, value: float) -> None:
        try:
            self.root.attributes("-alpha", value)
        except tk.TclError:
            pass

    def _toggle_lines(self) -> None:
        self.cfg.show_original = self._var_orig.get()
        self.cfg.show_translation = self._var_trans.get()
        if not self.cfg.show_original and not self.cfg.show_translation:
            self.cfg.show_original = True
            self._var_orig.set(True)
        self.text.tag_configure("orig", foreground=self.cfg.fg_dim if self.cfg.show_translation else self.cfg.fg)
        self._render()

    def apply_settings(self, s: Settings) -> None:
        """Apply the appearance part of the user settings at runtime."""
        self.cfg.opacity = s.opacity
        self.cfg.history = s.history
        self.cfg.show_original = s.show_original
        self.cfg.show_translation = s.show_translation
        if s.font_family:
            self.cfg.font_family = s.font_family
            for f in (self._font, self._font_partial, self._font_small):
                f.configure(family=s.font_family)
        self._set_font_size(s.font_size)
        self._var_orig.set(s.show_original)
        self._var_trans.set(s.show_translation)
        self.text.tag_configure("orig", foreground=self.cfg.fg_dim if self.cfg.show_translation else self.cfg.fg)
        self.set_opacity(s.opacity)
        sw = self.root.winfo_screenwidth()
        self._w = max(MIN_WIDTH, min(s.width, sw - 40))
        self._h = max(MIN_HEIGHT, s.height or self._auto_height())
        self._x = max(0, min(self._x, sw - self._w))
        self.cfg.width, self.cfg.height = s.width, s.height
        self._apply_geometry()
        del self._finals[: -self.cfg.history]
        self._render()

    def clear(self) -> None:
        self._finals.clear()
        self._partial = None
        self._last_final_id = 0
        self._render()
