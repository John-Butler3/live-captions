"""Floating always-on-top caption window (Tkinter, no extra dependencies)."""

from __future__ import annotations

import queue
import tkinter as tk
import tkinter.font as tkfont
from dataclasses import dataclass
from typing import Callable, Optional

from .engine import Caption


@dataclass
class OverlayConfig:
    width: int = 900
    font_family: str = "Helvetica"
    font_size: int = 22
    opacity: float = 0.85
    history: int = 3                 # finalised lines kept on screen
    show_original: bool = True
    show_translation: bool = True
    position: str = "bottom"         # bottom | top | center
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

    def __init__(self, config: OverlayConfig, on_close: Optional[Callable[[], None]] = None):
        self.cfg = config
        self.on_close = on_close or (lambda: None)
        self._queue: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self._finals: list[Caption] = []
        self._partial: Optional[Caption] = None
        self._status = "loading model…"

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

    # -- internals (Tk thread) -----------------------------------------------

    def _drain(self) -> None:
        changed = False
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == "caption":
                    cap: Caption = payload  # type: ignore[assignment]
                    if cap.final:
                        self._finals.append(cap)
                        del self._finals[: -self.cfg.history]
                        self._partial = None
                    else:
                        self._partial = cap
                    self._status = ""
                elif kind == "status":
                    self._status = str(payload)
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
        self._fit_height()

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

    def _fit_height(self) -> None:
        self.root.update_idletasks()
        display_lines = self.text.count("1.0", "end", "displaylines")
        count = display_lines[0] if isinstance(display_lines, tuple) else (display_lines or 1)
        self.text.configure(height=max(1, int(count)))
        self.root.update_idletasks()
        self._apply_geometry(keep_anchor=True)

    def _place_initial(self) -> None:
        self.root.update_idletasks()
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        w = min(self.cfg.width, sw - 2 * 20)
        x = (sw - w) // 2
        self._x, self._y, self._w = x, 0, w
        self._anchor_bottom = self.cfg.position == "bottom"
        self.root.geometry(f"{w}x100+{x}+0")
        self.root.update_idletasks()
        h = self.root.winfo_reqheight()
        if self.cfg.position == "bottom":
            self._y = sh - self.cfg.margin - h
        elif self.cfg.position == "top":
            self._y = self.cfg.margin
        else:
            self._y = (sh - h) // 2
        self._apply_geometry()
        try:  # macOS needs alpha re-applied once the window exists
            self.root.attributes("-alpha", self.cfg.opacity)
        except tk.TclError:
            pass
        self.root.focus_force()

    def _apply_geometry(self, keep_anchor: bool = False) -> None:
        h = self.root.winfo_reqheight()
        if keep_anchor and self._anchor_bottom:
            bottom = self._y + self._last_h if hasattr(self, "_last_h") else self._y + h
            self._y = bottom - h
        self._last_h = h
        self.root.geometry(f"{self._w}x{h}+{self._x}+{self._y}")

    # -- mouse: drag to move, grip to resize ----------------------------------

    def _bind_mouse(self) -> None:
        for widget in (self.root, self.frame, self.text):
            widget.bind("<ButtonPress-1>", self._drag_start)
            widget.bind("<B1-Motion>", self._drag_move)
        self._grip.bind("<ButtonPress-1>", self._resize_start)
        self._grip.bind("<B1-Motion>", self._resize_move)

    def _drag_start(self, event):
        self._drag = (event.x_root - self._x, event.y_root - self._y)
        self.root.focus_force()  # so the +/-/Esc keys work after a click
        return "break"           # stop the Text widget from selecting text

    def _drag_move(self, event):
        dx, dy = self._drag
        self._x, self._y = event.x_root - dx, event.y_root - dy
        self._apply_geometry()
        return "break"

    def _resize_start(self, event) -> None:
        self._resize = (event.x_root, self._w)
        return "break"

    def _resize_move(self, event) -> None:
        x0, w0 = self._resize
        self._w = max(300, w0 + (event.x_root - x0))
        self._render()
        return "break"

    # -- menu / settings ------------------------------------------------------

    def _build_menu(self) -> tk.Menu:
        m = tk.Menu(self.root, tearoff=0)
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
        self._font.configure(size=size)
        self._font_partial.configure(size=size)
        self._font_small.configure(size=max(9, size - 10))
        self._render()

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

    def clear(self) -> None:
        self._finals.clear()
        self._partial = None
        self._render()
