"""Settings window: microphone, languages, model and appearance."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

from .languages import ARGOS_LANGUAGES, argos_available, sorted_languages
from .settings import Settings
from .transcriber import MODEL_SIZES

DEFAULT_DEVICE_LABEL = "System default"
AUTO_LABEL = "Auto-detect"
OFF_LABEL = "Off (captions only)"


def list_input_devices() -> list[str]:
    try:
        import sounddevice as sd

        names = []
        for info in sd.query_devices():
            if info["max_input_channels"] > 0 and info["name"] not in names:
                names.append(info["name"])
        return names
    except Exception:
        return []


class SettingsDialog(tk.Toplevel):
    def __init__(self, master: tk.Misc, settings: Settings, on_apply: Callable[[Settings], None],
                 translation_note: Optional[str] = None):
        super().__init__(master)
        self.title("Live Captions – Settings")
        self.resizable(False, False)
        self.attributes("-topmost", True)
        self._settings = settings
        self._on_apply = on_apply
        self._models = [m for m in MODEL_SIZES]
        if settings.model not in self._models:
            self._models.insert(0, settings.model)

        pad = {"padx": 8, "pady": 4}
        body = ttk.Frame(self, padding=12)
        body.grid(row=0, column=0, sticky="nsew")
        row = 0

        def label(text):
            nonlocal row
            ttk.Label(body, text=text).grid(row=row, column=0, sticky="w", **pad)

        # -- microphone -------------------------------------------------------
        label("Microphone")
        self._devices = list_input_devices()
        dev_values = [DEFAULT_DEVICE_LABEL] + self._devices
        self.var_device = tk.StringVar(value=self._device_label(settings.input_device))
        cb = ttk.Combobox(body, textvariable=self.var_device, values=dev_values, width=44)
        cb.grid(row=row, column=1, sticky="ew", **pad)
        ttk.Button(body, text="↻", width=3, command=self._refresh_devices).grid(row=row, column=2, **pad)
        self._device_box = cb
        row += 1

        # -- spoken language --------------------------------------------------
        label("Spoken language")
        self._src = [("", AUTO_LABEL)] + sorted_languages()
        self.var_source = tk.StringVar(value=self._name_for(self._src, settings.language))
        ttk.Combobox(body, textvariable=self.var_source, values=[n for _, n in self._src],
                     state="readonly", width=44).grid(row=row, column=1, sticky="ew", **pad)
        row += 1

        # -- translate into ---------------------------------------------------
        label("Translate into")
        if argos_available():
            targets = sorted_languages(ARGOS_LANGUAGES)
            note = translation_note or "Offline translation via Argos Translate (language packs download on first use)."
        else:
            targets = [("en", "English")]
            note = translation_note or ("Only English is available: Whisper translates speech to English itself. "
                                        "Install Argos Translate (or the full build) for other languages.")
        self._tgt = [("", OFF_LABEL)] + targets
        current_target = settings.target if settings.target in dict(self._tgt) else ""
        self.var_target = tk.StringVar(value=self._name_for(self._tgt, current_target))
        ttk.Combobox(body, textvariable=self.var_target, values=[n for _, n in self._tgt],
                     state="readonly", width=44).grid(row=row, column=1, sticky="ew", **pad)
        row += 1
        ttk.Label(body, text=note, wraplength=380, foreground="#666666").grid(
            row=row, column=1, columnspan=2, sticky="w", padx=8, pady=(0, 8))
        row += 1

        self.var_partials = tk.BooleanVar(value=settings.translate_partials)
        ttk.Checkbutton(body, text="Translate the live line too (uses more CPU)",
                        variable=self.var_partials).grid(row=row, column=1, columnspan=2, sticky="w", **pad)
        row += 1
        self.var_show_orig = tk.BooleanVar(value=settings.show_original)
        ttk.Checkbutton(body, text="Show the original text above the translation",
                        variable=self.var_show_orig).grid(row=row, column=1, columnspan=2, sticky="w", **pad)
        row += 1

        ttk.Separator(body).grid(row=row, column=0, columnspan=3, sticky="ew", pady=6)
        row += 1

        # -- model ------------------------------------------------------------
        label("Whisper model")
        self.var_model = tk.StringVar(value=settings.model)
        ttk.Combobox(body, textvariable=self.var_model, values=self._models, width=20).grid(
            row=row, column=1, sticky="w", **pad)
        row += 1
        ttk.Label(body, text="tiny = fastest, small = good balance, medium/large = accurate but slow on CPU. "
                             "Models download on first use.", wraplength=380, foreground="#666666").grid(
            row=row, column=1, columnspan=2, sticky="w", padx=8, pady=(0, 8))
        row += 1

        ttk.Separator(body).grid(row=row, column=0, columnspan=3, sticky="ew", pady=6)
        row += 1

        # -- appearance -------------------------------------------------------
        label("Font size")
        self.var_font = tk.IntVar(value=settings.font_size)
        ttk.Spinbox(body, from_=10, to=72, textvariable=self.var_font, width=6).grid(
            row=row, column=1, sticky="w", **pad)
        row += 1
        label("Window width")
        self.var_width = tk.IntVar(value=settings.width)
        ttk.Spinbox(body, from_=300, to=4000, increment=50, textvariable=self.var_width, width=6).grid(
            row=row, column=1, sticky="w", **pad)
        row += 1
        label("Opacity")
        self.var_opacity = tk.DoubleVar(value=settings.opacity)
        ttk.Scale(body, from_=0.3, to=1.0, variable=self.var_opacity, length=200).grid(
            row=row, column=1, sticky="w", **pad)
        row += 1
        label("Lines kept")
        self.var_history = tk.IntVar(value=settings.history)
        ttk.Spinbox(body, from_=1, to=10, textvariable=self.var_history, width=6).grid(
            row=row, column=1, sticky="w", **pad)
        row += 1

        # -- buttons ----------------------------------------------------------
        btns = ttk.Frame(body)
        btns.grid(row=row, column=0, columnspan=3, sticky="e", pady=(12, 0))
        ttk.Button(btns, text="Cancel", command=self.destroy).pack(side="right", padx=4)
        ttk.Button(btns, text="Apply", command=self._apply).pack(side="right", padx=4)
        ttk.Button(btns, text="OK", command=lambda: (self._apply(), self.destroy())).pack(side="right", padx=4)

        self.bind("<Escape>", lambda e: self.destroy())
        self.bind("<Return>", lambda e: (self._apply(), self.destroy()))
        self.update_idletasks()
        self._center_on_screen()
        self.lift()
        self.focus_force()

    # -- helpers ----------------------------------------------------------------

    def _center_on_screen(self) -> None:
        w, h = self.winfo_reqwidth(), self.winfo_reqheight()
        x = (self.winfo_screenwidth() - w) // 2
        y = (self.winfo_screenheight() - h) // 2
        self.geometry(f"+{x}+{y}")

    @staticmethod
    def _name_for(pairs, code):
        for c, n in pairs:
            if c == code:
                return n
        return pairs[0][1]

    @staticmethod
    def _code_for(pairs, name):
        for c, n in pairs:
            if n == name:
                return c
        return ""

    def _device_label(self, value: str) -> str:
        if not value:
            return DEFAULT_DEVICE_LABEL
        for name in self._devices:
            if name == value or value.lower() in name.lower():
                return name
        return value  # unknown / unplugged device: keep the text

    def _refresh_devices(self) -> None:
        self._devices = list_input_devices()
        self._device_box["values"] = [DEFAULT_DEVICE_LABEL] + self._devices

    def _apply(self) -> None:
        s = self._settings.copy()
        dev = self.var_device.get().strip()
        s.input_device = "" if dev in ("", DEFAULT_DEVICE_LABEL) else dev
        s.language = self._code_for(self._src, self.var_source.get())
        s.target = self._code_for(self._tgt, self.var_target.get())
        s.translate_partials = bool(self.var_partials.get())
        s.show_original = bool(self.var_show_orig.get())
        s.model = self.var_model.get().strip() or s.model
        try:
            s.font_size = max(10, min(72, int(self.var_font.get())))
            s.width = max(300, int(self.var_width.get()))
            s.history = max(1, int(self.var_history.get()))
            s.opacity = round(max(0.3, min(1.0, float(self.var_opacity.get()))), 2)
        except (tk.TclError, ValueError):
            pass
        self._settings = s
        self._on_apply(s)
