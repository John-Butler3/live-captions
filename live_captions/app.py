"""Application controller: owns the overlay window and (re)starts the engine."""

from __future__ import annotations

import logging
import threading
from typing import Optional

from .engine import Caption, CaptionEngine, EngineConfig
from .settings import Settings, save_settings

log = logging.getLogger(__name__)


def engine_config(s: Settings) -> EngineConfig:
    return EngineConfig(
        language=s.language or None,
        target=s.target or None,
        step=s.step,
        max_segment=s.max_segment,
        silence=s.silence,
        silence_threshold=s.silence_threshold,
        min_segment=s.min_segment,
        translate_partials=s.translate_partials,
        commit_after=s.commit_after,
        commit_margin=s.commit_margin,
    )


class ModelCache:
    """Keeps the loaded Whisper model so changing the mic or language is instant."""

    def __init__(self):
        self._key = None
        self._transcriber = None

    def get(self, s: Settings, on_status):
        from .transcriber import create_transcriber

        key = tuple(getattr(s, k) for k in Settings.MODEL_FIELDS)
        if key != self._key:
            on_status(f"loading Whisper '{s.model}'…")
            self._transcriber = None  # free the old model before loading the next
            self._transcriber = create_transcriber(
                s.backend, s.model, device=s.device, compute_type=s.compute_type,
                threads=s.threads, beam_size=s.beam_size,
            )
            self._key = key
        return self._transcriber


class App:
    def __init__(self, settings: Settings, overlay, file: Optional[str] = None, realtime: bool = True):
        self.settings = settings
        self.overlay = overlay
        self.file = file
        self.realtime = realtime
        self.models = ModelCache()
        self._engine: Optional[CaptionEngine] = None
        self._lock = threading.Lock()
        self._gen = 0
        self._dialog = None
        overlay.on_close = self.shutdown
        overlay.on_settings = self.open_settings
        overlay.on_geometry = self._geometry_changed
        self._save_job = None

    # -- lifecycle ---------------------------------------------------------------

    def start(self) -> None:
        self._restart_async()

    def shutdown(self) -> None:
        self._gen += 1
        eng = self._engine
        if eng is not None:
            eng.stop()

    def open_settings(self) -> None:
        from .settings_dialog import SettingsDialog

        if self._dialog is not None and self._dialog.winfo_exists():
            self._dialog.lift()
            self._dialog.focus_force()
            return
        self._dialog = SettingsDialog(self.overlay.root, self.settings, self.apply_settings)

    def apply_settings(self, new: Settings) -> None:
        old = self.settings
        self.settings = new
        try:
            path = save_settings(new)
            log.info("settings saved to %s", path)
        except Exception:
            log.exception("could not save settings")
        self.overlay.apply_settings(new)
        if new.engine_changed(old):
            self._restart_async()

    def _geometry_changed(self, x: int, y: int, w: int, h: int) -> None:
        """Remember where the user put the window (saved shortly after the drag ends)."""
        s = self.settings
        s.pos_x, s.pos_y, s.width, s.height = x, y, w, h
        s.font_size = self.overlay.cfg.font_size
        if self._save_job is not None:
            self.overlay.root.after_cancel(self._save_job)
        self._save_job = self.overlay.root.after(1000, self._save_quiet)

    def _save_quiet(self) -> None:
        self._save_job = None
        try:
            save_settings(self.settings)
        except Exception:
            log.exception("could not save settings")

    # -- engine --------------------------------------------------------------------

    def _restart_async(self) -> None:
        self._gen += 1
        threading.Thread(target=self._restart, args=(self._gen,), name="engine-restart", daemon=True).start()

    def _restart(self, gen: int) -> None:
        with self._lock:
            if self._engine is not None:
                self._engine.stop()
                self._engine.join(3)
                self._engine = None
            if gen != self._gen:
                return  # a newer restart is queued
            s = self.settings
            try:
                transcriber = self.models.get(s, self.status)
                from .translator import create_translator
                translator = create_translator(s.translator, s.target or None)
                source = self._make_source(s)
            except SystemExit as exc:
                self.status(str(exc))
                return
            except Exception as exc:
                log.exception("engine start failed")
                self.status(f"could not start: {exc}")
                return
            if gen != self._gen:
                source.close()
                return
            engine = CaptionEngine(source, transcriber, translator, engine_config(s), self.caption, self.status)
            self._engine = engine
            engine.start()
            self.status(self._listening_text(s, translator))
            if self.file:
                while not engine.finished and gen == self._gen:
                    engine.join(0.5)
                if gen == self._gen:
                    self.status("done — press Esc to close")

    def _make_source(self, s: Settings):
        from .audio import FileSource, MicrophoneSource

        if self.file:
            return FileSource(self.file, realtime=self.realtime)
        return MicrophoneSource(s.input_device or None)

    @staticmethod
    def _listening_text(s: Settings, translator) -> str:
        from .languages import language_name

        parts = [f"listening ({s.model})"]
        if s.language:
            parts.append(language_name(s.language))
        if s.target:
            parts.append(f"→ {language_name(s.target)}")
        return " · ".join(parts)

    # -- callbacks (any thread) -----------------------------------------------

    def caption(self, cap: Caption) -> None:
        self.overlay.push(cap)

    def status(self, text: str) -> None:
        self.overlay.status(text)
