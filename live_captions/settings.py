"""Persistent user settings (JSON file in the platform config directory)."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import platform
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

APP_DIR_NAME = "LiveCaptions"


@dataclass
class Settings:
    # speech recognition
    model: str = "small"
    language: str = ""            # "" = auto-detect
    backend: str = "auto"
    device: str = "auto"
    compute_type: str = "default"
    threads: int = 0
    beam_size: int = 1
    # translation
    target: str = ""              # "" = captions only
    translator: str = "auto"
    translate_partials: bool = True
    # audio
    input_device: str = ""        # "" = system default; otherwise device name or index
    step: float = 1.5
    max_segment: float = 12.0
    silence: float = 0.7
    min_segment: float = 3.0
    silence_threshold: float = 0.01
    commit_after: float = 5.0
    commit_margin: float = 1.0
    # window
    width: int = 900
    height: int = 0               # pixels; 0 = automatic (4 lines at the font size)
    pos_x: int = -1               # remembered window position; -1 = use `position`
    pos_y: int = -1
    font_size: int = 22
    font_family: str = ""         # "" = platform default
    opacity: float = 0.85
    history: int = 3
    position: str = "bottom"
    show_original: bool = True
    show_translation: bool = True

    ENGINE_FIELDS = ("model", "language", "backend", "device", "compute_type", "threads", "beam_size",
                     "target", "translator", "translate_partials", "input_device", "step", "max_segment",
                     "silence", "min_segment", "silence_threshold", "commit_after", "commit_margin")
    MODEL_FIELDS = ("model", "backend", "device", "compute_type", "threads", "beam_size")

    def copy(self) -> "Settings":
        return dataclasses.replace(self)

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def engine_changed(self, other: "Settings") -> bool:
        return any(getattr(self, k) != getattr(other, k) for k in self.ENGINE_FIELDS)

    def model_changed(self, other: "Settings") -> bool:
        return any(getattr(self, k) != getattr(other, k) for k in self.MODEL_FIELDS)

    def update(self, values: dict[str, Any]) -> None:
        names = {f.name: f.type for f in fields(self)}
        for k, v in values.items():
            if k in names and v is not None:
                setattr(self, k, v)


def config_dir() -> Path:
    system = platform.system()
    if system == "Windows":
        base = Path(os.environ.get("APPDATA") or Path.home())
    elif system == "Darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / APP_DIR_NAME


def settings_path() -> Path:
    override = os.environ.get("LIVE_CAPTIONS_CONFIG")
    return Path(override) if override else config_dir() / "settings.json"


def load_settings(path: Path | None = None) -> tuple[Settings, bool]:
    """Return (settings, existed)."""
    path = path or settings_path()
    s = Settings()
    if not path.exists():
        return s, False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            s.update(_coerce(s, data))
    except Exception:
        log.exception("could not read %s; using defaults", path)
        return s, False
    return s, True


def save_settings(s: Settings, path: Path | None = None) -> Path:
    path = path or settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(s.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)
    return path


def _coerce(template: Settings, data: dict[str, Any]) -> dict[str, Any]:
    """Cast JSON values to the dataclass field types; drop anything that does not fit."""
    out: dict[str, Any] = {}
    for f in fields(template):
        if f.name not in data:
            continue
        v = data[f.name]
        current = getattr(template, f.name)
        try:
            if isinstance(current, bool):
                out[f.name] = bool(v)
            elif isinstance(current, int):
                out[f.name] = int(v)
            elif isinstance(current, float):
                out[f.name] = float(v)
            elif isinstance(current, str):
                out[f.name] = "" if v is None else str(v)
        except (TypeError, ValueError):
            log.warning("ignoring bad setting %s=%r", f.name, v)
    return out
