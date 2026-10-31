"""Command line entry point.

Settings are loaded from the user's settings file (written by the in-app
Settings window); any command line option given overrides the saved value for
this run.
"""

from __future__ import annotations

import argparse
import logging
import os
import platform
import sys

from . import __version__
from .engine import Caption
from .settings import Settings, load_settings, settings_path
from .transcriber import MODEL_SIZES

S = argparse.SUPPRESS
D = Settings()  # defaults for help text


def default_font() -> str:
    system = platform.system()
    if system == "Windows":
        return "Segoe UI"
    if system == "Darwin":
        return "Helvetica"
    return "DejaVu Sans"


def _windowed() -> bool:
    """True when running as a GUI executable without a console (PyInstaller --windowed)."""
    return getattr(sys, "frozen", False) and (sys.stdout is None or sys.stderr is None)


def _log_dir() -> str:
    if platform.system() == "Windows":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    elif platform.system() == "Darwin":
        base = os.path.expanduser("~/Library/Logs")
    else:
        base = os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~/.local/state")
    path = os.path.join(base, "LiveCaptions")
    os.makedirs(path, exist_ok=True)
    return path


def _redirect_streams_if_windowed() -> str | None:
    """Without a console, stdout/stderr are None; send them to a log file instead."""
    if not _windowed():
        return None
    log_path = os.path.join(_log_dir(), "live-captions.log")
    f = open(log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stdout or f
    sys.stderr = sys.stderr or f
    return log_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="live-captions",
        description="Floating live captions (and translation) using OpenAI Whisper. "
                    "Options override the saved settings (see the in-app Settings window).",
    )
    g = p.add_argument_group("speech recognition")
    g.add_argument("-m", "--model", default=S, metavar="SIZE",
                   help=f"Whisper model: tiny, base, small, medium, large-v3, ... (default: {D.model}). "
                        f"Known sizes: {', '.join(MODEL_SIZES)}")
    g.add_argument("-l", "--language", default=S, metavar="CODE",
                   help="spoken language code, e.g. en, de, ja (default: auto-detect)")
    g.add_argument("--backend", choices=("auto", "faster-whisper", "openai-whisper"), default=S,
                   help="speech backend (default: faster-whisper if installed, else openai-whisper)")
    g.add_argument("--device", default=S, help="cpu, cuda or auto (default: auto)")
    g.add_argument("--compute-type", default=S, dest="compute_type",
                   help="faster-whisper compute type: int8, int8_float16, float16, float32")
    g.add_argument("--threads", type=int, default=S, help="CPU threads for inference (0 = library default)")
    g.add_argument("--beam-size", type=int, default=S, dest="beam_size", help="beam size; 1 is fastest")

    g = p.add_argument_group("translation")
    g.add_argument("-t", "--translate", default=S, metavar="CODE", dest="target",
                   help="translate captions into this language, e.g. en, es, fr ('' to turn off)")
    g.add_argument("--translator", choices=("auto", "argos", "whisper", "none"), default=S,
                   help="argos = offline Argos Translate (any language); whisper = Whisper's built-in "
                        "speech translation (English only, replaces the original line)")
    g.add_argument("--final-only", action="store_false", default=S, dest="translate_partials",
                   help="only translate finished sentences, not the live partial line (saves CPU)")

    g = p.add_argument_group("audio")
    g.add_argument("-i", "--input", default=S, metavar="DEVICE", dest="input_device",
                   help="input device index or name (default: system default microphone)")
    g.add_argument("--list-devices", action="store_true", help="list audio input devices and exit")
    g.add_argument("-f", "--file", default=None, metavar="PATH",
                   help="caption an audio/video file instead of the microphone")
    g.add_argument("--fast", action="store_true", help="with --file: process as fast as possible, not real time")
    g.add_argument("--step", type=float, default=S, help=f"seconds between live updates (default: {D.step})")
    g.add_argument("--max-segment", type=float, default=S, dest="max_segment",
                   help=f"max seconds of speech per caption line (default: {D.max_segment:g})")
    g.add_argument("--silence", type=float, default=S,
                   help=f"seconds of silence that ends a sentence (default: {D.silence})")
    g.add_argument("--min-segment", type=float, default=S, dest="min_segment",
                   help=f"seconds before a short pause may end a sentence (default: {D.min_segment:g})")
    g.add_argument("--silence-threshold", type=float, default=S, dest="silence_threshold",
                   help=f"RMS level treated as silence; raise in noisy rooms (default: {D.silence_threshold})")

    g = p.add_argument_group("window")
    g.add_argument("--no-gui", action="store_true", help="print captions to the terminal instead of a window")
    g.add_argument("--settings", action="store_true", help="open the Settings window on start")
    g.add_argument("--width", type=int, default=S, help=f"window width in pixels (default: {D.width})")
    g.add_argument("--font-size", type=int, default=S, dest="font_size",
                   help=f"caption font size (default: {D.font_size})")
    g.add_argument("--font", default=S, dest="font_family", help="caption font family")
    g.add_argument("--opacity", type=float, default=S, help=f"window opacity 0.3-1.0 (default: {D.opacity})")
    g.add_argument("--history", type=int, default=S, help=f"finished lines kept on screen (default: {D.history})")
    g.add_argument("--position", choices=("bottom", "top", "center"), default=S)
    g.add_argument("--hide-original", action="store_false", default=S, dest="show_original",
                   help="show only the translation")

    p.add_argument("--config", default=None, metavar="PATH",
                   help=f"settings file (default: {settings_path()})")
    p.add_argument("--reset-settings", action="store_true", help="ignore the saved settings file")
    p.add_argument("-v", "--verbose", action="count", default=0, help="-v for info, -vv for debug")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def main(argv=None) -> int:
    log_path = _redirect_streams_if_windowed()
    args = build_parser().parse_args(argv)
    if log_path:
        # GUI-only executable: terminal modes make no sense, keep the window.
        args.no_gui = False
    logging.basicConfig(
        level=logging.DEBUG if args.verbose > 1 else logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    for noisy in ("faster_whisper", "filelock", "httpcore", "httpcore2", "httpx", "urllib3",
                  "huggingface_hub", "argostranslate", "stanza"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.config:
        os.environ["LIVE_CAPTIONS_CONFIG"] = args.config

    if args.list_devices:
        from .audio import list_devices
        devices = list_devices()
        if log_path:
            try:
                import tkinter as tk
                from tkinter import messagebox
                root = tk.Tk()
                root.withdraw()
                messagebox.showinfo("Live Captions – input devices", devices)
            except Exception:
                print(devices)
        else:
            print(devices)
        return 0

    # Saved settings, overridden by whatever was given on the command line.
    if args.reset_settings:
        settings, existed = Settings(), True
    else:
        settings, existed = load_settings()
    overrides = {k: v for k, v in vars(args).items() if k in settings.to_dict()}
    settings.update(overrides)
    if settings.target and settings.language and settings.target == settings.language:
        print("warning: translation target is the same as the spoken language", file=sys.stderr)

    if args.no_gui:
        return _run_terminal(settings, args)

    try:
        from .overlay import CaptionOverlay, OverlayConfig
    except ImportError as exc:  # tkinter missing
        print(f"error: Tkinter is not available ({exc}); install python3-tk or use --no-gui", file=sys.stderr)
        return 2
    from .app import App

    overlay = CaptionOverlay(OverlayConfig(
        width=settings.width, font_family=settings.font_family or default_font(),
        font_size=settings.font_size, opacity=max(0.3, min(1.0, settings.opacity)),
        history=settings.history, show_original=settings.show_original,
        show_translation=settings.show_translation, position=settings.position,
    ))
    app = App(settings, overlay, file=args.file, realtime=not args.fast)
    app.start()
    if args.settings or not existed:
        # First run: let the user pick the microphone and languages.
        overlay.root.after(300, app.open_settings)
    try:
        overlay.run()
    except KeyboardInterrupt:
        overlay.close()
    return 0


def _run_terminal(settings: Settings, args) -> int:
    from .app import engine_config
    from .audio import FileSource, MicrophoneSource
    from .engine import CaptionEngine
    from .transcriber import create_transcriber
    from .translator import create_translator

    def on_status(text: str) -> None:
        print(f"[{text}]", file=sys.stderr)

    on_status(f"loading Whisper '{settings.model}'…")
    transcriber = create_transcriber(
        settings.backend, settings.model, device=settings.device, compute_type=settings.compute_type,
        threads=settings.threads, beam_size=settings.beam_size,
    )
    translator = create_translator(settings.translator, settings.target or None)
    if settings.target:
        on_status(f"translating → {settings.target} ({translator.name})")
    if args.file:
        source = FileSource(args.file, realtime=not args.fast)
        on_status(f"playing {args.file}")
    else:
        source = MicrophoneSource(settings.input_device or None)
        on_status("listening…")
    engine = CaptionEngine(source, transcriber, translator, engine_config(settings), _print_caption, on_status)
    engine.start()
    try:
        while not engine.finished:
            engine.join(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
    return 0


def _print_caption(cap: Caption) -> None:
    if cap.final:
        sys.stdout.write("\r\033[K")
        if cap.text:
            print(f"[{cap.language or '??'}] {cap.text}")
        if cap.translation:
            print(f"   → {cap.translation}")
        sys.stdout.flush()
    else:
        line = cap.text or cap.translation
        if cap.translation and cap.text:
            line = f"{cap.text}  →  {cap.translation}"
        sys.stdout.write("\r\033[K… " + line[-150:])
        sys.stdout.flush()


if __name__ == "__main__":
    sys.exit(main())
