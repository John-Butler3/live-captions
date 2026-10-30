"""Command line entry point."""

from __future__ import annotations

import argparse
import logging
import sys
import threading

from . import __version__
from .engine import Caption, CaptionEngine, EngineConfig
from .transcriber import MODEL_SIZES


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="live-captions",
        description="Floating live captions (and translation) using OpenAI Whisper.",
    )
    g = p.add_argument_group("speech recognition")
    g.add_argument("-m", "--model", default="small", metavar="SIZE",
                   help="Whisper model: tiny, base, small, medium, large-v3, ... (default: small). "
                        f"Known sizes: {', '.join(MODEL_SIZES)}")
    g.add_argument("-l", "--language", default=None, metavar="CODE",
                   help="spoken language code, e.g. en, de, ja (default: auto-detect)")
    g.add_argument("--backend", choices=("auto", "faster-whisper", "openai-whisper"), default="auto",
                   help="speech backend (default: faster-whisper if installed, else openai-whisper)")
    g.add_argument("--device", default="auto", help="cpu, cuda or auto (default: auto)")
    g.add_argument("--compute-type", default="default",
                   help="faster-whisper compute type: int8, int8_float16, float16, float32")
    g.add_argument("--threads", type=int, default=0, help="CPU threads for inference (0 = library default)")
    g.add_argument("--beam-size", type=int, default=1, help="beam size; 1 is fastest (default: 1)")

    g = p.add_argument_group("translation")
    g.add_argument("-t", "--translate", default=None, metavar="CODE",
                   help="translate captions into this language, e.g. en, es, fr")
    g.add_argument("--translator", choices=("auto", "argos", "whisper", "none"), default="auto",
                   help="argos = offline Argos Translate (any language); whisper = Whisper's built-in "
                        "speech translation (English only, replaces the original line)")
    g.add_argument("--final-only", action="store_true",
                   help="only translate finished sentences, not the live partial line (saves CPU)")

    g = p.add_argument_group("audio")
    g.add_argument("-i", "--input", default=None, metavar="DEVICE",
                   help="input device index or name substring (default: system default microphone)")
    g.add_argument("--list-devices", action="store_true", help="list audio input devices and exit")
    g.add_argument("-f", "--file", default=None, metavar="PATH",
                   help="caption an audio/video file instead of the microphone")
    g.add_argument("--fast", action="store_true", help="with --file: process as fast as possible, not real time")
    g.add_argument("--step", type=float, default=1.5, help="seconds between live updates (default: 1.5)")
    g.add_argument("--max-segment", type=float, default=12.0,
                   help="max seconds of speech per caption line (default: 12)")
    g.add_argument("--silence", type=float, default=0.7,
                   help="seconds of silence that ends a sentence (default: 0.7)")
    g.add_argument("--min-segment", type=float, default=3.0,
                   help="seconds before a short pause may end a sentence (default: 3)")
    g.add_argument("--silence-threshold", type=float, default=0.01,
                   help="RMS level treated as silence; raise in noisy rooms (default: 0.01)")

    g = p.add_argument_group("window")
    g.add_argument("--no-gui", action="store_true", help="print captions to the terminal instead of a window")
    g.add_argument("--width", type=int, default=900, help="window width in pixels (default: 900)")
    g.add_argument("--font-size", type=int, default=22, help="caption font size (default: 22)")
    g.add_argument("--font", default="Helvetica", help="caption font family")
    g.add_argument("--opacity", type=float, default=0.85, help="window opacity 0.3-1.0 (default: 0.85)")
    g.add_argument("--history", type=int, default=3, help="finished lines kept on screen (default: 3)")
    g.add_argument("--position", choices=("bottom", "top", "center"), default="bottom")
    g.add_argument("--hide-original", action="store_true", help="show only the translation")

    p.add_argument("-v", "--verbose", action="count", default=0, help="-v for info, -vv for debug")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose > 1 else logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    for noisy in ("faster_whisper", "filelock", "httpcore", "httpcore2", "httpx", "urllib3",
                  "huggingface_hub", "argostranslate", "stanza"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.list_devices:
        from .audio import list_devices
        print(list_devices())
        return 0

    if args.translate and args.language and args.translate == args.language:
        print("warning: --translate is the same as --language; nothing to translate", file=sys.stderr)

    engine_cfg = EngineConfig(
        language=args.language,
        target=args.translate,
        step=args.step,
        max_segment=args.max_segment,
        silence=args.silence,
        min_segment=args.min_segment,
        silence_threshold=args.silence_threshold,
        translate_partials=not args.final_only,
    )

    overlay = None
    if not args.no_gui:
        try:
            from .overlay import CaptionOverlay, OverlayConfig
        except ImportError as exc:  # tkinter missing
            print(f"error: Tkinter is not available ({exc}); install python3-tk or use --no-gui", file=sys.stderr)
            return 2
        overlay = CaptionOverlay(OverlayConfig(
            width=args.width, font_family=args.font, font_size=args.font_size,
            opacity=max(0.3, min(1.0, args.opacity)), history=args.history,
            show_original=not args.hide_original, position=args.position,
        ))

    engine_holder: dict[str, CaptionEngine] = {}

    def on_caption(cap: Caption) -> None:
        if overlay is not None:
            overlay.push(cap)
        else:
            _print_caption(cap)

    def on_status(text: str) -> None:
        if overlay is not None:
            overlay.status(text)
        else:
            print(f"[{text}]", file=sys.stderr)

    def start_engine() -> None:
        """Load models and start streaming (runs off the UI thread)."""
        from .audio import FileSource, MicrophoneSource
        from .transcriber import create_transcriber
        from .translator import create_translator

        try:
            on_status(f"loading Whisper '{args.model}'…")
            transcriber = create_transcriber(
                args.backend, args.model, device=args.device, compute_type=args.compute_type,
                threads=args.threads, beam_size=args.beam_size,
            )
            translator = create_translator(args.translator, args.translate)
            if args.translate:
                on_status(f"translating → {args.translate} ({translator.name})")
            if args.file:
                source = FileSource(args.file, realtime=not args.fast)
            else:
                source = MicrophoneSource(args.input)
        except SystemExit as exc:
            on_status(str(exc))
            if overlay is None:
                raise
            return
        except Exception as exc:
            logging.getLogger(__name__).exception("startup failed")
            on_status(f"startup failed: {exc}")
            if overlay is None:
                raise SystemExit(1)
            return
        engine = CaptionEngine(source, transcriber, translator, engine_cfg, on_caption, on_status)
        engine_holder["engine"] = engine
        on_status("listening…" if not args.file else f"playing {args.file}")
        engine.start()
        if overlay is None:
            # Terminal mode: block until the stream ends (file) or Ctrl-C.
            try:
                while not engine.finished:
                    engine.join(0.5)
            except KeyboardInterrupt:
                pass
            finally:
                engine.stop()
        elif args.file:
            # Keep the window open after a file finishes so the text can be read.
            while not engine.finished:
                engine.join(0.5)
            on_status("done — press Esc to close")

    if overlay is None:
        start_engine()
        return 0

    worker = threading.Thread(target=start_engine, name="startup", daemon=True)
    worker.start()
    overlay.on_close = lambda: engine_holder.get("engine") and engine_holder["engine"].stop()
    try:
        overlay.run()
    except KeyboardInterrupt:
        overlay.close()
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
