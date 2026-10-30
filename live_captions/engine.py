"""Streaming engine: turns a stream of audio blocks into caption events.

Strategy (the usual "rolling window" approach for Whisper, which is not a true
streaming model):

* Audio blocks are appended to the current segment buffer.
* Every ``step`` seconds of new audio, the whole buffer is transcribed and the
  result is published as a *partial* caption (it may change as more context
  arrives).
* When ``silence`` seconds of quiet follow speech, or the buffer reaches
  ``max_segment`` seconds, the segment is finalised: transcribed one last time,
  translated, published as a *final* caption, and the buffer is cleared.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .audio import SAMPLE_RATE, AudioSource
from .transcriber import Transcriber
from .translator import Translator

log = logging.getLogger(__name__)


@dataclass
class Caption:
    text: str                 # transcript in the spoken language ("" in whisper-translate mode)
    translation: str = ""     # translated text, if any
    language: Optional[str] = None
    final: bool = False
    timestamp: float = field(default_factory=time.time)


@dataclass
class EngineConfig:
    language: Optional[str] = None       # force source language, None = auto-detect
    target: Optional[str] = None         # translation target, None = captions only
    step: float = 1.5                    # seconds of new audio between partial updates
    max_segment: float = 12.0            # hard limit of a segment in seconds
    silence: float = 0.7                 # seconds of silence that ends a segment
    min_segment: float = 3.0             # don't end on a short pause before this many seconds
    silence_threshold: float = 0.01      # RMS below this counts as silence
    min_speech: float = 0.3              # ignore segments shorter than this
    translate_partials: bool = True


class CaptionEngine:
    def __init__(self, source: AudioSource, transcriber: Transcriber, translator: Translator,
                 config: EngineConfig, on_caption: Callable[[Caption], None],
                 on_status: Optional[Callable[[str], None]] = None):
        self.source = source
        self.transcriber = transcriber
        self.translator = translator
        self.cfg = config
        self.on_caption = on_caption
        self.on_status = on_status or (lambda s: None)
        self._stop = threading.Event()
        self._blocks: "queue.Queue[np.ndarray]" = queue.Queue()
        self._threads: list[threading.Thread] = []
        # Whisper's own translate mode: one pass that outputs English.
        self._whisper_translate = translator.name == "whisper"
        # Adaptive noise floor so the silence threshold follows the room.
        self._noise_floor = config.silence_threshold

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        self._threads = [
            threading.Thread(target=self._capture_loop, name="capture", daemon=True),
            threading.Thread(target=self._process_loop, name="process", daemon=True),
        ]
        for t in self._threads:
            t.start()

    def stop(self) -> None:
        self._stop.set()
        self.source.close()

    def join(self, timeout: Optional[float] = None) -> None:
        for t in self._threads:
            t.join(timeout)

    @property
    def finished(self) -> bool:
        return all(not t.is_alive() for t in self._threads)

    # -- threads -------------------------------------------------------------

    def _capture_loop(self) -> None:
        try:
            for block in self.source.blocks():
                if self._stop.is_set():
                    break
                self._blocks.put(block)
        except Exception as exc:
            log.exception("audio capture failed")
            self.on_status(f"audio error: {exc}")
        finally:
            self._blocks.put(None)  # sentinel: end of stream

    def _process_loop(self) -> None:
        cfg = self.cfg
        buf: list[np.ndarray] = []
        buf_len = 0                    # samples in buf
        since_partial = 0              # new samples since last partial transcription
        speech_seen = False
        silence_run = 0.0              # seconds of consecutive silence
        last_partial_text = ""
        stream_ended = False

        while not self._stop.is_set() and not stream_ended:
            try:
                block = self._blocks.get(timeout=0.5)
            except queue.Empty:
                continue
            if block is None:
                stream_ended = True
                block = None
            else:
                buf.append(block)
                buf_len += len(block)
                since_partial += len(block)
                is_speech = self._is_speech(block)
                if is_speech:
                    speech_seen = True
                    silence_run = 0.0
                else:
                    silence_run += len(block) / SAMPLE_RATE

            seconds = buf_len / SAMPLE_RATE
            end_segment = False
            if stream_ended:
                end_segment = speech_seen
            elif speech_seen and silence_run >= cfg.silence:
                # Short pauses only end a segment once it has enough context; a
                # longer pause (2x) always ends it so short replies show promptly.
                end_segment = seconds >= cfg.min_segment or silence_run >= 2 * cfg.silence
            elif seconds >= cfg.max_segment:
                end_segment = True

            if not speech_seen and not end_segment:
                # Pure silence so far: keep only the tail so the buffer doesn't grow.
                if seconds > 1.0:
                    buf, buf_len = _tail(buf, int(0.5 * SAMPLE_RATE))
                    since_partial = 0
                continue

            if end_segment:
                audio = np.concatenate(buf) if buf else np.zeros(0, np.float32)
                buf, buf_len, since_partial = [], 0, 0
                speech_seen, silence_run = False, 0.0
                if len(audio) / SAMPLE_RATE >= cfg.min_speech:
                    self._emit(audio, final=True, previous=last_partial_text)
                last_partial_text = ""
                continue

            if since_partial >= cfg.step * SAMPLE_RATE:
                since_partial = 0
                audio = np.concatenate(buf)
                text = self._emit(audio, final=False, previous=last_partial_text)
                last_partial_text = text

    # -- helpers -------------------------------------------------------------

    def _is_speech(self, block: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(block.astype(np.float64) ** 2)) + 1e-9)
        # Track the noise floor slowly downward, quickly upward (but never above the
        # configured threshold * 4, so loud rooms still stop at something sane).
        if rms < self._noise_floor:
            self._noise_floor = 0.9 * self._noise_floor + 0.1 * rms
        else:
            self._noise_floor = 0.995 * self._noise_floor + 0.005 * rms
        threshold = max(self.cfg.silence_threshold, min(self._noise_floor * 3.0, self.cfg.silence_threshold * 4))
        return rms > threshold

    def _emit(self, audio: np.ndarray, final: bool, previous: str = "") -> str:
        cfg = self.cfg
        t0 = time.monotonic()
        try:
            if self._whisper_translate:
                result = self.transcriber.transcribe(audio, cfg.language, task="translate")
                text, translation = "", result.text
            else:
                result = self.transcriber.transcribe(audio, cfg.language, task="transcribe")
                text = result.text
                translation = ""
                if text and cfg.target and (final or cfg.translate_partials):
                    translation = self.translator.translate(text, result.language, cfg.target)
        except Exception as exc:
            log.exception("transcription failed")
            self.on_status(f"error: {exc}")
            return previous
        dt = time.monotonic() - t0
        log.debug("%s %.1fs audio in %.2fs: %r", "FINAL" if final else "partial",
                  len(audio) / SAMPLE_RATE, dt, text or translation)
        shown = text or translation
        if not shown:
            if final and previous:
                # Final pass heard nothing but partials did: keep what we showed.
                self.on_caption(Caption(previous, "", result.language, final=True))
            return ""
        self.on_caption(Caption(text, translation, result.language, final=final))
        return text or translation


def _tail(blocks: list[np.ndarray], keep: int) -> tuple[list[np.ndarray], int]:
    out: list[np.ndarray] = []
    total = 0
    for b in reversed(blocks):
        out.insert(0, b)
        total += len(b)
        if total >= keep:
            break
    return out, total
