"""Streaming engine: turns a stream of audio blocks into caption events.

Strategy (the usual "rolling window" approach for Whisper, which is not a true
streaming model):

* Audio blocks are appended to the current segment buffer.
* Every ``step`` seconds of new audio, the whole buffer is transcribed and the
  result is published as a *partial* caption (it may change as more context
  arrives).
* When ``silence`` seconds of quiet follow speech, the segment is finalised:
  transcribed one last time, published as a *final* caption, and the buffer is
  cleared.
* During continuous speech (no pause), once the buffer is longer than
  ``commit_after`` seconds, Whisper's own sentence segments are used: every
  segment that ended more than ``commit_margin`` seconds before the end of the
  buffer is *committed* as a final caption and its audio dropped, so the window
  stays short and lines come out sentence by sentence. ``max_segment`` is a hard
  cap: the buffer is then cut at the quietest point near its end.

Translation runs on its own thread so it never delays the captions: a caption
is published immediately and re-published (same ``id``) once its translation
is ready. Partial captions are translated on a best-effort basis; if the
translator is still busy, the newest partial simply waits for the next pass.
"""

from __future__ import annotations

import itertools
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
    id: int = 0               # segment number; a later caption with the same id replaces it
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
    commit_after: float = 5.0            # continuous speech: start committing sentences after this
    commit_margin: float = 1.0           # ...that ended this long before the buffer end (stable)


PARTIAL = "partial"  # queue marker: "a new partial caption is pending"

# A leading "!" marks a sticky status: the window keeps it until the next status.
SLOW_HINT = ("!Whisper is slower than real time on this PC – pick a smaller model "
             "(tiny or base) in Settings")


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
        # Finals are queued in order; partials are signalled with a marker and only the
        # newest pending partial is translated (stale ones are dropped).
        self._jobs: "queue.Queue[Optional[Caption | str]]" = queue.Queue()
        self._pending_partial: Optional[Caption] = None
        self._pending_lock = threading.Lock()
        self._threads: list[threading.Thread] = []
        self._ids = itertools.count(1)
        self._segment_id = next(self._ids)
        self._slow_hint_shown = False
        # Whisper's own translate mode: one pass that outputs English.
        self._whisper_translate = translator.name == "whisper"
        self._translate = bool(config.target) and not self._whisper_translate
        # Adaptive noise floor so the silence threshold follows the room.
        self._noise_floor = config.silence_threshold

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        self._threads = [
            threading.Thread(target=self._capture_loop, name="capture", daemon=True),
            threading.Thread(target=self._process_loop, name="process", daemon=True),
        ]
        if self._translate:
            self._threads.append(threading.Thread(target=self._translate_loop, name="translate", daemon=True))
        for t in self._threads:
            t.start()

    def stop(self) -> None:
        self._stop.set()
        self.source.close()
        self._jobs.put(None)

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

        try:
            while not self._stop.is_set() and not stream_ended:
                try:
                    block = self._blocks.get(timeout=0.5)
                except queue.Empty:
                    continue
                if block is None:
                    stream_ended = True
                else:
                    buf.append(block)
                    buf_len += len(block)
                    since_partial += len(block)
                    if self._is_speech(block):
                        speech_seen = True
                        silence_run = 0.0
                    else:
                        silence_run += len(block) / SAMPLE_RATE

                seconds = buf_len / SAMPLE_RATE
                end_reason = None
                if stream_ended:
                    end_reason = "stream" if speech_seen else None
                elif speech_seen and silence_run >= cfg.silence:
                    # Short pauses only end a segment once it has enough context; a
                    # longer pause (2x) always ends it so short replies show promptly.
                    if seconds >= cfg.min_segment or silence_run >= 2 * cfg.silence:
                        end_reason = "pause"
                elif seconds >= cfg.max_segment:
                    end_reason = "cap"

                if not speech_seen and end_reason is None:
                    # Pure silence so far: keep only the tail so the buffer doesn't grow.
                    if seconds > 1.0:
                        buf, buf_len = _tail(buf, int(0.5 * SAMPLE_RATE))
                        since_partial = 0
                    continue

                if end_reason in ("pause", "stream"):
                    audio = np.concatenate(buf) if buf else np.zeros(0, np.float32)
                    buf, buf_len, since_partial = [], 0, 0
                    speech_seen, silence_run = False, 0.0
                    if len(audio) / SAMPLE_RATE >= cfg.min_speech:
                        self._emit(audio, final=True, previous=last_partial_text)
                    last_partial_text = ""
                    self._segment_id = next(self._ids)
                    continue

                if end_reason == "cap":
                    # Hard cap without a pause: cut at the quietest point in the last part
                    # of the buffer, finalise what is before it and carry the rest over.
                    audio = np.concatenate(buf)
                    cut = _quietest_cut(audio, int(0.6 * len(audio)))
                    head, tail = audio[:cut], audio[cut:]
                    self._emit(head, final=True, previous=last_partial_text)
                    buf, buf_len, since_partial = [tail], len(tail), len(tail)
                    last_partial_text = ""
                    silence_run = 0.0
                    self._segment_id = next(self._ids)
                    continue

                if since_partial >= cfg.step * SAMPLE_RATE:
                    since_partial = 0
                    audio = np.concatenate(buf)
                    last_partial_text, drop = self._emit_partial(audio, last_partial_text)
                    if drop:
                        rest = audio[drop:]
                        buf, buf_len = ([rest] if len(rest) else []), len(rest)
        finally:
            self._jobs.put(None)  # let the translation thread drain and exit

    def _translate_loop(self) -> None:
        """Translate finals in order; partials best-effort (only the newest one)."""
        cfg = self.cfg
        if cfg.language:
            self._prepare_translator(cfg.language)
        prepared = {cfg.language} if cfg.language else set()
        while True:
            job = self._jobs.get()
            if job is None:
                # Sentinel: finish whatever final jobs are queued, then exit.
                while True:
                    try:
                        j = self._jobs.get_nowait()
                    except queue.Empty:
                        break
                    if isinstance(j, Caption):
                        self._run_translation(j, prepared)
                return
            if job == PARTIAL:
                job = self._take_pending_partial()
                if job is None:
                    continue  # superseded by a newer partial or a final
            self._run_translation(job, prepared)

    # -- helpers -------------------------------------------------------------

    def _prepare_translator(self, language: str) -> None:
        try:
            self.translator.prepare(language, self.cfg.target, self.on_status)
        except Exception as exc:
            log.exception("translator preparation failed")
            self.on_status(f"translation unavailable: {exc}")

    def _run_translation(self, cap: Caption, prepared: set) -> None:
        if not cap.text or not cap.language:
            return
        if cap.language not in prepared:
            prepared.add(cap.language)
            if cap.language != self.cfg.target:
                self._prepare_translator(cap.language)
        t0 = time.monotonic()
        try:
            translation = self.translator.translate(cap.text, cap.language, self.cfg.target)
        except Exception:
            log.exception("translation failed")
            return
        log.debug("translated %s in %.2fs: %r", "final" if cap.final else "partial",
                  time.monotonic() - t0, translation)
        if translation and not self._stop.is_set():
            self.on_caption(Caption(cap.text, translation, cap.language, cap.final, cap.id))

    def _take_pending_partial(self) -> Optional[Caption]:
        with self._pending_lock:
            cap, self._pending_partial = self._pending_partial, None
        return cap

    def _queue_translation(self, cap: Caption) -> None:
        if cap.final:
            with self._pending_lock:
                self._pending_partial = None  # the final supersedes any pending partial
            self._jobs.put(cap)
        elif self.cfg.translate_partials:
            with self._pending_lock:
                self._pending_partial = cap
            self._jobs.put(PARTIAL)

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

    def _emit_partial(self, audio: np.ndarray, previous: str) -> tuple[str, int]:
        """Publish a partial caption; commit stable leading sentences during long speech.

        Returns (partial_text, samples_to_drop_from_buffer).
        """
        cfg = self.cfg
        seconds = len(audio) / SAMPLE_RATE
        result = self._transcribe(audio)
        if result is None:
            return previous, 0
        drop = 0
        segs = result.segments
        if seconds >= cfg.commit_after and len(segs) >= 2:
            stable = [sg for sg in segs[:-1] if sg.end <= seconds - cfg.commit_margin]
            if stable:
                committed = " ".join(sg.text for sg in stable if sg.text).strip()
                drop = min(len(audio), int(stable[-1].end * SAMPLE_RATE))
                if committed:
                    self._publish(committed, result.language, final=True)
                    self._segment_id = next(self._ids)
                rest_text = " ".join(sg.text for sg in segs[len(stable):] if sg.text).strip()
                if rest_text:
                    self._publish(rest_text, result.language, final=False)
                return rest_text, drop
        text = self._shown_text(result)
        if text:
            self._publish(text, result.language, final=False)
        return text or previous, 0

    def _shown_text(self, result) -> str:
        return result.text

    def _transcribe(self, audio: np.ndarray):
        cfg = self.cfg
        t0 = time.monotonic()
        try:
            task = "translate" if self._whisper_translate else "transcribe"
            result = self.transcriber.transcribe(audio, cfg.language, task=task)
        except Exception as exc:
            log.exception("transcription failed")
            self.on_status(f"error: {exc}")
            return None
        dt = time.monotonic() - t0
        log.debug("transcribed %.1fs audio in %.2fs: %r", len(audio) / SAMPLE_RATE, dt, result.text)
        if dt > max(2.0 * cfg.step, 3.0) and not self._slow_hint_shown:
            self._slow_hint_shown = True
            log.warning("transcription took %.1fs for %.1fs of audio", dt, len(audio) / SAMPLE_RATE)
            self.on_status(SLOW_HINT)
        return result

    def _publish(self, text: str, language, final: bool) -> None:
        lang = self.cfg.language or language
        if self._whisper_translate:
            cap = Caption("", text, lang, final=final, id=self._segment_id)
        else:
            cap = Caption(text, "", lang, final=final, id=self._segment_id)
        self.on_caption(cap)
        if self._translate:
            self._queue_translation(cap)

    def _emit(self, audio: np.ndarray, final: bool, previous: str = "") -> str:
        cfg = self.cfg
        t0 = time.monotonic()
        try:
            if self._whisper_translate:
                result = self.transcriber.transcribe(audio, cfg.language, task="translate")
                text, translation = "", result.text
            else:
                result = self.transcriber.transcribe(audio, cfg.language, task="transcribe")
                text, translation = result.text, ""
        except Exception as exc:
            log.exception("transcription failed")
            self.on_status(f"error: {exc}")
            return previous
        dt = time.monotonic() - t0
        log.debug("%s %.1fs audio in %.2fs: %r", "FINAL" if final else "partial",
                  len(audio) / SAMPLE_RATE, dt, text or translation)
        if dt > max(2.0 * cfg.step, 3.0) and not self._slow_hint_shown:
            self._slow_hint_shown = True
            log.warning("transcription took %.1fs for %.1fs of audio", dt, len(audio) / SAMPLE_RATE)
            self.on_status(SLOW_HINT)
        shown = text or translation
        seg = self._segment_id
        lang = cfg.language or result.language
        if not shown:
            if final and previous:
                # Final pass heard nothing but partials did: keep what we showed.
                cap = Caption(previous, "", lang, final=True, id=seg)
                self.on_caption(cap)
                if self._translate:
                    self._queue_translation(cap)
            return ""
        cap = Caption(text, translation, lang, final=final, id=seg)
        self.on_caption(cap)
        if self._translate:
            self._queue_translation(cap)
        return text or translation


def _quietest_cut(audio: np.ndarray, start: int, window: int = SAMPLE_RATE // 10) -> int:
    """Index of the quietest `window`-sized block at or after `start` (for a soft cut)."""
    best, best_rms = len(audio), float("inf")
    for i in range(start, len(audio) - window, window):
        rms = float(np.sqrt(np.mean(audio[i:i + window] ** 2)))
        if rms < best_rms:
            best, best_rms = i + window // 2, rms
    return best


def _tail(blocks: list[np.ndarray], keep: int) -> tuple[list[np.ndarray], int]:
    out: list[np.ndarray] = []
    total = 0
    for b in reversed(blocks):
        out.insert(0, b)
        total += len(b)
        if total >= keep:
            break
    return out, total
