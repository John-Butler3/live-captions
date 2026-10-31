"""Whisper speech-to-text backends.

The default backend is faster-whisper (CTranslate2), which runs the same OpenAI
Whisper weights several times faster on CPU. The original ``openai-whisper``
package is supported as a fallback with the same interface.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

MODEL_SIZES = ("tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en",
               "large-v2", "large-v3", "turbo", "distil-small.en", "distil-medium.en",
               "distil-large-v3")


@dataclass
class Segment:
    start: float
    end: float
    text: str
    compression_ratio: float = 0.0


@dataclass
class Transcript:
    text: str
    language: Optional[str]  # ISO 639-1 code detected or forced
    segments: list = field(default_factory=list)  # Segment objects with times in seconds


class Transcriber:
    """Common interface for the two backends."""

    def transcribe(self, audio: np.ndarray, language: Optional[str], task: str = "transcribe") -> Transcript:
        raise NotImplementedError


class FasterWhisperTranscriber(Transcriber):
    def __init__(self, model: str, device: str = "auto", compute_type: str = "default",
                 threads: int = 0, beam_size: int = 1):
        from faster_whisper import WhisperModel

        if compute_type == "default":
            compute_type = "int8" if device in ("cpu", "auto") else "float16"
        log.info("loading faster-whisper %s (device=%s, compute=%s)", model, device, compute_type)
        if not threads:
            # CTranslate2 defaults to 4 threads; use the cores we have (capped).
            threads = max(1, min(os.cpu_count() or 4, 8))
        kwargs = {"device": device, "compute_type": compute_type, "cpu_threads": threads}
        log.info("using %d CPU threads", threads)
        try:
            self._model = WhisperModel(model, **kwargs)
        except Exception:
            if device == "auto":
                # CUDA detected but unusable (missing cuDNN etc.): fall back to CPU.
                log.warning("GPU init failed, falling back to CPU", exc_info=True)
                kwargs.update(device="cpu", compute_type="int8")
                self._model = WhisperModel(model, **kwargs)
            else:
                raise
        self.beam_size = beam_size

    def transcribe(self, audio, language, task="transcribe"):
        segments, info = self._model.transcribe(
            audio,
            language=language,
            task=task,
            beam_size=self.beam_size,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300, "speech_pad_ms": 200},
            condition_on_previous_text=False,
            temperature=0.0,
            compression_ratio_threshold=None,  # no retries; we handle loops ourselves
            log_prob_threshold=None,
            no_speech_threshold=0.6,
        )
        segs = [Segment(s.start, s.end, _clean_segment(s.text, s.compression_ratio), s.compression_ratio)
                for s in segments]
        return _build_transcript(segs, info.language)


class OpenAIWhisperTranscriber(Transcriber):
    def __init__(self, model: str, device: str = "auto", **_ignored):
        import torch
        import whisper

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("loading openai-whisper %s on %s", model, device)
        self._model = whisper.load_model(model, device=device)
        self._fp16 = device != "cpu"

    def transcribe(self, audio, language, task="transcribe"):
        result = self._model.transcribe(
            audio.astype(np.float32),
            language=language,
            task=task,
            fp16=self._fp16,
            temperature=0.0,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
        )
        segs = [Segment(s["start"], s["end"], _clean_segment(s["text"], s.get("compression_ratio", 0.0)),
                        s.get("compression_ratio", 0.0)) for s in result.get("segments", [])]
        return _build_transcript(segs, result.get("language"))


def create_transcriber(backend: str, model: str, **kwargs) -> Transcriber:
    if backend == "auto":
        try:
            import faster_whisper  # noqa: F401
            backend = "faster-whisper"
        except ImportError:
            backend = "openai-whisper"
    if backend == "faster-whisper":
        return FasterWhisperTranscriber(model, **kwargs)
    if backend == "openai-whisper":
        return OpenAIWhisperTranscriber(model, **kwargs)
    raise ValueError(f"unknown backend {backend!r}")


_HALLUCINATIONS = {
    "thank you.", "thanks for watching.", "thank you for watching.", "you", "you.",
    "subtitles by the amara.org community", "bye.", ".", "...", "thank you for watching!",
}


def _build_transcript(segs: list, language: Optional[str]) -> Transcript:
    segs = [s for s in segs if s.text]
    text = collapse_repeats(" ".join(s.text for s in segs))
    return Transcript(text=_clean(text), language=language, segments=segs)


def _clean_segment(text: str, compression_ratio: float) -> str:
    text = _clean(text)
    if compression_ratio and compression_ratio > 2.4:
        # Very compressible text = the decoder looped ("and the and the and the…").
        text = collapse_repeats(text)
        if _compression_ratio(text) > 2.4:
            log.debug("dropping looping segment: %r", text[:80])
            return ""
    return text


def _clean(text: str) -> str:
    text = " ".join(text.split())
    # Whisper tends to emit these on silence/noise; drop them when they are the whole output.
    if text.lower() in _HALLUCINATIONS:
        return ""
    return text


def _compression_ratio(text: str) -> float:
    import zlib

    data = text.encode("utf-8")
    return len(data) / max(1, len(zlib.compress(data))) if data else 0.0


def collapse_repeats(text: str, max_ngram: int = 8) -> str:
    """Collapse a phrase repeated back-to-back (a Whisper decoding loop).

    Single words may repeat twice ("no no"), longer phrases are kept once.
    """
    words = text.split()
    changed = True
    while changed:
        changed = False
        for n in range(1, max_ngram + 1):
            i = 0
            out: list[str] = []
            while i < len(words):
                gram = words[i:i + n]
                count = 1
                while len(gram) == n and words[i + count * n:i + (count + 1) * n] == gram:
                    count += 1
                allowed = 2 if n == 1 else 1
                if count > allowed:
                    out.extend(gram * allowed)
                    i += count * n
                    changed = True
                else:
                    out.append(words[i])
                    i += 1
            words = out
    return " ".join(words)
