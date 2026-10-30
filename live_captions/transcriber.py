"""Whisper speech-to-text backends.

The default backend is faster-whisper (CTranslate2), which runs the same OpenAI
Whisper weights several times faster on CPU. The original ``openai-whisper``
package is supported as a fallback with the same interface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

MODEL_SIZES = ("tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en",
               "large-v2", "large-v3", "turbo", "distil-small.en", "distil-medium.en",
               "distil-large-v3")


@dataclass
class Transcript:
    text: str
    language: Optional[str]  # ISO 639-1 code detected or forced


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
        kwargs = {"device": device, "compute_type": compute_type}
        if threads:
            kwargs["cpu_threads"] = threads
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
            without_timestamps=True,
            temperature=0.0,
        )
        text = " ".join(s.text.strip() for s in segments if s.text.strip())
        return Transcript(text=_clean(text), language=info.language)


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
        return Transcript(text=_clean(result.get("text", "")), language=result.get("language"))


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


def _clean(text: str) -> str:
    text = " ".join(text.split())
    # Whisper tends to emit these on silence/noise; drop them when they are the whole output.
    if text.lower() in _HALLUCINATIONS:
        return ""
    return text
