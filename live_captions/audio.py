"""Audio sources: microphone (sounddevice) and audio files (for testing)."""

from __future__ import annotations

import queue
import subprocess
import threading
import time
import wave
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

SAMPLE_RATE = 16000  # Whisper's native sample rate
BLOCK_SECONDS = 0.1  # size of each chunk handed to the engine
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_SECONDS)


class AudioSource:
    """Yields float32 mono blocks at 16 kHz."""

    def blocks(self) -> Iterator[np.ndarray]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class MicrophoneSource(AudioSource):
    def __init__(self, device: Optional[str | int] = None):
        import sounddevice as sd  # imported lazily so file mode works without PortAudio

        self._sd = sd
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self._closed = threading.Event()
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=BLOCK_SIZE,
            device=_resolve_device(sd, device),
            callback=self._callback,
        )
        self._stream.start()

    def _callback(self, indata, frames, time_info, status):  # noqa: D401
        if status:
            # Over/underruns are not fatal; keep going.
            pass
        self._queue.put(indata[:, 0].copy())

    def blocks(self) -> Iterator[np.ndarray]:
        while not self._closed.is_set():
            try:
                yield self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

    def close(self) -> None:
        self._closed.set()
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass


class FileSource(AudioSource):
    """Streams an audio file as if it were being spoken live.

    WAV files are read with the standard library; anything else is decoded via
    ffmpeg (which must be on PATH). With realtime=True the file is paced at 1x so
    the streaming behaviour matches a microphone.
    """

    def __init__(self, path: str | Path, realtime: bool = True):
        self.path = Path(path)
        self.realtime = realtime
        self._closed = threading.Event()
        self._audio = load_audio(self.path)

    def blocks(self) -> Iterator[np.ndarray]:
        start = time.monotonic()
        pos = 0
        n = len(self._audio)
        while pos < n and not self._closed.is_set():
            block = self._audio[pos : pos + BLOCK_SIZE]
            if len(block) < BLOCK_SIZE:
                block = np.pad(block, (0, BLOCK_SIZE - len(block)))
            pos += BLOCK_SIZE
            if self.realtime:
                due = start + pos / SAMPLE_RATE
                delay = due - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            yield block
        # Trailing silence so the engine flushes the final segment.
        for _ in range(int(2.0 / BLOCK_SECONDS)):
            if self._closed.is_set():
                break
            if self.realtime:
                time.sleep(BLOCK_SECONDS)
            yield np.zeros(BLOCK_SIZE, dtype=np.float32)

    def close(self) -> None:
        self._closed.set()


def load_audio(path: Path) -> np.ndarray:
    """Decode an audio file to float32 mono 16 kHz."""
    if path.suffix.lower() == ".wav":
        try:
            return _load_wav(path)
        except Exception:
            pass  # fall through to ffmpeg for exotic WAV encodings
    cmd = [
        "ffmpeg", "-nostdin", "-threads", "0", "-i", str(path),
        "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(SAMPLE_RATE), "-",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required to decode non-WAV files") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed: {exc.stderr.decode(errors='ignore')}") from exc
    return np.frombuffer(out, np.int16).astype(np.float32) / 32768.0


def _load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        channels, width, rate = wf.getnchannels(), wf.getsampwidth(), wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    if width == 2:
        data = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, np.int32).astype(np.float32) / 2147483648.0
    elif width == 1:
        data = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported WAV sample width {width}")
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    if rate != SAMPLE_RATE:
        data = _resample(data, rate, SAMPLE_RATE)
    return data.astype(np.float32)


def _resample(data: np.ndarray, src: int, dst: int) -> np.ndarray:
    n_out = int(round(len(data) * dst / src))
    x_old = np.linspace(0.0, 1.0, num=len(data), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, data)


def _resolve_device(sd, device):
    if device is None or device == "":
        return None
    if isinstance(device, int) or (isinstance(device, str) and device.isdigit()):
        return int(device)
    # Exact name first, then substring match; input devices only. On Windows the
    # same device appears once per host API; prefer WASAPI over MME/DirectSound.
    needle = device.lower()
    devices = [(i, d) for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]
    apis = sd.query_hostapis()
    for matcher in (lambda n: n == needle, lambda n: needle in n):
        matches = [(i, d) for i, d in devices if matcher(d["name"].lower())]
        if matches:
            for i, d in matches:
                if "wasapi" in apis[d["hostapi"]]["name"].lower():
                    return i
            return matches[0][0]
    raise ValueError(f"no input device matching {device!r}; use --list-devices")


def list_devices() -> str:
    import sounddevice as sd

    lines = []
    default_in = sd.default.device[0]
    for idx, info in enumerate(sd.query_devices()):
        if info["max_input_channels"] <= 0:
            continue
        marker = "*" if idx == default_in else " "
        lines.append(f"{marker} {idx:3d}  {info['name']}  ({info['max_input_channels']} ch)")
    return "\n".join(lines) if lines else "no input devices found"
