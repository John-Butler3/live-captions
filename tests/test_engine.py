"""Segmentation tests for CaptionEngine using a fake transcriber (no models needed)."""

import sys
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live_captions.audio import BLOCK_SIZE, SAMPLE_RATE, AudioSource  # noqa: E402
from live_captions.engine import CaptionEngine, EngineConfig  # noqa: E402
from live_captions.transcriber import Transcriber, Transcript  # noqa: E402
from live_captions.translator import Translator  # noqa: E402


class ArraySource(AudioSource):
    def __init__(self, audio):
        self.audio = audio.astype(np.float32)

    def blocks(self):
        for i in range(0, len(self.audio), BLOCK_SIZE):
            b = self.audio[i:i + BLOCK_SIZE]
            if len(b) < BLOCK_SIZE:
                b = np.pad(b, (0, BLOCK_SIZE - len(b)))
            yield b


class FakeTranscriber(Transcriber):
    """Returns the duration of the audio it was given, so tests can check segmenting."""

    def __init__(self):
        self.calls = []

    def transcribe(self, audio, language, task="transcribe"):
        secs = len(audio) / SAMPLE_RATE
        self.calls.append((secs, task))
        return Transcript(text=f"{secs:.1f}s", language="en")


class UpperTranslator(Translator):
    name = "fake"

    def translate(self, text, source, target):
        return f"{target}:{text}".upper()


def tone(seconds, amp=0.3):
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def silence(seconds):
    return np.zeros(int(seconds * SAMPLE_RATE), np.float32)


def run_engine(audio, cfg, translator=None):
    captions = []
    tr = FakeTranscriber()
    eng = CaptionEngine(ArraySource(audio), tr, translator or Translator(), cfg, captions.append)
    eng.start()
    deadline = time.time() + 10
    while not eng.finished and time.time() < deadline:
        eng.join(0.1)
    return captions, tr


class EngineTests(unittest.TestCase):
    def test_pause_splits_segments_and_emits_partials(self):
        audio = np.concatenate([tone(4.0), silence(1.5), tone(4.0), silence(1.5)])
        caps, tr = run_engine(audio, EngineConfig(step=1.0, silence=0.7, min_segment=3.0))
        finals = [c for c in caps if c.final]
        partials = [c for c in caps if not c.final]
        self.assertEqual(len(finals), 2)
        self.assertGreaterEqual(len(partials), 4)
        for c in finals:
            self.assertGreaterEqual(float(c.text[:-1]), 4.0)
            self.assertLess(float(c.text[:-1]), 6.0)

    def test_short_pause_does_not_split_before_min_segment(self):
        audio = np.concatenate([tone(1.0), silence(0.8), tone(1.0), silence(2.0)])
        caps, _ = run_engine(audio, EngineConfig(step=1.0, silence=0.7, min_segment=3.0))
        finals = [c for c in caps if c.final]
        self.assertEqual(len(finals), 1)
        self.assertGreaterEqual(float(finals[0].text[:-1]), 2.8)

    def test_long_pause_ends_short_utterance(self):
        audio = np.concatenate([tone(0.8), silence(2.0), tone(0.8), silence(2.0)])
        caps, _ = run_engine(audio, EngineConfig(step=1.0, silence=0.7, min_segment=3.0))
        self.assertEqual(len([c for c in caps if c.final]), 2)

    def test_max_segment_forces_split(self):
        audio = np.concatenate([tone(20.0), silence(2.0)])
        caps, _ = run_engine(audio, EngineConfig(step=2.0, max_segment=8.0))
        finals = [c for c in caps if c.final]
        self.assertGreaterEqual(len(finals), 2)
        for c in finals:
            self.assertLessEqual(float(c.text[:-1]), 8.2)

    def test_silence_only_produces_nothing(self):
        caps, tr = run_engine(silence(5.0), EngineConfig(step=1.0))
        self.assertEqual(caps, [])
        self.assertEqual(tr.calls, [])

    def test_translation_attached(self):
        audio = np.concatenate([tone(3.5), silence(2.0)])
        caps, _ = run_engine(audio, EngineConfig(step=1.0, target="es"), UpperTranslator())
        final = [c for c in caps if c.final][-1]
        self.assertTrue(final.translation.startswith("ES:"))
        partial = [c for c in caps if not c.final][0]
        self.assertTrue(partial.translation.startswith("ES:"))

    def test_final_only_skips_partial_translation(self):
        audio = np.concatenate([tone(3.5), silence(2.0)])
        caps, _ = run_engine(audio, EngineConfig(step=1.0, target="es", translate_partials=False), UpperTranslator())
        self.assertTrue(all(c.translation == "" for c in caps if not c.final))
        self.assertTrue(all(c.translation for c in caps if c.final))

    def test_whisper_translate_mode_uses_translate_task(self):
        t = Translator()
        t.name = "whisper"
        audio = np.concatenate([tone(3.5), silence(2.0)])
        caps, tr = run_engine(audio, EngineConfig(step=1.0, target="en"), t)
        self.assertTrue(all(task == "translate" for _, task in tr.calls))
        final = [c for c in caps if c.final][-1]
        self.assertEqual(final.text, "")
        self.assertTrue(final.translation)


if __name__ == "__main__":
    unittest.main()
