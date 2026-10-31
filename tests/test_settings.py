import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live_captions.settings import Settings, load_settings, save_settings  # noqa: E402
from live_captions.languages import WHISPER_LANGUAGES, sorted_languages, language_name  # noqa: E402
from live_captions.cli import build_parser  # noqa: E402


class SettingsTests(unittest.TestCase):
    def test_round_trip(self):
        p = Path(tempfile.mkdtemp()) / "settings.json"
        s = Settings(target="en", language="de", input_device="USB Mic", font_size=30, opacity=0.5)
        save_settings(s, p)
        loaded, existed = load_settings(p)
        self.assertTrue(existed)
        self.assertEqual(loaded, s)

    def test_missing_file_gives_defaults(self):
        loaded, existed = load_settings(Path(tempfile.mkdtemp()) / "nope.json")
        self.assertFalse(existed)
        self.assertEqual(loaded, Settings())

    def test_bad_values_are_ignored(self):
        p = Path(tempfile.mkdtemp()) / "settings.json"
        p.write_text(json.dumps({"font_size": "huge", "model": "tiny", "unknown": 1, "opacity": "0.6"}))
        loaded, _ = load_settings(p)
        self.assertEqual(loaded.model, "tiny")
        self.assertEqual(loaded.font_size, Settings().font_size)
        self.assertEqual(loaded.opacity, 0.6)

    def test_change_detection(self):
        a, b = Settings(), Settings()
        self.assertFalse(a.engine_changed(b))
        b.input_device = "Mic 2"
        self.assertTrue(a.engine_changed(b))
        self.assertFalse(a.model_changed(b))
        b.model = "tiny"
        self.assertTrue(a.model_changed(b))
        c = Settings(font_size=40)
        self.assertFalse(a.engine_changed(c))

    def test_cli_overrides_only_given_options(self):
        args = build_parser().parse_args(["-t", "en", "--hide-original", "--final-only"])
        s = Settings(model="tiny", input_device="Mic")
        s.update({k: v for k, v in vars(args).items() if k in s.to_dict()})
        self.assertEqual(s.target, "en")
        self.assertFalse(s.show_original)
        self.assertFalse(s.translate_partials)
        self.assertEqual(s.model, "tiny")        # untouched
        self.assertEqual(s.input_device, "Mic")  # untouched

    def test_languages(self):
        self.assertEqual(sorted_languages()[0], ("en", "English"))
        self.assertEqual(language_name(""), "Auto-detect")
        self.assertEqual(language_name("de"), "German")
        self.assertGreaterEqual(len(WHISPER_LANGUAGES), 99)


if __name__ == "__main__":
    unittest.main()
