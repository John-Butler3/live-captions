# Live Captions

A floating, always-on-top window that shows live captions of whatever your
microphone (or a virtual audio device) hears, optionally translated into
another language. Speech recognition runs locally with OpenAI Whisper
(`tiny` or `small` are the recommended sizes for real-time use on a CPU).

```
┌──────────────────────────────────────────────────────┐
│  Ask not what your country can do for you.           │  ← finished line
│  Pregunta no qué puede hacer tu país por ti.         │  ← translation
│  what you can do for your…                           │  ← live partial
└──────────────────────────────────────────────────────┘
```

* Borderless, semi-transparent, stays above every other window
* Settings window (⚙ button, double-click, or right-click → Settings…) to pick
  the microphone, the spoken language, the language to translate into, the
  Whisper model and the look; settings are remembered between runs
* Fixed-size box: new lines scroll in at the bottom, the window never grows
  or jumps. Drag anywhere to move, drag the ◢ corner to resize; size and
  position are remembered. `+` / `-` for font size, right-click (or
  Ctrl-click) for the menu, `Esc` to quit
* Works on Windows, macOS and Linux (Python + Tkinter, no Electron)
* Everything runs offline once the models are downloaded

## Install

Python 3.9+ with Tkinter (python.org installers include it; on Debian/Ubuntu
`sudo apt install python3-tk portaudio19-dev`).

```bash
git clone https://github.com/John-Butler3/live-captions.git
cd live-captions
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .                   # faster-whisper + sounddevice + numpy
pip install -e ".[translate]"      # optional: Argos Translate for translation
```

`faster-whisper` is the default backend: it runs the same OpenAI Whisper
weights through CTranslate2, several times faster on CPU and with much lower
memory use. The original `openai-whisper` package also works
(`pip install -e ".[openai]"` then `--backend openai-whisper`); it needs
PyTorch and ffmpeg.

Models download automatically on first run (tiny ≈ 75 MB, small ≈ 460 MB) and
are cached in `~/.cache/huggingface`.

## Run

```bash
live-captions                              # opens the window; first run opens Settings
live-captions -m tiny                      # captions, auto-detect language, 'small' model
live-captions -m tiny                      # fastest, lower accuracy
live-captions -l de -t en                  # German speech, English translation
live-captions -t es                        # any language → Spanish (Argos)
live-captions -l ja -t en --translator whisper   # Whisper's built-in translation (English only)
live-captions --hide-original -t en        # show only the translated line
live-captions -f talk.mp4 -t fr            # caption a file in real time (needs ffmpeg for non-WAV)
live-captions --no-gui -f talk.wav --fast  # terminal output, as fast as possible
live-captions --list-devices               # find a microphone / loopback device
live-captions -i "BlackHole"               # capture from a named input device
```

Command line options override the saved settings for that run. Settings
live in `%APPDATA%\LiveCaptions\settings.json` on Windows,
`~/Library/Application Support/LiveCaptions/settings.json` on macOS and
`~/.config/LiveCaptions/settings.json` on Linux (`--reset-settings` ignores
the file, `--settings` opens the Settings window on start).

Run `live-captions --help` for everything. Useful knobs:

| Flag | Default | Meaning |
|------|---------|---------|
| `-m/--model` | `small` | `tiny`, `base`, `small`, `medium`, `large-v3`, `tiny.en`, `distil-small.en`, … |
| `-l/--language` | auto | force the spoken language (faster and more stable than auto-detect) |
| `-t/--translate` | off | target language code |
| `--step` | 1.5 | seconds of new audio between live updates |
| `--silence` | 0.7 | pause length that ends a sentence |
| `--min-segment` | 3 | a pause only ends a sentence once it is this long (keeps context) |
| `--max-segment` | 12 | hard limit per caption line (cut at the quietest point near the end) |
| `--commit-after` / `--commit-margin` | 5 / 1 | during continuous speech, finish sentences this early |
| `--silence-threshold` | 0.01 | raise in noisy rooms if lines never end |
| `--final-only` | off | translate finished lines only, saves CPU |
| `--width`, `--height`, `--font-size`, `--opacity`, `--position`, `--history` | | window appearance (height 0 = fit 4 lines) |
| `--device cuda` | auto | use an NVIDIA GPU (install CUDA/cuDNN libs for CTranslate2) |

### Translation

* **Argos Translate** (`--translator argos`, used automatically when installed)
  translates between any pair of its ~40 languages, fully offline. Language
  packs are downloaded on first use (≈100 MB each; non-English pairs pivot
  through English). Captions are split into sentences with a simple
  punctuation rule instead of Argos' Stanza models, which keeps start-up fast
  and works for languages Stanza lacks (Malay, Tagalog, …). The package still
  depends on PyTorch via Stanza, so the install is large.
* **Whisper** (`--translator whisper`) uses the model's own speech translation.
  It only outputs English and shows just the English line, but needs no extra
  packages and no second model.

### Captioning system audio (videos, calls)

Whisper needs the audio as an *input* device. Route your system output into a
virtual input and select it with `-i`:

* macOS: [BlackHole](https://github.com/ExistentialAudio/BlackHole) + a Multi-Output Device
* Windows: enable "Stereo Mix" in Sound settings, or VB-Cable
* Linux (PulseAudio/PipeWire): use the `Monitor of …` source

### Performance

Whisper is not a streaming model, so the app transcribes a rolling buffer of
the current sentence every `--step` seconds and replaces the live line; when a
pause is detected the sentence is finalised. During continuous talking with no
pauses, Whisper's own sentence boundaries are used instead: once the buffer is
longer than `--commit-after` seconds, every sentence that ended at least
`--commit-margin` seconds ago is committed as a finished line and its audio
dropped, so lines come out sentence by sentence and the window never has to
chop a long speech mid-word. Phrases that Whisper repeats in a loop (a known
failure mode on long windows) are collapsed. Translation runs on its own
thread, so captions are never delayed by it: a line appears first and its
translation follows a moment later (Argos takes well under 0.1 s per sentence
once loaded).

Each Whisper pass costs roughly the same regardless of audio length, so the
model size decides whether the app keeps up. If a pass takes longer than the
update interval the window shows a hint to pick a smaller model. Rough CPU
times per pass with int8 on a recent laptop: `tiny` ≈ 0.3 s, `base` ≈ 0.5 s,
`small` ≈ 1–2 s, `medium` ≈ 5 s. Fixing the spoken language (instead of
auto-detect) and `--final-only` also save work. With an NVIDIA GPU
(`--device cuda`, CUDA 12 + cuDNN 9 libraries installed) `small` runs in a
fraction of a second.

### First run and troubleshooting

* **Model download**: the first start downloads the Whisper model (tiny ≈ 75 MB,
  small ≈ 460 MB). The window shows "loading Whisper …" until it is ready.
* **Language packs**: with Argos the first use of a language pair downloads a
  ≈100 MB pack and the window shows "downloading language pack …". With
  auto-detected speech the download happens after the first sentence is heard;
  fixing the spoken language in Settings lets it download at start-up.
* **Nothing is translated**: check the status line under the captions. English
  speech with "Translate into: English" has nothing to translate; the plain exe
  only translates *into* English (Whisper mode shows just the English line);
  the full exe is needed for other target languages.
* **No captions**: make sure the right microphone is selected (Settings) and
  that Windows allows microphone access for desktop apps. The log is at
  `%LOCALAPPDATA%\LiveCaptions\live-captions.log` (`-v` on the command line
  for more detail).

## Windows executable

Ready-to-run executables (no Python needed) are on the
[releases page](https://github.com/John-Butler3/live-captions/releases):

| File | Translation | Size |
|------|-------------|------|
| `LiveCaptions.exe` | into English only (Whisper's built-in translation) | ≈ 100 MB |
| `LiveCaptions-full.exe` | into any of ~40 languages (Argos Translate bundled; language packs download on first use) | ≈ 400 MB |

Both are built with PyInstaller by `.github/workflows/build-windows.yml`.

**Build it yourself** on Windows 10/11 with Python 3.9+ installed from
python.org (tick "Add python.exe to PATH"):

```powershell
git clone https://github.com/John-Butler3/live-captions.git
cd live-captions
powershell -ExecutionPolicy Bypass -File scripts\build_windows.ps1
```

The script creates `.venv-build`, installs the dependencies, runs the tests and
writes `dist\LiveCaptions.exe` (roughly 250 MB, single file). Options:

| Switch | Effect |
|--------|--------|
| `-OneDir` | folder build `dist\LiveCaptions\` that starts faster (nothing to unpack on launch) |
| `-WithArgos` | bundle Argos Translate for translation into languages other than English (adds PyTorch, several hundred MB) |
| `-Console` | keep a console window, so `--no-gui`, `--list-devices` and logs print to the terminal |

**GitHub Actions**: `.github/workflows/build-windows.yml` builds the exe on a
Windows runner. Run it from the Actions tab (workflow_dispatch) to get the exe
as a downloadable artifact, or push a tag such as `v0.1.0` to attach it to a
GitHub release.

**Using the exe**

* Double-click `LiveCaptions.exe`. The first start takes a few seconds while the
  single-file bundle unpacks, then a few more while the Whisper model downloads
  to `%USERPROFILE%\.cache\huggingface` (once).
* Create a shortcut and add arguments in its *Target* field, for example
  `LiveCaptions.exe -m tiny -l de -t en`. All command line options work.
* Pick the microphone and languages in the Settings window (⚙). The plain exe
  translates into English only; use `LiveCaptions-full.exe` for other targets.
* The windowed exe has no console; its log is written to
  `%LOCALAPPDATA%\LiveCaptions\live-captions.log`, and `--list-devices` shows
  a message box.
* Windows SmartScreen will warn about an unsigned exe the first time; choose
  "More info" → "Run anyway". Microphone access must be allowed for desktop
  apps under *Settings → Privacy & security → Microphone*.

## How it is built

```
live_captions/
  audio.py        microphone / file sources (16 kHz mono float32 blocks)
  transcriber.py  faster-whisper and openai-whisper backends
  translator.py   Argos Translate, Whisper translate mode
  engine.py       rolling-buffer segmentation, partial/final captions
  overlay.py      Tkinter floating window
  settings.py     persisted user settings (JSON)
  settings_dialog.py  Tk settings window
  app.py          controller: overlay + engine restart on settings change
  languages.py    Whisper / Argos language tables
  cli.py          argument parsing and wiring
tests/            engine segmentation tests (no models needed)
run.py            entry point used by PyInstaller
live_captions.spec, scripts/build_windows.ps1, .github/workflows/build-windows.yml
```

```bash
python -m unittest discover -s tests
```

## License

MIT
