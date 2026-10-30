# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Live Captions.

Build:  pyinstaller live_captions.spec            (one-file LiveCaptions.exe)
        pyinstaller live_captions.spec -- --onedir  (folder build, faster start)
        pyinstaller live_captions.spec -- --with-argos  (bundle Argos Translate; very large)

The spec collects the native libraries that the import analysis cannot see:
CTranslate2, onnxruntime, PortAudio (sounddevice), PyAV, and the Silero VAD
model that ships inside faster-whisper.
"""

import argparse
import os
import sys

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_dynamic_libs

parser = argparse.ArgumentParser()
parser.add_argument("--onedir", action="store_true")
parser.add_argument("--with-argos", action="store_true")
parser.add_argument("--console", action="store_true", help="keep a console window (debugging)")
opts = parser.parse_args()

APP_NAME = "LiveCaptions"
ROOT = os.path.abspath(SPECPATH)
ICON = os.path.join(ROOT, "assets", "live-captions.ico")

datas, binaries, hiddenimports = [], [], []

for pkg in ("faster_whisper", "ctranslate2", "onnxruntime", "av", "tokenizers", "huggingface_hub"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# sounddevice ships PortAudio as a data package on Windows/macOS wheels.
try:
    import _sounddevice_data  # noqa: F401
    datas += collect_data_files("_sounddevice_data")
except ImportError:
    pass
binaries += collect_dynamic_libs("sounddevice")
hiddenimports += ["sounddevice", "_cffi_backend"]

excludes = ["matplotlib", "scipy", "pandas", "IPython", "jupyter", "pytest", "setuptools", "PyQt5", "PySide6"]
if opts.with_argos:
    for pkg in ("argostranslate", "stanza", "sentencepiece", "ctranslate2", "torch"):
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
else:
    excludes += ["argostranslate", "stanza", "torch", "sentencepiece"]

a = Analysis(
    [os.path.join(ROOT, "run.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe_kwargs = dict(
    name=APP_NAME,
    debug=False,
    strip=False,
    upx=False,
    console=opts.console,
    disable_windowed_traceback=False,
    icon=ICON if os.path.exists(ICON) and sys.platform == "win32" else None,
)

if opts.onedir:
    exe = EXE(pyz, a.scripts, exclude_binaries=True, **exe_kwargs)
    coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name=APP_NAME)
else:
    exe = EXE(pyz, a.scripts, a.binaries, a.datas, **exe_kwargs)

if sys.platform == "darwin" and opts.onedir:
    app = BUNDLE(coll, name=f"{APP_NAME}.app", icon=None, bundle_identifier="dev.live-captions",
                 info_plist={"NSMicrophoneUsageDescription": "Live Captions transcribes your microphone.",
                             "NSHighResolutionCapable": True})
