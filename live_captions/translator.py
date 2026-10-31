"""Text translation backends.

* ``argos``   - Argos Translate, fully offline, any installed language pair
                (pivots through English automatically).
* ``whisper`` - Whisper's built-in speech translation. Only produces English, and
                replaces the transcript rather than adding to it. Used when Argos
                is not installed and the target language is English.
* ``none``    - captions only.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

log = logging.getLogger(__name__)


class Translator:
    name = "none"

    def prepare(self, source: Optional[str], target: str, on_status=None) -> None:
        """Download/load whatever is needed for source -> target (may take a while)."""

    def translate(self, text: str, source: Optional[str], target: str) -> str:
        return ""


class SimpleSentencizer:
    """Punctuation-based sentence splitter used instead of Argos' Stanza/MiniSBD models.

    Captions are a few seconds of speech, so a regex is accurate enough, and it
    avoids loading a neural splitter per language (Stanza has no model for some
    languages, e.g. Malay, and fails with "language ms is currently unsupported").
    """

    _split = re.compile(r"(?<=[.!?。！？])\s+")

    def split_sentences(self, text: str) -> list[str]:
        return [p for p in self._split.split(text.strip()) if p]

    def __str__(self) -> str:
        return "SimpleSentencizer"


def _use_simple_sbd(translation) -> None:
    """Swap the sentencizer on a (possibly pivoted/composite) Argos translation."""
    if hasattr(translation, "sentencizer"):
        translation.sentencizer = SimpleSentencizer()
    # CachedTranslation.underlying, CompositeTranslation.t1/t2 (pivot via English)
    for attr in ("underlying", "t1", "t2"):
        sub = getattr(translation, attr, None)
        if sub is not None:
            _use_simple_sbd(sub)


class ArgosTranslator(Translator):
    name = "argos"

    def __init__(self, target: str, auto_install: bool = True):
        import os

        os.environ.setdefault("ARGOS_DEVICE_TYPE", "cpu")
        import argostranslate.translate as at

        self._at = at
        self.target = target
        self.auto_install = auto_install
        self._cache: dict[tuple[str, str], object] = {}
        self._missing: set[tuple[str, str]] = set()

    def prepare(self, source, target, on_status=None):
        if not source or source == target:
            return
        tr = self._get(source, target, on_status)
        if tr is not None:
            if on_status:
                on_status(f"loading translation {source} → {target}…")
            tr.translate("Hello.")  # loads the CTranslate2 model and sentence splitter

    def _get(self, source: str, target: str, on_status=None):
        key = (source, target)
        if key in self._cache:
            return self._cache[key]
        if key in self._missing:
            return None
        tr = _argos_lookup(self._at, source, target)
        if tr is None and self.auto_install:
            if on_status:
                on_status(f"downloading language pack {source} → {target} (one-time, ~100 MB)…")
            if install_argos_pair(source, target):
                tr = _argos_lookup(self._at, source, target)
        if tr is None:
            log.warning("no Argos translation %s -> %s available", source, target)
            if on_status:
                on_status(f"no translation available for {source} → {target}")
            self._missing.add(key)
            return None
        _use_simple_sbd(tr)
        self._cache[key] = tr
        return tr

    def translate(self, text, source, target):
        if not text or not source or source == target:
            return ""
        tr = self._get(source, target)
        if tr is None:
            return ""
        try:
            return tr.translate(text).strip()
        except Exception:
            log.exception("argos translation failed")
            return ""


def _argos_lookup(at, source: str, target: str):
    try:
        return at.get_translation_from_codes(source, target)
    except Exception:
        return None


def install_argos_pair(source: str, target: str) -> bool:
    """Download and install the Argos package(s) needed for source -> target."""
    import argostranslate.package as pkg

    log.info("installing Argos Translate package(s) for %s -> %s", source, target)
    try:
        pkg.update_package_index()
        available = pkg.get_available_packages()
    except Exception:
        log.exception("could not fetch Argos package index")
        return False

    def install(src, dst) -> bool:
        for p in available:
            if p.from_code == src and p.to_code == dst:
                pkg.install_from_path(p.download())
                return True
        return False

    if install(source, target):
        return True
    # Pivot through English.
    ok = True
    if source != "en":
        ok &= install(source, "en")
    if target != "en":
        ok &= install("en", target)
    return ok


def create_translator(kind: str, target: Optional[str]) -> Translator:
    """Return (translator). kind is 'auto', 'argos', 'whisper' or 'none'."""
    if not target or kind == "none":
        return Translator()
    if kind in ("auto", "argos"):
        try:
            return ArgosTranslator(target)
        except ImportError:
            if kind == "argos":
                raise SystemExit("argostranslate is not installed: pip install argostranslate")
    if kind in ("auto", "whisper"):
        if target == "en":
            t = Translator()
            t.name = "whisper"
            return t
        if kind == "whisper":
            raise SystemExit("Whisper can only translate into English; install argostranslate for other targets")
    raise SystemExit(
        f"No translator available for target '{target}'. Install Argos Translate "
        "(pip install argostranslate) or use --translate en for Whisper's built-in translation."
    )
