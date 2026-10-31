"""Language codes understood by Whisper (ISO 639-1, plus a few 639-2 codes)."""

from __future__ import annotations

WHISPER_LANGUAGES: dict[str, str] = {
    "en": "English", "zh": "Chinese", "de": "German", "es": "Spanish", "ru": "Russian",
    "ko": "Korean", "fr": "French", "ja": "Japanese", "pt": "Portuguese", "tr": "Turkish",
    "pl": "Polish", "ca": "Catalan", "nl": "Dutch", "ar": "Arabic", "sv": "Swedish",
    "it": "Italian", "id": "Indonesian (Bahasa Indonesia)", "hi": "Hindi", "fi": "Finnish", "vi": "Vietnamese",
    "he": "Hebrew", "uk": "Ukrainian", "el": "Greek", "ms": "Malay (Bahasa Melayu)", "cs": "Czech",
    "ro": "Romanian", "da": "Danish", "hu": "Hungarian", "ta": "Tamil", "no": "Norwegian",
    "th": "Thai", "ur": "Urdu", "hr": "Croatian", "bg": "Bulgarian", "lt": "Lithuanian",
    "la": "Latin", "mi": "Maori", "ml": "Malayalam", "cy": "Welsh", "sk": "Slovak",
    "te": "Telugu", "fa": "Persian", "lv": "Latvian", "bn": "Bengali", "sr": "Serbian",
    "az": "Azerbaijani", "sl": "Slovenian", "kn": "Kannada", "et": "Estonian", "mk": "Macedonian",
    "br": "Breton", "eu": "Basque", "is": "Icelandic", "hy": "Armenian", "ne": "Nepali",
    "mn": "Mongolian", "bs": "Bosnian", "kk": "Kazakh", "sq": "Albanian", "sw": "Swahili",
    "gl": "Galician", "mr": "Marathi", "pa": "Punjabi", "si": "Sinhala", "km": "Khmer",
    "sn": "Shona", "yo": "Yoruba", "so": "Somali", "af": "Afrikaans", "oc": "Occitan",
    "ka": "Georgian", "be": "Belarusian", "tg": "Tajik", "sd": "Sindhi", "gu": "Gujarati",
    "am": "Amharic", "yi": "Yiddish", "lo": "Lao", "uz": "Uzbek", "fo": "Faroese",
    "ht": "Haitian Creole", "ps": "Pashto", "tk": "Turkmen", "nn": "Nynorsk", "mt": "Maltese",
    "sa": "Sanskrit", "lb": "Luxembourgish", "my": "Myanmar", "bo": "Tibetan", "tl": "Tagalog",
    "mg": "Malagasy", "as": "Assamese", "tt": "Tatar", "haw": "Hawaiian", "ln": "Lingala",
    "ha": "Hausa", "ba": "Bashkir", "jw": "Javanese", "su": "Sundanese", "yue": "Cantonese",
}

# Languages Argos Translate has packages for (pivoting through English).
ARGOS_LANGUAGES: tuple[str, ...] = (
    "en", "ar", "az", "bg", "bn", "ca", "cs", "da", "de", "el", "es", "et", "fa", "fi", "fr",
    "ga", "he", "hi", "hu", "id", "it", "ja", "ko", "lt", "lv", "ms", "nb", "nl", "pl", "pt",
    "ro", "ru", "sk", "sl", "sq", "sv", "th", "tl", "tr", "uk", "ur", "zh", "zt",
)


def language_name(code: str | None) -> str:
    if not code:
        return "Auto-detect"
    return WHISPER_LANGUAGES.get(code, code)


def sorted_languages(codes=None) -> list[tuple[str, str]]:
    """(code, name) pairs sorted by name; English first."""
    codes = list(codes) if codes is not None else list(WHISPER_LANGUAGES)
    pairs = [(c, WHISPER_LANGUAGES.get(c, c)) for c in codes if c != "en"]
    pairs.sort(key=lambda p: p[1])
    return ([("en", "English")] if "en" in codes else []) + pairs


def argos_available() -> bool:
    try:
        import argostranslate  # noqa: F401
        return True
    except ImportError:
        return False
