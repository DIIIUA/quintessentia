"""Lightweight zero-dependency English / Russian localization for Quintessentia CLI, TUI, and Web UI.

Language resolution priority:
  1. Explicit CLI flag (`--lang en|ru` / `--language en|ru`)
  2. Environment variable (`QUINTESSENTIA_LANG=en|ru`)
  3. Saved user preference in `cli_config.json` (`"language": "en"|"ru"`)
  4. Operating system locale (POSIX `LC_ALL`/`LANG` or Win32 `GetUserDefaultUILanguage`)
  5. Fallback: English (`"en"`)
"""
from __future__ import annotations

import json
import locale
import os
from pathlib import Path
from typing import Any, Sequence


SUPPORTED_LANGS: tuple[str, str] = ("en", "ru")

_ACTIVE_LANG: str | None = None


def normalize_lang(value: Any) -> str | None:
    """Normalize a language string into `'en'`, `'ru'`, or `None` if unrecognized."""
    if value is None:
        return None
    raw = str(value).strip().lower()
    if not raw or raw == "auto":
        return None
    if raw in ("en", "eng", "english") or raw.startswith(("en_", "en-")):
        return "en"
    if raw in ("ru", "rus", "russian", "рус", "русский") or raw.startswith(("ru_", "ru-")):
        return "ru"
    return None


def detect_system_lang() -> str:
    """Detect preferred UI language (`'en'` or `'ru'`) from environment or OS locale."""
    env_lang = normalize_lang(os.environ.get("QUINTESSENTIA_LANG"))
    if env_lang:
        return env_lang

    for var in ("LC_ALL", "LC_MESSAGES", "LANG", "LANGUAGE"):
        val = os.environ.get(var)
        if not val:
            continue
        first = val.split(":", 1)[0]
        norm = normalize_lang(first)
        if norm:
            return norm

    if os.name == "nt":
        try:
            import ctypes
            lang_id = int(ctypes.windll.kernel32.GetUserDefaultUILanguage())
            # Primary language ID 0x19 = LANG_RUSSIAN
            if (lang_id & 0x3FF) == 0x19:
                return "ru"
            return "en"
        except Exception:
            pass

    try:
        loc = locale.getlocale()[0]
        norm = normalize_lang(loc)
        if norm:
            return norm
    except Exception:
        pass

    return "en"


def extract_lang_from_argv(argv: Sequence[str] | None) -> str | None:
    """Extract the last explicit `--lang` / `--language` value from an argv sequence before `--`."""
    if not argv:
        return None
    found: str | None = None
    items = list(argv)
    i = 0
    while i < len(items):
        tok = items[i]
        if tok == "--":
            break
        if tok in ("--lang", "--language") and i + 1 < len(items):
            norm = normalize_lang(items[i + 1])
            if norm:
                found = norm
            i += 2
            continue
        for prefix in ("--lang=", "--language="):
            if tok.startswith(prefix):
                norm = normalize_lang(tok[len(prefix):])
                if norm:
                    found = norm
                break
        i += 1
    return found


def load_saved_lang(config_path: Path | None = None) -> str | None:
    """Read saved `"language"` preference from `cli_config.json` if available."""
    if config_path is None or not config_path.is_file():
        return None
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return normalize_lang(data.get("language"))
    except Exception:
        pass
    return None


def resolve_cli_lang(
    argv: Sequence[str] | None = None,
    *,
    config_path: Path | None = None,
) -> str:
    """Resolve effective CLI language (`'en'` or `'ru'`) using the full priority chain."""
    from_argv = extract_lang_from_argv(argv)
    if from_argv:
        return from_argv
    from_env = normalize_lang(os.environ.get("QUINTESSENTIA_LANG"))
    if from_env:
        return from_env
    from_cfg = load_saved_lang(config_path)
    if from_cfg:
        return from_cfg
    return detect_system_lang()


def set_lang(lang: str | None) -> str:
    """Set the process-wide active CLI language (`'en'` or `'ru'`), or reset to auto when `None`."""
    global _ACTIVE_LANG
    if lang is None:
        _ACTIVE_LANG = None
        return get_lang()
    norm = normalize_lang(lang)
    if norm is None:
        raise ValueError(f"Unsupported language {lang!r}; expected one of {SUPPORTED_LANGS}")
    _ACTIVE_LANG = norm
    return _ACTIVE_LANG


def get_lang() -> str:
    """Return the active CLI language (`'en'` or `'ru'`)."""
    if _ACTIVE_LANG in SUPPORTED_LANGS:
        return _ACTIVE_LANG
    return detect_system_lang()


def tr(en: str, ru: str, *, lang: str | None = None, **kwargs: Any) -> str:
    """Return localized string in English or Russian with optional `.format(**kwargs)`."""
    eff = normalize_lang(lang) if lang is not None else get_lang()
    text = ru if eff == "ru" else en
    return text.format(**kwargs) if kwargs else text
