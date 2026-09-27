"""Instant dictionary lookup and lazy BanglaT5 sentence translation."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from backend.config_loader import get_config, resolve_models_dir
from ml.translation.translate import BanglaTranslator, TranslationResult


class DictionaryNotFoundError(FileNotFoundError):
    pass


def resolve_bangla_dictionary_path(config: dict[str, Any] | None = None) -> Path:
    cfg = config or get_config()
    paths = cfg.get("paths", {})
    models_dir = resolve_models_dir(cfg)
    filename = paths.get("bangla_dictionary_filename", "bangla_dictionary.json")
    return models_dir / filename


@lru_cache(maxsize=1)
def load_bangla_dictionary(config_path: str | None = None) -> dict[str, str]:
    config = get_config(config_path)
    path = resolve_bangla_dictionary_path(config)
    if not path.is_file():
        raise DictionaryNotFoundError(
            f"Bangla dictionary missing at {path}. "
            "Run ml/translation/build_word_dictionary.py"
        )
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    dictionary = payload.get("dictionary")
    if not isinstance(dictionary, dict):
        raise ValueError(f"Invalid dictionary payload in {path}")
    return {str(k): str(v) for k, v in dictionary.items()}


def lookup_instant_bangla(english: str, dictionary: dict[str, str] | None = None) -> str | None:
    mapping = dictionary if dictionary is not None else load_bangla_dictionary()
    stripped = english.strip()
    if not stripped:
        return None
    lowered = stripped.lower()
    if lowered in mapping:
        return mapping[lowered]
    if stripped in mapping:
        return mapping[stripped]
    return None


_translator: BanglaTranslator | None = None


def get_translator() -> BanglaTranslator:
    global _translator
    if _translator is None:
        _translator = BanglaTranslator()
    return _translator


def translate_sentence(english: str) -> TranslationResult:
    return get_translator().translate(english)


def reset_translation_singletons_for_tests() -> None:
    global _translator
    _translator = None
    load_bangla_dictionary.cache_clear()
