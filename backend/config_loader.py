"""Load and resolve paths from config/config.yaml."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "config.yaml"


def repo_root() -> Path:
    return _REPO_ROOT


def resolve_config_path(config_path: Path | str | None = None) -> Path:
    if config_path is None:
        return _DEFAULT_CONFIG_PATH
    return Path(config_path)


def load_yaml_config(config_path: Path | str | None = None) -> dict[str, Any]:
    path = resolve_config_path(config_path)
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at root of {path}")
    return data


def resolve_data_root(config: dict[str, Any]) -> Path:
    paths = config.get("paths", {})
    env_var = paths.get("data_root_env_var", "SIGN_LANGUAGE_DATA_ROOT")
    env_value = os.environ.get(env_var)
    if env_value:
        return Path(env_value).expanduser().resolve()
    repo_relative = paths.get("repo_relative_data_root", "data")
    return (_REPO_ROOT / repo_relative).resolve()


def resolve_reports_dir(config: dict[str, Any]) -> Path:
    paths = config.get("paths", {})
    reports_name = paths.get("reports_dir", "reports")
    return (_REPO_ROOT / reports_name).resolve()


@lru_cache(maxsize=4)
def get_config(config_path: str | None = None) -> dict[str, Any]:
    return load_yaml_config(config_path)


def get_inference_settings(config: dict[str, Any]) -> dict[str, float | int | bool]:
    inference = config.get("inference", {})
    sentence = config.get("sentence", {})
    return {
        "confidence_threshold": float(inference.get("confidence_threshold", 0.60)),
        "debounce_cooldown_ms": int(inference.get("debounce_cooldown_ms", 800)),
        "temporal_ema_windows": int(inference.get("temporal_ema_windows", 3)),
        "duplicate_suppression": bool(sentence.get("duplicate_suppression", True)),
    }
