"""Resolve data and config paths from config.yaml and environment."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _REPO_ROOT / "config" / "config.yaml"


def repo_root() -> Path:
    return _REPO_ROOT


def load_config() -> dict[str, Any]:
    with _CONFIG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_data_root(config: dict[str, Any] | None = None) -> Path:
    cfg = config if config is not None else load_config()
    env_var = cfg["paths"]["data_root_env_var"]
    env_val = os.environ.get(env_var, "").strip()
    if env_val:
        return Path(env_val)
    return _REPO_ROOT / cfg["paths"]["repo_relative_data_root"]


def processed_dir(config: dict[str, Any] | None = None) -> Path:
    root = resolve_data_root(config)
    cfg = config if config is not None else load_config()
    return root / cfg["paths"]["processed_dir"]


def reports_dir(config: dict[str, Any] | None = None) -> Path:
    cfg = config if config is not None else load_config()
    return _REPO_ROOT / cfg["paths"]["reports_dir"]


def kaggle_raw_dir(config: dict[str, Any] | None = None) -> Path:
    cfg = config if config is not None else load_config()
    return resolve_data_root(cfg) / cfg["paths"]["raw_dir"] / cfg["datasets"]["kaggle_raw_subdir"]
