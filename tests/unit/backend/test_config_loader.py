"""Unit tests for config loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.config_loader import (
    get_config,
    get_inference_settings,
    load_yaml_config,
    repo_root,
    resolve_data_root,
    resolve_reports_dir,
)


def test_repo_root_is_workspace(config_path: Path) -> None:
    root = repo_root()
    assert (root / "config" / "config.yaml") == config_path.resolve()


def test_load_yaml_config_has_inference_thresholds(config_path: Path) -> None:
    cfg = load_yaml_config(config_path)
    assert cfg["inference"]["confidence_threshold"] == 0.60
    assert cfg["inference"]["debounce_cooldown_ms"] == 800
    assert cfg["inference"]["temporal_ema_windows"] == 3


def test_resolve_data_root_uses_repo_relative_by_default(
    config_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SIGN_LANGUAGE_DATA_ROOT", raising=False)
    cfg = load_yaml_config(config_path)
    assert resolve_data_root(cfg) == (repo_root / "data").resolve()


def test_resolve_data_root_env_override(
    config_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SIGN_LANGUAGE_DATA_ROOT", str(tmp_path))
    cfg = load_yaml_config(config_path)
    assert resolve_data_root(cfg) == tmp_path.resolve()


def test_get_inference_settings(config_path: Path) -> None:
    cfg = load_yaml_config(config_path)
    settings = get_inference_settings(cfg)
    assert settings["confidence_threshold"] == 0.60
    assert settings["debounce_cooldown_ms"] == 800
    assert settings["temporal_ema_windows"] == 3
    assert settings["duplicate_suppression"] is True


def test_get_config_cached(config_path: Path) -> None:
    get_config.cache_clear()
    a = get_config(str(config_path))
    b = get_config(str(config_path))
    assert a is b
