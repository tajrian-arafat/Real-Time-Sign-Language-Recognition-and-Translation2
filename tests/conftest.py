"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def config_path() -> Path:
    return CONFIG_PATH


@pytest.fixture
def synthetic_pose_frame() -> np.ndarray:
    """Minimal pose-like tensor: 33 landmarks x 3 (x,y,z)."""
    frame = np.zeros((33, 3), dtype=np.float32)
    frame[11, :2] = [0.4, 0.5]
    frame[12, :2] = [0.6, 0.5]
    frame[0, :2] = [0.5, 0.3]
    return frame


@pytest.fixture
def synthetic_sequence(synthetic_pose_frame: np.ndarray) -> np.ndarray:
    t = 8
    seq = np.stack([synthetic_pose_frame.copy() for _ in range(t)], axis=0)
    for i in range(t):
        seq[i, :, 0] += i * 0.01
    return seq
