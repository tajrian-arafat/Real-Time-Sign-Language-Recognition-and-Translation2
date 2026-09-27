"""Synthetic tests for landmark normalization math."""

from __future__ import annotations

import numpy as np
import pytest

from ml.landmark_normalize import (
    normalize_frame,
    normalize_sequence,
    shoulder_midpoint_and_width,
)


def test_shoulder_midpoint_and_width(synthetic_pose_frame: np.ndarray) -> None:
    mid, width = shoulder_midpoint_and_width(synthetic_pose_frame)
    np.testing.assert_allclose(mid, [0.5, 0.5], rtol=1e-5)
    assert width == pytest.approx(0.2)


def test_normalize_frame_centers_shoulders(synthetic_pose_frame: np.ndarray) -> None:
    out = normalize_frame(synthetic_pose_frame)
    np.testing.assert_allclose(out[11, :2], [-0.5, 0.0], rtol=1e-5)
    np.testing.assert_allclose(out[12, :2], [0.5, 0.0], rtol=1e-5)


def test_normalize_sequence_preserves_shape(
    synthetic_sequence: np.ndarray,
) -> None:
    out = normalize_sequence(synthetic_sequence)
    assert out.shape == synthetic_sequence.shape
    assert out.dtype == np.float32


def test_normalize_is_translation_invariant_in_x() -> None:
    frame = np.zeros((33, 3), dtype=np.float32)
    frame[11, :2] = [1.0, 0.5]
    frame[12, :2] = [2.0, 0.5]
    shifted = frame.copy()
    shifted[:, 0] += 10.0
    a = normalize_frame(frame)
    b = normalize_frame(shifted)
    np.testing.assert_allclose(a[:, :2], b[:, :2], rtol=1e-5)
