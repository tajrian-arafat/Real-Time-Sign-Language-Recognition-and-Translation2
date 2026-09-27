"""Per-frame landmark normalization (shoulder midpoint origin, shoulder-width scale)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# MediaPipe pose indices (subset used for normalization anchor).
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12


def shoulder_midpoint_and_width(
    frame: NDArray[np.floating],
    left_idx: int = POSE_LEFT_SHOULDER,
    right_idx: int = POSE_RIGHT_SHOULDER,
) -> tuple[NDArray[np.floating], float]:
    """
    frame: shape (num_landmarks, D) with at least x,y in first two dims.
    Returns (midpoint[D], shoulder_width scalar).
    """
    left = frame[left_idx, :2].astype(np.float64)
    right = frame[right_idx, :2].astype(np.float64)
    midpoint = (left + right) / 2.0
    width = float(np.linalg.norm(right - left))
    if width < 1e-6:
        width = 1.0
    return midpoint, width


def normalize_frame(
    frame: NDArray[np.floating],
    left_idx: int = POSE_LEFT_SHOULDER,
    right_idx: int = POSE_RIGHT_SHOULDER,
    eps: float = 1e-6,
) -> NDArray[np.float32]:
    """Translate to shoulder midpoint; scale so shoulder width == 1."""
    frame = np.asarray(frame, dtype=np.float64)
    midpoint, width = shoulder_midpoint_and_width(frame, left_idx, right_idx)
    out = frame.copy()
    out[:, :2] = (out[:, :2] - midpoint) / max(width, eps)
    return out.astype(np.float32)


def normalize_sequence(
    sequence: NDArray[np.floating],
    left_idx: int = POSE_LEFT_SHOULDER,
    right_idx: int = POSE_RIGHT_SHOULDER,
) -> NDArray[np.float32]:
    """Normalize each frame independently. sequence shape (T, L, D)."""
    seq = np.asarray(sequence, dtype=np.float64)
    if seq.ndim != 3:
        raise ValueError(f"Expected (T, L, D), got shape {seq.shape}")
    normalized = np.stack(
        [
            normalize_frame(seq[t], left_idx=left_idx, right_idx=right_idx)
            for t in range(seq.shape[0])
        ],
        axis=0,
    )
    return normalized.astype(np.float32)
