"""Finite-value guards for holistic landmark tensors."""
from __future__ import annotations

import numpy as np

from ml.preprocess.landmark_spec import (
    HOLISTIC_LANDMARK_COUNT,
    SHOULDER_LEFT_HOLISTIC,
    SHOULDER_RIGHT_HOLISTIC,
)

# MediaPipe pose hip indices (holistic pose segment).
HIP_LEFT_HOLISTIC = 23
HIP_RIGHT_HOLISTIC = 24

DEFAULT_CLIP = 20.0


def sanitize_holistic_frames(
    frames: np.ndarray,
    *,
    clip: float = DEFAULT_CLIP,
) -> np.ndarray:
    """
    Replace non-finite coords with 0 and clip extreme values before normalization.

    Args:
        frames: (T, 543, 3) float array from parquet loader.
    """
    if frames.ndim != 3 or frames.shape[1] != HOLISTIC_LANDMARK_COUNT or frames.shape[2] != 3:
        raise ValueError(f"Expected (T, 543, 3), got {frames.shape}")

    out = np.asarray(frames, dtype=np.float32)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    if clip > 0:
        np.clip(out, -clip, clip, out=out)
    return out


def _point_finite(frame: np.ndarray, idx: int) -> bool:
    pt = frame[idx]
    return bool(np.all(np.isfinite(pt)) and np.linalg.norm(pt) > 1e-8)


def shoulder_anchor(frame: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Return (midpoint, width) from shoulders, else hips, else None."""
    pairs = (
        (SHOULDER_LEFT_HOLISTIC, SHOULDER_RIGHT_HOLISTIC),
        (HIP_LEFT_HOLISTIC, HIP_RIGHT_HOLISTIC),
    )
    for left_idx, right_idx in pairs:
        if not (_point_finite(frame, left_idx) and _point_finite(frame, right_idx)):
            continue
        left = frame[left_idx].astype(np.float64)
        right = frame[right_idx].astype(np.float64)
        mid = (left + right) * 0.5
        width = float(np.linalg.norm(left - right))
        if width < 1e-6:
            width = 1.0
        return mid, width
    return None


def ensure_finite_features(features: np.ndarray, *, clip: float = DEFAULT_CLIP) -> np.ndarray:
    """Last-resort guard on packed (T, F) tensors."""
    out = np.asarray(features, dtype=np.float32)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    if clip > 0:
        np.clip(out, -clip, clip, out=out)
    return out
