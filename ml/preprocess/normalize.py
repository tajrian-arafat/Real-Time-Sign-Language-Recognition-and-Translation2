"""Per-frame shoulder-centered normalization."""
from __future__ import annotations

import numpy as np

from ml.preprocess.landmark_spec import (
    HAND_LEFT_INDICES,
    HAND_RIGHT_INDICES,
    SELECTED_HOLISTIC_INDICES,
    SHOULDER_LEFT_HOLISTIC,
    SHOULDER_RIGHT_HOLISTIC,
)


def _index_in_selected(holistic_index: int) -> int:
    return SELECTED_HOLISTIC_INDICES.index(holistic_index)


def normalize_sequence(
    frames_holistic: np.ndarray,
    *,
    shoulder_left: int = SHOULDER_LEFT_HOLISTIC,
    shoulder_right: int = SHOULDER_RIGHT_HOLISTIC,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Translate so shoulder midpoint is origin; scale by shoulder width.

    Args:
        frames_holistic: (T, 543, 3) raw coordinates from Kaggle parquet layout.
    Returns:
        (T, 130, 3) subset in normalized space.
    """
    if frames_holistic.ndim != 3 or frames_holistic.shape[1] != 543:
        raise ValueError(f"Expected (T, 543, 3), got {frames_holistic.shape}")

    selected = frames_holistic[:, SELECTED_HOLISTIC_INDICES, :].astype(np.float32, copy=True)

    left_sel = _index_in_selected(shoulder_left)
    right_sel = _index_in_selected(shoulder_right)

    for t in range(selected.shape[0]):
        left = selected[t, left_sel]
        right = selected[t, right_sel]
        mid = (left + right) * 0.5
        width = float(np.linalg.norm(left - right))
        if width < eps:
            width = 1.0
        selected[t] = (selected[t] - mid) / width

    return selected


def pack_features(normalized_selected: np.ndarray) -> np.ndarray:
    """
    Flatten to (T, 392): xyz for 130 landmarks + left/right hand presence flags.
    """
    t_len = normalized_selected.shape[0]
    flat = normalized_selected.reshape(t_len, -1)

    left_start = _index_in_selected(HAND_LEFT_INDICES[0])
    right_start = _index_in_selected(HAND_RIGHT_INDICES[0])
    left_pts = normalized_selected[:, left_start : left_start + 21, :]
    right_pts = normalized_selected[:, right_start : right_start + 21, :]
    left_present = (np.linalg.norm(left_pts.reshape(t_len, -1), axis=1) > 1e-8).astype(
        np.float32
    )
    right_present = (np.linalg.norm(right_pts.reshape(t_len, -1), axis=1) > 1e-8).astype(
        np.float32
    )
    flags = np.stack([left_present, right_present], axis=1)
    return np.concatenate([flat, flags], axis=1).astype(np.float32)


def normalize_and_pack(frames_holistic: np.ndarray) -> np.ndarray:
    return pack_features(normalize_sequence(frames_holistic))
