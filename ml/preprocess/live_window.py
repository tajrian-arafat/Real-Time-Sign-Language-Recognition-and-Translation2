"""Convert live / video landmark windows into training-aligned packed tensors."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ml.preprocess.landmark_spec import FEATURES_PER_FRAME
from ml.preprocess.normalize import normalize_and_pack
from ml.preprocess.tasks_holistic import (
    HOLISTIC_FLAT_DIM,
    LEGACY_TASKS_FLAT_DIM,
    holistic_flat_to_frame,
    infer_frame_layout,
    legacy_tasks_flat_to_holistic,
)


def frame_flat_to_holistic(flat: Sequence[float]) -> np.ndarray:
    layout = infer_frame_layout(flat)
    if layout == "holistic_flat":
        return holistic_flat_to_frame(flat)
    if layout == "legacy_tasks_v1":
        return legacy_tasks_flat_to_holistic(flat)
    if layout == "packed392":
        raise ValueError("Frame is already packed; use as-is without holistic conversion")
    arr = np.asarray(flat, dtype=np.float32).reshape(-1)
    if arr.size >= HOLISTIC_FLAT_DIM:
        return holistic_flat_to_frame(arr)
    if arr.size == LEGACY_TASKS_FLAT_DIM:
        return legacy_tasks_flat_to_holistic(arr)
    raise ValueError(f"Unsupported landmark flat length {arr.size}")


def preprocess_landmark_frames(frames: list[list[float]]) -> np.ndarray:
    """
    Turn a window of per-frame flats into (T, 392) training features.

    Accepts holistic flats (543×3), legacy browser flats, or pre-packed rows.
    """
    if not frames:
        raise ValueError("landmark window must include at least one frame")

    rows: list[np.ndarray] = []
    for frame in frames:
        layout = infer_frame_layout(frame)
        if layout == "packed392":
            arr = np.asarray(frame, dtype=np.float32).reshape(-1)
            if arr.size < FEATURES_PER_FRAME:
                padded = np.zeros(FEATURES_PER_FRAME, dtype=np.float32)
                padded[: arr.size] = arr
                arr = padded
            rows.append(arr[:FEATURES_PER_FRAME].astype(np.float32))
            continue

        holistic = frame_flat_to_holistic(frame)
        packed = normalize_and_pack(holistic[np.newaxis, ...])[0]
        rows.append(packed.astype(np.float32))

    return np.stack(rows, axis=0)
