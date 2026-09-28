"""Build MediaPipe Holistic-layout tensors from Tasks Vision detections."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from ml.preprocess.landmark_spec import (
    HOLISTIC_LANDMARK_COUNT,
    LEFT_HAND_HOLISTIC_START,
    POSE_LANDMARK_COUNT,
    RIGHT_HAND_HOLISTIC_START,
)

HOLISTIC_FLAT_DIM = HOLISTIC_LANDMARK_COUNT * 3
LEGACY_TASKS_FLAT_DIM = 392


def _write_landmark(
    holistic: np.ndarray,
    index: int,
    point: Sequence[float] | object,
) -> None:
    if index < 0 or index >= HOLISTIC_LANDMARK_COUNT:
        return
    if hasattr(point, "x"):
        x = float(point.x)
        y = float(point.y)
        z = float(getattr(point, "z", 0.0))
    else:
        coords = np.asarray(point, dtype=np.float64).reshape(-1)
        if coords.size < 2:
            return
        x, y = float(coords[0]), float(coords[1])
        z = float(coords[2]) if coords.size > 2 else 0.0
    holistic[index, 0] = x
    holistic[index, 1] = y
    holistic[index, 2] = z


def _hand_side_label(handedness: object | None) -> str | None:
    if handedness is None:
        return None
    category = getattr(handedness, "category_name", None) or getattr(
        handedness, "display_name", None
    )
    if category is None and isinstance(handedness, str):
        category = handedness
    if not category:
        return None
    text = str(category).lower()
    if "left" in text:
        return "left"
    if "right" in text:
        return "right"
    return None


def detections_to_holistic_frame(
    hand_landmarks: Iterable[Sequence[object]] | None,
    *,
    handednesses: Iterable[object] | None = None,
    pose_landmarks: Sequence[object] | None = None,
    face_landmarks: Sequence[object] | None = None,
) -> np.ndarray:
    """
    Map Tasks outputs into (543, 3) Holistic index layout used by Kaggle parquets.

    Pose occupies indices 0–32, face mesh 33–500, left hand 501–521, right 522–542.
    """
    holistic = np.zeros((HOLISTIC_LANDMARK_COUNT, 3), dtype=np.float32)

    if pose_landmarks:
        for i, pt in enumerate(pose_landmarks[:POSE_LANDMARK_COUNT]):
            _write_landmark(holistic, i, pt)

    if face_landmarks:
        for i, pt in enumerate(face_landmarks):
            _write_landmark(holistic, POSE_LANDMARK_COUNT + i, pt)

    hands = list(hand_landmarks or [])
    labels = list(handednesses or [])
    left_hand: Sequence[object] | None = None
    right_hand: Sequence[object] | None = None
    unlabeled: list[Sequence[object]] = []

    for idx, hand in enumerate(hands[:2]):
        side = _hand_side_label(labels[idx]) if idx < len(labels) else None
        if side == "left":
            left_hand = hand
        elif side == "right":
            right_hand = hand
        else:
            unlabeled.append(hand)

    for hand in unlabeled:
        if left_hand is None:
            left_hand = hand
        elif right_hand is None:
            right_hand = hand

    if left_hand is not None:
        for i, pt in enumerate(left_hand[:21]):
            _write_landmark(holistic, LEFT_HAND_HOLISTIC_START + i, pt)
    if right_hand is not None:
        for i, pt in enumerate(right_hand[:21]):
            _write_landmark(holistic, RIGHT_HAND_HOLISTIC_START + i, pt)

    return holistic


def holistic_flat_to_frame(flat: Sequence[float]) -> np.ndarray:
    arr = np.asarray(flat, dtype=np.float32).reshape(-1)
    if arr.size < HOLISTIC_FLAT_DIM:
        padded = np.zeros(HOLISTIC_FLAT_DIM, dtype=np.float32)
        padded[: arr.size] = arr
        arr = padded
    return arr[:HOLISTIC_FLAT_DIM].reshape(HOLISTIC_LANDMARK_COUNT, 3)


def legacy_tasks_flat_to_holistic(flat: Sequence[float]) -> np.ndarray:
    """
    Parse browser legacy layout: hand0, hand1, pose33, face40, zero pad to 392.
    """
    arr = np.asarray(flat, dtype=np.float32).reshape(-1)
    holistic = np.zeros((HOLISTIC_LANDMARK_COUNT, 3), dtype=np.float32)
    idx = 0

    def read_block(count: int, holistic_start: int) -> None:
        nonlocal idx
        for i in range(count):
            if idx + 2 >= arr.size:
                return
            holistic[holistic_start + i, 0] = arr[idx]
            holistic[holistic_start + i, 1] = arr[idx + 1]
            holistic[holistic_start + i, 2] = arr[idx + 2]
            idx += 3

    read_block(21, LEFT_HAND_HOLISTIC_START)
    read_block(21, RIGHT_HAND_HOLISTIC_START)
    read_block(POSE_LANDMARK_COUNT, 0)
    read_block(40, POSE_LANDMARK_COUNT)
    return holistic


def infer_frame_layout(flat: Sequence[float]) -> str:
    arr = np.asarray(flat, dtype=np.float64).reshape(-1)
    if arr.size >= HOLISTIC_FLAT_DIM:
        return "holistic_flat"
    if arr.size == LEGACY_TASKS_FLAT_DIM:
        tail = arr[345:390]
        if np.max(np.abs(tail)) < 1e-5:
            return "legacy_tasks_v1"
        last_two = arr[-2:]
        if np.all((last_two == 0.0) | (last_two == 1.0)):
            body = arr[:390]
            outside_unit = float(np.mean((body < 0.0) | (body > 1.0)))
            if outside_unit > 0.05:
                return "packed392"
        return "legacy_tasks_v1"
    return "unknown"
