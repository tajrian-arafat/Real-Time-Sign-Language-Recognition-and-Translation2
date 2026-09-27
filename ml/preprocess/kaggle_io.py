"""Load Kaggle asl-signs landmark parquets (543 × xyz layout)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HOLISTIC_LANDMARK_COUNT = 543
META_COLUMNS = {"frame", "row_id", "sequence_id", "type", "landmark_index"}

# MediaPipe Holistic flat layout: pose(33) + face(468) + left_hand(21) + right_hand(21) = 543.
_POSE_COUNT = 33
_FACE_COUNT = 468
_HAND_COUNT = 21
_TYPE_TO_HOLISTIC_OFFSET: dict[str, int] = {
    "pose": 0,
    "face": _POSE_COUNT,
    "left_hand": _POSE_COUNT + _FACE_COUNT,
    "right_hand": _POSE_COUNT + _FACE_COUNT + _HAND_COUNT,
}


def _load_long_format_landmarks(df: pd.DataFrame) -> np.ndarray:
    """Competition parquets: one row per (frame, type, landmark_index) with x,y,z."""
    required = {"frame", "type", "landmark_index", "x", "y", "z"}
    if not required.issubset(df.columns):
        raise ValueError(f"missing columns for long format: {required - set(df.columns)}")

    types = df["type"].astype(str)
    offsets = types.map(_TYPE_TO_HOLISTIC_OFFSET)
    valid = offsets.notna()
    if not valid.any():
        raise ValueError("no recognized landmark types in parquet")

    frame_ids = df.loc[valid, "frame"].to_numpy(dtype=np.int64)
    holistic_idx = (
        offsets.loc[valid].astype(np.int32).to_numpy()
        + df.loc[valid, "landmark_index"].to_numpy(dtype=np.int32)
    )
    in_range = (holistic_idx >= 0) & (holistic_idx < HOLISTIC_LANDMARK_COUNT)
    frame_ids = frame_ids[in_range]
    holistic_idx = holistic_idx[in_range]
    coords = df.loc[valid, ["x", "y", "z"]].to_numpy(dtype=np.float32)[in_range]

    unique_frames, frame_index = np.unique(frame_ids, return_inverse=True)
    out = np.zeros((len(unique_frames), HOLISTIC_LANDMARK_COUNT, 3), dtype=np.float32)
    out[frame_index, holistic_idx, 0] = coords[:, 0]
    out[frame_index, holistic_idx, 1] = coords[:, 1]
    out[frame_index, holistic_idx, 2] = coords[:, 2]
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def load_parquet_landmarks(path: Path) -> np.ndarray:
    """
    Read one sequence parquet into (T, 543, 3).

    Kaggle competition files store 1629 coordinate columns (543 landmarks × xyz),
    plus ``frame`` / ``row_id`` metadata columns, or long-format type/landmark_index rows.
    """
    df = pd.read_parquet(path)
    if {"type", "landmark_index", "x", "y", "z"}.issubset(df.columns):
        return _load_long_format_landmarks(df)

    if "frame" in df.columns:
        df = df.sort_values("frame")

    coord_cols = [c for c in df.columns if c not in META_COLUMNS]
    if len(coord_cols) == HOLISTIC_LANDMARK_COUNT * 3:
        values = df[coord_cols].to_numpy(dtype=np.float32)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        return values.reshape(len(df), HOLISTIC_LANDMARK_COUNT, 3)

    # Named columns x_i, y_i, z_i
    if all(f"x_{i}" in df.columns for i in (0, 1)):
        frames = []
        for _, row in df.iterrows():
            pts = np.zeros((HOLISTIC_LANDMARK_COUNT, 3), dtype=np.float32)
            for i in range(HOLISTIC_LANDMARK_COUNT):
                pts[i, 0] = row.get(f"x_{i}", 0.0)
                pts[i, 1] = row.get(f"y_{i}", 0.0)
                pts[i, 2] = row.get(f"z_{i}", 0.0)
            frames.append(pts)
        return np.stack(frames, axis=0)

    raise ValueError(
        f"Unrecognized parquet schema in {path}: {len(coord_cols)} coord columns, "
        f"columns sample {list(df.columns[:8])}"
    )


def find_train_csv(kaggle_dir: Path) -> Path | None:
    matches = list(kaggle_dir.rglob("train.csv"))
    return matches[0] if matches else None


def kaggle_data_ready(kaggle_dir: Path) -> tuple[bool, str]:
    """Return whether verified Kaggle landmarks are available for full preprocessing."""
    if not kaggle_dir.is_dir():
        return False, "kaggle raw directory missing"

    parquets = list(kaggle_dir.rglob("*.parquet"))
    train_csv = find_train_csv(kaggle_dir)
    if not train_csv:
        return False, "train.csv not found"
    if not parquets:
        return False, "no parquet landmark files"

    import csv

    with train_csv.open(newline="", encoding="utf-8") as f:
        row_count = sum(1 for _ in csv.reader(f)) - 1
    if row_count != len(parquets):
        return False, f"parquet count {len(parquets)} != train.csv rows {row_count}"

    return True, "verified"
