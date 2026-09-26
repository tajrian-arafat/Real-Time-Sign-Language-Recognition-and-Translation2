"""Load Kaggle asl-signs landmark parquets (543 × xyz layout)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HOLISTIC_LANDMARK_COUNT = 543
META_COLUMNS = {"frame", "row_id", "sequence_id"}


def load_parquet_landmarks(path: Path) -> np.ndarray:
    """
    Read one sequence parquet into (T, 543, 3).

    Kaggle competition files store 1629 coordinate columns (543 landmarks × xyz),
    plus ``frame`` / ``row_id`` metadata columns.
    """
    df = pd.read_parquet(path)
    if "frame" in df.columns:
        df = df.sort_values("frame")

    coord_cols = [c for c in df.columns if c not in META_COLUMNS]
    if len(coord_cols) == HOLISTIC_LANDMARK_COUNT * 3:
        values = df[coord_cols].to_numpy(dtype=np.float32)
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
