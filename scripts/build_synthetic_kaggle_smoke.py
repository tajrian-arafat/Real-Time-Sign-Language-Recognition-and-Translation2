#!/usr/bin/env python3
"""Build a tiny long-format Kaggle tree with NaN landmarks for preprocessing smoke tests."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ml.preprocess.paths import load_config, resolve_data_root  # noqa: E402


def _write_sequence_parquet(path: Path, *, frames: int, rng: np.random.Generator) -> None:
    rows: list[dict] = []
    row_id = 0
    for frame in range(frames):
        for i in range(33):
            x, y, z = rng.normal(size=3).astype(float)
            if i in (11, 12) and frame % 3 == 0:
                x, y, z = np.nan, np.nan, np.nan
            rows.append(
                {
                    "frame": frame,
                    "row_id": row_id,
                    "type": "pose",
                    "landmark_index": i,
                    "x": x,
                    "y": y,
                    "z": z,
                }
            )
            row_id += 1
        for i in range(5):
            rows.append(
                {
                    "frame": frame,
                    "row_id": row_id,
                    "type": "left_hand",
                    "landmark_index": i,
                    "x": float(rng.normal()),
                    "y": float(rng.normal()),
                    "z": float(rng.normal()),
                }
            )
            row_id += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def build_synthetic_kaggle(
    out_dir: Path,
    *,
    num_sequences: int,
    signs: list[str],
    seed: int = 0,
) -> None:
    rng = np.random.default_rng(seed)
    landmark_root = out_dir / "train_landmark_files"
    landmark_root.mkdir(parents=True, exist_ok=True)
    train_csv = out_dir / "train.csv"
    with train_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["path", "participant_id", "sequence_id", "sign"]
        )
        writer.writeheader()
        for i in range(num_sequences):
            pid = f"p{i % 5}"
            sid = f"s{i:05d}"
            rel = f"train_landmark_files/{pid}/{sid}.parquet"
            _write_sequence_parquet(
                out_dir / rel, frames=int(rng.integers(8, 24)), rng=rng
            )
            writer.writerow(
                {
                    "path": rel,
                    "participant_id": pid,
                    "sequence_id": sid,
                    "sign": signs[i % len(signs)],
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-sequences", type=int, default=120)
    args = parser.parse_args()
    config = load_config()
    root = resolve_data_root(config)
    kaggle_dir = root / config["paths"]["raw_dir"] / config["datasets"]["kaggle_raw_subdir"]
    signs = [f"word_{i}" for i in range(12)]
    build_synthetic_kaggle(kaggle_dir, num_sequences=args.num_sequences, signs=signs)
    print(f"Wrote synthetic Kaggle tree under {kaggle_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
