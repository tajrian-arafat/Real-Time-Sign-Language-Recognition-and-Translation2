#!/usr/bin/env python3
"""Stratified participant-level splits for Kaggle asl-signs (Agent 3)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ml.preprocess.kaggle_io import find_train_csv  # noqa: E402
from ml.preprocess.paths import load_config, processed_dir, resolve_data_root  # noqa: E402
from ml.preprocess.splits import load_train_rows, split_by_participant, write_split_manifest  # noqa: E402


def main() -> int:
    config = load_config()
    kaggle_dir = resolve_data_root(config) / config["paths"]["raw_dir"] / config["datasets"]["kaggle_raw_subdir"]
    train_csv = find_train_csv(kaggle_dir)
    if not train_csv:
        print("train.csv not found — Kaggle data not ready", file=sys.stderr)
        return 2

    rows = load_train_rows(train_csv)
    split_cfg = config["datasets"]["splits"]
    split_data = split_by_participant(
        rows,
        train_ratio=float(split_cfg["train_ratio"]),
        val_ratio=float(split_cfg["val_ratio"]),
        test_ratio=float(split_cfg["test_ratio"]),
    )
    out_dir = processed_dir(config) / "kaggle_asl_signs" / "splits"
    write_split_manifest(split_data, out_dir)
    summary = {k: v for k, v in split_data.items() if k != "sequences"}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
