#!/usr/bin/env python3
"""Build label_map.json from Kaggle train.csv (Agent 3)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ml.preprocess.kaggle_io import find_train_csv  # noqa: E402
from ml.preprocess.paths import load_config, processed_dir, resolve_data_root  # noqa: E402
from ml.preprocess.vocabulary import build_label_map_from_train_csv, write_label_map  # noqa: E402


def main() -> int:
    config = load_config()
    kaggle_dir = resolve_data_root(config) / config["paths"]["raw_dir"] / config["datasets"]["kaggle_raw_subdir"]
    train_csv = find_train_csv(kaggle_dir)
    if not train_csv:
        print("train.csv not found — Kaggle data not ready", file=sys.stderr)
        return 2

    label_map = build_label_map_from_train_csv(train_csv)
    out = processed_dir(config) / "kaggle_asl_signs" / config["paths"]["label_map_filename"]
    write_label_map(label_map, out)
    print(json.dumps({"path": str(out), "vocabulary_size": label_map["vocabulary_size"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
