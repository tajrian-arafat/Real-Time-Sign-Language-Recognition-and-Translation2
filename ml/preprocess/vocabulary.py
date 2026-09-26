"""Build label_map.json from Kaggle train.csv."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def build_label_map_from_train_csv(train_csv: Path) -> dict[str, Any]:
    signs: list[str] = []
    with train_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "sign" not in reader.fieldnames:
            raise ValueError(f"train.csv must contain a 'sign' column: {train_csv}")
        for row in reader:
            signs.append(row["sign"].strip())

    unique = sorted(set(signs))
    label_to_idx = {gloss: i for i, gloss in enumerate(unique)}
    idx_to_label = {str(i): gloss for gloss, i in label_to_idx.items()}

    return {
        "vocabulary_size": len(unique),
        "source": "kaggle_asl_signs",
        "label_to_index": label_to_idx,
        "index_to_label": idx_to_label,
        "sign_counts": dict(Counter(signs)),
    }


def write_label_map(label_map: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(label_map, indent=2) + "\n", encoding="utf-8")
