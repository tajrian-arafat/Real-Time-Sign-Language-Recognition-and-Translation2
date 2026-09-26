"""Participant-level stratified train/val/test splits."""
from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _modal_sign(signs: list[str]) -> str:
    return Counter(signs).most_common(1)[0][0]


def load_train_rows(train_csv: Path) -> list[dict[str, str]]:
    with train_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"path", "sign", "participant_id"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"train.csv missing columns {required - set(reader.fieldnames or [])}")
        return [dict(row) for row in reader]


def split_by_participant(
    rows: list[dict[str, str]],
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, Any]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1")

    by_participant: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_participant[row["participant_id"]].append(row)

    participants = list(by_participant.keys())
    strata: dict[str, list[str]] = defaultdict(list)
    for pid in participants:
        signs = [r["sign"] for r in by_participant[pid]]
        strata[_modal_sign(signs)].append(pid)

    rng = random.Random(seed)
    train_ids: set[str] = set()
    val_ids: set[str] = set()
    test_ids: set[str] = set()

    for _stratum, pids in strata.items():
        rng.shuffle(pids)
        n = len(pids)
        if n == 1:
            train_ids.add(pids[0])
            continue
        n_test = max(1, int(round(n * test_ratio))) if n >= 3 else 0
        n_val = max(1, int(round(n * val_ratio))) if n - n_test >= 2 else 0
        n_train = n - n_val - n_test
        if n_train <= 0:
            n_train = n - n_val - n_test
        idx = 0
        for pid in pids[idx : idx + n_train]:
            train_ids.add(pid)
        idx += n_train
        for pid in pids[idx : idx + n_val]:
            val_ids.add(pid)
        idx += n_val
        for pid in pids[idx : idx + n_test]:
            test_ids.add(pid)

    def collect(ids: set[str]) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for pid in ids:
            out.extend(by_participant[pid])
        return out

    train_rows = collect(train_ids)
    val_rows = collect(val_ids)
    test_rows = collect(test_ids)

    return {
        "seed": seed,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "participants": {
            "train": sorted(train_ids),
            "val": sorted(val_ids),
            "test": sorted(test_ids),
        },
        "sequences": {
            "train": train_rows,
            "val": val_rows,
            "test": test_rows,
        },
        "counts": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
    }


def write_split_manifest(split_data: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        k: v
        for k, v in split_data.items()
        if k != "sequences"
    }
    (out_dir / "splits.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for name in ("train", "val", "test"):
        path = out_dir / f"{name}_manifest.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in split_data["sequences"][name]:
                f.write(json.dumps(row) + "\n")
