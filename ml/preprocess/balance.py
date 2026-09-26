"""Class balancing: drop rare classes; oversample floor via temporal crops."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def apply_class_balance(
    entries: list[dict[str, Any]],
    *,
    oversample_floor: int,
    drop_below: int,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Filter and expand manifest entries referencing cached tensor paths.

    Each entry: {path, sign, label_index, participant_id, ...}
    Oversampling duplicates entries with ``aug_tag`` for traceability (temporal crop).
    """
    rng = np.random.default_rng(seed)
    counts = Counter(e["sign"] for e in entries)
    dropped = sorted([s for s, c in counts.items() if c < drop_below])
    kept = [e for e in entries if counts[e["sign"]] >= drop_below]

    log: dict[str, Any] = {
        "drop_below": drop_below,
        "oversample_floor": oversample_floor,
        "dropped_classes": dropped,
        "dropped_class_reason": "fewer than drop_below examples in split pool",
    }

    balanced: list[dict[str, Any]] = list(kept)
    for sign, count in Counter(e["sign"] for e in kept).items():
        if count >= oversample_floor:
            continue
        sign_entries = [e for e in kept if e["sign"] == sign]
        need = oversample_floor - count
        for i in range(need):
            src = sign_entries[int(rng.integers(0, len(sign_entries)))]
            aug = dict(src)
            aug["aug_tag"] = f"temporal_crop_{i}"
            balanced.append(aug)

    log["before"] = len(entries)
    log["after"] = len(balanced)
    return balanced, log


def save_balance_log(log: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
