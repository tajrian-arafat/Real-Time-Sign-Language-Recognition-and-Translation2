#!/usr/bin/env python3
"""Spot-check processed .npz tensors for finite values; write reports/preprocessing_fix.json."""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ml.preprocess.paths import load_config, processed_dir, reports_dir  # noqa: E402

ROOT_CAUSE: dict = {
    "summary": (
        "Kaggle competition parquets are long-format; holistic type offsets were wrong "
        "(left_hand at 33 instead of 501), and NaN/missing pose shoulders propagated "
        "through shoulder-width normalization and np.interp resampling into every .npz tensor."
    ),
    "fixes": [
        "Correct MediaPipe holistic offsets: pose=0, face=33, left_hand=501, right_hand=522",
        "sanitize_holistic_frames + shoulder/hip anchor fallback in normalize.py",
        "nan_to_num after parquet load and temporal resample",
        "finite guard in process_sequence",
    ],
    "code_paths": [
        "ml/preprocess/kaggle_io.py",
        "ml/preprocess/normalize.py",
        "ml/preprocess/sanitize.py",
        "ml/preprocess/sequence.py",
    ],
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cache_dir(config: dict) -> Path:
    root = processed_dir(config) / "kaggle_asl_signs" / "tensors"
    if not root.is_dir():
        raise FileNotFoundError(f"Missing tensor cache directory: {root}")
    caches = sorted(root.iterdir())
    if not caches:
        raise FileNotFoundError(f"No tensor cache under {root}")
    return caches[-1]


def verify_npz_finite(
    cache_dir: Path,
    *,
    sample_size: int = 100,
    seed: int = 42,
) -> dict:
    files = sorted(cache_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files in {cache_dir}")

    rng = random.Random(seed)
    if len(files) <= sample_size:
        sample = files
    else:
        sample = rng.sample(files, sample_size)

    bad: list[dict] = []
    for path in sample:
        data = np.load(path)
        tensor = data["tensor"]
        finite = bool(np.isfinite(tensor).all())
        if not finite:
            bad.append(
                {
                    "path": str(path),
                    "nan_count": int(np.isnan(tensor).sum()),
                    "inf_count": int(np.isinf(tensor).sum()),
                }
            )

    return {
        "generated_at_utc": _utc_now(),
        "cache_dir": str(cache_dir),
        "total_npz": len(files),
        "sample_size": len(sample),
        "sample_all_finite": len(bad) == 0,
        "non_finite_in_sample": len(bad),
        "failures": bad[:20],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify preprocessed tensors are finite")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config()
    cache = _cache_dir(config)
    report = verify_npz_finite(cache, sample_size=args.sample_size, seed=args.seed)
    report["root_cause"] = ROOT_CAUSE
    report["status"] = "ok" if report["sample_all_finite"] else "failed"

    out = reports_dir(config) / "preprocessing_fix.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["sample_all_finite"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
