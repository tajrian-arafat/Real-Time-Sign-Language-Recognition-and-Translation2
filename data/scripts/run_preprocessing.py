#!/usr/bin/env python3
"""Run Kaggle preprocessing or emit waiting_for_kaggle status (Agent 3)."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ml.preprocess.kaggle_io import kaggle_data_ready  # noqa: E402
from ml.preprocess.paths import load_config, reports_dir, resolve_data_root  # noqa: E402
from ml.preprocess.pipeline_kaggle import run_kaggle_preprocessing  # noqa: E402


def write_waiting_status(reason: str, kaggle_dir: Path) -> None:
    config = load_config()
    report = {
        "status": "waiting_for_kaggle",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "agent": "Agent 3 — Data Preprocessing",
        "reason": reason,
        "kaggle_dir": str(kaggle_dir),
        "message": (
            "Preprocessing pipeline code is ready. Full run requires verified Kaggle "
            "asl-signs parquets under raw/kaggle_asl_signs (Kaggle API token + competition join)."
        ),
        "unit_tests": "tests/unit/test_preprocessing.py",
    }
    out = reports_dir(config) / "preprocessing_status.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def main() -> int:
    config = load_config()
    data_root = resolve_data_root(config)
    kaggle_dir = data_root / config["paths"]["raw_dir"] / config["datasets"]["kaggle_raw_subdir"]
    ready, reason = kaggle_data_ready(kaggle_dir)
    if not ready:
        write_waiting_status(reason, kaggle_dir)
        return 0

    report = run_kaggle_preprocessing()
    print(json.dumps(report, indent=2))
    return 0 if report.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
