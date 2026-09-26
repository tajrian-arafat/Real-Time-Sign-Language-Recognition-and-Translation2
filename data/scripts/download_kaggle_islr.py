#!/usr/bin/env python3
"""Download Kaggle asl-signs competition data (Agent 2)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def resolve_data_root() -> Path:
    env = os.environ.get("SIGN_LANGUAGE_DATA_ROOT")
    if env:
        return Path(env)
    repo = Path(__file__).resolve().parents[2]
    return repo / "data"


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from kaggle_auth import kaggle_credentials_message, kaggle_credentials_present

    if not kaggle_credentials_present():
        print(f"BLOCKER: {kaggle_credentials_message()}", file=sys.stderr)
        return 2

    out_dir = resolve_data_root() / "raw" / "kaggle_asl_signs"
    out_dir.mkdir(parents=True, exist_ok=True)

    kaggle_bin = os.environ.get("KAGGLE_BIN", "kaggle")
    cmd = [
        kaggle_bin,
        "competitions",
        "download",
        "-c",
        "asl-signs",
        "-p",
        str(out_dir),
    ]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
