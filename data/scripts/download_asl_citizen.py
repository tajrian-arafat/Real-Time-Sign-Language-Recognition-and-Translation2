#!/usr/bin/env python3
"""Download ASL Citizen dataset from official Microsoft URL (Agent 2)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Verified from live page 2026-09-26:
# https://www.microsoft.com/en-us/research/project/asl-citizen/
ASL_CITIZEN_OFFICIAL_URL = (
    "https://download.microsoft.com/download/b/8/8/"
    "b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip"
)
ASL_CITIZEN_SOURCE_PAGE = (
    "https://www.microsoft.com/en-us/research/project/asl-citizen/"
)


def resolve_data_root() -> Path:
    env = os.environ.get("SIGN_LANGUAGE_DATA_ROOT")
    if env:
        return Path(env)
    repo = Path(__file__).resolve().parents[2]
    return repo / "data"


def main() -> int:
    out_dir = resolve_data_root() / "raw" / "asl_citizen"
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "ASL_Citizen.zip"

    cmd = [
        "wget",
        "-c",
        "--tries=3",
        "--timeout=30",
        "-O",
        str(zip_path),
        ASL_CITIZEN_OFFICIAL_URL,
    ]
    print("Source page:", ASL_CITIZEN_SOURCE_PAGE)
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
