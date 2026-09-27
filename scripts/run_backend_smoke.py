#!/usr/bin/env python3
"""Run backend smoke checks and write reports/backend_smoke.json."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/integration/test_backend_websocket.py",
        "-q",
    ]
    proc = subprocess.run(cmd, cwd=ROOT, check=False)
    report = ROOT / "reports" / "backend_smoke.json"
    if report.is_file():
        print(report.read_text(encoding="utf-8"))
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
