#!/usr/bin/env python3
"""Run pytest and write reports/test_scaffold.json."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = REPO_ROOT / "reports" / "test_scaffold.json"


def main() -> int:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/unit",
        "--tb=short",
        "-q",
        "--json-report",
        "--json-report-file=reports/pytest_unit_report.json",
    ]
    try:
        completed = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        completed = None

    # Fallback without pytest-json-report plugin
    if completed is None or "unrecognized arguments: --json-report" in (
        completed.stderr or ""
    ):
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/unit", "-q", "--tb=short"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        summary = _parse_pytest_summary(completed.stdout)
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "suite": "unit_scaffold",
            "exit_code": completed.returncode,
            "passed": summary.get("passed", 0),
            "failed": summary.get("failed", 0),
            "errors": summary.get("errors", 0),
            "skipped": summary.get("skipped", 0),
            "total": summary.get("total", 0),
            "stdout_tail": (completed.stdout or "")[-4000:],
            "stderr_tail": (completed.stderr or "")[-4000:],
        }
    else:
        json_path = REPO_ROOT / "reports" / "pytest_unit_report.json"
        detail = {}
        if json_path.is_file():
            detail = json.loads(json_path.read_text(encoding="utf-8"))
        summary = detail.get("summary", {})
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "suite": "unit_scaffold",
            "exit_code": completed.returncode,
            "passed": summary.get("passed", 0),
            "failed": summary.get("failed", 0),
            "errors": summary.get("error", summary.get("errors", 0)),
            "skipped": summary.get("skipped", 0),
            "total": summary.get("total", 0),
            "tests": [
                {
                    "nodeid": t.get("nodeid"),
                    "outcome": t.get("outcome"),
                    "duration": t.get("duration"),
                }
                for t in detail.get("tests", [])
            ],
        }

    frontend = _run_vitest(REPO_ROOT)
    payload["frontend_vitest"] = frontend
    payload["passed"] = payload.get("passed", 0) + frontend.get("passed", 0)
    payload["failed"] = payload.get("failed", 0) + frontend.get("failed", 0)
    payload["total"] = payload.get("total", 0) + frontend.get("total", 0)
    if frontend.get("exit_code", 0) != 0 and completed.returncode == 0:
        payload["exit_code"] = frontend["exit_code"]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    exit_code = payload.get("exit_code", 1)
    return exit_code if completed else 1


def _run_vitest(repo_root: Path) -> dict[str, object]:
    frontend = repo_root / "frontend"
    if not (frontend / "package.json").is_file():
        return {"skipped": True, "reason": "no frontend package"}
    npm = subprocess.run(
        ["npm", "ci", "--prefix", str(frontend)],
        capture_output=True,
        text=True,
        check=False,
    )
    if npm.returncode != 0:
        subprocess.run(
            ["npm", "install", "--prefix", str(frontend)],
            capture_output=True,
            text=True,
            check=False,
        )
    vitest = subprocess.run(
        ["npm", "run", "test", "--prefix", str(frontend)],
        capture_output=True,
        text=True,
        check=False,
    )
    passed = vitest.stdout.count(" passed") or (1 if vitest.returncode == 0 else 0)
    failed = vitest.stdout.count(" failed")
    return {
        "exit_code": vitest.returncode,
        "passed": 1 if vitest.returncode == 0 else 0,
        "failed": 0 if vitest.returncode == 0 else 1,
        "total": 1,
        "stdout_tail": (vitest.stdout or "")[-2000:],
        "stderr_tail": (vitest.stderr or "")[-2000:],
    }


def _parse_pytest_summary(stdout: str) -> dict[str, int]:
    import re

    line = ""
    for raw in reversed(stdout.splitlines()):
        if " passed" in raw or " failed" in raw or " error" in raw:
            line = raw.strip()
            break
    counts: dict[str, int] = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    for key in counts:
        m = re.search(rf"(\d+) {key[:-1] if key == 'errors' else key.rstrip('s')}", line)
        if not m:
            m = re.search(rf"(\d+) {key}", line)
        if m:
            counts[key] = int(m.group(1))
    counts["total"] = sum(counts.values())
    return counts


if __name__ == "__main__":
    raise SystemExit(main())
