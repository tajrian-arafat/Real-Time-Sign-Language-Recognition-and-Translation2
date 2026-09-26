#!/usr/bin/env python3
"""Orchestrate Tier-1 dataset acquisition and write reports/data_acquisition.json."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

MAX_RETRIES = 3
ASL_CITIZEN_SOURCE_PAGE = (
    "https://www.microsoft.com/en-us/research/project/asl-citizen/"
)
ASL_CITIZEN_OFFICIAL_URL = (
    "https://download.microsoft.com/download/b/8/8/"
    "b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip"
)
WINDOWS_LOCAL_DATA_ROOT = r"D:\PROJECTS\sign language"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_data_root() -> Path:
    env = os.environ.get("SIGN_LANGUAGE_DATA_ROOT")
    if env:
        return Path(env)
    return repo_root() / "data"


def run_cmd(cmd: list[str], cwd: Path | None = None) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, combined.strip()


def kaggle_attempts() -> dict:
    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    api_key = os.environ.get("KAGGLE_KEY", "").strip()
    out: dict = {
        "competition": "asl-signs",
        "target_dir": str(resolve_data_root() / "raw" / "kaggle_asl_signs"),
        "credentials_present": bool(username and api_key),
        "attempts": [],
        "status": "blocked",
        "fallback_tier": None,
    }

    if not out["credentials_present"]:
        out["status"] = "blocked_missing_credentials"
        out["fallback_tier"] = (
            "Tier 3 — WLASL pretrained I3D checkpoint (see FALLBACK LOGIC: Kaggle failure path)"
        )
        out["message"] = (
            "KAGGLE_USERNAME and KAGGLE_KEY are not set. "
            "Obtain a free Kaggle API token and re-run acquisition."
        )
        return out

    kaggle_bin = os.environ.get("KAGGLE_BIN", "kaggle")
    if not shutil_which(kaggle_bin):
        kaggle_bin = str(Path.home() / ".local" / "bin" / "kaggle")
    out_dir = Path(out["target_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, MAX_RETRIES + 1):
        code, log = run_cmd(
            [
                kaggle_bin,
                "competitions",
                "download",
                "-c",
                "asl-signs",
                "-p",
                str(out_dir),
            ]
        )
        out["attempts"].append({"attempt": attempt, "exit_code": code, "log_tail": log[-2000:]})
        if code == 0:
            out["status"] = "downloaded"
            out["fallback_tier"] = None
            return out

    out["status"] = "failed_after_retries"
    out["fallback_tier"] = (
        "Tier 3 — WLASL pretrained I3D checkpoint (see FALLBACK LOGIC: Kaggle failure path)"
    )
    return out


def shutil_which(name: str) -> str | None:
    from shutil import which

    return which(name)


def asl_citizen_status() -> dict:
    data_root = resolve_data_root()
    out_dir = data_root / "raw" / "asl_citizen"
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "ASL_Citizen.zip"
    pid_file = out_dir / ".download.pid"

    status: dict = {
        "source_page": ASL_CITIZEN_SOURCE_PAGE,
        "official_url": ASL_CITIZEN_OFFICIAL_URL,
        "target_dir": str(out_dir),
        "zip_path": str(zip_path),
        "background_pid": None,
        "wget_log": str(out_dir / "wget_asl_citizen.log"),
        "status": "not_started",
        "fallback_tier": None,
    }

    # Expected full zip ~42.8 GB per Microsoft dataset page
    expected_zip_bytes = 40_000_000_000

    if pid_file.is_file():
        try:
            pid = int(pid_file.read_text().strip())
            if Path(f"/proc/{pid}").exists():
                status["background_pid"] = pid
                status["status"] = "in_progress"
                if zip_path.is_file():
                    status["zip_bytes_so_far"] = zip_path.stat().st_size
                return status
        except ValueError:
            pass

    if zip_path.is_file():
        size = zip_path.stat().st_size
        status["zip_bytes_so_far"] = size
        if size >= expected_zip_bytes:
            status["status"] = "download_complete_pending_integrity"
            return status
        # Partial file without active pid — may have failed mid-download
        if size > 0:
            status["status"] = "partial_or_stalled"
            status["fallback_tier"] = "Tier 2 — Hugging Face Voxel51/WLASL (video mirror)"

    log_path = Path(status["wget_log"])
    wget_cmd = (
        f"export SIGN_LANGUAGE_DATA_ROOT={data_root}; "
        f"wget -c --tries={MAX_RETRIES} --timeout=30 "
        f"-O {zip_path} {ASL_CITIZEN_OFFICIAL_URL} "
        f">> {log_path} 2>&1"
    )
    code, log = run_cmd(
        [
            "bash",
            "-lc",
            f"nohup bash -lc {json.dumps(wget_cmd)} </dev/null >/dev/null 2>&1 & echo $!",
        ]
    )
    if code != 0:
        status["status"] = "failed_to_start"
        status["log"] = log
        status["fallback_tier"] = "Tier 2 — Hugging Face Voxel51/WLASL (video mirror)"
        return status

    pid = log.strip().splitlines()[-1] if log else ""
    if pid.isdigit():
        pid_file.write_text(pid, encoding="utf-8")
        status["background_pid"] = int(pid)
        status["status"] = "started_background"
    else:
        status["status"] = "failed_to_start"
        status["log"] = log
        status["fallback_tier"] = "Tier 2 — Hugging Face Voxel51/WLASL (video mirror)"
    return status


def main() -> int:
    root = repo_root()
    sys.path.insert(0, str(root / "data" / "scripts"))
    from verify_integrity import verify_asl_citizen, verify_kaggle  # noqa: E402

    data_root = resolve_data_root()
    os.environ.setdefault("SIGN_LANGUAGE_DATA_ROOT", str(data_root))

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "agent": "Agent 2 — Data Acquisition",
        "data_root_cloud": str(data_root),
        "data_root_windows_local_documentation": WINDOWS_LOCAL_DATA_ROOT,
        "sign_language_data_root_env": os.environ.get("SIGN_LANGUAGE_DATA_ROOT"),
        "kaggle": kaggle_attempts(),
        "asl_citizen": asl_citizen_status(),
        "integrity": {
            "kaggle_asl_signs": verify_kaggle(data_root),
            "asl_citizen": verify_asl_citizen(data_root),
        },
        "max_retries": MAX_RETRIES,
    }

    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "data_acquisition.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
