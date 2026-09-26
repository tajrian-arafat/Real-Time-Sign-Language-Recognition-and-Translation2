#!/usr/bin/env python3
"""Integrity checks for downloaded datasets (Agent 2)."""
from __future__ import annotations

import csv
import json
import os
import zipfile
from pathlib import Path
from typing import Any


def resolve_data_root() -> Path:
    env = os.environ.get("SIGN_LANGUAGE_DATA_ROOT")
    if env:
        return Path(env)
    repo = Path(__file__).resolve().parents[2]
    return repo / "data"


def verify_kaggle(root: Path) -> dict[str, Any]:
    kaggle_dir = root / "raw" / "kaggle_asl_signs"
    result: dict[str, Any] = {
        "path": str(kaggle_dir),
        "status": "missing",
        "parquet_count": 0,
        "train_csv_rows": None,
        "match": None,
        "notes": [],
    }
    if not kaggle_dir.is_dir():
        result["notes"].append("Directory does not exist")
        return result

    archives = list(kaggle_dir.glob("*.zip"))
    if archives and not any(kaggle_dir.glob("*.parquet")):
        result["notes"].append(
            f"Found archive(s) {', '.join(a.name for a in archives)}; extract before parquet check"
        )

    parquets = list(kaggle_dir.rglob("*.parquet"))
    result["parquet_count"] = len(parquets)

    train_csv_candidates = list(kaggle_dir.rglob("train.csv"))
    if not train_csv_candidates:
        result["status"] = "incomplete"
        result["notes"].append("train.csv not found")
        return result

    train_csv = train_csv_candidates[0]
    with train_csv.open(newline="", encoding="utf-8") as f:
        row_count = sum(1 for _ in csv.reader(f)) - 1  # header
    result["train_csv_rows"] = row_count
    result["match"] = row_count == len(parquets)
    result["status"] = "verified" if result["match"] else "mismatch"
    if not result["match"]:
        result["notes"].append(
            f"Expected parquet count {row_count}, found {len(parquets)}"
        )
    return result


def verify_asl_citizen(root: Path) -> dict[str, Any]:
    citizen_dir = root / "raw" / "asl_citizen"
    zip_path = citizen_dir / "ASL_Citizen.zip"
    # ~84k videos per master prompt / Microsoft page
    expected_video_min = 70_000
    expected_video_max = 95_000

    result: dict[str, Any] = {
        "path": str(citizen_dir),
        "status": "missing",
        "zip_path": str(zip_path),
        "zip_bytes": None,
        "zip_complete": None,
        "extracted_video_count": 0,
        "expected_video_range": [expected_video_min, expected_video_max],
        "in_expected_range": None,
        "notes": [],
    }

    if not citizen_dir.is_dir():
        result["notes"].append("Directory does not exist")
        return result

    pid_file = citizen_dir / ".download.pid"
    if pid_file.is_file():
        try:
            pid = int(pid_file.read_text().strip())
            if Path(f"/proc/{pid}").exists():
                result["status"] = "in_progress"
                result["notes"].append(f"Background download active (pid {pid})")
                if zip_path.is_file():
                    result["zip_bytes"] = zip_path.stat().st_size
                return result
        except ValueError:
            pass

    if zip_path.is_file():
        size = zip_path.stat().st_size
        result["zip_bytes"] = size
        if size == 0:
            result["status"] = "in_progress"
            result["notes"].append("Zip file exists but is empty (download starting)")
            return result
        try:
            with zipfile.ZipFile(zip_path) as zf:
                names = [n for n in zf.namelist() if n.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
                result["extracted_video_count"] = len(names)
                bad = zf.testzip()
                result["zip_complete"] = bad is None
                if bad:
                    result["notes"].append(f"Zip corrupt at member: {bad}")
                    result["status"] = "corrupt"
                    return result
        except zipfile.BadZipFile:
            result["status"] = "in_progress"
            result["notes"].append("Zip incomplete or still downloading (BadZipFile)")
            return result

        if result["zip_complete"]:
            in_range = expected_video_min <= result["extracted_video_count"] <= expected_video_max
            result["in_expected_range"] = in_range
            result["status"] = "verified" if in_range else "mismatch"
            if not in_range:
                result["notes"].append(
                    f"Video count {result['extracted_video_count']} outside expected range"
                )
        return result

    videos = list(citizen_dir.rglob("*.mp4"))
    result["extracted_video_count"] = len(videos)
    if videos:
        in_range = expected_video_min <= len(videos) <= expected_video_max
        result["in_expected_range"] = in_range
        result["status"] = "verified" if in_range else "mismatch"
    else:
        result["notes"].append("No zip or extracted videos found")
    return result


def main() -> int:
    root = resolve_data_root()
    report = {
        "data_root": str(root),
        "kaggle_asl_signs": verify_kaggle(root),
        "asl_citizen": verify_asl_citizen(root),
    }
    reports_dir = Path(__file__).resolve().parents[2] / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "data_acquisition_integrity.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
