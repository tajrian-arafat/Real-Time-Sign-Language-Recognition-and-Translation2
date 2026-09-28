#!/usr/bin/env python3
"""Restore served ONNX + extended checkpoint from HF hub when data exists but models missing."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_REPO = os.environ.get(
    "SIGN_LANGUAGE_RESTORE_HF_REPO", "Taalvi/sign-language-cloud-restore"
)


def main() -> int:
    onnx = REPO / "models" / "served" / "model.onnx"
    extended = REPO / "models" / "kaggle_extended_v1" / "best.pt"
    if onnx.is_file() and extended.is_file():
        print("restore_served_artifacts: models already present", file=sys.stderr)
        return 0

    hf_repo = DEFAULT_REPO
    dest = REPO / "artifacts" / "hf_restore"
    dest.mkdir(parents=True, exist_ok=True)

    cmd = [
        "huggingface-cli",
        "download",
        hf_repo,
        "--local-dir",
        str(dest),
        "--local-dir-use-symlinks",
        "False",
    ]
    proc = subprocess.run(cmd, cwd=REPO, check=False)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hf_repo": hf_repo,
        "download_exit_code": proc.returncode,
    }
    if proc.returncode != 0:
        report["status"] = "download_failed"
        _write(report)
        return proc.returncode

    src_onnx = dest / "model.onnx"
    src_pt = dest / "best.pt"
    if src_onnx.is_file():
        (REPO / "models" / "served").mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_onnx, onnx)
    if src_pt.is_file():
        (REPO / "models" / "kaggle_extended_v1").mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_pt, extended)

    report["served_onnx"] = onnx.is_file()
    report["extended_best_pt"] = extended.is_file()
    report["status"] = "ok" if report["served_onnx"] and report["extended_best_pt"] else "partial"
    _write(report)
    return 0 if report["status"] == "ok" else 1


def _write(report: dict) -> None:
    out = REPO / "reports" / "restore_served_artifacts.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
