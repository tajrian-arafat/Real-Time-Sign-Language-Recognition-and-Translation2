#!/usr/bin/env python3
"""Bootstrap cloud agents when snapshot disk is empty: download/preprocess if creds exist."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

MIN_NPZ = 90_000
REPO = Path(__file__).resolve().parents[1]


def _venv_python() -> Path:
    py = REPO / ".venv" / "bin" / "python"
    if py.is_file():
        return py
    return Path(sys.executable)


def _npz_count(data_root: Path) -> int:
    processed = data_root / "processed"
    if not processed.is_dir():
        return 0
    return sum(1 for _ in processed.rglob("*.npz"))


def _kaggle_creds_ok() -> bool:
    sys.path.insert(0, str(REPO / "data" / "scripts"))
    from kaggle_auth import kaggle_credentials_present  # noqa: WPS433

    return bool(kaggle_credentials_present())


def _run(py: Path, args: list[str], env: dict[str, str]) -> int:
    proc = subprocess.run(
        [str(py), *args],
        cwd=REPO,
        env=env,
        check=False,
    )
    return int(proc.returncode)


def main() -> int:
    data_root = Path(
        os.environ.get("SIGN_LANGUAGE_DATA_ROOT", str(REPO / "data"))
    )
    os.environ["SIGN_LANGUAGE_DATA_ROOT"] = str(data_root)
    py = _venv_python()

    report: dict = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "npz_count": _npz_count(data_root),
        "min_npz": MIN_NPZ,
        "action": "none",
    }

    onnx = REPO / "models" / "served" / "model.onnx"
    extended = REPO / "models" / "kaggle_extended_v1" / "best.pt"
    report["served_onnx"] = onnx.is_file()
    report["extended_best_pt"] = extended.is_file()

    if report["npz_count"] >= MIN_NPZ:
        report["action"] = "skip_data_ready"
        report["status"] = "ok"
        _write_report(report)
        print(
            f"ensure_cloud_pipeline: OK ({report['npz_count']} NPZ >= {MIN_NPZ})",
            file=sys.stderr,
        )
        return 0

    if not _kaggle_creds_ok():
        report["action"] = "blocked_missing_kaggle_credentials"
        report["status"] = "blocked"
        report["message"] = (
            "Set KAGGLE_API_TOKEN (or KAGGLE_USERNAME+KAGGLE_KEY) in Cursor My Secrets, "
            "then re-run or trigger a new environment build."
        )
        _write_report(report)
        print(report["message"], file=sys.stderr)
        return 1

    env = os.environ.copy()
    report["action"] = "download_and_preprocess"
    print("ensure_cloud_pipeline: running data acquisition...", file=sys.stderr)
    code = _run(py, ["scripts/run_data_acquisition.py"], env)
    if code != 0:
        report["status"] = "acquisition_failed"
        report["exit_code"] = code
        _write_report(report)
        return code

    kaggle_dir = data_root / "raw" / "kaggle_asl_signs"
    sys.path.insert(0, str(REPO))
    from ml.preprocess.kaggle_io import kaggle_data_ready  # noqa: WPS433

    if not kaggle_data_ready(kaggle_dir):
        report["status"] = "kaggle_raw_not_ready_after_download"
        _write_report(report)
        return 1

    print("ensure_cloud_pipeline: running preprocessing...", file=sys.stderr)
    code = _run(py, ["data/scripts/run_preprocessing.py"], env)
    report["npz_count_after"] = _npz_count(data_root)
    report["status"] = "ok" if report["npz_count_after"] >= MIN_NPZ else "preprocess_incomplete"
    report["exit_code"] = code
    _write_report(report)
    return 0 if report["npz_count_after"] >= MIN_NPZ else code or 1


def _write_report(report: dict) -> None:
    out = REPO / "reports" / "ensure_cloud_pipeline.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
