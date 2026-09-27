#!/usr/bin/env python3
"""
MediaPipe HolisticLandmarker extraction for ASL Citizen (Agent 3).

Does NOT run until Agent 2 verifies the ASL_Citizen.zip download (integrity verified).
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import importlib.util

_verify_path = Path(__file__).parent / "verify_integrity.py"
_spec = importlib.util.spec_from_file_location("verify_integrity", _verify_path)
assert _spec and _spec.loader
verify_integrity = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verify_integrity)


def main() -> int:
    root = verify_integrity.resolve_data_root()
    citizen = verify_integrity.verify_asl_citizen(root)
    if citizen.get("status") != "verified":
        msg = {
            "status": "blocked",
            "reason": "ASL Citizen zip not integrity-verified",
            "integrity": citizen,
        }
        print(json.dumps(msg, indent=2), file=sys.stderr)
        return 2

    zip_path = Path(citizen["zip_path"])
    with zipfile.ZipFile(zip_path) as zf:
        video_count = sum(
            1 for n in zf.namelist() if n.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
        )
    print(
        json.dumps(
            {
                "status": "not_implemented",
                "message": (
                    "Extraction pipeline stub: zip verified with "
                    f"{video_count} videos. Implement HolisticLandmarker batching in a follow-up."
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
