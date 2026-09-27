"""Unit tests for pipeline status suggestions."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ml.preprocess.resume_hints import suggest_next_command  # noqa: E402


def test_suggest_download_when_kaggle_missing() -> None:
    status = {"kaggle_data_ready": False, "processed_tensors_ready": False, "checkpoints": {}}
    assert "download_kaggle_islr" in suggest_next_command(status)


def test_suggest_preprocess_when_raw_ready() -> None:
    status = {
        "kaggle_data_ready": True,
        "processed_tensors_ready": False,
        "checkpoints": {},
        "run_name": "kaggle_baseline",
    }
    assert suggest_next_command(status) == "python data/scripts/run_preprocessing.py"


def test_suggest_train_resume_when_last_exists() -> None:
    status = {
        "kaggle_data_ready": True,
        "processed_tensors_ready": True,
        "served_onnx_exists": False,
        "run_name": "kaggle_baseline",
        "checkpoints": {
            "last.pt": {"path": "models/kaggle_baseline/last.pt"},
            "best.pt": None,
        },
    }
    cmd = suggest_next_command(status)
    assert "--resume models/kaggle_baseline/last.pt" in cmd


def test_suggest_export_when_best_but_no_onnx() -> None:
    status = {
        "kaggle_data_ready": True,
        "processed_tensors_ready": True,
        "served_onnx_exists": False,
        "run_name": "kaggle_baseline",
        "checkpoints": {
            "last.pt": {"path": "models/kaggle_baseline/last.pt"},
            "best.pt": {"path": "models/kaggle_baseline/best.pt"},
        },
    }
    cmd = suggest_next_command(status)
    assert "export_onnx" in cmd
    assert "best.pt" in cmd
