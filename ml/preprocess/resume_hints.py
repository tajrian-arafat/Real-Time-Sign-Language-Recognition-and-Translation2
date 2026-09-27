"""Pipeline stage hints (no heavy ML imports)."""
from __future__ import annotations

from typing import Any


def suggest_next_command(status: dict[str, Any]) -> str:
    if not status["kaggle_data_ready"]:
        return "python data/scripts/download_kaggle_islr.py"

    if not status["processed_tensors_ready"]:
        return "python data/scripts/run_preprocessing.py"

    ckpt = status["checkpoints"]
    last = ckpt.get("last.pt")
    best = ckpt.get("best.pt")

    if status.get("served_onnx_exists") and best is not None:
        return "# Pipeline artifacts look complete; run backend/frontend or re-train if needed"

    if best is not None and not status.get("served_onnx_exists"):
        path = best["path"]
        return f"python -m ml.export_onnx --checkpoint {path}"

    if last is not None:
        path = last["path"]
        run_name = status.get("run_name", "kaggle_baseline")
        return f"python -m ml.train --resume {path} --run-name {run_name}"

    run_name = status.get("run_name", "kaggle_baseline")
    return f"python -m ml.train --run-name {run_name}"
