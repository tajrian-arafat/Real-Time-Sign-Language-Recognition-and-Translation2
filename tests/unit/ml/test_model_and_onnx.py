"""Unit tests for classifier forward pass and ONNX parity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ml.export_onnx import export_onnx, verify_onnx_parity
from ml.model import SignLanguageClassifier
from ml.train import train_main


def test_sign_language_classifier_forward() -> None:
    model = SignLanguageClassifier(num_classes=8, input_dim=392)
    x = torch.randn(2, 64, 392)
    mask = torch.ones(2, 64)
    logits = model(x, mask)
    assert logits.shape == (2, 8)


def test_synthetic_train_export_parity(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "models" / "smoke"
    ckpt_dir.mkdir(parents=True)
    status = train_main(
        ["--force-synthetic", "--max-steps", "5", "--epochs", "1", "--run-name", "smoke"]
    )
    assert status["status"] == "waiting_for_kaggle_data"
    best = Path(status["checkpoint_best"])
    assert best.is_file()

    onnx_path = tmp_path / "model.onnx"
    export_onnx(best, output_path=onnx_path)
    parity = verify_onnx_parity(best, onnx_path)
    assert parity["parity_passed"] is True
    assert parity["max_abs_diff"] < 1e-3
