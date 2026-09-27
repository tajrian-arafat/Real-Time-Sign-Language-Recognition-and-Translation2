"""PyTorch → ONNX export with numerical parity check (Agent 4)."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch

from ml.model import SignLanguageClassifier, build_model_from_config
from ml.preprocess.paths import load_config, processed_dir, reports_dir, repo_root


class _OnnxExportWrapper(torch.nn.Module):
    """Single-input export graph matching backend/inference.py (landmarks only)."""

    def __init__(self, model: SignLanguageClassifier) -> None:
        super().__init__()
        self.model = model

    def forward(self, landmarks: torch.Tensor) -> torch.Tensor:
        mask = torch.ones(
            landmarks.shape[0],
            landmarks.shape[1],
            device=landmarks.device,
            dtype=landmarks.dtype,
        )
        return self.model(landmarks, mask)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def export_onnx(
    checkpoint: Path,
    *,
    output_path: Path | None = None,
    opset: int = 17,
) -> Path:
    config = load_config()
    landmarks = config.get("landmarks", {})
    sequence_t = int(landmarks.get("sequence_length_T", 64))
    feature_dim = int(landmarks.get("input_dim_per_frame", 392))

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    num_classes = int(ckpt["num_classes"])
    model = build_model_from_config(config, num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    export_model = _OnnxExportWrapper(model)
    export_model.eval()

    models_dir = repo_root() / config["paths"]["models_dir"]
    if output_path is None:
        output_path = models_dir / config["inference"]["served_onnx"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(1, sequence_t, feature_dim, dtype=torch.float32)

    export_kwargs: dict[str, Any] = {
        "input_names": ["landmarks"],
        "output_names": ["logits"],
        "dynamic_axes": {
            "landmarks": {0: "batch"},
            "logits": {0: "batch"},
        },
        "opset_version": opset,
        "do_constant_folding": True,
    }
    # Legacy exporter avoids dynamo batch-shape bugs with TransformerEncoder on ORT.
    try:
        torch.onnx.export(
            export_model,
            dummy,
            str(output_path),
            dynamo=False,
            **export_kwargs,
        )
    except TypeError:
        torch.onnx.export(export_model, dummy, str(output_path), **export_kwargs)

    run_dir = checkpoint.parent
    label_src = processed_dir(config) / "kaggle_asl_signs" / config["paths"]["label_map_filename"]
    label_dst = output_path.parent / "label_map.json"
    if label_src.is_file():
        shutil.copy2(label_src, label_dst)
    elif (run_dir / "label_map.json").is_file():
        shutil.copy2(run_dir / "label_map.json", label_dst)
    else:
        stub = repo_root() / "config" / "stub_label_map.json"
        if num_classes == len(json.loads(stub.read_text())["labels"]):
            shutil.copy2(stub, label_dst)
        else:
            labels = [f"class_{i}" for i in range(num_classes)]
            label_dst.write_text(
                json.dumps({"labels": labels, "num_classes": num_classes}, indent=2),
                encoding="utf-8",
            )

    return output_path


def verify_onnx_parity(
    checkpoint: Path,
    onnx_path: Path,
    *,
    atol: float = 1e-4,
    rtol: float = 1e-4,
    batch_size: int = 4,
) -> dict[str, Any]:
    config = load_config()
    landmarks = config.get("landmarks", {})
    sequence_t = int(landmarks.get("sequence_length_T", 64))
    feature_dim = int(landmarks.get("input_dim_per_frame", 392))

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    num_classes = int(ckpt["num_classes"])
    model = build_model_from_config(config, num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    export_model = _OnnxExportWrapper(model)
    export_model.eval()

    rng = np.random.default_rng(7)
    landmarks_np = rng.standard_normal((batch_size, sequence_t, feature_dim)).astype(
        np.float32
    )

    with torch.no_grad():
        torch_logits = export_model(torch.from_numpy(landmarks_np)).cpu().numpy()

    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    ort_out = session.run(None, {"landmarks": landmarks_np})[0]

    max_abs = float(np.max(np.abs(torch_logits - ort_out)))
    ok = bool(np.allclose(torch_logits, ort_out, atol=atol, rtol=rtol))

    # Also verify batch=1 (minimum serving shape).
    if batch_size != 1:
        one_np = landmarks_np[:1]
        with torch.no_grad():
            torch_one = export_model(torch.from_numpy(one_np)).cpu().numpy()
        ort_one = session.run(None, {"landmarks": one_np})[0]
        one_ok = bool(np.allclose(torch_one, ort_one, atol=atol, rtol=rtol))
        ok = ok and one_ok
        max_abs = max(max_abs, float(np.max(np.abs(torch_one - ort_one))))

    return {
        "generated_at_utc": _utc_now(),
        "checkpoint": str(checkpoint),
        "onnx_path": str(onnx_path),
        "batch_size": batch_size,
        "max_abs_diff": max_abs,
        "atol": atol,
        "rtol": rtol,
        "parity_passed": ok,
    }


def export_main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Export checkpoint to ONNX")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    onnx_path = export_onnx(args.checkpoint, output_path=args.output)
    parity = verify_onnx_parity(args.checkpoint, onnx_path)
    report = {
        "generated_at_utc": _utc_now(),
        "agent": "Agent 4 — Model Training & Evaluation",
        "onnx_export": str(onnx_path),
        "parity": parity,
    }
    _write_json(reports_dir() / "onnx_export.json", report)
    if not parity["parity_passed"]:
        raise RuntimeError(
            f"ONNX parity failed: max_abs_diff={parity['max_abs_diff']}"
        )
    return report


if __name__ == "__main__":
    export_main()
