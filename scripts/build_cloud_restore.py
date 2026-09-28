#!/usr/bin/env python3
"""Assemble artifacts/cloud_restore for Hugging Face upload."""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
BUNDLE = REPO / "artifacts" / "cloud_restore"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    checkpoint = REPO / "models" / "kaggle_overnight_v1" / "best.pt"
    if not checkpoint.is_file():
        checkpoint = REPO / "models" / "kaggle_extended_v1" / "best.pt"
    onnx_src = REPO / "models" / "served" / "model.onnx"
    from ml.preprocess.paths import load_config, processed_dir

    config = load_config()
    processed = processed_dir(config)
    label_src = processed / config["paths"]["label_map_filename"]
    splits_src = processed / "splits"
    if not checkpoint.is_file():
        raise SystemExit(f"Missing checkpoint: {checkpoint}")
    if not onnx_src.is_file():
        raise SystemExit(f"Missing ONNX: {onnx_src}")
    if not label_src.is_file():
        raise SystemExit(f"Missing label_map: {label_src}")

    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    BUNDLE.mkdir(parents=True)
    (BUNDLE / "splits").mkdir()

    shutil.copy2(checkpoint, BUNDLE / "best.pt")
    shutil.copy2(onnx_src, BUNDLE / "model.onnx")
    shutil.copy2(label_src, BUNDLE / "label_map.json")
    for name in ("train_tensor_index.json", "val_tensor_index.json", "test_tensor_index.json"):
        src = splits_src / name
        if src.is_file():
            shutil.copy2(src, BUNDLE / "splits" / name)

    meta = torch.load(checkpoint, map_location="cpu", weights_only=False)
    manifest: dict = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_vm": "overnight cloud agent",
        "data_root_note": "NPZ tensors NOT included; restore requires snapshot or re-preprocess",
        "extended_checkpoint": {
            "best_val_acc": float(meta.get("best_val_acc", 0)),
            "epoch": int(meta.get("epoch", -1)),
            "run_mode": meta.get("run_mode", "unknown"),
        },
        "files": [],
    }
    for rel in (
        "best.pt",
        "label_map.json",
        "model.onnx",
        "splits/train_tensor_index.json",
        "splits/val_tensor_index.json",
        "splits/test_tensor_index.json",
    ):
        p = BUNDLE / rel
        if p.is_file():
            manifest["files"].append(
                {"path": rel, "size_bytes": p.stat().st_size, "sha256": _sha256(p)}
            )
    (BUNDLE / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
