"""PyTorch datasets for preprocessed Kaggle tensors and synthetic smoke data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ml.preprocess.paths import load_config, processed_dir


class KaggleLandmarkDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        split: str = "train",
        *,
        config: dict[str, Any] | None = None,
        transform: Any | None = None,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split: {split}")
        self.config = config or load_config()
        self.split = split
        self.transform = transform
        root = processed_dir(self.config) / "kaggle_asl_signs"
        index_path = root / "splits" / f"{split}_tensor_index.json"
        if not index_path.is_file():
            raise FileNotFoundError(
                f"Missing tensor index {index_path}. "
                "Run data/scripts/run_preprocessing.py after Kaggle download."
            )
        self.entries = json.loads(index_path.read_text(encoding="utf-8"))
        self.root = root

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        entry = self.entries[idx]
        npz_path = self.root / entry["path"]
        data = np.load(npz_path)
        tensor = torch.from_numpy(data["tensor"].astype(np.float32))
        mask = torch.from_numpy(data["mask"].astype(np.float32))
        label = int(data["label_index"])
        sample = {
            "tensor": tensor,
            "mask": mask,
            "label": label,
            "sign": str(data["sign"]),
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class SyntheticLandmarkDataset(Dataset[dict[str, Any]]):
    """Deterministic random sequences for pipeline smoke tests without real data."""

    def __init__(
        self,
        num_samples: int,
        num_classes: int,
        *,
        sequence_length: int = 64,
        feature_dim: int = 392,
        seed: int = 42,
    ) -> None:
        if num_samples < 1 or num_classes < 2:
            raise ValueError("num_samples >= 1 and num_classes >= 2 required")
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        rng = np.random.default_rng(seed)
        self._labels = rng.integers(0, num_classes, size=num_samples)
        self._tensors = rng.standard_normal(
            (num_samples, sequence_length, feature_dim)
        ).astype(np.float32)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "tensor": torch.from_numpy(self._tensors[idx]),
            "mask": torch.ones(self.sequence_length, dtype=torch.float32),
            "label": int(self._labels[idx]),
            "sign": f"synthetic_{self._labels[idx]}",
        }


def processed_tensors_ready(config: dict[str, Any] | None = None) -> bool:
    cfg = config or load_config()
    root = processed_dir(cfg) / "kaggle_asl_signs"
    train_index = root / "splits" / "train_tensor_index.json"
    label_map = root / cfg["paths"]["label_map_filename"]
    return train_index.is_file() and label_map.is_file()


def load_num_classes_from_label_map(config: dict[str, Any] | None = None) -> int:
    cfg = config or load_config()
    root = processed_dir(cfg) / "kaggle_asl_signs"
    label_map_path = root / cfg["paths"]["label_map_filename"]
    if not label_map_path.is_file():
        raise FileNotFoundError(label_map_path)
    data = json.loads(label_map_path.read_text(encoding="utf-8"))
    if "num_classes" in data:
        return int(data["num_classes"])
    labels = data.get("labels") or data.get("label_to_index")
    if isinstance(labels, list):
        return len(labels)
    if isinstance(labels, dict):
        return len(labels)
    raise ValueError(f"Cannot infer num_classes from {label_map_path}")
