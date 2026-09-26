"""PyTorch Dataset over preprocessed Kaggle landmark tensors."""
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
                f"Missing tensor index {index_path}. Run data/scripts/run_preprocessing.py first."
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
