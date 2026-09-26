"""Metrics and confusion matrix (Agent 4)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.dataset import KaggleLandmarkDataset, SyntheticLandmarkDataset, processed_tensors_ready
from ml.model import build_model_from_config
from ml.preprocess.paths import load_config, processed_dir, reports_dir, repo_root
from ml.train import _collate, _resolve_device


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_label_names(config: dict[str, Any]) -> list[str]:
    root = processed_dir(config) / "kaggle_asl_signs"
    label_map_path = root / config["paths"]["label_map_filename"]
    if label_map_path.is_file():
        data = json.loads(label_map_path.read_text(encoding="utf-8"))
        if isinstance(data.get("labels"), list):
            return [str(x) for x in data["labels"]]
        idx_map = data.get("index_to_label") or data.get("label_to_index")
        if isinstance(idx_map, dict):
            if "label_to_index" in data:
                inv = {int(v): str(k) for k, v in data["label_to_index"].items()}
                return [inv[i] for i in sorted(inv)]
    stub = repo_root() / "config" / "stub_label_map.json"
    if stub.is_file():
        return [str(x) for x in json.loads(stub.read_text())["labels"]]
    return []


@torch.no_grad()
def _collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_labels: list[int] = []
    all_top1: list[int] = []
    all_top5: list[list[int]] = []
    for batch in loader:
        x = batch["tensor"].to(device)
        mask = batch["mask"].to(device)
        y = batch["label"].to(device)
        logits = model(x, mask)
        probs = torch.softmax(logits, dim=1)
        top5 = torch.topk(probs, k=min(5, probs.shape[1]), dim=1).indices.cpu().numpy()
        top1 = top5[:, 0]
        all_labels.extend(y.cpu().numpy().tolist())
        all_top1.extend(top1.tolist())
        all_top5.extend(top5.tolist())
    return (
        np.asarray(all_labels, dtype=np.int64),
        np.asarray(all_top1, dtype=np.int64),
        np.asarray(all_top5, dtype=np.int64),
    )


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    f1s: list[float] = []
    for c in range(num_classes):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        if tp == 0 and fp == 0 and fn == 0:
            continue
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision + recall == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * precision * recall / (precision + recall))
    return float(np.mean(f1s)) if f1s else 0.0


def _per_class_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    num_classes = len(labels)
    for c in range(num_classes):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        support = int((y_true == c).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        rows.append(
            {
                "class_index": c,
                "label": labels[c] if c < len(labels) else str(c),
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "support": support,
            }
        )
    return rows


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> list[list[int]]:
    mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred, strict=False):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            mat[t, p] += 1
    return mat.tolist()


def _top_k_accuracy(y_true: np.ndarray, top_k: np.ndarray, k: int) -> float:
    if top_k.shape[1] < k:
        k = top_k.shape[1]
    hits = 0
    for i, label in enumerate(y_true):
        if label in top_k[i, :k]:
            hits += 1
    return hits / max(len(y_true), 1)


def evaluate_main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Evaluate ASL landmark classifier")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)

    config = load_config()
    device = _resolve_device(bool(config.get("hardware", {}).get("prefer_cuda", True)))
    real_data = processed_tensors_ready(config)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    num_classes = int(ckpt["num_classes"])

    model = build_model_from_config(config, num_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    if real_data:
        dataset = KaggleLandmarkDataset(args.split, config=config)
        labels = _load_label_names(config)
        if len(labels) != num_classes:
            labels = [str(i) for i in range(num_classes)]
    else:
        dataset = SyntheticLandmarkDataset(64, num_classes, seed=123)
        labels = [f"synthetic_{i}" for i in range(num_classes)]

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate,
    )

    y_true, y_pred, y_top5 = _collect_predictions(model, loader, device)
    top1 = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    top5 = _top_k_accuracy(y_true, y_top5, 5)
    macro_f1 = _macro_f1(y_true, y_pred, num_classes)

    metrics: dict[str, Any] = {
        "generated_at_utc": _utc_now(),
        "agent": "Agent 4 — Model Training & Evaluation",
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "real_data_eval": real_data,
        "num_samples": int(len(y_true)),
        "num_classes": num_classes,
        "top1_accuracy": round(top1, 6),
        "top5_accuracy": round(top5, 6),
        "macro_f1": round(macro_f1, 6),
        "per_class": _per_class_metrics(y_true, y_pred, labels),
        "confusion_matrix": _confusion_matrix(y_true, y_pred, num_classes),
        "label_order": labels,
    }

    if not real_data:
        metrics["note"] = (
            "Metrics computed on synthetic smoke data only; not representative of ASL accuracy."
        )

    out_path = reports_dir() / "metrics.json"
    _write_json(out_path, metrics)

    try:
        import matplotlib.pyplot as plt

        fig_path = reports_dir() / "confusion_matrix.png"
        mat = np.asarray(metrics["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(mat, cmap="Blues")
        ax.set_title(f"Confusion matrix ({args.split})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig.tight_layout()
        fig.savefig(fig_path, dpi=120)
        plt.close(fig)
        metrics["confusion_matrix_image"] = str(fig_path)
        _write_json(out_path, metrics)
    except ImportError:
        pass

    return metrics


if __name__ == "__main__":
    evaluate_main()
