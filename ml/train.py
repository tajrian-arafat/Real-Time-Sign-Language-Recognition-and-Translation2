"""Resumable training loop (Agent 4)."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ml.dataset import (
    KaggleLandmarkDataset,
    SyntheticLandmarkDataset,
    load_num_classes_from_label_map,
    processed_tensors_ready,
)
from ml.model import SignLanguageClassifier, build_model_from_config
from ml.preprocess.paths import load_config, processed_dir, reports_dir, repo_root


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve_device(prefer_cuda: bool) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _batch_size_for_device(config: dict[str, Any], device: torch.device) -> int:
    training = config.get("training", {})
    if device.type == "cuda":
        return int(training.get("batch_size", 128))
    return int(training.get("batch_size_low_memory", 32))


def _scaled_lr(config: dict[str, Any], batch_size: int) -> float:
    training = config.get("training", {})
    base_lr = float(training.get("base_lr", 3e-4))
    base_bs = int(training.get("base_lr_batch_size", 128))
    return base_lr * (batch_size / base_bs)


class _WarmupCosineScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        total_steps: int,
        warmup_steps: int,
        base_lr: float,
    ) -> None:
        self.optimizer = optimizer
        self.total_steps = max(total_steps, 1)
        self.warmup_steps = max(warmup_steps, 0)
        self.base_lr = base_lr
        self.step_num = 0

    def step(self) -> None:
        self.step_num += 1
        if self.step_num <= self.warmup_steps and self.warmup_steps > 0:
            lr = self.base_lr * (self.step_num / self.warmup_steps)
        else:
            progress = (self.step_num - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps, 1
            )
            progress = min(max(progress, 0.0), 1.0)
            lr = self.base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in self.optimizer.param_groups:
            group["lr"] = lr


def _collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    tensors = torch.stack([b["tensor"] for b in batch], dim=0)
    masks = torch.stack([b["mask"] for b in batch], dim=0)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return {"tensor": tensors, "mask": masks, "label": labels}


@torch.no_grad()
def _top1_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum().item()
    return correct / max(labels.numel(), 1)


def _run_epoch(
    model: SignLanguageClassifier,
    loader: DataLoader,
    *,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: _WarmupCosineScheduler | None,
    max_steps: int | None,
    global_step: int,
    grad_clip_norm: float = 0.0,
) -> tuple[float, float, int]:
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_acc = 0.0
    batches = 0
    steps = global_step

    for batch in loader:
        x = batch["tensor"].to(device)
        mask = batch["mask"].to(device)
        y = batch["label"].to(device)
        with torch.set_grad_enabled(is_train):
            logits = model(x, mask)
            loss = criterion(logits, y)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                steps += 1

        total_loss += float(loss.item())
        total_acc += _top1_accuracy(logits.detach(), y)
        batches += 1

        if is_train and (not math.isfinite(float(loss.item()))):
            raise RuntimeError(f"Non-finite training loss at step {steps}")

        if is_train and max_steps is not None and steps >= max_steps:
            break

    if batches == 0:
        return 0.0, 0.0, steps
    return total_loss / batches, total_acc / batches, steps


def _save_checkpoint(
    path: Path,
    *,
    model: SignLanguageClassifier,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    best_val_acc: float,
    config: dict[str, Any],
    num_classes: int,
    run_mode: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "best_val_acc": best_val_acc,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "num_classes": num_classes,
            "run_mode": run_mode,
            "config_snapshot": {
                "landmarks": config.get("landmarks", {}),
                "model": config.get("model", {}),
                "training": config.get("training", {}),
            },
        },
        path,
    )


def _load_checkpoint(
    path: Path,
    model: SignLanguageClassifier,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(data["model_state_dict"])
    optimizer.load_state_dict(data["optimizer_state_dict"])
    return data


def train_main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Train ASL landmark classifier")
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint to resume")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after N train steps")
    parser.add_argument("--run-name", type=str, default="kaggle_baseline")
    parser.add_argument("--force-synthetic", action="store_true")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override LR (default: scaled from config; use 1e-4 for stable Kaggle runs)",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=None,
        help="Max gradient norm (default: training.grad_clip_norm or 1.0 on real data)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size (default: config training.batch_size on CUDA, batch_size_low_memory on CPU)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers (default: max(1, cpu_count-1) on CPU; config training.num_workers on CUDA)",
    )
    args = parser.parse_args(argv)

    config = load_config()
    training_cfg = config.get("training", {})
    hardware = config.get("hardware", {})
    device = _resolve_device(bool(hardware.get("prefer_cuda", True)))
    if args.batch_size is not None:
        batch_size = int(args.batch_size)
    else:
        batch_size = _batch_size_for_device(config, device)
    if args.num_workers is not None:
        num_workers = int(args.num_workers)
    elif device.type == "cpu":
        num_workers = max(1, (os.cpu_count() or 2) - 1)
    else:
        num_workers = int(training_cfg.get("num_workers", 3))

    real_data = processed_tensors_ready(config) and not args.force_synthetic
    run_mode = "kaggle_processed" if real_data else "synthetic_smoke"

    if real_data:
        num_classes = load_num_classes_from_label_map(config)
        train_ds: Dataset = KaggleLandmarkDataset("train", config=config)
        val_ds: Dataset = KaggleLandmarkDataset("val", config=config)
    else:
        num_classes = 16
        train_ds = SyntheticLandmarkDataset(128, num_classes)
        val_ds = SyntheticLandmarkDataset(32, num_classes, seed=99)

    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "collate_fn": _collate,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    model = build_model_from_config(config, num_classes).to(device)
    label_smoothing = float(training_cfg.get("label_smoothing", 0.1))
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    weight_decay = float(training_cfg.get("weight_decay", 0.01))
    if args.learning_rate is not None:
        lr = float(args.learning_rate)
    else:
        lr = _scaled_lr(config, batch_size)
    grad_clip = args.grad_clip_norm
    if grad_clip is None:
        grad_clip = float(training_cfg.get("grad_clip_norm", 1.0 if real_data else 0.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    max_epochs = args.epochs
    if max_epochs is None:
        max_epochs = int(training_cfg.get("max_epochs", 60)) if real_data else 2
    if not real_data and args.max_steps is None:
        args.max_steps = 20

    steps_per_epoch = max(len(train_loader), 1)
    total_steps = steps_per_epoch * max_epochs
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    warmup_ratio = float(training_cfg.get("lr_warmup_ratio", 0.05))
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = _WarmupCosineScheduler(
        optimizer,
        total_steps=max(total_steps, 1),
        warmup_steps=warmup_steps,
        base_lr=lr,
    )

    ckpt_dir = repo_root() / config["paths"]["models_dir"] / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best.pt"
    last_path = ckpt_dir / "last.pt"

    start_epoch = 0
    global_step = 0
    best_val_acc = -1.0
    patience = int(training_cfg.get("early_stopping_patience", 8))
    stale_epochs = 0

    if args.resume and args.resume.is_file():
        meta = _load_checkpoint(args.resume, model, optimizer)
        start_epoch = int(meta.get("epoch", 0)) + 1
        global_step = int(meta.get("global_step", 0))
        best_val_acc = float(meta.get("best_val_acc", -1.0))

    started = time.perf_counter()
    history: list[dict[str, Any]] = []

    for epoch in range(start_epoch, max_epochs):
        train_loss, train_acc, global_step = _run_epoch(
            model,
            train_loader,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            max_steps=args.max_steps,
            global_step=global_step,
            grad_clip_norm=grad_clip,
        )
        val_loss, val_acc, _ = _run_epoch(
            model,
            val_loader,
            device=device,
            criterion=criterion,
            optimizer=None,
            scheduler=None,
            max_steps=None,
            global_step=global_step,
            grad_clip_norm=0.0,
        )
        if epoch == 0 and not math.isfinite(train_loss):
            raise RuntimeError(f"Non-finite train loss after epoch 0: {train_loss}")
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_top1_acc": train_acc,
                "val_loss": val_loss,
                "val_top1_acc": val_acc,
                "global_step": global_step,
            }
        )
        _save_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            best_val_acc=best_val_acc,
            config=config,
            num_classes=num_classes,
            run_mode=run_mode,
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            stale_epochs = 0
            _save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                best_val_acc=best_val_acc,
                config=config,
                num_classes=num_classes,
                run_mode=run_mode,
            )
        else:
            stale_epochs += 1

        if args.max_steps is not None and global_step >= args.max_steps:
            break
        if real_data and stale_epochs >= patience:
            break

    elapsed = time.perf_counter() - started
    status: dict[str, Any] = {
        "generated_at_utc": _utc_now(),
        "agent": "Agent 4 — Model Training & Evaluation",
        "run_mode": run_mode,
        "real_data_available": real_data,
        "device": str(device),
        "batch_size": batch_size,
        "num_workers": num_workers,
        "learning_rate": lr,
        "grad_clip_norm": grad_clip if grad_clip > 0 else None,
        "num_classes": num_classes,
        "epochs_completed": len(history),
        "global_step": global_step,
        "best_val_top1_acc": best_val_acc if best_val_acc >= 0 else None,
        "checkpoint_best": str(best_path),
        "checkpoint_last": str(last_path),
        "elapsed_seconds": round(elapsed, 3),
        "history": history,
    }

    if not real_data:
        status["status"] = "waiting_for_kaggle_data"
        status["message"] = (
            "Synthetic smoke training completed; full 250-word training requires "
            "processed Kaggle tensors (accept competition rules and re-run acquisition + preprocessing)."
        )
    else:
        status["status"] = "completed"
        status["message"] = "Training finished on processed Kaggle tensors."

    _write_json(reports_dir() / "training_status.json", status)
    return status


if __name__ == "__main__":
    train_main()
