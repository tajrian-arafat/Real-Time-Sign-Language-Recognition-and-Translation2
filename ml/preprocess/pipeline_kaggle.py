"""End-to-end Kaggle landmark preprocessing to cached tensors."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ml.preprocess.balance import apply_class_balance, save_balance_log
from ml.preprocess.kaggle_io import find_train_csv, kaggle_data_ready, load_parquet_landmarks
from ml.preprocess.landmark_spec import FEATURES_PER_FRAME, sync_landmark_indices_to_config_yaml
from ml.preprocess.normalize import normalize_and_pack
from ml.preprocess.paths import load_config, processed_dir, reports_dir, resolve_data_root
from ml.preprocess.sequence import resample_sequence
from ml.preprocess.splits import load_train_rows, split_by_participant, write_split_manifest
from ml.preprocess.vocabulary import build_label_map_from_train_csv, write_label_map


def _config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(
        {
            "T": config["landmarks"]["sequence_length_T"],
            "indices": config["landmarks"].get("pose_landmark_indices", []),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _resolve_parquet_path(kaggle_dir: Path, rel_path: str) -> Path:
    candidate = kaggle_dir / rel_path
    if candidate.is_file():
        return candidate
    # train.csv paths are often relative to landmark root subfolder
    matches = list(kaggle_dir.rglob(Path(rel_path).name))
    if len(matches) == 1:
        return matches[0]
    if matches:
        return matches[0]
    raise FileNotFoundError(rel_path)


def process_sequence(
    parquet_path: Path,
    *,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    holistic = load_parquet_landmarks(parquet_path)
    features = normalize_and_pack(holistic)
    resampled, mask = resample_sequence(features, target_length=sequence_length)
    if not np.isfinite(resampled).all():
        raise ValueError(f"non-finite tensor after preprocessing: {parquet_path}")
    return resampled, mask


def run_kaggle_preprocessing(
    *,
    max_sequences: int | None = None,
) -> dict[str, Any]:
    sync_landmark_indices_to_config_yaml()
    config = load_config()
    data_root = resolve_data_root(config)
    kaggle_dir = data_root / config["paths"]["raw_dir"] / config["datasets"]["kaggle_raw_subdir"]
    out_processed = processed_dir(config) / "kaggle_asl_signs"
    sequence_length = int(config["landmarks"]["sequence_length_T"])

    ready, reason = kaggle_data_ready(kaggle_dir)
    if not ready:
        return {
            "status": "waiting_for_kaggle",
            "reason": reason,
            "kaggle_dir": str(kaggle_dir),
        }

    train_csv = find_train_csv(kaggle_dir)
    assert train_csv is not None

    label_map = build_label_map_from_train_csv(train_csv)
    label_map_path = out_processed / config["paths"]["label_map_filename"]
    write_label_map(label_map, label_map_path)

    rows = load_train_rows(train_csv)
    split_cfg = config["datasets"]["splits"]
    split_data = split_by_participant(
        rows,
        train_ratio=float(split_cfg["train_ratio"]),
        val_ratio=float(split_cfg["val_ratio"]),
        test_ratio=float(split_cfg["test_ratio"]),
    )
    write_split_manifest(split_data, out_processed / "splits")

    cache_dir = out_processed / "tensors" / _config_hash(config)
    cache_dir.mkdir(parents=True, exist_ok=True)

    entries_by_split: dict[str, list[dict[str, Any]]] = {k: [] for k in ("train", "val", "test")}
    processed_count = 0

    for split_name, split_rows in split_data["sequences"].items():
        for row in split_rows:
            if max_sequences is not None and processed_count >= max_sequences:
                break
            rel = row["path"]
            pq = _resolve_parquet_path(kaggle_dir, rel)
            tensor, mask = process_sequence(pq, sequence_length=sequence_length)
            sign = row["sign"]
            label_index = label_map["label_to_index"][sign]
            seq_id = Path(rel).stem
            out_path = cache_dir / f"{seq_id}.npz"
            np.savez_compressed(
                out_path,
                tensor=tensor,
                mask=mask,
                label_index=label_index,
                sign=sign,
            )
            entries_by_split[split_name].append(
                {
                    "path": str(out_path.relative_to(out_processed)),
                    "sign": sign,
                    "label_index": label_index,
                    "participant_id": row["participant_id"],
                    "sequence_id": row.get("sequence_id", seq_id),
                }
            )
            processed_count += 1

    balance_cfg = config["datasets"]["class_balance"]
    balance_logs: dict[str, Any] = {}
    for split_name in ("train", "val", "test"):
        if split_name == "train":
            balanced, blog = apply_class_balance(
                entries_by_split[split_name],
                oversample_floor=int(balance_cfg["oversample_floor"]),
                drop_below=int(balance_cfg["drop_below"]),
            )
            entries_by_split[split_name] = balanced
            save_balance_log(blog, out_processed / "balance_train.json")
            balance_logs["train"] = blog
        manifest_path = out_processed / "splits" / f"{split_name}_tensor_index.json"
        manifest_path.write_text(
            json.dumps(entries_by_split[split_name], indent=2) + "\n", encoding="utf-8"
        )

    report = {
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "agent": "Agent 3 — Data Preprocessing",
        "vocabulary_size": label_map["vocabulary_size"],
        "sequences_processed": processed_count,
        "tensor_shape": [sequence_length, FEATURES_PER_FRAME],
        "label_map_path": str(label_map_path),
        "processed_root": str(out_processed),
        "cache_dir": str(cache_dir),
        "split_counts": split_data["counts"],
        "balance": balance_logs,
    }
    status_path = reports_dir(config) / "preprocessing_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
