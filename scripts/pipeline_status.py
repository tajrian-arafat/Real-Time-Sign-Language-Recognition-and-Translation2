#!/usr/bin/env python3
"""Print pipeline artifact counts and suggest the next command."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ml.preprocess.kaggle_io import find_train_csv, kaggle_data_ready  # noqa: E402
from ml.preprocess.paths import (  # noqa: E402
    kaggle_raw_dir,
    load_config,
    processed_dir,
    repo_root,
    resolve_data_root,
)
from ml.preprocess.pipeline_kaggle import _config_hash  # noqa: E402
from ml.preprocess.resume_hints import suggest_next_command  # noqa: E402


def _count_parquets(kaggle_dir: Path) -> int:
    if not kaggle_dir.is_dir():
        return 0
    return sum(1 for _ in kaggle_dir.rglob("*.parquet"))


def _count_npz(cache_dir: Path) -> int:
    if not cache_dir.is_dir():
        return 0
    return sum(1 for _ in cache_dir.glob("*.npz"))


def _checkpoint_info(run_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"run_dir": str(run_dir)}
    for name in ("last.pt", "best.pt"):
        path = run_dir / name
        if path.is_file():
            stat = path.stat()
            out[name] = {
                "path": str(path),
                "size_bytes": stat.st_size,
                "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            }
        else:
            out[name] = None
    return out


def collect_status(
    *,
    config: dict[str, Any] | None = None,
    run_name: str = "kaggle_baseline",
) -> dict[str, Any]:
    cfg = config or load_config()
    data_root = resolve_data_root(cfg)
    kaggle_dir = kaggle_raw_dir(cfg)
    processed_root = processed_dir(cfg) / "kaggle_asl_signs"
    cache_dir = processed_root / "tensors" / _config_hash(cfg)
    models_dir = repo_root() / cfg["paths"]["models_dir"] / run_name
    served_onnx = repo_root() / cfg["paths"]["models_dir"] / cfg["inference"]["served_onnx"]

    from ml.dataset import processed_tensors_ready

    kaggle_ready, kaggle_reason = kaggle_data_ready(kaggle_dir)
    train_csv = find_train_csv(kaggle_dir)
    tensors_ready = processed_tensors_ready(cfg)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "kaggle_dir": str(kaggle_dir),
        "kaggle_data_ready": kaggle_ready,
        "kaggle_ready_detail": kaggle_reason,
        "train_csv": str(train_csv) if train_csv else None,
        "parquet_count": _count_parquets(kaggle_dir),
        "preprocess_config_hash": _config_hash(cfg),
        "tensor_cache_dir": str(cache_dir),
        "npz_count": _count_npz(cache_dir),
        "processed_tensors_ready": tensors_ready,
        "processed_root": str(processed_root),
        "checkpoints": _checkpoint_info(models_dir),
        "served_onnx": str(served_onnx),
        "served_onnx_exists": served_onnx.is_file(),
        "run_name": run_name,
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Kaggle pipeline status")
    parser.add_argument("--run-name", default="kaggle_baseline")
    parser.add_argument("--json", action="store_true", help="Emit JSON only")
    args = parser.parse_args()

    status = collect_status(run_name=args.run_name)
    status["suggested_next"] = suggest_next_command(status)

    if args.json:
        print(json.dumps(status, indent=2))
    else:
        print(f"Data root: {status['data_root']}")
        print(f"Kaggle raw: {status['kaggle_dir']}")
        print(
            f"  parquets: {status['parquet_count']} "
            f"(ready={status['kaggle_data_ready']}: {status['kaggle_ready_detail']})"
        )
        if status["train_csv"]:
            print(f"  train.csv: {status['train_csv']}")
        print(f"Preprocess hash: {status['preprocess_config_hash']}")
        print(f"  NPZ cache: {status['npz_count']} under {status['tensor_cache_dir']}")
        print(f"  tensors ready for training: {status['processed_tensors_ready']}")
        ck = status["checkpoints"]
        print(f"Checkpoints ({ck['run_dir']}):")
        for key in ("last.pt", "best.pt"):
            info = ck.get(key)
            if info:
                print(f"  {key}: {info['path']} ({info['size_bytes']} bytes)")
            else:
                print(f"  {key}: (missing)")
        print(f"Served ONNX: {status['served_onnx']} exists={status['served_onnx_exists']}")
        print(f"Suggested next: {status['suggested_next']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
