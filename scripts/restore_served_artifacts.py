#!/usr/bin/env python3
"""Restore served ONNX + extended checkpoint from HF hub when data exists but models missing."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_REPO = os.environ.get(
    "SIGN_LANGUAGE_RESTORE_HF_REPO", "Taalvi/sign-language-cloud-restore"
)


def _refresh_requested() -> bool:
    return os.environ.get("SIGN_LANGUAGE_RESTORE_REFRESH", "").lower() in (
        "1",
        "true",
        "yes",
    )


def _extended_checkpoint_present() -> bool:
    for rel in (
        "kaggle_extended_v3/best.pt",
        "kaggle_extended_v2/best.pt",
        "kaggle_extended_v1/best.pt",
    ):
        if (REPO / "models" / rel).is_file():
            return True
    return False


def main() -> int:
    served_dir = REPO / "models" / "served"
    onnx = served_dir / "model.onnx"
    label_map = served_dir / "label_map.json"
    extended = REPO / "models" / "kaggle_extended_v1" / "best.pt"

    if (
        not _refresh_requested()
        and onnx.is_file()
        and label_map.is_file()
        and _extended_checkpoint_present()
    ):
        print("restore_served_artifacts: served bundle complete", file=sys.stderr)
        return 0

    if onnx.is_file() and extended.is_file() and not label_map.is_file():
        _copy_label_map_from_bundle(REPO / "artifacts" / "hf_restore", label_map)
        if label_map.is_file():
            print("restore_served_artifacts: restored missing label_map.json", file=sys.stderr)
            return 0

    if onnx.is_file() and not label_map.is_file():
        print(
            "restore_served_artifacts: model.onnx without label_map.json — restoring labels",
            file=sys.stderr,
        )

    hf_repo = DEFAULT_REPO
    dest = REPO / "artifacts" / "hf_restore"
    dest.mkdir(parents=True, exist_ok=True)

    exit_code, method = _download_hf_dataset(hf_repo, dest)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hf_repo": hf_repo,
        "download_exit_code": exit_code,
        "download_method": method,
    }
    if exit_code != 0:
        report["status"] = "download_failed"
        _write(report)
        return exit_code

    src_onnx = dest / "model.onnx"
    src_pt = dest / "best.pt"
    src_label = dest / "label_map.json"
    served_dir.mkdir(parents=True, exist_ok=True)
    if src_onnx.is_file():
        shutil.copy2(src_onnx, onnx)
    if src_pt.is_file():
        ckpt_dir = REPO / "models" / "kaggle_extended_v3"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_pt, ckpt_dir / "best.pt")
        if not extended.is_file():
            (REPO / "models" / "kaggle_extended_v1").mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_pt, extended)
    _copy_label_map_from_bundle(dest, label_map, overwrite=True)

    report["served_onnx"] = onnx.is_file()
    report["served_label_map"] = label_map.is_file()
    report["extended_best_pt"] = extended.is_file()
    report["status"] = (
        "ok"
        if report["served_onnx"] and report["served_label_map"] and report["extended_best_pt"]
        else "partial"
    )
    _write(report)
    return 0 if report["status"] == "ok" else 1


def _copy_label_map_from_bundle(bundle_dir: Path, label_dst: Path, *, overwrite: bool = False) -> None:
    src = bundle_dir / "label_map.json"
    if not src.is_file():
        processed = _processed_label_map_path()
        if processed is not None:
            src = processed
    if src.is_file() and (overwrite or not label_dst.is_file()):
        label_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, label_dst)


def _processed_label_map_path() -> Path | None:
    """Fallback when HF bundle lacks label_map but preprocess data exists on disk."""
    try:
        from backend.config_loader import get_config, resolve_data_root

        cfg = get_config()
        root = resolve_data_root(cfg)
        candidate = (
            root
            / cfg["paths"]["processed_dir"]
            / "kaggle_asl_signs"
            / cfg["paths"]["label_map_filename"]
        )
        if candidate.is_file():
            return candidate
    except Exception:
        pass
    return None


def _download_hf_dataset(hf_repo: str, dest: Path) -> tuple[int, str]:
    """Prefer huggingface_hub API (no deprecated huggingface-cli)."""
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=hf_repo,
            repo_type="dataset",
            local_dir=str(dest),
        )
        return 0, "snapshot_download"
    except Exception as exc:
        print(f"restore_served_artifacts: snapshot_download failed: {exc}", file=sys.stderr)

    cmd = [
        "hf",
        "download",
        hf_repo,
        "--repo-type",
        "dataset",
        "--local-dir",
        str(dest),
    ]
    proc = subprocess.run(cmd, cwd=REPO, check=False, env=os.environ)
    if proc.returncode == 0:
        return 0, "hf_cli"
    legacy = [
        "huggingface-cli",
        "download",
        hf_repo,
        "--repo-type",
        "dataset",
        "--local-dir",
        str(dest),
        "--local-dir-use-symlinks",
        "False",
    ]
    proc = subprocess.run(legacy, cwd=REPO, check=False, env=os.environ)
    return proc.returncode, "huggingface-cli" if proc.returncode == 0 else "failed"


def _write(report: dict) -> None:
    out = REPO / "reports" / "restore_served_artifacts.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
