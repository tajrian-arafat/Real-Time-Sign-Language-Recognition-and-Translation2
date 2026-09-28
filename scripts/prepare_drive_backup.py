#!/usr/bin/env python3
"""Stage cloud-restore artifacts for Google Drive secondary backup (zip + manifest)."""

from __future__ import annotations

import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BUNDLE = REPO / "artifacts" / "cloud_restore"
STAGE = REPO / "artifacts" / "drive_backup_staging"
DRIVE_FOLDER_ID = "1vShHvKqIx0ojZsvAIWqli5LkdA4np2Ls"
DRIVE_SUBFOLDER = "model-backups"

REQUIRED = (
    "model.onnx",
    "best.pt",
    "label_map.json",
    "MANIFEST.json",
)


def main() -> int:
    if not BUNDLE.is_dir():
        print(f"Missing bundle dir {BUNDLE}; run build_cloud_restore.py first.", flush=True)
        return 1

    missing = [name for name in REQUIRED if not (BUNDLE / name).is_file()]
    if missing:
        print(f"Bundle incomplete, missing: {missing}", flush=True)
        return 1

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = "unknown"
    manifest = json.loads((BUNDLE / "MANIFEST.json").read_text(encoding="utf-8"))
    ext = manifest.get("extended_checkpoint") or {}
    best_val = ext.get("best_val_acc")

    training = REPO / "reports" / "training_status.json"
    if training.is_file():
        ts = json.loads(training.read_text(encoding="utf-8"))
        ckpt = ts.get("checkpoint_best") or ""
        if "kaggle_" in ckpt:
            run_name = Path(ckpt).parent.name

    if STAGE.is_dir():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    for name in REQUIRED:
        shutil.copy2(BUNDLE / name, STAGE / name)

    metrics_src = BUNDLE / "metrics.json"
    if not metrics_src.is_file():
        metrics_src = REPO / "reports" / "metrics.json"
    if metrics_src.is_file():
        shutil.copy2(metrics_src, STAGE / "metrics.json")
    else:
        print("Warning: metrics.json not found in bundle or reports/", flush=True)

    zip_name = f"sign-language-backup-{run_name}-{stamp}.zip"
    zip_path = REPO / "artifacts" / "drive_backup" / zip_name
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(STAGE.iterdir()):
            if path.is_file():
                zf.write(path, arcname=path.name)

    subfolder_id = DRIVE_FOLDER_ID
    subfolder_meta = REPO / "reports" / "drive_model_backups_folder_id.json"
    if subfolder_meta.is_file():
        subfolder_id = json.loads(subfolder_meta.read_text(encoding="utf-8")).get(
            "model_backups_folder_id", subfolder_id
        )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "drive_parent_folder_id": DRIVE_FOLDER_ID,
        "drive_subfolder_name": DRIVE_SUBFOLDER,
        "drive_subfolder_id": subfolder_id,
        "drive_subfolder_url": f"https://drive.google.com/drive/folders/{subfolder_id}",
        "drive_parent_url": f"https://drive.google.com/drive/folders/{DRIVE_FOLDER_ID}",
        "zip_local_path": str(zip_path),
        "zip_size_bytes": zip_path.stat().st_size,
        "staging_files": [p.name for p in sorted(STAGE.iterdir()) if p.is_file()],
        "best_val_acc": best_val,
        "run_name": run_name,
        "hf_primary": "Taalvi/sign-language-cloud-restore",
        "upload_note": "Upload zip (or individual staging files) to model-backups/ via Google Drive MCP create_file.",
    }
    out = REPO / "reports" / "drive_backup_manifest.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
