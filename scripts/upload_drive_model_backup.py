#!/usr/bin/env python3
"""Upload staged model backup zip to Google Drive (secondary to HF)."""

from __future__ import annotations

import json
import mimetypes
import os
import sys
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[1]
SUBFOLDER_FILE = REPO / "reports" / "drive_model_backups_folder_id.json"
PARENT_FALLBACK = "1vShHvKqIx0ojZsvAIWqli5LkdA4np2Ls"
UPLOAD_URL = "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable&supportsAllDrives=true"


def _parent_id() -> str:
    if SUBFOLDER_FILE.is_file():
        data = json.loads(SUBFOLDER_FILE.read_text(encoding="utf-8"))
        sub = data.get("model_backups_folder_id")
        if sub:
            return str(sub)
    return PARENT_FALLBACK


def _access_token() -> str | None:
    for key in (
        "GOOGLE_DRIVE_ACCESS_TOKEN",
        "GOOGLE_ACCESS_TOKEN",
        "DRIVE_ACCESS_TOKEN",
    ):
        val = os.environ.get(key)
        if val:
            return val.strip()
    return None


def _upload_resumable(token: str, path: Path, parent_id: str) -> dict:
    mime, _ = mimetypes.guess_type(path.name)
    mime = mime or "application/octet-stream"
    meta = {"name": path.name, "parents": [parent_id]}
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=UTF-8",
        "X-Upload-Content-Type": mime,
        "X-Upload-Content-Length": str(path.stat().st_size),
    }
    init_resp = requests.post(UPLOAD_URL, headers=headers, json=meta, timeout=60)
    init_resp.raise_for_status()
    session_url = init_resp.headers["Location"]
    with path.open("rb") as body:
        put_resp = requests.put(
            session_url,
            data=body,
            headers={"Content-Length": str(path.stat().st_size)},
            timeout=600,
        )
    put_resp.raise_for_status()
    return put_resp.json()


def main() -> int:
    manifest_path = REPO / "reports" / "drive_backup_manifest.json"
    if not manifest_path.is_file():
        print("Missing reports/drive_backup_manifest.json", file=sys.stderr)
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    zip_path = Path(manifest["zip_local_path"])
    stage = REPO / "artifacts" / "drive_backup_staging"
    token = _access_token()
    parent_id = _parent_id()
    result_path = REPO / "reports" / "drive_upload_result.json"

    if not token:
        pending = {
            "status": "pending_no_token",
            "zip_local_path": str(zip_path) if zip_path.is_file() else None,
            "staging_dir": str(stage) if stage.is_dir() else None,
            "drive_parent_folder_id": PARENT_FALLBACK,
            "drive_subfolder_id": parent_id,
            "drive_subfolder_url": f"https://drive.google.com/drive/folders/{parent_id}",
            "hint": "Set GOOGLE_DRIVE_ACCESS_TOKEN in Cursor secrets for unattended resumable upload after six_hour_train, or upload staged zip via Google Drive MCP create_file.",
        }
        result_path.write_text(json.dumps(pending, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(pending))
        return 0

    uploaded: list[dict] = []
    targets: list[Path] = []
    if stage.is_dir():
        targets.extend(sorted(p for p in stage.iterdir() if p.is_file()))
    elif zip_path.is_file():
        targets.append(zip_path)

    if not targets:
        print("No staging files or zip to upload", file=sys.stderr)
        return 1

    prefix = manifest.get("run_name", "kaggle")
    stamp = manifest.get("generated_at_utc", "")[:19].replace(":", "").replace("-", "")

    for path in targets:
        drive_name = path.name
        if path.parent == stage:
            drive_name = f"sign-language-{prefix}-{stamp}-{path.name}"
        try:
            meta = _upload_resumable(token, path, parent_id)
        except requests.HTTPError as exc:
            err = {
                "status": "error",
                "message": str(exc),
                "response": getattr(exc.response, "text", ""),
                "failed_file": str(path),
                "uploaded_so_far": uploaded,
            }
            result_path.write_text(json.dumps(err, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(err), file=sys.stderr)
            return 1
        uploaded.append(
            {
                "local_path": str(path),
                "drive_file_id": meta.get("id"),
                "drive_file_name": meta.get("name"),
                "drive_web_view_link": meta.get("webViewLink"),
            }
        )

    ok = {
        "status": "uploaded",
        "drive_parent_folder_id": PARENT_FALLBACK,
        "drive_subfolder_id": parent_id,
        "drive_subfolder_url": f"https://drive.google.com/drive/folders/{parent_id}",
        "files": uploaded,
        "zip_local_path": str(zip_path) if zip_path.is_file() else None,
    }
    result_path.write_text(json.dumps(ok, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(ok))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
