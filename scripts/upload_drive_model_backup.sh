#!/usr/bin/env bash
# Secondary Drive backup after HF (primary). Requires staged zip from prepare_drive_backup.py.
# Uses Google Drive MCP via upload_drive_model_backup.py when CURSOR_MCP_DRIVE=1 is set by agent;
# otherwise leaves reports/drive_backup_manifest.json for manual/MCP upload.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${ROOT}/.venv/bin/python"
MANIFEST="${ROOT}/reports/drive_backup_manifest.json"

if [[ ! -f "$MANIFEST" ]]; then
  echo "No $MANIFEST — run prepare_drive_backup.py first" >&2
  exit 1
fi

if [[ -f "${ROOT}/scripts/upload_drive_model_backup.py" ]]; then
  "$PY" "${ROOT}/scripts/upload_drive_model_backup.py" "$@"
else
  echo "upload_drive_model_backup.py missing; see $MANIFEST for local zip path" >&2
  exit 0
fi
