#!/usr/bin/env bash
# Upload /workspace/artifacts/cloud_restore to Hugging Face (needs HF_TOKEN in env).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLE="${ROOT}/artifacts/cloud_restore"
REPO="${HF_RESTORE_REPO:-Taalvi/sign-language-cloud-restore}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set. Add it in Cursor My Secrets and use a cloud agent started after that." >&2
  exit 1
fi
if [[ ! -d "$BUNDLE" ]] || [[ ! -f "$BUNDLE/model.onnx" ]]; then
  echo "Missing $BUNDLE (model.onnx). Build bundle on pipeline VM first." >&2
  exit 1
fi

HF_BIN="${ROOT}/.venv/bin/hf"
if [[ ! -x "$HF_BIN" ]]; then
  HF_BIN="$(command -v hf || true)"
fi
if [[ -z "$HF_BIN" ]] || [[ ! -x "$HF_BIN" ]]; then
  echo "Missing Hugging Face CLI (expected ${ROOT}/.venv/bin/hf)" >&2
  exit 1
fi
"$HF_BIN" auth login --token "$HF_TOKEN"
"$HF_BIN" upload "$REPO" "$BUNDLE" . --repo-type dataset
echo "Uploaded to https://huggingface.co/datasets/${REPO}"
