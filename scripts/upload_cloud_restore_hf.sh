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

if command -v hf >/dev/null 2>&1; then
  hf auth login --token "$HF_TOKEN"
  hf upload "$REPO" "$BUNDLE" . --repo-type dataset
else
  huggingface-cli login --token "$HF_TOKEN"
  huggingface-cli upload "$REPO" "$BUNDLE" . --repo-type dataset
fi
echo "Uploaded to https://huggingface.co/datasets/${REPO}"
