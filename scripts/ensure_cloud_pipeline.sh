#!/usr/bin/env bash
# After setup_env: restore pipeline data on empty cloud VMs (snapshot fallback).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export SIGN_LANGUAGE_DATA_ROOT="${SIGN_LANGUAGE_DATA_ROOT:-/workspace/data}"
export PATH="${HOME}/.local/bin:${PATH}"
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
python scripts/ensure_cloud_pipeline.py
python scripts/restore_served_artifacts.py || true
