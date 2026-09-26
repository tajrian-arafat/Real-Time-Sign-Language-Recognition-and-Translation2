#!/usr/bin/env bash
# Agent 1 skeleton: create Python venv (3.11 when available) and install Node deps for frontend.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PATH="${HOME}/.local/bin:${PATH}"

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if command -v uv >/dev/null 2>&1; then
    PYTHON="$(uv python find 3.11 2>/dev/null || true)"
  fi
  if [[ -z "$PYTHON" || ! -x "$PYTHON" ]]; then
    if command -v python3.11 >/dev/null 2>&1; then
      PYTHON=python3.11
    else
      PYTHON=python3
    fi
  fi
fi

echo "Using Python: $PYTHON ($("$PYTHON" --version))"

if [[ ! -d .venv ]]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv .venv --python 3.11
  else
    "$PYTHON" -m venv .venv
  fi
fi
if [[ ! -x .venv/bin/pip ]]; then
  .venv/bin/python -m ensurepip --upgrade
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

if command -v node >/dev/null 2>&1 && [[ -f frontend/package.json ]]; then
  echo "Installing frontend dependencies..."
  npm --prefix frontend install
else
  echo "Skipping frontend npm install (node or frontend/package.json missing)."
fi

echo "Setup complete. Activate with: source .venv/bin/activate"
