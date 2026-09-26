#!/usr/bin/env bash
# Train → evaluate → export ONNX (Agent 4).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
RUN_NAME="${RUN_NAME:-kaggle_baseline}"
python3 -m ml.train --run-name "$RUN_NAME" "$@"
CKPT="models/${RUN_NAME}/best.pt"
python3 -m ml.evaluate --checkpoint "$CKPT" --split test
python3 -m ml.export_onnx --checkpoint "$CKPT"
echo "Artifacts: reports/training_status.json, reports/metrics.json, reports/onnx_export.json"
