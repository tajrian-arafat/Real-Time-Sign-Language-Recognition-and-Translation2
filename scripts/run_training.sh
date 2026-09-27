#!/usr/bin/env bash
# Train → evaluate → export ONNX (Agent 4).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
RUN_NAME="${RUN_NAME:-kaggle_baseline}"
EPOCHS="${TRAIN_EPOCHS:-}"
EVAL_SPLIT="${EVAL_SPLIT:-val}"
TRAIN_ARGS=()
if [[ -n "$EPOCHS" ]]; then
  TRAIN_ARGS+=(--epochs "$EPOCHS")
fi
python3 -m ml.train --run-name "$RUN_NAME" "${TRAIN_ARGS[@]}" "$@"
CKPT="models/${RUN_NAME}/best.pt"
python3 -m ml.evaluate --checkpoint "$CKPT" --split "$EVAL_SPLIT"
python3 -m ml.export_onnx --checkpoint "$CKPT"
echo "Artifacts: reports/training_status.json, reports/metrics.json, reports/onnx_export.json"
