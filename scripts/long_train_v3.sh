#!/usr/bin/env bash
# Long CPU fine-tune v3: resume v2 best, disable early stop, full wall budget.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${ROOT}/.venv/bin/python"
LOG="${ROOT}/reports/long_train_v3.log"
mkdir -p reports

export SIGN_LANGUAGE_DATA_ROOT="${SIGN_LANGUAGE_DATA_ROOT:-/workspace/data}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

echo "=== long_train_v3 start $(date -u +%Y-%m-%dT%H:%M:%SZ) main=$(git rev-parse --short HEAD) ===" | tee -a "$LOG"
"$PY" scripts/pipeline_status.py 2>&1 | tee -a "$LOG"

RESUME="${RESUME_CKPT:-models/kaggle_extended_v2/best.pt}"
RUN_NAME="${RUN_NAME:-kaggle_extended_v3}"
EPOCHS="${TRAIN_EPOCHS:-120}"
LR="${LEARNING_RATE:-3e-5}"
WALL="${MAX_WALL_SECONDS:-21600}"
PATIENCE="${EARLY_STOPPING_PATIENCE:-999}"
NUM_WORKERS="${TRAIN_NUM_WORKERS:-0}"

echo "device=cpu batch=32 lr=$LR epochs=$EPOCHS wall=${WALL}s patience=$PATIENCE workers=$NUM_WORKERS run=$RUN_NAME resume=$RESUME" | tee -a "$LOG"

TRAIN_ARGS=(
  --run-name "$RUN_NAME"
  --resume "$RESUME"
  --epochs "$EPOCHS"
  --learning-rate "$LR"
  --num-workers "$NUM_WORKERS"
  --early-stopping-patience "$PATIENCE"
  --export-on-best
  --max-wall-seconds "$WALL"
)
"$PY" -m ml.train "${TRAIN_ARGS[@]}" 2>&1 | tee -a "$LOG"

CKPT="models/${RUN_NAME}/best.pt"
if [[ -f "$CKPT" ]]; then
  "$PY" -m ml.evaluate --checkpoint "$CKPT" --split val 2>&1 | tee -a "$LOG"
  "$PY" -m ml.export_onnx --checkpoint "$CKPT" 2>&1 | tee -a "$LOG"
fi

"$PY" scripts/build_cloud_restore.py 2>&1 | tee -a "$LOG"

if [[ -n "${HF_TOKEN:-}" ]] && [[ -f artifacts/cloud_restore/model.onnx ]]; then
  bash scripts/upload_cloud_restore_hf.sh 2>&1 | tee -a "$LOG" || true
else
  echo "Skipping HF upload (HF_TOKEN unset or bundle missing)" | tee -a "$LOG"
fi

echo "=== long_train_v3 end $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
