#!/usr/bin/env bash
# Six-hour CPU fine-tune: resume extended v1 → v2, log, export on best, HF bundle.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${ROOT}/.venv/bin/python"
LOG="${ROOT}/reports/six_hour_train.log"
mkdir -p reports

export SIGN_LANGUAGE_DATA_ROOT="${SIGN_LANGUAGE_DATA_ROOT:-/workspace/data}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

echo "=== six_hour_train start $(date -u +%Y-%m-%dT%H:%M:%SZ) main=$(git rev-parse --short HEAD) ===" | tee -a "$LOG"
"$PY" scripts/pipeline_status.py 2>&1 | tee -a "$LOG"

RESUME="${RESUME_CKPT:-models/kaggle_extended_v1/best.pt}"
RUN_NAME="${RUN_NAME:-kaggle_extended_v2}"
EPOCHS="${TRAIN_EPOCHS:-120}"
LR="${LEARNING_RATE:-5e-5}"
WALL="${MAX_WALL_SECONDS:-21600}"
# num_workers=0: stable on cloud agents (avoid fork + persistent_workers issues)
NUM_WORKERS="${TRAIN_NUM_WORKERS:-0}"

echo "device=cpu batch=32 lr=$LR epochs=$EPOCHS wall=${WALL}s workers=$NUM_WORKERS run=$RUN_NAME resume=$RESUME" | tee -a "$LOG"

"$PY" -m ml.train \
  --run-name "$RUN_NAME" \
  --resume "$RESUME" \
  --epochs "$EPOCHS" \
  --learning-rate "$LR" \
  --num-workers "$NUM_WORKERS" \
  --export-on-best \
  --max-wall-seconds "$WALL" \
  2>&1 | tee -a "$LOG"

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

if [[ -f artifacts/cloud_restore/model.onnx ]]; then
  "$PY" scripts/prepare_drive_backup.py 2>&1 | tee -a "$LOG" || true
  if [[ -x scripts/upload_drive_model_backup.sh ]]; then
    bash scripts/upload_drive_model_backup.sh 2>&1 | tee -a "$LOG" || true
  else
    echo "Drive backup staged; upload via upload_drive_model_backup.sh or Google Drive MCP" | tee -a "$LOG"
  fi
fi

echo "=== six_hour_train end $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
