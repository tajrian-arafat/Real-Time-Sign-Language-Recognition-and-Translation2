#!/usr/bin/env bash
# End-to-end pipeline: acquisition → preprocess → train → evaluate → export ONNX.
# Use SKIP_* env vars when data and checkpoints already exist (cloud snapshot / HF restore).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export SIGN_LANGUAGE_DATA_ROOT="${SIGN_LANGUAGE_DATA_ROOT:-/workspace/data}"
export PATH="${HOME}/.local/bin:${PATH}"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

RUN_NAME="${RUN_NAME:-kaggle_extended_v1}"
SERVED_ONNX="models/served/model.onnx"
SERVED_LABEL="models/served/label_map.json"
CHECKPOINT="models/${RUN_NAME}/best.pt"

log() { echo "[run_full_pipeline] $*"; }

# --- Optional: wipe-proof model restore (no training data required) ---
if [[ "${SKIP_RESTORE:-0}" != "1" ]]; then
  python scripts/restore_served_artifacts.py || true
fi

# --- 1. Data acquisition ---
if [[ "${SKIP_ACQUISITION:-0}" == "1" ]]; then
  log "SKIP_ACQUISITION=1 — skipping run_data_acquisition.py"
else
  log "Step 1/5: data acquisition"
  python scripts/run_data_acquisition.py
fi

# --- 2. Preprocessing ---
NPZ_COUNT="$(find "${SIGN_LANGUAGE_DATA_ROOT}/processed" -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')"
if [[ "${SKIP_PREPROCESS:-0}" == "1" ]] || [[ "${NPZ_COUNT}" -ge 90000 ]]; then
  if [[ "${NPZ_COUNT}" -ge 90000 ]]; then
    log "SKIP_PREPROCESS implied: ${NPZ_COUNT} NPZ tensors already under ${SIGN_LANGUAGE_DATA_ROOT}/processed"
  else
    log "SKIP_PREPROCESS=1 — skipping preprocessing"
  fi
else
  log "Step 2/5: preprocessing (NPZ count=${NPZ_COUNT})"
  python data/scripts/run_preprocessing.py
fi

# --- 3. Training ---
if [[ "${SKIP_TRAIN:-0}" == "1" ]] || [[ -f "${CHECKPOINT}" ]]; then
  if [[ -f "${CHECKPOINT}" ]]; then
    log "SKIP_TRAIN implied: checkpoint exists at ${CHECKPOINT}"
  else
    log "SKIP_TRAIN=1 — skipping training"
  fi
else
  log "Step 3/5: training (run_name=${RUN_NAME})"
  RUN_NAME="${RUN_NAME}" bash scripts/run_training.sh
fi

if [[ ! -f "${CHECKPOINT}" ]] && [[ -f "${SERVED_ONNX}" ]]; then
  log "No ${CHECKPOINT}; using served ONNX for evaluate/export only"
  CHECKPOINT="${SERVED_ONNX}"
fi

# --- 4. Evaluation ---
if [[ "${SKIP_EVAL:-0}" == "1" ]]; then
  log "SKIP_EVAL=1 — skipping ml.evaluate"
else
  if [[ -f "models/kaggle_extended_v1/best.pt" ]]; then
    EVAL_CKPT="models/kaggle_extended_v1/best.pt"
  elif [[ -f "${CHECKPOINT}" ]] && [[ "${CHECKPOINT}" == *.pt ]]; then
    EVAL_CKPT="${CHECKPOINT}"
  else
    EVAL_CKPT=""
  fi
  if [[ -n "${EVAL_CKPT}" ]]; then
    log "Step 4/5: evaluate ${EVAL_CKPT}"
    python -m ml.evaluate --checkpoint "${EVAL_CKPT}" --split "${EVAL_SPLIT:-val}"
  else
    log "Step 4/5: skipped evaluate (no .pt checkpoint)"
  fi
fi

# --- 5. ONNX export to models/served ---
if [[ "${SKIP_EXPORT:-0}" == "1" ]] || [[ -f "${SERVED_ONNX}" && -f "${SERVED_LABEL}" ]]; then
  if [[ -f "${SERVED_ONNX}" ]]; then
    log "SKIP_EXPORT implied: ${SERVED_ONNX} already present"
  else
    log "SKIP_EXPORT=1 — skipping export_onnx"
  fi
else
  if [[ -f "models/kaggle_extended_v1/best.pt" ]]; then
    log "Step 5/5: export ONNX from models/kaggle_extended_v1/best.pt"
    python -m ml.export_onnx --checkpoint models/kaggle_extended_v1/best.pt
  elif [[ -f "${CHECKPOINT}" ]] && [[ "${CHECKPOINT}" == *.pt ]]; then
    log "Step 5/5: export ONNX from ${CHECKPOINT}"
    python -m ml.export_onnx --checkpoint "${CHECKPOINT}"
  else
    log "Step 5/5: skipped export (no checkpoint)"
  fi
fi

# Ensure label_map beside served ONNX (HF restore bundle or processed copy)
if [[ -f "${SERVED_ONNX}" ]] && [[ ! -f "${SERVED_LABEL}" ]]; then
  if [[ -f artifacts/hf_restore/label_map.json ]]; then
    cp artifacts/hf_restore/label_map.json "${SERVED_LABEL}"
  elif [[ -f "${SIGN_LANGUAGE_DATA_ROOT}/processed/kaggle_asl_signs/label_map.json" ]]; then
    cp "${SIGN_LANGUAGE_DATA_ROOT}/processed/kaggle_asl_signs/label_map.json" "${SERVED_LABEL}"
  fi
fi

log "Done. Artifacts: reports/data_acquisition.json, reports/preprocessing_status.json,"
log "  reports/metrics.json, reports/onnx_export.json, models/served/model.onnx"
