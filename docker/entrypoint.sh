#!/usr/bin/env bash
set -euo pipefail
cd /app
if [[ ! -f models/served/model.onnx ]] || [[ ! -f models/served/label_map.json ]]; then
  echo "entrypoint: served ONNX or label_map missing — running restore_served_artifacts.py"
  python scripts/restore_served_artifacts.py || true
fi
if [[ ! -f models/served/label_map.json ]] && [[ -f /data/processed/kaggle_asl_signs/label_map.json ]]; then
  cp /data/processed/kaggle_asl_signs/label_map.json models/served/label_map.json
fi
exec "$@"
