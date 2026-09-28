# Deployment guide

Production stack: **FastAPI + ONNX Runtime** (backend) and **React/Vite** (frontend), per [ASL-Bangla-System-Architecture.md](ASL-Bangla-System-Architecture.md) Part 12.

## One-command local / staging (Docker Compose)

```bash
export SIGN_LANGUAGE_DATA_ROOT=/workspace/data   # or host path with processed NPZ + raw Kaggle data
export HF_TOKEN=hf_...                           # optional; restores ONNX + checkpoint on empty disk

npm --prefix frontend run build
VITE_API_URL=http://localhost:8000 npm --prefix frontend run build
docker compose up --build
```

- API: http://localhost:8000/health  
- UI (nginx): http://localhost:8080  

Mount `models/served/` read-only if you already have ONNX locally; otherwise the backend entrypoint runs `scripts/restore_served_artifacts.py` when `HF_TOKEN` is set.

### SKIP flags (full pipeline on VM)

```bash
bash scripts/run_full_pipeline.sh
```

| Variable | Effect |
| -------- | ------ |
| `SKIP_RESTORE=1` | Do not run HF restore |
| `SKIP_ACQUISITION=1` | Skip Kaggle / ASL Citizen download |
| `SKIP_PREPROCESS=1` | Skip preprocessing (auto-skipped when ≥90k NPZ exist) |
| `SKIP_TRAIN=1` | Skip training (auto-skipped when `models/<RUN_NAME>/best.pt` exists) |
| `SKIP_EVAL=1` | Skip `python -m ml.evaluate` |
| `SKIP_EXPORT=1` | Skip ONNX export (auto-skipped when `models/served/model.onnx` exists) |

## Render.com

1. Connect this repository in Render and apply `render.yaml`.
2. Set **HF_TOKEN** on the `asl-bangla-api` service (for wipe-proof model restore).
3. Attach a persistent disk at `/data` if you host preprocessed tensors on the instance; otherwise rely on HF restore for **models only** (inference works; re-training needs data on disk).
4. Set **VITE_API_URL** on the static site to the API’s public URL before build (Render `fromService` in blueprint).

Health check: `GET /health` — expect `model_loaded: true`, `is_stub: false` when `models/served/model.onnx` and `label_map.json` are present.

## Environment variables

| Name | Required | Purpose |
| ---- | -------- | ------- |
| `SIGN_LANGUAGE_DATA_ROOT` | Recommended | Root for `raw/` and `processed/` (default `/workspace/data` in cloud) |
| `HF_TOKEN` | For empty-disk restore | Download `Taalvi/sign-language-cloud-restore` dataset |
| `SIGN_LANGUAGE_RESTORE_HF_REPO` | Optional | Override HF dataset id |
| `VITE_API_URL` | Frontend prod build | Backend origin (e.g. `https://your-api.onrender.com`) |
| `VITE_MOCK_WS` | Optional | Set `true` for offline UI mock (not for production) |
| `KAGGLE_USERNAME` / `KAGGLE_KEY` or `KAGGLE_API_TOKEN` | Training only | Kaggle `asl-signs` download |

No paid APIs; BanglaT5 loads from Hugging Face on first sentence translation (CPU).

## Limitations (honest scope)

- **CPU inference** — suitable for demo/staging; latency higher than GPU.
- **Isolated-sign recognition** (~250-word Kaggle vocabulary), not continuous ASL translation.
- **Validation accuracy** — see `reports/metrics.json` (extended checkpoint ~64% top-1 on val at time of last evaluate; not a production SLA).
- **Webcam E2E** requires browser + camera; automated runs document WebSocket smoke only (`reports/live_websocket_smoke.json`, `docs/manual-e2e-checklist.md`).

## Wipe-proof bootstrap (cloud agents)

```bash
bash scripts/setup_env.sh
bash scripts/ensure_cloud_pipeline.sh
python scripts/restore_served_artifacts.py
```

See [integration-environment.md](integration-environment.md) and [pipeline-resilience.md](pipeline-resilience.md).
