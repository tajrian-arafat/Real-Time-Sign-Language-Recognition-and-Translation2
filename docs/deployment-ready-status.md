# Deployment-ready status

**Branch:** `cursor/deployment-ready-integration-f1a1`  
**Date (UTC):** 2026-09-28  

## Boot verification

| Check | Result |
| ----- | ------ |
| NPZ tensors | **94,477** under `SIGN_LANGUAGE_DATA_ROOT=/workspace/data` |
| `models/served/model.onnx` | Present |
| `models/served/label_map.json` | Present (HF bundle / restore) |
| `/health` `is_stub` | **false** |

## Model metrics (re-run)

Command: `python -m ml.evaluate --checkpoint models/kaggle_extended_v1/best.pt --split val`

| Metric | Value |
| ------ | ----- |
| Top-1 | **64.05%** |
| Top-5 | **86.28%** |
| Macro F1 | 0.630 |

Artifact: `reports/metrics.json`

## Tests

| Suite | Result |
| ----- | ------ |
| `pytest -q` | **39 passed**, 1 skipped (BanglaT5 integration) |
| `npm --prefix frontend test` | **4 passed** |
| Frontend prod build | `npm run build` OK |

## New / updated deliverables

- `scripts/run_full_pipeline.sh` — wired pipeline with `SKIP_*` flags
- `Dockerfile`, `docker-compose.yml`, `docker/entrypoint.sh`, `docker/nginx-frontend.conf`
- `render.yaml` — API + static frontend
- `docs/deployment.md`
- `tests/integration/test_backend_video_upload.py`
- `reports/agent9_acceptance.json` (regenerated)
- `reports/webcam_e2e_status.md`
- `scripts/restore_served_artifacts.py` — restores missing `label_map.json`

## Remaining blockers (honest)

1. **Live browser webcam E2E** — backend verified; camera session not automated in CI.
2. **ASL Citizen** — not downloaded on this VM (250-word Kaggle scope only).
3. **Overnight ~67% val** — not reproduced on `kaggle_extended_v1/best.pt` (measured **64.05%** on this run); cite `reports/metrics.json` as source of truth.

## Quick commands

```bash
bash scripts/run_full_pipeline.sh    # respects SKIP when data/models exist
docker compose up --build            # after frontend build + optional HF_TOKEN
curl -s http://127.0.0.1:8000/health | jq .
```

Acceptance checklist: `reports/agent9_acceptance.json`.
