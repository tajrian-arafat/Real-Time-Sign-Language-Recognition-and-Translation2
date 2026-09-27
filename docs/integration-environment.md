# Integration environment (Agent 9)

Cloud agents and local developers should use the same env vars documented here. **Never commit Kaggle tokens or API keys.**

## Required / recommended environment variables

| Variable | Purpose |
| -------- | ------- |
| `SIGN_LANGUAGE_DATA_ROOT` | Root for `raw/` and `processed/` datasets. On this VM use `/workspace/data`. On Windows (author machine) use `D:\PROJECTS\sign language`. Read via `config/config.yaml` → `paths.data_root_env_var`. |
| `KAGGLE_USERNAME` + `KAGGLE_KEY` | Legacy Kaggle API credentials for `asl-signs` download. |
| `KAGGLE_API_TOKEN` | Newer Kaggle token (also read from `~/.kaggle/access_token`). Either this **or** username/key is required for Agent 2 acquisition. |

Store secrets in **Cursor Cloud → My Secrets** (or your shell profile locally), not in the repository.

## Served ONNX model (real inference)

Training (Agent 4) writes the production artifact to:

`models/served/model.onnx`

This path is configured in `config/config.yaml` → `inference.served_onnx`. When that file exists, the FastAPI backend loads it automatically and `/health` reports `is_stub: false`.

Until the file is present:

- Backend uses `models/stub/stub_classifier.onnx` when `inference.allow_stub_model: true` (default for integration).
- WebSocket and REST tests use the stub; predictions are deterministic, not random.

After a pipeline worker lands `model.onnx`:

```bash
source .venv/bin/activate
export SIGN_LANGUAGE_DATA_ROOT="${SIGN_LANGUAGE_DATA_ROOT:-/workspace/data}"
python scripts/run_backend_smoke.py
pytest tests/integration/test_backend_websocket.py -q
```

Optional heavy translation integration:

```bash
RUN_BANGLAT5_INTEGRATION=1 pytest tests/integration/test_backend_translation.py -q
```

## Frontend mock vs live backend

Copy `frontend/.env.example` to `frontend/.env.local`:

- `VITE_MOCK_WS=true` — offline UI development (no backend).
- `VITE_MOCK_WS=false` — live WebSocket to backend (default); run `bash scripts/run_backend.sh` and `bash scripts/run_frontend.sh`.

## Cloud agent bootstrap

Repository-managed setup is in [`.cursor/environment.json`](../.cursor/environment.json) (`install` runs `scripts/setup_env.sh`). Pair with **My Secrets** for `KAGGLE_API_TOKEN` when running data acquisition on fresh pods.

See also [pipeline-resilience.md](./pipeline-resilience.md) for data retention and snapshots.
