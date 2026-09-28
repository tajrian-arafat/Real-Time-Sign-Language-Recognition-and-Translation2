# Real-Time ASL Recognition and Bangla Translation

Isolated-sign ASL recognition with English gloss accumulation and Bangla translation. See [docs/ASL-Bangla-System-Architecture.md](docs/ASL-Bangla-System-Architecture.md) and [docs/Cursor-Master-Prompt-and-Agents.md](docs/Cursor-Master-Prompt-and-Agents.md).

## Data directory (Windows)

On the author's Windows PC, **all dataset and large downloads must live only under:**

`D:\PROJECTS\sign language`

Configure this via `config/config.yaml` → `paths.data_root`, or override at runtime:

```powershell
set SIGN_LANGUAGE_DATA_ROOT=D:\PROJECTS\sign language
```

Application code must read the data root from config/environment — **do not hardcode** this path in Python/TypeScript source.

On this cloud development VM, use repo-relative storage under `/workspace/data/` by setting:

```bash
export SIGN_LANGUAGE_DATA_ROOT=/workspace/data
```

Kaggle `asl-signs` download accepts either legacy `KAGGLE_USERNAME` / `KAGGLE_KEY` or the newer `KAGGLE_API_TOKEN` (also read from `~/.kaggle/access_token`). Never commit tokens.

(or leave unset once path resolution helpers are added in later agents; the config file documents both patterns).

## Quick start (Agent 1 scaffolding)

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
python scripts/probe_environment.py
node --version
```

Integration branch agents: see [docs/integration-environment.md](docs/integration-environment.md) for `SIGN_LANGUAGE_DATA_ROOT`, Kaggle secrets, stub vs served ONNX, and frontend mock mode.

```bash
source .venv/bin/activate
pytest -q
npm --prefix frontend test
bash scripts/run_backend.sh   # terminal 1
bash scripts/run_frontend.sh  # terminal 2
```

Full training requires processed Kaggle tensors under `$SIGN_LANGUAGE_DATA_ROOT`; see `scripts/run_training.sh` and `scripts/run_data_acquisition.py`.

## Served model and validation metrics (measured)

Production inference loads ONNX from:

`models/served/model.onnx`

(relative to repo root; absolute path on the integration VM when artifacts are present: `/workspace/models/served/model.onnx`). Label map: `models/served/label_map.json`. When this file is missing, the backend falls back to `models/stub/stub_classifier.onnx` and `/health` reports `is_stub: true` — see [docs/integration-environment.md](docs/integration-environment.md).

**Validation split (Kaggle `asl-signs`, 250 classes, 9,211 samples)** — artifact `reports/metrics.json`:

| Checkpoint | Top-1 | Top-5 | Macro F1 |
| ---------- | ----- | ----- | -------- |
| `models/kaggle_baseline/best.pt` (baseline) | 55.48% | 82.10% | 0.537 |
| `models/kaggle_extended_v1/best.pt` (served / HF restore) | **64.05%** | see metrics file | see metrics file |

Reproduce: `python -m ml.evaluate --checkpoint models/kaggle_extended_v1/best.pt --split val`. Do not expect these numbers from the stub ONNX.

## Wipe-proof model restore

On empty cloud VMs, after data is present:

```bash
python scripts/restore_served_artifacts.py   # needs HF_TOKEN for hub download
```

Bundle: Hugging Face dataset `Taalvi/sign-language-cloud-restore` (ONNX + `best.pt` + `label_map.json`). See `scripts/build_cloud_restore.py` to rebuild the bundle.

## Deployment

Docker Compose, Render blueprint, and env vars: **[docs/deployment.md](docs/deployment.md)**. Status summary: **[docs/deployment-ready-status.md](docs/deployment-ready-status.md)**.

Full pipeline (with SKIP flags when data/models exist): `bash scripts/run_full_pipeline.sh`.

## Inference latency benchmark

WebSocket round-trip latency (synthetic landmark windows, default N=100):

```bash
source .venv/bin/activate
python scripts/run_latency_benchmark.py
```

Output: `reports/latency_benchmark.json` (p50/p95 for round-trip and server-reported `latency_ms`).

## Honest scope (integration / QA)

- **Tier-1 data on integration VM:** Kaggle `asl-signs` only; ASL Citizen may be absent unless downloaded separately.
- **Live webcam / upload-video E2E:** requires a browser session with camera or a sample clip; see [docs/manual-e2e-checklist.md](docs/manual-e2e-checklist.md).
- **Bangla:** 250-word instant dictionary plus optional BanglaT5 sentence path (on-demand model download).
