# Manual end-to-end checklist (Agent 9)

Use this when automated browser smoke is not run in CI or on a headless agent. Confirms **real** backend inference (`is_stub: false`) and live frontend WebSocket (`VITE_MOCK_WS=0`).

## Prerequisites

- Served ONNX at `models/served/model.onnx` and `models/served/label_map.json`
- Python venv: `source .venv/bin/activate`
- Frontend deps: `npm --prefix frontend install`
- `frontend/.env.local` with `VITE_MOCK_WS=false` (copy from `frontend/.env.example`)

## 1. Backend health

```bash
export SIGN_LANGUAGE_DATA_ROOT="${SIGN_LANGUAGE_DATA_ROOT:-/workspace/data}"
bash scripts/run_backend.sh
```

In another terminal:

```bash
curl -s http://127.0.0.1:8000/health | python -m json.tool
```

Expect: `"model_loaded": true`, `"is_stub": false`, `"model_path"` ending in `models/served/model.onnx`.

## 2. Frontend (live WebSocket)

```bash
bash scripts/run_frontend.sh
```

Open http://127.0.0.1:5173 in a desktop browser.

- Allow camera when prompted (webcam mode).
- Confirm the recognition panel shows **non-empty** gloss predictions and confidence (not static mock labels).
- Optional: upload a short ASL clip via the video control and verify segments in the UI.

## 3. Bangla path

- Type or accumulate an English gloss (e.g. `hello`) and trigger instant dictionary translation — expect Bangla script in the Bangla panel.
- Optional sentence translation: ensure network allows BanglaT5 download on first use.

## 4. Sentence bar

- Hold a stable sign until debounce commits a word to the sentence bar.
- Use remove-last and clear-all controls; confirm duplicate suppression (same sign should not spam repeats within cooldown).

## 5. Record evidence (optional)

- Screenshot of UI with prediction + Bangla.
- Save `curl` health output and `reports/latency_benchmark.json` from `python scripts/run_latency_benchmark.py`.

## Stub-only environments

If `/health` shows `is_stub: true`, WebSocket still works but predictions are **deterministic stub outputs**, not the 55.48% val model. Deploy or copy `models/served/model.onnx` before treating this checklist as production E2E sign-off.
