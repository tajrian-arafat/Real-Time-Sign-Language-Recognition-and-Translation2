# How to test the ASL → Bangla product (no manual setup)

This guide is for **Tajrian** and anyone who wants to try the app **without** installing Python, downloading datasets by hand, or running long training jobs. The project already ships Docker and a **Cursor Cloud Active Build** that includes data and a real ONNX model.

## What you are testing

- **Isolated ASL signs** from the Kaggle ~250-word vocabulary (one sign at a time), **not** full continuous ASL conversation.
- **Bangla output** via dictionary lookup and optional sentence translation (BanglaT5 loads on first use).
- **Real model** when `/health` shows `"is_stub": false` — predictions use `models/served/model.onnx`, not the tiny stub used only for empty disks.
- **Accuracy:** validation top-1 is about **64–67%** on held-out data (see `reports/metrics.json`). That is an honest research/demo level, **not** an 80% production guarantee.

---

## Option A — Cursor Cloud Agent (recommended)

Use the **Active Build** for this repository so the VM already has ~94k preprocessed samples and served ONNX weights.

1. Open **[Cursor Dashboard](https://cursor.com/dashboard)** → **Cloud Agents** → **Environments** (or **Settings → Cloud Agents → Environments**).
2. Select this repo: **Real-Time-Sign-Language-Recognition-and-Translation2**.
3. Confirm the **Active Build** snapshot id is **`17f076bc`** (or newer if your team updated it).
4. Click **New Agent** (or **Run agent** on that environment). Wait until the agent finishes booting.

Environment link pattern (replace with your team’s ids when shown in the UI):

`https://cursor.com/agents?environment=<environment-id>`

5. In the agent’s terminal (or ask the agent to run), verify **data + model** without extra setup:

```bash
cd /workspace
export SIGN_LANGUAGE_DATA_ROOT=/workspace/data
source .venv/bin/activate
echo "NPZ count: $(find /workspace/data/processed -name '*.npz' 2>/dev/null | wc -l)"
python scripts/restore_served_artifacts.py || true
bash scripts/run_backend.sh &
sleep 8
curl -s http://127.0.0.1:8000/health | python -m json.tool
```

6. **Pass criteria** for the health JSON:

   - `"status": "ok"`
   - `"model_loaded": true`
   - **`"is_stub": false`** ← means the real trained ONNX is loaded

7. Try the UI (same VM): in another terminal, `bash scripts/run_frontend.sh` and open the URL the script prints (or use the agent’s **Try Live** desktop link if offered).

**What `is_stub: false` means:** the backend found `models/served/model.onnx` and `label_map.json` and is **not** using the stub classifier. You are testing the same kind of model used in deployment docs.

---

## Option B — Docker on your machine

Requires **Docker** and **Docker Compose** installed locally.

1. Clone **main** (deployment from PR #12 includes `Dockerfile` and `docker-compose.yml`):

```bash
git clone https://github.com/tajrian-arafat/Real-Time-Sign-Language-Recognition-and-Translation2.git
cd Real-Time-Sign-Language-Recognition-and-Translation2
```

2. *(Optional)* If you have **no** `models/served/model.onnx` yet, set a Hugging Face token so the container entrypoint can restore weights:

```bash
export HF_TOKEN=hf_...   # read-only access to the project restore dataset
export SIGN_LANGUAGE_DATA_ROOT=/path/to/data   # optional; omit if you only need inference
```

3. Build and run:

```bash
docker compose up --build
```

4. Open in a browser:

   - **Health (API):** [http://localhost:8000/health](http://localhost:8000/health) — expect `model_loaded: true`, `is_stub: false` after restore or if you mounted `models/served/`.
   - **Frontend (nginx):** [http://localhost:8080](http://localhost:8080)

5. Stop: `docker compose down`

More detail: [deployment.md](deployment.md).

---

## Quick checklist

| Step | Expect |
|------|--------|
| NPZ count (Active Build) | Tens of thousands (e.g. **94,477** on the standard cloud disk) |
| `models/served/model.onnx` | File exists (~11 MB) |
| `GET /health` | `is_stub: false`, `model_loaded: true` |
| Scope | Isolated signs, ~**64–67%** val accuracy, CPU inference |

---

## If something fails

- **`is_stub: true`** — ONNX or label map missing; run `python scripts/restore_served_artifacts.py` with `HF_TOKEN` set, or use Active Build **17f076bc**.
- **Docker errors** — ensure Docker Desktop/daemon is running; on some cloud agent VMs Docker is not installed (use Option A instead).
- **Camera / webcam** — use a real browser with permission; see [manual-e2e-checklist.md](manual-e2e-checklist.md) for a full UI walkthrough.

Automated verification artifact from the last post-merge run: `reports/post_merge_health.json`.
