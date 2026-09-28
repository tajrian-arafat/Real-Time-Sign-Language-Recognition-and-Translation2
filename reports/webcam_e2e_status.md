# Webcam end-to-end status

**Automated coverage (this deployment-ready pass):**

- WebSocket round-trip with **real ONNX** when `models/served/model.onnx` is present: `tests/integration/test_backend_websocket.py`, artifact `reports/backend_smoke.json`.
- Latency benchmark (synthetic windows): `python scripts/run_latency_benchmark.py` → `reports/latency_benchmark.json`.
- Video upload path: `tests/integration/test_backend_video_upload.py` (MediaPipe extraction stubbed; classifier and API contract are real).

**Not run in CI (browser + camera required):**

- Live `getUserMedia` webcam session with MediaPipe Tasks in the browser.
- Manual checklist: [docs/manual-e2e-checklist.md](../docs/manual-e2e-checklist.md).

**Acceptance note:** Agent 9 criterion `live_webcam_real_predictions` remains **partial** until a recorded browser session or coordinator snapshot includes camera capture; backend inference is verified non-stub when served artifacts exist.
