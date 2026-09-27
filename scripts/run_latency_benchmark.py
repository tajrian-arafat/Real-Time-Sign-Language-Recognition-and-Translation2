#!/usr/bin/env python3
"""Measure WebSocket inference latency (p50/p95) and write reports/latency_benchmark.json.

Uses FastAPI TestClient against /ws/recognize with synthetic landmark windows (N=100 by default).
Run from repo root with venv active:

    source .venv/bin/activate
    python scripts/run_latency_benchmark.py

Optional: LATENCY_BENCHMARK_N=200 python scripts/run_latency_benchmark.py
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
REPORT_PATH = ROOT / "reports" / "latency_benchmark.json"


def _synthetic_window(seed: float, num_frames: int = 16) -> dict:
    rng = np.random.default_rng(int(seed * 1000))
    frames = []
    for i in range(num_frames):
        vec = rng.random(392, dtype=np.float32).tolist()
        frames.append({"timestamp_ms": i * 66, "landmarks": vec})
    return {"type": "landmark_window", "frames": frames}


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def main() -> int:
    n = int(os.environ.get("LATENCY_BENCHMARK_N", "100"))

    from backend.bangla_service import reset_translation_singletons_for_tests
    from backend.inference import reset_classifier_for_tests
    from backend.main import app

    reset_classifier_for_tests()
    reset_translation_singletons_for_tests()

    round_trip_ms: list[float] = []
    server_latency_ms: list[float] = []

    with TestClient(app) as client:
        health = client.get("/health").json()
        with client.websocket_connect("/ws/recognize") as ws:
            status = ws.receive_json()
            if status.get("type") != "status" or status.get("connection") != "connected":
                raise RuntimeError(f"Unexpected WS status: {status}")

            for i in range(n):
                window = _synthetic_window(seed=0.1 + i * 0.001)
                t0 = time.perf_counter()
                ws.send_json(window)
                prediction = ws.receive_json()
                elapsed = (time.perf_counter() - t0) * 1000.0
                if prediction.get("type") != "prediction":
                    raise RuntimeError(f"Expected prediction, got: {prediction}")
                round_trip_ms.append(elapsed)
                lat = prediction.get("latency_ms")
                if lat is not None:
                    server_latency_ms.append(float(lat))

    round_trip_ms.sort()
    server_latency_ms.sort()

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "fastapi_testclient_websocket",
        "endpoint": "/ws/recognize",
        "num_samples": n,
        "health": {
            "is_stub": health.get("is_stub"),
            "model_path": health.get("model_path"),
            "model_loaded": health.get("model_loaded"),
        },
        "round_trip_ms": {
            "min": round(min(round_trip_ms), 4),
            "max": round(max(round_trip_ms), 4),
            "mean": round(statistics.mean(round_trip_ms), 4),
            "p50": round(_percentile(round_trip_ms, 50), 4),
            "p95": round(_percentile(round_trip_ms, 95), 4),
        },
        "server_reported_latency_ms": {
            "min": round(min(server_latency_ms), 4) if server_latency_ms else None,
            "max": round(max(server_latency_ms), 4) if server_latency_ms else None,
            "mean": round(statistics.mean(server_latency_ms), 4)
            if server_latency_ms
            else None,
            "p50": round(_percentile(server_latency_ms, 50), 4)
            if server_latency_ms
            else None,
            "p95": round(_percentile(server_latency_ms, 95), 4)
            if server_latency_ms
            else None,
        },
        "notes": (
            "round_trip_ms includes TestClient JSON serialize/deserialize; "
            "server_reported_latency_ms is classifier.predict_window only."
        ),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
