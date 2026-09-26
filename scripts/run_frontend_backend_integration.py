#!/usr/bin/env python3
"""Smoke-test frontend↔backend contract (REST + WebSocket landmarks)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import httpx
import numpy as np
from fastapi.testclient import TestClient

REPO = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO / "reports" / "frontend_backend_integration.json"


def _synthetic_window(num_frames: int = 16) -> dict:
    rng = np.random.default_rng(42)
    frames = []
    for i in range(num_frames):
        vec = rng.random(392, dtype=np.float32).tolist()
        frames.append({"timestamp_ms": i * 66, "landmarks": vec})
    return {"type": "landmark_window", "frames": frames}


def main() -> int:
    sys.path.insert(0, str(REPO))
    from backend.bangla_service import reset_translation_singletons_for_tests
    from backend.inference import reset_classifier_for_tests
    from backend.main import app

    reset_classifier_for_tests()
    reset_translation_singletons_for_tests()

    report: dict = {
        "backend_base": "http://127.0.0.1:8000",
        "vite_proxy": {"api": "/api", "ws": "/ws"},
        "vite_mock_default": "false (opt-in mock via VITE_MOCK_WS=true)",
        "websocket_path": "/ws/recognize",
        "landmark_feature_dim": 392,
        "landmark_stream": "browser MediaPipe → landmark_window JSON (no video)",
    }

    window = _synthetic_window()

    with TestClient(app) as client:
        health_resp = client.get("/health")
        report["health_ok"] = health_resp.status_code == 200
        report["health"] = health_resp.json()

        with client.websocket_connect("/ws/recognize") as ws:
            status = ws.receive_json()
            ws.send_json(window)
            prediction = ws.receive_json()

        report["websocket_status"] = status
        report["websocket_ok"] = (
            status.get("type") == "status"
            and prediction.get("type") == "prediction"
        )
        report["sample_prediction"] = prediction

        word_resp = client.post("/api/translate/word", json={"english": "hello"})
        words_resp = client.post(
            "/api/translate/words", json={"words": ["hello", "thank you"]}
        )
        sentence_resp = client.post(
            "/api/translate/sentence",
            json={"english": "hello friend"},
        )

    report["translate_word_ok"] = word_resp.status_code == 200 and bool(
        word_resp.json().get("bangla")
    )
    report["translate_words_ok"] = words_resp.status_code == 200
    report["translate_sentence_ok"] = sentence_resp.status_code == 200 and bool(
        sentence_resp.json().get("bangla")
    )
    report["sample_translate_word"] = word_resp.json()
    report["sample_translate_sentence"] = sentence_resp.json()

    env_example = (REPO / "frontend" / ".env.example").read_text(encoding="utf-8")
    report["env_example_mock_ws_false"] = "VITE_MOCK_WS=false" in env_example

    frontend_test = subprocess.run(
        ["npm", "run", "test", "--", "--run"],
        cwd=REPO / "frontend",
        capture_output=True,
        text=True,
        check=False,
    )
    report["frontend_vitest_ok"] = frontend_test.returncode == 0
    if not report["frontend_vitest_ok"]:
        report["frontend_vitest_stderr"] = frontend_test.stderr[-2000:]

    report["ok"] = all(
        [
            report["health_ok"],
            report["websocket_ok"],
            report["translate_word_ok"],
            report["translate_words_ok"],
            report["translate_sentence_ok"],
            report["env_example_mock_ws_false"],
            report["frontend_vitest_ok"],
        ]
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
