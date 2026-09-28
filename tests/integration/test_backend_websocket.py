"""WebSocket integration test with synthetic landmark windows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.bangla_service import load_bangla_dictionary, reset_translation_singletons_for_tests
from backend.inference import reset_classifier_for_tests
from backend.main import app


def _synthetic_window(seed: float = 0.42, num_frames: int = 32) -> dict:
    rng = np.random.default_rng(int(seed * 1000))
    holistic = rng.normal(size=(num_frames, 543, 3)).astype(np.float32) * 0.05
    holistic[:, 11, :] = np.array([-0.2, 0.0, 0.0], dtype=np.float32)
    holistic[:, 12, :] = np.array([0.2, 0.0, 0.0], dtype=np.float32)
    frames = []
    for i in range(num_frames):
        flat = holistic[i].reshape(-1).tolist()
        frames.append({"timestamp_ms": i * 66, "landmarks": flat})
    return {"type": "landmark_window", "frames": frames}


@pytest.fixture()
def client() -> TestClient:
    reset_classifier_for_tests()
    reset_translation_singletons_for_tests()
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_model(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_loaded"] is True
    served_onnx = Path("models/served/model.onnx")
    if served_onnx.is_file():
        assert body["is_stub"] is False
    else:
        assert body["is_stub"] is True


def test_websocket_prediction_round_trip(client: TestClient, tmp_path: Path) -> None:
    report_path = Path("reports/backend_smoke.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    window = _synthetic_window()
    with client.websocket_connect("/ws/recognize") as ws:
        status = ws.receive_json()
        assert status["type"] == "status"
        assert status["connection"] == "connected"

        ws.send_json(window)
        prediction = ws.receive_json()
        assert prediction["type"] == "prediction"
        assert isinstance(prediction["word"], str)
        assert prediction["word"]
        assert 0.0 <= prediction["confidence"] <= 1.0
        assert len(prediction["top_k"]) >= 1
        assert prediction["top_k"][0]["word"] == prediction["word"]

        # Deterministic for identical input
        ws.send_json(window)
        prediction_repeat = ws.receive_json()
        assert prediction_repeat["word"] == prediction["word"]
        assert prediction_repeat["confidence"] == pytest.approx(
            prediction["confidence"], rel=1e-5
        )

    dict_resp = client.post("/api/translate/word", json={"english": "hello"})
    instant_ok = dict_resp.status_code == 200 and dict_resp.json().get("bangla")

    health = client.get("/health").json()
    report = {
        "health_ok": True,
        "websocket_ok": True,
        "deterministic_repeat_ok": True,
        "instant_dictionary_ok": bool(instant_ok),
        "sample_prediction": prediction,
        "sample_instant_translate": dict_resp.json() if instant_ok else None,
        "model_version": prediction.get("model_version"),
        "is_stub": bool(health.get("is_stub")),
        "onnx_path": health.get("model_path"),
        "bangla_dictionary_words": len(load_bangla_dictionary()),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
