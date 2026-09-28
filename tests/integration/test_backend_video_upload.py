"""Video upload recognition with synthetic landmarks (no real MP4 fixture required)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.inference import reset_classifier_for_tests
from backend.landmark_server import ExtractedFrame
from backend.main import app


def _synthetic_frames(count: int = 24, seed: float = 1.0) -> list[ExtractedFrame]:
    rng = np.random.default_rng(int(seed * 1000))
    return [
        ExtractedFrame(
            timestamp_ms=i * 66,
            landmarks=rng.random(392, dtype=np.float32).tolist(),
        )
        for i in range(count)
    ]


@pytest.fixture()
def client() -> TestClient:
    reset_classifier_for_tests()
    with TestClient(app) as test_client:
        yield test_client


def test_recognize_video_with_stubbed_extraction(client: TestClient) -> None:
    health = client.get("/health")
    assert health.status_code == 200
    if not health.json().get("model_loaded"):
        pytest.skip("Model not loaded on this checkout")

    frames = _synthetic_frames()
    with patch(
        "backend.main.VideoLandmarkExtractor.extract_from_bytes",
        return_value=frames,
    ):
        resp = client.post(
            "/api/recognize/video",
            files={"file": ("clip.mp4", b"\x00\x00\x00\x18ftypmp42", "video/mp4")},
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["vocab_size"] >= 1
    assert body["latency_ms"] >= 0.0
    assert isinstance(body["segments"], list)
    if body["segments"]:
        seg = body["segments"][0]
        assert isinstance(seg["word"], str) and seg["word"]
        assert 0.0 <= seg["confidence"] <= 1.0
