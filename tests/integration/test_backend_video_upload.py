"""Video upload recognition with synthetic landmarks (no real MP4 fixture required)."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.inference import reset_classifier_for_tests
from backend.landmark_server import ExtractedFrame
from backend.main import app

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE = REPO_ROOT / "artifacts" / "user_test_videos" / "WATER - 1080.mp4"


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


def test_recognize_video_alias_matches_primary(client: TestClient) -> None:
    health = client.get("/health")
    if not health.json().get("model_loaded"):
        pytest.skip("Model not loaded on this checkout")

    frames = _synthetic_frames()
    with patch(
        "backend.main.VideoLandmarkExtractor.extract_from_bytes",
        return_value=frames,
    ):
        payload = b"\x00\x00\x00\x18ftypmp42"
        primary = client.post(
            "/api/recognize/video",
            files={"file": ("clip.mp4", payload, "video/mp4")},
        )
        alias = client.post(
            "/api/video",
            files={"file": ("clip.mp4", payload, "video/mp4")},
        )
    assert primary.status_code == 200
    assert alias.status_code == 200
    assert alias.json()["segments"] == primary.json()["segments"]


@pytest.mark.integration
def test_recognize_video_user_fixture_when_present(client: TestClient) -> None:
    """Optional local/Drive clip; skipped in CI when fixture path is absent."""
    fixture_path = Path(
        os.environ.get("SIGN_LANGUAGE_TEST_VIDEO_FIXTURE", str(DEFAULT_FIXTURE))
    )
    if not fixture_path.is_file():
        pytest.skip(f"No video fixture at {fixture_path}")

    health = client.get("/health")
    if not health.json().get("model_loaded"):
        pytest.skip("Model not loaded on this checkout")

    data = fixture_path.read_bytes()
    resp = client.post(
        "/api/recognize/video",
        files={"file": (fixture_path.name, data, "video/mp4")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["segments"], "Expected at least one recognition segment"
    assert body["segments"][0]["confidence"] > 0.0
