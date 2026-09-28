"""Tests for live landmark window preprocessing parity with training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from backend.inference import OnnxSignClassifier, reset_classifier_for_tests
from ml.preprocess.landmark_spec import FEATURES_PER_FRAME
from ml.preprocess.live_window import preprocess_landmark_frames
from ml.preprocess.normalize import normalize_and_pack
from ml.preprocess.tasks_holistic import (
    HOLISTIC_FLAT_DIM,
    LEGACY_TASKS_FLAT_DIM,
    holistic_flat_to_frame,
    legacy_tasks_flat_to_holistic,
)


def _synthetic_holistic_frames(t: int = 12) -> np.ndarray:
    rng = np.random.default_rng(7)
    frames = rng.normal(size=(t, 543, 3)).astype(np.float32) * 0.05
    frames[:, 11, :] = np.array([-0.2, 0.0, 0.0], dtype=np.float32)
    frames[:, 12, :] = np.array([0.2, 0.0, 0.0], dtype=np.float32)
    return frames


def test_holistic_flat_preprocess_matches_training_pack() -> None:
    holistic = _synthetic_holistic_frames(6)
    expected = normalize_and_pack(holistic)
    flats = [row.reshape(-1).tolist() for row in holistic]
    got = preprocess_landmark_frames(flats)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


def test_legacy_flat_converts_to_finite_packed() -> None:
    holistic = _synthetic_holistic_frames(1)[0]
    legacy = np.zeros(LEGACY_TASKS_FLAT_DIM, dtype=np.float32)
    idx = 0
    for hand_start in (501, 522):
        for i in range(21):
            legacy[idx : idx + 3] = holistic[hand_start + i]
            idx += 3
    for i in range(33):
        legacy[idx : idx + 3] = holistic[i]
        idx += 3
    for i in range(40):
        legacy[idx : idx + 3] = holistic[33 + i]
        idx += 3
    packed = preprocess_landmark_frames([legacy.tolist()])
    assert packed.shape == (1, FEATURES_PER_FRAME)
    assert np.isfinite(packed).all()


def test_hello_label_index_from_val_index() -> None:
    val_index = Path("artifacts/hf_restore/splits/val_tensor_index.json")
    if not val_index.is_file():
        pytest.skip("val tensor index not present in this checkout")
    rows = json.loads(val_index.read_text(encoding="utf-8"))
    hello_rows = [r for r in rows if r.get("sign") == "hello"]
    assert hello_rows, "expected hello samples in val index"
    assert hello_rows[0]["label_index"] >= 0


def test_synthetic_holistic_hello_class_stub_inference() -> None:
    """Upper bound with stub model: preprocessed finite features run end-to-end."""
    reset_classifier_for_tests()
    classifier = OnnxSignClassifier()
    holistic = _synthetic_holistic_frames(48)
    flats = [f.reshape(-1).tolist() for f in holistic]
    result = classifier.predict_window(flats)
    assert result.word
    assert 0.0 <= result.confidence <= 1.0


def test_holistic_flat_roundtrip_shape() -> None:
    holistic = _synthetic_holistic_frames(1)[0]
    flat = holistic.reshape(-1)
    assert flat.size == HOLISTIC_FLAT_DIM
    roundtrip = holistic_flat_to_frame(flat.tolist())
    np.testing.assert_allclose(roundtrip, holistic, rtol=1e-6)


def test_legacy_parser_places_hands_in_holistic_slots() -> None:
    holistic = _synthetic_holistic_frames(1)[0]
    legacy = np.zeros(LEGACY_TASKS_FLAT_DIM, dtype=np.float32)
    idx = 0
    for hand_start in (501, 522):
        for i in range(21):
            legacy[idx : idx + 3] = holistic[hand_start + i]
            idx += 3
    for i in range(33):
        legacy[idx : idx + 3] = holistic[i]
        idx += 3
    parsed = legacy_tasks_flat_to_holistic(legacy.tolist())
    np.testing.assert_allclose(parsed[501], holistic[501], rtol=1e-5)
    np.testing.assert_allclose(parsed[11], holistic[11], rtol=1e-5)
