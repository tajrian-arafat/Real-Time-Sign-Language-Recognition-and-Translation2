"""Unit tests for landmark preprocessing (synthetic parquets)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ml.preprocess.landmark_spec import FEATURES_PER_FRAME, SELECTED_HOLISTIC_INDICES
from ml.preprocess.normalize import normalize_and_pack, normalize_sequence
from ml.preprocess.sequence import resample_sequence
from ml.preprocess.kaggle_io import load_parquet_landmarks
from ml.preprocess.splits import split_by_participant
from ml.preprocess.vocabulary import build_label_map_from_train_csv


def _synthetic_holistic_frames(t: int = 10) -> np.ndarray:
    rng = np.random.default_rng(0)
    frames = rng.normal(size=(t, 543, 3)).astype(np.float32)
    # Stable shoulders for normalization
    frames[:, 11, :] = np.array([-0.2, 0.0, 0.0], dtype=np.float32)
    frames[:, 12, :] = np.array([0.2, 0.0, 0.0], dtype=np.float32)
    return frames


def test_landmark_subset_count() -> None:
    assert len(SELECTED_HOLISTIC_INDICES) == 130
    assert FEATURES_PER_FRAME == 392


def test_normalize_and_pack_shape() -> None:
    holistic = _synthetic_holistic_frames(8)
    packed = normalize_and_pack(holistic)
    assert packed.shape == (8, 392)
    assert np.isfinite(packed).all()
    # Shoulders centered: left shoulder x near -0.5, right near +0.5 after scale
    selected = normalize_sequence(holistic)
    left_idx = SELECTED_HOLISTIC_INDICES.index(11)
    right_idx = SELECTED_HOLISTIC_INDICES.index(12)
    assert selected[0, left_idx, 0] == pytest.approx(-0.5, abs=0.05)
    assert selected[0, right_idx, 0] == pytest.approx(0.5, abs=0.05)


def test_resample_to_64_with_mask() -> None:
    packed = normalize_and_pack(_synthetic_holistic_frames(5))
    out, mask = resample_sequence(packed, target_length=64)
    assert out.shape == (64, 392)
    assert mask.shape == (64,)
    assert mask.sum() >= 1


def test_load_synthetic_parquet(tmp_path: Path) -> None:
    t = 6
    cols = ["frame", "row_id"]
    data = {"frame": list(range(t)), "row_id": list(range(t))}
    for i in range(543 * 3):
        data[str(i)] = np.random.randn(t).astype(np.float32)
    df = pd.DataFrame(data)
    pq = tmp_path / "seq.parquet"
    df.to_parquet(pq, index=False)
    arr = load_parquet_landmarks(pq)
    assert arr.shape == (t, 543, 3)


def test_label_map_and_splits(tmp_path: Path) -> None:
    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        "path,participant_id,sequence_id,sign\n"
        "a.parquet,p1,s1,hello\n"
        "b.parquet,p1,s2,hello\n"
        "c.parquet,p2,s3,world\n"
        "d.parquet,p3,s4,world\n"
        "e.parquet,p4,s5,thanks\n",
        encoding="utf-8",
    )
    label_map = build_label_map_from_train_csv(train_csv)
    assert label_map["vocabulary_size"] == 3
    rows = [
        {"path": "a.parquet", "participant_id": "p1", "sequence_id": "s1", "sign": "hello"},
        {"path": "b.parquet", "participant_id": "p1", "sequence_id": "s2", "sign": "hello"},
        {"path": "c.parquet", "participant_id": "p2", "sequence_id": "s3", "sign": "world"},
        {"path": "d.parquet", "participant_id": "p3", "sequence_id": "s4", "sign": "world"},
        {"path": "e.parquet", "participant_id": "p4", "sequence_id": "s5", "sign": "thanks"},
    ]
    splits = split_by_participant(rows, seed=0)
    assert splits["counts"]["train"] + splits["counts"]["val"] + splits["counts"]["test"] == 5
    participants = set(splits["participants"]["train"]) | set(splits["participants"]["val"]) | set(
        splits["participants"]["test"]
    )
    assert participants == {"p1", "p2", "p3", "p4"}


def test_normalize_nan_shoulders_produces_finite_packed() -> None:
    holistic = _synthetic_holistic_frames(4)
    holistic[:, 11, :] = np.nan
    holistic[:, 12, :] = np.nan
    packed = normalize_and_pack(holistic)
    assert packed.shape == (4, 392)
    assert np.isfinite(packed).all()


def test_long_format_holistic_offsets(tmp_path: Path) -> None:
    """Left hand landmark_index 0 must map to holistic index 501, not 33."""
    rows = []
    frame = 0
    for i in range(33):
        rows.append(
            {
                "frame": frame,
                "row_id": len(rows),
                "type": "pose",
                "landmark_index": i,
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
            }
        )
    rows.append(
        {
            "frame": frame,
            "row_id": len(rows),
            "type": "left_hand",
            "landmark_index": 0,
            "x": 1.0,
            "y": 2.0,
            "z": 3.0,
        }
    )
    df = pd.DataFrame(rows)
    pq = tmp_path / "long.parquet"
    df.to_parquet(pq, index=False)
    arr = load_parquet_landmarks(pq)
    assert arr.shape == (1, 543, 3)
    np.testing.assert_allclose(arr[0, 501], [1.0, 2.0, 3.0])
    assert np.allclose(arr[0, 33], 0.0)


def test_run_preprocessing_waiting_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIGN_LANGUAGE_DATA_ROOT", str(tmp_path))
    (tmp_path / "raw" / "kaggle_asl_signs").mkdir(parents=True)
    from ml.preprocess.paths import reports_dir, load_config
    import data.scripts.run_preprocessing as rp

    rp.main()
    status_path = reports_dir(load_config()) / "preprocessing_status.json"
    assert status_path.is_file()
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["status"] == "waiting_for_kaggle"
