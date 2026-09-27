"""Unit tests for sequence length fitting."""

from __future__ import annotations

import numpy as np

from ml.sequence_padding import fit_sequence_length, pad_sequence_edge, resize_sequence_linear


def test_resize_sequence_linear_longer_to_shorter(
    synthetic_sequence: np.ndarray,
) -> None:
    out = resize_sequence_linear(synthetic_sequence, 4)
    assert out.shape == (4, synthetic_sequence.shape[1], synthetic_sequence.shape[2])


def test_pad_sequence_edge_mask(synthetic_sequence: np.ndarray) -> None:
    short = synthetic_sequence[:3]
    padded, mask = pad_sequence_edge(short, 8)
    assert padded.shape[0] == 8
    assert mask.tolist() == [True, True, True, False, False, False, False, False]
    np.testing.assert_array_equal(padded[3], padded[2])


def test_fit_sequence_length_downsample(synthetic_sequence: np.ndarray) -> None:
    out, mask = fit_sequence_length(synthetic_sequence, 4)
    assert out.shape[0] == 4
    assert mask.all()
