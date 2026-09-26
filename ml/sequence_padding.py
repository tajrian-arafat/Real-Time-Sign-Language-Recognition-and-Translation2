"""Resize/pad landmark sequences to fixed length T."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def resize_sequence_linear(
    sequence: NDArray[np.floating], target_length: int
) -> NDArray[np.float32]:
    """Linearly resample along time axis to target_length."""
    seq = np.asarray(sequence, dtype=np.float64)
    if seq.ndim != 3:
        raise ValueError(f"Expected (T, L, D), got {seq.shape}")
    t_in = seq.shape[0]
    if t_in == target_length:
        return seq.astype(np.float32)
    if t_in == 0:
        raise ValueError("Cannot resize empty sequence")
    x_old = np.linspace(0.0, 1.0, t_in)
    x_new = np.linspace(0.0, 1.0, target_length)
    flat = seq.reshape(t_in, -1)
    resampled = np.stack(
        [np.interp(x_new, x_old, flat[:, j]) for j in range(flat.shape[1])],
        axis=1,
    )
    l_dim, d_dim = seq.shape[1], seq.shape[2]
    return resampled.reshape(target_length, l_dim, d_dim).astype(np.float32)


def pad_sequence_edge(
    sequence: NDArray[np.floating], target_length: int
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Edge-pad shorter sequences; returns (padded, attention_mask)."""
    seq = np.asarray(sequence, dtype=np.float64)
    if seq.ndim != 3:
        raise ValueError(f"Expected (T, L, D), got {seq.shape}")
    t_in = seq.shape[0]
    if t_in > target_length:
        return resize_sequence_linear(seq, target_length), np.ones(
            target_length, dtype=np.bool_
        )
    if t_in == target_length:
        return seq.astype(np.float32), np.ones(target_length, dtype=np.bool_)
    pad_count = target_length - t_in
    last = np.repeat(seq[-1:, ...], pad_count, axis=0)
    padded = np.concatenate([seq, last], axis=0).astype(np.float32)
    mask = np.zeros(target_length, dtype=np.bool_)
    mask[:t_in] = True
    return padded, mask


def fit_sequence_length(
    sequence: NDArray[np.floating], target_length: int
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Longer -> linear resize; shorter -> edge pad with mask."""
    seq = np.asarray(sequence, dtype=np.float64)
    t_in = seq.shape[0]
    if t_in > target_length:
        out = resize_sequence_linear(seq, target_length)
        return out, np.ones(target_length, dtype=np.bool_)
    return pad_sequence_edge(seq, target_length)
