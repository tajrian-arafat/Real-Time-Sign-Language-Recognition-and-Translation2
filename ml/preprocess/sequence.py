"""Fixed-length temporal resampling (T=64) with attention mask."""
from __future__ import annotations

import numpy as np


def resample_sequence(
    features: np.ndarray,
    target_length: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Linearly resample along time to ``target_length`` frames.

    Args:
        features: (T_src, F) per-frame feature vectors.
    Returns:
        (target_length, F) resampled features and (target_length,) mask (1=valid).
    """
    if features.ndim != 2:
        raise ValueError(f"Expected (T, F), got {features.shape}")

    src_len, feat_dim = features.shape
    if src_len == 0:
        out = np.zeros((target_length, feat_dim), dtype=np.float32)
        mask = np.zeros((target_length,), dtype=np.float32)
        return out, mask

    if src_len == target_length:
        return features.astype(np.float32, copy=False), np.ones((target_length,), dtype=np.float32)

    src_x = np.linspace(0.0, 1.0, src_len, dtype=np.float64)
    dst_x = np.linspace(0.0, 1.0, target_length, dtype=np.float64)

    out = np.zeros((target_length, feat_dim), dtype=np.float32)
    for j in range(feat_dim):
        out[:, j] = np.interp(dst_x, src_x, features[:, j])

    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    if src_len >= target_length:
        mask = np.ones((target_length,), dtype=np.float32)
    else:
        # Shorter clips: valid frames map to the resampled prefix; tail is padded.
        valid_dst = int(round((src_len - 1) / max(src_len - 1, 1) * (target_length - 1))) + 1
        valid_dst = min(valid_dst, target_length)
        mask = np.zeros((target_length,), dtype=np.float32)
        mask[:valid_dst] = 1.0

    return out, mask
