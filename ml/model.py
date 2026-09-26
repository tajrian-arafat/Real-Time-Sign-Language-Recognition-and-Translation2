"""Conv1D + Transformer classifier (Agent 4)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class AttentionPooling(nn.Module):
    """Single learned query attention over time; respects padding mask."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D), mask: (B, T) float/bool, 1 = valid
        scores = torch.matmul(x, self.query.transpose(1, 2)).squeeze(-1)
        valid = mask.bool()
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=1)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)


class SignLanguageClassifier(nn.Module):
    """
    Conv1D stem → Transformer encoder → attention pooling → linear head.

    Input landmarks: (batch, T, input_dim) with optional padding mask (batch, T).
    """

    def __init__(
        self,
        *,
        num_classes: int,
        input_dim: int = 392,
        d_model: int = 256,
        nhead: int = 4,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        conv_kernel: int = 3,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        padding = conv_kernel // 2
        self.conv_stem = nn.Sequential(
            nn.Conv1d(input_dim, d_model, kernel_size=conv_kernel, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_model, d_model, kernel_size=conv_kernel, padding=padding),
            nn.ReLU(inplace=True),
        )
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )
        self.pool = AttentionPooling(d_model)
        self.head_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected (B, T, F), got {tuple(x.shape)}")
        batch, seq_len, feat = x.shape
        if feat != self.input_dim:
            raise ValueError(f"Expected feature dim {self.input_dim}, got {feat}")
        if mask is None:
            mask = torch.ones(batch, seq_len, device=x.device, dtype=x.dtype)
        # Conv1d over time
        h = self.conv_stem(x.transpose(1, 2)).transpose(1, 2)
        h = self.pos_encoder(h)
        # PyTorch: True in src_key_padding_mask = ignore position
        pad_mask = ~mask.bool()
        h = self.transformer(h, src_key_padding_mask=pad_mask)
        pooled = self.pool(h, mask)
        pooled = self.head_norm(pooled)
        return self.classifier(pooled)


def build_model_from_config(
    config: dict, num_classes: int
) -> SignLanguageClassifier:
    model_cfg = config.get("model", {})
    landmarks = config.get("landmarks", {})
    return SignLanguageClassifier(
        num_classes=num_classes,
        input_dim=int(landmarks.get("input_dim_per_frame", 392)),
        d_model=int(model_cfg.get("d_model", 256)),
        nhead=int(model_cfg.get("nhead", 4)),
        num_encoder_layers=int(model_cfg.get("num_encoder_layers", 4)),
        dim_feedforward=int(model_cfg.get("dim_feedforward", 512)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        conv_kernel=int(model_cfg.get("conv1d_kernel", 3)),
    )
