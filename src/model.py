from __future__ import annotations
import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)

class TransformerClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        seq_len: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.1,
        pooling: str = "mean",
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pooling = pooling

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max(512, seq_len + 2), dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, F]
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.encoder(h)

        if self.pooling == "cls":
            pooled = h[:, 0]
        else:
            pooled = h.mean(dim=1)
        return self.classifier(pooled)
