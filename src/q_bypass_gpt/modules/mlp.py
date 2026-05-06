from __future__ import annotations

import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        hidden = 4 * d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
