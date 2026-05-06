from __future__ import annotations

import torch
import torch.nn as nn


class QuantumPaperSelfAttention(nn.Module):
    """
    Practical baseline inspired by the quantum-paper attention rule:
    A = Q K^T, without softmax, with causal masking by zeroing future entries.
    This version is stabilized for practical PyTorch training.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        max_seq_len: int,
        dropout: float = 0.1,
        bias: bool = False,
    ):
        super().__init__()
        assert d_model % n_head == 0

        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.resid_dropout = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer(
            "causal_mask",
            mask.view(1, 1, max_seq_len, max_seq_len),
            persistent=False,
        )

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        x = x.view(bsz, seq_len, self.n_head, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, n_head, seq_len, head_dim = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(bsz, seq_len, n_head * head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = x.shape

        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        scores = scores * mask.to(dtype=scores.dtype)

        # Stabilization for practical training
        scores = torch.tanh(scores)
        denom = scores.abs().sum(dim=-1, keepdim=True).clamp_min(1e-6)
        scores = scores / denom

        out = scores @ v
        out = self._merge_heads(out)
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        return out