from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def naive_interference_scores(q_amp, k_amp, q_phi, k_phi):
    phase_diff = q_phi.unsqueeze(-2) - k_phi.unsqueeze(-3)
    pairwise = q_amp.unsqueeze(-2) * k_amp.unsqueeze(-3) * torch.cos(phase_diff)
    scores = pairwise.sum(dim=-1) / math.sqrt(q_amp.size(-1))
    return scores


class NaiveInterferenceSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        max_seq_len: int,
        dropout: float = 0.1,
        bias: bool = False,
        amplitude_activation: str = "softplus",
        phase_bound: str = "pi_tanh",
        disable_phase: bool = False,
    ):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.amplitude_activation = amplitude_activation
        self.phase_bound = phase_bound
        self.disable_phase = disable_phase

        self.q_amp_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_amp_proj = nn.Linear(d_model, d_model, bias=bias)
        self.q_phase_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_phase_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, max_seq_len, max_seq_len), persistent=False)

    def _split_heads(self, x):
        bsz, seq_len, _ = x.shape
        x = x.view(bsz, seq_len, self.n_head, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x):
        bsz, n_head, seq_len, head_dim = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(bsz, seq_len, n_head * head_dim)

    def _amplitude(self, x):
        if self.amplitude_activation == "softplus":
            return F.softplus(x)
        if self.amplitude_activation == "relu":
            return F.relu(x)
        if self.amplitude_activation == "exp":
            return torch.exp(torch.clamp(x, max=8.0))
        raise ValueError(f"Unknown amplitude_activation: {self.amplitude_activation}")

    def _phase(self, x):
        if self.disable_phase:
            return torch.zeros_like(x)
        if self.phase_bound == "pi_tanh":
            return math.pi * torch.tanh(x)
        if self.phase_bound == "none":
            return x
        raise ValueError(f"Unknown phase_bound: {self.phase_bound}")

    def forward(self, x):
        _, seq_len, _ = x.shape

        q_amp = self._split_heads(self._amplitude(self.q_amp_proj(x)))
        k_amp = self._split_heads(self._amplitude(self.k_amp_proj(x)))
        q_phi = self._split_heads(self._phase(self.q_phase_proj(x)))
        k_phi = self._split_heads(self._phase(self.k_phase_proj(x)))
        v = self._split_heads(self.v_proj(x))

        scores = naive_interference_scores(q_amp, k_amp, q_phi, k_phi)
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        scores = scores.masked_fill(~mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ v
        out = self._merge_heads(out)
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        return out
