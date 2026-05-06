from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class InterferenceBypassSelfAttention(nn.Module):
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
        nn.init.normal_(self.q_phase_proj.weight, mean=0.0, std=0.005)
        nn.init.normal_(self.k_phase_proj.weight, mean=0.0, std=0.005)

        if self.q_phase_proj.bias is not None:
            nn.init.zeros_(self.q_phase_proj.bias)
        if self.k_phase_proj.bias is not None:
            nn.init.zeros_(self.k_phase_proj.bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.score_log_scale = nn.Parameter(torch.zeros(self.n_head))
        self.phase_head_scale_logit = nn.Parameter(torch.full((self.n_head,), -1.0986123))  # sigmoid -> ~0.25
        self.amp_norm_eps = 1e-6

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
       if self.phase_bound == "half_pi_tanh":
          return 0.5 * math.pi * torch.tanh(x)
       if self.phase_bound == "learned_half_pi_tanh":
          return 0.5 * math.pi * torch.tanh(x)
       if self.phase_bound == "none":
          return x
       raise ValueError(f"Unknown phase_bound: {self.phase_bound}")

    def compute_scores(self, x):
        q_amp = self._split_heads(self._amplitude(self.q_amp_proj(x)))
        k_amp = self._split_heads(self._amplitude(self.k_amp_proj(x)))
        q_phi = self._split_heads(self._phase(self.q_phase_proj(x)))
        k_phi = self._split_heads(self._phase(self.k_phase_proj(x)))

        # RMS-style amplitude normalization per head
        q_amp = q_amp / torch.sqrt(q_amp.pow(2).mean(dim=-1, keepdim=True) + self.amp_norm_eps)
        k_amp = k_amp / torch.sqrt(k_amp.pow(2).mean(dim=-1, keepdim=True) + self.amp_norm_eps)

        # Optional learnable per-head phase scaling
        if self.phase_bound == "learned_half_pi_tanh":
            phase_scale = torch.sigmoid(self.phase_head_scale_logit).view(1, self.n_head, 1, 1)
            q_phi = phase_scale * q_phi
            k_phi = phase_scale * k_phi

        q_cos = q_amp * torch.cos(q_phi)
        q_sin = q_amp * torch.sin(q_phi)
        k_cos = k_amp * torch.cos(k_phi)
        k_sin = k_amp * torch.sin(k_phi)

        scores = (
            q_cos @ k_cos.transpose(-2, -1) +
            q_sin @ k_sin.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        # Learnable per-head score temperature
        score_scale = torch.exp(self.score_log_scale).view(1, self.n_head, 1, 1)
        scores = score_scale * scores

        return scores
    
    def forward(self, x):
        _, seq_len, _ = x.shape
        scores = self.compute_scores(x)
        v = self._split_heads(self.v_proj(x))

        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        scores = scores.masked_fill(~mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ v
        out = self._merge_heads(out)
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        return out
