from __future__ import annotations

import torch.nn as nn

from q_bypass_gpt.modules.baseline_attention import BaselineSelfAttention
from q_bypass_gpt.modules.hybrid_interference_attention import HybridInterferenceSelfAttention
from q_bypass_gpt.modules.interference_attention import InterferenceBypassSelfAttention
from q_bypass_gpt.modules.mlp import MLP
from q_bypass_gpt.modules.naive_interference_attention import NaiveInterferenceSelfAttention
from q_bypass_gpt.modules.quantum_paper_attention import QuantumPaperSelfAttention


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.d_model)
        self.ln_2 = nn.LayerNorm(cfg.d_model)

        if cfg.attention_type == "baseline":
            self.attn = BaselineSelfAttention(
                d_model=cfg.d_model,
                n_head=cfg.n_head,
                max_seq_len=cfg.max_seq_len,
                dropout=cfg.dropout,
                bias=cfg.bias,
            )
        elif cfg.attention_type == "quantum_paper":
            self.attn = QuantumPaperSelfAttention(
                d_model=cfg.d_model,
                n_head=cfg.n_head,
                max_seq_len=cfg.max_seq_len,
                dropout=cfg.dropout,
                bias=cfg.bias,
            )
        elif cfg.attention_type == "interference_bypass":
            self.attn = InterferenceBypassSelfAttention(
                d_model=cfg.d_model,
                n_head=cfg.n_head,
                max_seq_len=cfg.max_seq_len,
                dropout=cfg.dropout,
                bias=cfg.bias,
                amplitude_activation=cfg.amplitude_activation,
                phase_bound=cfg.phase_bound,
                disable_phase=cfg.disable_phase,
            )
        elif cfg.attention_type == "interference_naive":
            self.attn = NaiveInterferenceSelfAttention(
                d_model=cfg.d_model,
                n_head=cfg.n_head,
                max_seq_len=cfg.max_seq_len,
                dropout=cfg.dropout,
                bias=cfg.bias,
                amplitude_activation=cfg.amplitude_activation,
                phase_bound=cfg.phase_bound,
                disable_phase=cfg.disable_phase,
            )
        elif cfg.attention_type == "interference_hybrid":
            self.attn = HybridInterferenceSelfAttention(
                d_model=cfg.d_model,
                n_head=cfg.n_head,
                max_seq_len=cfg.max_seq_len,
                dropout=cfg.dropout,
                bias=cfg.bias,
                amplitude_activation=cfg.amplitude_activation,
                phase_bound=cfg.phase_bound,
                disable_phase=cfg.disable_phase,
                hybrid_lambda_init=cfg.hybrid_lambda_init,
                hybrid_lambda_max=cfg.hybrid_lambda_max,
            )
        else:
            raise ValueError(f"Unknown attention_type: {cfg.attention_type}")

        self.mlp = MLP(cfg.d_model, cfg.dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x