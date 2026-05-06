from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from q_bypass_gpt.models.blocks import TransformerBlock
from q_bypass_gpt.models.config import ModelConfig


class QBypassGPT(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = ModelConfig(**config_dict)

        self.token_emb = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.pos_emb = nn.Embedding(self.config.max_seq_len, self.config.d_model)
        self.drop = nn.Dropout(self.config.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(self.config) for _ in range(self.config.n_layer)])
        self.ln_f = nn.LayerNorm(self.config.d_model)
        self.lm_head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)

        self.lm_head.weight = self.token_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        _, seq_len = idx.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len {self.config.max_seq_len}"
            )

        pos = torch.arange(0, seq_len, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )

        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.max_seq_len :]
            out = self(idx_cond)
            logits = out["logits"][:, -1, :] / max(temperature, 1e-6)

            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx
