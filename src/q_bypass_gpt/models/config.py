from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int
    max_seq_len: int
    n_layer: int
    n_head: int
    d_model: int
    dropout: float = 0.1
    bias: bool = False
    attention_type: str = "baseline"
    amplitude_activation: str = "softplus"
    phase_bound: str = "pi_tanh"
    disable_phase: bool = False

    # new fields for hybrid attention
    hybrid_lambda_init: float = 0.0
    hybrid_lambda_max: float = 0.5