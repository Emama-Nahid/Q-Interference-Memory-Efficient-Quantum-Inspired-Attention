from __future__ import annotations

import torch

from q_bypass_gpt.models.gpt_baseline import BaselineGPT
from q_bypass_gpt.models.qbypass_gpt import QBypassGPT


def _base_config(attention_type: str):
    return {
        "vocab_size": 100,
        "max_seq_len": 32,
        "n_layer": 2,
        "n_head": 4,
        "d_model": 64,
        "dropout": 0.0,
        "bias": False,
        "attention_type": attention_type,
        "amplitude_activation": "softplus",
        "phase_bound": "pi_tanh",
        "disable_phase": False,
    }


def test_baseline_shapes():
    model = BaselineGPT(_base_config("baseline"))
    x = torch.randint(0, 100, (2, 16))
    y = torch.randint(0, 100, (2, 16))
    out = model(x, y)
    assert out["logits"].shape == (2, 16, 100)
    assert out["loss"].ndim == 0


def test_qbypass_shapes():
    model = QBypassGPT(_base_config("interference_bypass"))
    x = torch.randint(0, 100, (2, 16))
    y = torch.randint(0, 100, (2, 16))
    out = model(x, y)
    assert out["logits"].shape == (2, 16, 100)
    assert out["loss"].ndim == 0
