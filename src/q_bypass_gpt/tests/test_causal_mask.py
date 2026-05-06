from __future__ import annotations

import torch

from q_bypass_gpt.models.gpt_baseline import BaselineGPT


def test_causal_mask_prevents_future_leakage():
    config = {
        "vocab_size": 50,
        "max_seq_len": 16,
        "n_layer": 2,
        "n_head": 2,
        "d_model": 32,
        "dropout": 0.0,
        "bias": False,
        "attention_type": "baseline",
        "amplitude_activation": "softplus",
        "phase_bound": "pi_tanh",
        "disable_phase": False,
    }
    model = BaselineGPT(config).eval()

    x1 = torch.randint(0, 50, (1, 10))
    x2 = x1.clone()
    x2[0, -1] = (x2[0, -1] + 1) % 50

    with torch.no_grad():
        out1 = model(x1)["logits"]
        out2 = model(x2)["logits"]

    assert torch.allclose(out1[:, :-1, :], out2[:, :-1, :], atol=1e-5)
