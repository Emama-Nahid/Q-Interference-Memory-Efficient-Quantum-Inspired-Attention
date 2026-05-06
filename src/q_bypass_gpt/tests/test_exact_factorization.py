from __future__ import annotations

import math

import torch

from q_bypass_gpt.modules.naive_interference_attention import naive_interference_scores


def test_exact_factorization_matches_naive():
    torch.manual_seed(42)
    b, h, t, dh = 2, 3, 8, 16

    q_amp = torch.rand(b, h, t, dh)
    k_amp = torch.rand(b, h, t, dh)
    q_phi = math.pi * (2 * torch.rand(b, h, t, dh) - 1)
    k_phi = math.pi * (2 * torch.rand(b, h, t, dh) - 1)

    naive = naive_interference_scores(q_amp, k_amp, q_phi, k_phi)

    bypass = (
        (q_amp * torch.cos(q_phi)) @ (k_amp * torch.cos(k_phi)).transpose(-2, -1)
        + (q_amp * torch.sin(q_phi)) @ (k_amp * torch.sin(k_phi)).transpose(-2, -1)
    ) / math.sqrt(dh)

    assert torch.allclose(naive, bypass, atol=1e-6), (naive - bypass).abs().max().item()
