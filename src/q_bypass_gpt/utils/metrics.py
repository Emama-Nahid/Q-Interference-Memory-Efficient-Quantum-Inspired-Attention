from __future__ import annotations

import math

import torch


def perplexity_from_loss(loss: float) -> float:
    return math.exp(loss) if loss < 20 else float("inf")


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def compute_grad_norm(model) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        param_norm = p.grad.detach().data.norm(2)
        total += param_norm.item() ** 2
    return total ** 0.5
