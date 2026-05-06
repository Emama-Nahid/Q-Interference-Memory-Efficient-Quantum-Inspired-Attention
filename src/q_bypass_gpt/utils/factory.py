from __future__ import annotations
from q_bypass_gpt.models.hf_causal_lm import HFCausalLMWrapper
from q_bypass_gpt.modules.quantum_paper_attention import QuantumPaperSelfAttention

import math

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from q_bypass_gpt.models.gpt_baseline import BaselineGPT
from q_bypass_gpt.models.qbypass_gpt import QBypassGPT
from q_bypass_gpt.data.datamodule import TextDataModule


def build_datamodule(config: dict):
    return TextDataModule(config["data"])


def build_model(config: dict, vocab_size: int):
    model_cfg = dict(config["model"])

    # Hugging Face external causal LM baseline path.
    model_type = model_cfg.get("model_type", "local_gpt")

    if model_type == "hf_causal_lm":
        return HFCausalLMWrapper(
            model_name=model_cfg["pretrained_model_name"],
            vocab_size=vocab_size,
        )

    # Local GPT / quantum-paper / Q-Bypass path.
    model_cfg["vocab_size"] = vocab_size
    model_cfg["max_seq_len"] = config["data"]["block_size"]

    attention_type = model_cfg["attention_type"]

    if attention_type in {"baseline", "quantum_paper"}:
        return BaselineGPT(model_cfg)

    if attention_type in {
        "interference_bypass",
        "interference_naive",
        "interference_hybrid",
    }:
        return QBypassGPT(model_cfg)

    raise ValueError(f"Unknown attention_type: {attention_type}")


def build_optimizer(config: dict, model):
    train_cfg = config["train"]
    return AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        betas=tuple(train_cfg["betas"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )


def build_scheduler(config: dict, optimizer):
    train_cfg = config["train"]
    warmup_steps = int(train_cfg["warmup_steps"])
    max_steps = int(train_cfg["max_steps"])
    min_lr = float(train_cfg["min_lr"])
    base_lr = float(train_cfg["lr"])

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return max(current_step, 1) / max(warmup_steps, 1)

        progress = (current_step - warmup_steps) / max(max_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = min_lr + (base_lr - min_lr) * cosine
        return lr / base_lr

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
