from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


class HFCausalLMWrapper(nn.Module):
    """
    Hugging Face causal LM wrapper for this project.

    This wrapper makes external Hugging Face causal language models behave like
    the local QBypassGPT model.

    Your local project expects:
        out = model(x, y)
        out["logits"]
        out["loss"]

    Important:
    We do NOT pass labels to the Hugging Face model because this project
    already creates shifted next-token labels as y.

    Important for AMP:
    We force the loaded model parameters to FP32. Some Hugging Face checkpoints
    may load with FP16 parameters depending on the checkpoint/config. If trainable
    parameters are FP16, PyTorch GradScaler can crash with:
        ValueError: Attempting to unscale FP16 gradients.
    AMP autocast will still use FP16 operations where safe during forward pass.
    """

    def __init__(self, model_name: str, vocab_size: Optional[int] = None):
        super().__init__()
        self.model_name = model_name

        # Load the pretrained causal LM.
        # Do not pass torch_dtype=torch.float16 here for training.
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # Force trainable weights to FP32 for safe AMP training.
        # This fixes OPT-125M "Attempting to unscale FP16 gradients" crashes.
        self.model = self.model.float()

        # Disable KV cache during training to avoid unnecessary memory use.
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False

        # Safety check. If tokenizer vocab and model vocab differ, resize embeddings.
        if vocab_size is not None:
            current_vocab_size = self.model.get_input_embeddings().weight.shape[0]
            if current_vocab_size != vocab_size:
                self.model.resize_token_embeddings(vocab_size)

                # Newly resized embeddings should also be FP32.
                self.model = self.model.float()

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        outputs = self.model(input_ids=idx, use_cache=False)
        logits = outputs.logits

        loss = None
        if targets is not None:
            # Compute loss manually because the project already shifts labels.
            # Use logits.float() for numerical stability under AMP.
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )

        return {
            "logits": logits,
            "loss": loss,
        }

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ):
        do_sample = temperature > 0

        generation_kwargs = {
            "input_ids": idx,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature if do_sample else None,
            "pad_token_id": self.model.config.eos_token_id,
        }

        if top_k is not None:
            generation_kwargs["top_k"] = top_k

        generation_kwargs = {
            k: v for k, v in generation_kwargs.items() if v is not None
        }

        return self.model.generate(**generation_kwargs)