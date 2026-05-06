from __future__ import annotations

import torch
from transformers import AutoTokenizer

from q_bypass_gpt.models.hf_causal_lm import HFCausalLMWrapper


def main():
    model_name = "facebook/opt-125m"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    print(f"Loading model: {model_name}")
    model = HFCausalLMWrapper(
        model_name=model_name,
        vocab_size=tokenizer.vocab_size,
    ).to(device)

    model.eval()

    text = "Quantum-inspired attention may help language models"
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)

    x = input_ids[:, :-1]
    y = input_ids[:, 1:]

    with torch.no_grad():
        out = model(x, y)

    print("Input shape:", tuple(x.shape))
    print("Target shape:", tuple(y.shape))
    print("Logits shape:", tuple(out["logits"].shape))
    print("Loss:", float(out["loss"].item()))

    generated = model.generate(
        input_ids,
        max_new_tokens=30,
        temperature=0.8,
        top_k=50,
    )

    print("\nGenerated text:")
    print(tokenizer.decode(generated[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()