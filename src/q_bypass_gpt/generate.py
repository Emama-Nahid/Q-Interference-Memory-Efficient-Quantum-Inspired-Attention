from __future__ import annotations

import argparse

import torch

from q_bypass_gpt.utils.checkpoint import load_checkpoint
from q_bypass_gpt.utils.config import load_config
from q_bypass_gpt.utils.factory import build_datamodule, build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datamodule = build_datamodule(config)
    datamodule.setup()
    tokenizer = datamodule.tokenizer

    model = build_model(config, vocab_size=tokenizer.vocab_size).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    prompt_ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"].to(device)

    gen_cfg = config["generation"]
    output_ids = model.generate(
        prompt_ids,
        max_new_tokens=args.max_new_tokens or int(gen_cfg["max_new_tokens"]),
        temperature=args.temperature if args.temperature is not None else float(gen_cfg["temperature"]),
        top_k=args.top_k if args.top_k is not None else int(gen_cfg["top_k"]),
    )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("\n=== GENERATED TEXT ===\n")
    print(text)


if __name__ == "__main__":
    main()
