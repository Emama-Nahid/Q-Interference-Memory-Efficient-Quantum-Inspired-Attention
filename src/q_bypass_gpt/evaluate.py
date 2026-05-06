from __future__ import annotations

import argparse
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table

from q_bypass_gpt.train import evaluate
from q_bypass_gpt.utils.checkpoint import load_checkpoint
from q_bypass_gpt.utils.config import load_config
from q_bypass_gpt.utils.factory import build_datamodule, build_model
from q_bypass_gpt.utils.metrics import count_parameters, perplexity_from_loss

console = Console()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, choices=["val", "test"], default="test")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datamodule = build_datamodule(config)
    datamodule.setup()
    model = build_model(config, vocab_size=datamodule.tokenizer.vocab_size).to(device)

    load_checkpoint(args.checkpoint, model, map_location=device)

    loader = datamodule.get_dataloader(args.split, distributed=False)
    loss, elapsed = evaluate(model, loader, device, max_batches=config["eval"]["max_eval_batches"])

    table = Table(title=f"Evaluation: {Path(args.checkpoint).name}")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Split", args.split)
    table.add_row("Loss", f"{loss:.4f}")
    table.add_row("Perplexity", f"{perplexity_from_loss(loss):.4f}")
    table.add_row("Eval time (sec)", f"{elapsed:.2f}")
    table.add_row("Parameters", f"{count_parameters(model):,}")
    console.print(table)


if __name__ == "__main__":
    main()
