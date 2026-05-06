from __future__ import annotations

import argparse
import time
import torch
import yaml

from q_bypass_gpt.utils.factory import build_model


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def benchmark_once(model, x, y=None, train_mode=True):
    device = x.device
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    if train_mode:
        model.train()
        start = time.perf_counter()
        out = model(x, y)
        loss = out["loss"] if isinstance(out, dict) else out[1]
        loss.backward()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
    else:
        model.eval()
        with torch.no_grad():
            start = time.perf_counter()
            _ = model(x, y)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start

    peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    return elapsed, peak_alloc, peak_reserved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cfg = load_config(args.config)
    vocab_size = 50257
    model = build_model(cfg, vocab_size=vocab_size).to(args.device)

    total_params = sum(p.numel() for p in model.parameters())

    print(f"Attention: {cfg['model']['attention_type']}")
    print(f"Parameters: {total_params:,}")
    print()

    for seq_len in args.seq_lens:
        x = torch.randint(0, vocab_size, (args.batch_size, seq_len), device=args.device)
        y = torch.randint(0, vocab_size, (args.batch_size, seq_len), device=args.device)

        # warmup
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            try:
                _ = benchmark_once(model, x, y, train_mode=True)
            except RuntimeError:
                torch.cuda.empty_cache()

        model.zero_grad(set_to_none=True)
        train_elapsed, train_alloc, train_reserved = benchmark_once(model, x, y, train_mode=True)
        model.zero_grad(set_to_none=True)
        eval_elapsed, eval_alloc, eval_reserved = benchmark_once(model, x, y, train_mode=False)

        tokens = args.batch_size * seq_len
        train_tps = tokens / train_elapsed
        eval_tps = tokens / eval_elapsed

        print(
            f"seq={seq_len:4d} | "
            f"train_alloc={train_alloc:8.2f} MB | "
            f"train_reserved={train_reserved:8.2f} MB | "
            f"eval_alloc={eval_alloc:8.2f} MB | "
            f"eval_reserved={eval_reserved:8.2f} MB | "
            f"train_toks/s={train_tps:8.2f} | "
            f"eval_toks/s={eval_tps:8.2f}"
        )


if __name__ == "__main__":
    main()