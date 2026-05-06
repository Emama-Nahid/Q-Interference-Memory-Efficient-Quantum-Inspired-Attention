from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch


def naive_score(q_amp: torch.Tensor, k_amp: torch.Tensor, q_phi: torch.Tensor, k_phi: torch.Tensor) -> torch.Tensor:
    # q_amp, k_amp, q_phi, k_phi: [T, dh]
    phase_diff = q_phi.unsqueeze(1) - k_phi.unsqueeze(0)         # [T, T, dh]
    pairwise = q_amp.unsqueeze(1) * k_amp.unsqueeze(0) * torch.cos(phase_diff)
    return pairwise.sum(dim=-1) / math.sqrt(q_amp.size(-1))      # [T, T]


def factored_score(q_amp: torch.Tensor, k_amp: torch.Tensor, q_phi: torch.Tensor, k_phi: torch.Tensor) -> torch.Tensor:
    q_c = q_amp * torch.cos(q_phi)
    q_s = q_amp * torch.sin(q_phi)
    k_c = k_amp * torch.cos(k_phi)
    k_s = k_amp * torch.sin(k_phi)
    return (q_c @ k_c.T + q_s @ k_s.T) / math.sqrt(q_amp.size(-1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_lens", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--head_dim", type=int, default=60)  # matches d_model=720, n_head=12
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out_csv", type=Path, default=Path("outputs/exactness_check.csv"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    print("\nNumerical exactness check (FP32)\n")
    print("| Seq Len | Head Dim | Mean Abs. Error | Max Abs. Error |")
    print("|---:|---:|---:|---:|")

    for T in args.seq_lens:
        mean_errs = []
        max_errs = []

        for _ in range(args.trials):
            q_amp = torch.nn.functional.softplus(torch.randn(T, args.head_dim, dtype=torch.float32))
            k_amp = torch.nn.functional.softplus(torch.randn(T, args.head_dim, dtype=torch.float32))
            q_phi = math.pi * torch.tanh(torch.randn(T, args.head_dim, dtype=torch.float32))
            k_phi = math.pi * torch.tanh(torch.randn(T, args.head_dim, dtype=torch.float32))

            s_naive = naive_score(q_amp, k_amp, q_phi, k_phi)
            s_fact = factored_score(q_amp, k_amp, q_phi, k_phi)

            diff = (s_naive - s_fact).abs()
            mean_errs.append(diff.mean().item())
            max_errs.append(diff.max().item())

        row = {
            "seq_len": T,
            "head_dim": args.head_dim,
            "mean_abs_error": sum(mean_errs) / len(mean_errs),
            "max_abs_error": sum(max_errs) / len(max_errs),
        }
        rows.append(row)

        print(
            f"| {row['seq_len']} | {row['head_dim']} | "
            f"{row['mean_abs_error']:.10e} | {row['max_abs_error']:.10e} |"
        )

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seq_len", "head_dim", "mean_abs_error", "max_abs_error"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {args.out_csv}")


if __name__ == "__main__":
    main()