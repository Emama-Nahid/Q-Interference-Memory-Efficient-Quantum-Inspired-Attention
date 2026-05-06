from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from statistics import mean, pstdev


def parse_best_val_loss(metrics_csv: Path) -> float:
    vals = []
    with open(metrics_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"].strip() == "val":
                vals.append(float(row["loss"]))
    if not vals:
        raise ValueError(f"No val rows found in {metrics_csv}")
    return min(vals)


def parse_test_eval(test_eval_path: Path) -> tuple[float, float]:
    text = test_eval_path.read_text()

    loss_match = re.search(r"Loss\s+([0-9.]+)", text)
    ppl_match = re.search(r"Perplexity\s+([0-9.]+)", text)

    if not loss_match or not ppl_match:
        raise ValueError(f"Could not parse test metrics from {test_eval_path}")

    return float(loss_match.group(1)), float(ppl_match.group(1))


def fmt(mu: float, sigma: float) -> str:
    return f"{mu:.4f} ± {sigma:.4f}"


def summarize(prefix: str, seeds: list[int], output_root: Path) -> dict[str, str]:
    best_vals = []
    test_losses = []
    test_ppls = []

    for seed in seeds:
        exp = f"{prefix}_seed{seed}"
        exp_dir = output_root / exp

        metrics_csv = exp_dir / "logs" / "metrics.csv"
        test_eval = exp_dir / "metrics" / "test_eval.txt"

        best_vals.append(parse_best_val_loss(metrics_csv))
        loss, ppl = parse_test_eval(test_eval)
        test_losses.append(loss)
        test_ppls.append(ppl)

    return {
        "best_val": fmt(mean(best_vals), pstdev(best_vals)),
        "test_loss": fmt(mean(test_losses), pstdev(test_losses)),
        "test_ppl": fmt(mean(test_ppls), pstdev(test_ppls)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 42, 123])
    parser.add_argument("--output_root", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    baseline = summarize("wikitext103_baseline", args.seeds, args.output_root)
    qint = summarize("wikitext103_qinterference", args.seeds, args.output_root)

    print("\nWikiText-103 3-seed summary\n")
    print("| Model | Best Val Loss | Test Loss | Test PPL |")
    print("|---|---:|---:|---:|")
    print(f"| Baseline GPT | {baseline['best_val']} | {baseline['test_loss']} | {baseline['test_ppl']} |")
    print(f"| Q-Interference (ours) | {qint['best_val']} | {qint['test_loss']} | {qint['test_ppl']} |")


if __name__ == "__main__":
    main()