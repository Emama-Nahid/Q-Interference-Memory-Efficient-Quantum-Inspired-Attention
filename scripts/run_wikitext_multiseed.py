from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def run_and_log(cmd: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[RUN] {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        check=False,
    )
    print(proc.stdout)
    log_path.write_text(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with code {proc.returncode}: {' '.join(cmd)}")


def make_seed_config(template_cfg: dict, seed: int, exp_name: str) -> dict:
    cfg = yaml.safe_load(yaml.safe_dump(template_cfg))  # deep copy
    cfg["experiment"]["name"] = exp_name
    cfg["train"]["seed"] = int(seed)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_config", type=Path, default=Path("configs/main_baseline.yaml"))
    parser.add_argument("--qint_config", type=Path, default=Path("configs/main_qhybrid.yaml"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 42, 123])
    parser.add_argument("--config_outdir", type=Path, default=Path("configs/multiseed"))
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    baseline_template = load_yaml(args.baseline_config)
    qint_template = load_yaml(args.qint_config)

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    jobs = [
        ("baseline", baseline_template, "wikitext103_baseline"),
        ("qinterference", qint_template, "wikitext103_qinterference"),
    ]

    for tag, template, prefix in jobs:
        for seed in args.seeds:
            exp_name = f"{prefix}_seed{seed}"
            cfg = make_seed_config(template, seed, exp_name)
            cfg_path = args.config_outdir / f"{exp_name}.yaml"
            save_yaml(cfg, cfg_path)

            out_dir = Path(cfg["experiment"]["output_dir"]) / exp_name
            ckpt_path = out_dir / "checkpoints" / "best.pt"
            train_log = Path("outputs/manual_logs") / f"{exp_name}_train_stdout.txt"
            eval_log = out_dir / "metrics" / "test_eval.txt"

            if args.skip_existing and eval_log.exists():
                print(f"[SKIP] {exp_name} already evaluated.")
                continue

            train_cmd = [sys.executable, "-m", "q_bypass_gpt.train", "--config", str(cfg_path)]
            run_and_log(train_cmd, train_log, env)

            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

            eval_cmd = [
                sys.executable,
                "-m",
                "q_bypass_gpt.evaluate",
                "--config",
                str(cfg_path),
                "--checkpoint",
                str(ckpt_path),
                "--split",
                "test",
            ]
            run_and_log(eval_cmd, eval_log, env)

    print("\nDone.")


if __name__ == "__main__":
    main()