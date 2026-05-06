from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from q_bypass_gpt.utils.checkpoint import save_checkpoint
from q_bypass_gpt.utils.config import ensure_dir, load_config, save_json
from q_bypass_gpt.utils.distributed import (
    cleanup_distributed,
    is_main_process,
    reduce_mean,
    setup_distributed,
    unwrap_model,
)
from q_bypass_gpt.utils.factory import build_datamodule, build_model, build_optimizer, build_scheduler
from q_bypass_gpt.utils.logger import ExperimentLogger
from q_bypass_gpt.utils.metrics import count_parameters, perplexity_from_loss
from q_bypass_gpt.utils.seed import set_seed

console = Console()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--local-rank", "--local_rank", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, dataloader, device, max_batches: int | None = None):
    model.eval()
    losses = []
    start = time.time()
    for batch_idx, (x, y) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x, y)
        loss = out["loss"]
        if torch.isnan(loss):
            raise RuntimeError("NaN loss detected during evaluation.")
        losses.append(loss.detach())
    elapsed = time.time() - start
    mean_loss = torch.stack(losses).mean()
    mean_loss = reduce_mean(mean_loss)
    model.train()
    return float(mean_loss.item()), elapsed


def main():
    args = parse_args()
    config = load_config(args.config)

    distributed, rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    set_seed(int(config["train"]["seed"]) + rank)

    exp_name = config["experiment"]["name"]
    out_root = ensure_dir(config["experiment"]["output_dir"])
    exp_dir = ensure_dir(Path(out_root) / exp_name)
    ckpt_dir = ensure_dir(exp_dir / "checkpoints")
    log_dir = ensure_dir(exp_dir / "logs")
    ensure_dir(exp_dir / "plots")
    ensure_dir(exp_dir / "metrics")

    if is_main_process():
        save_json(exp_dir / "resolved_config.json", config)

    logger = ExperimentLogger(log_dir) if is_main_process() else None

    datamodule = build_datamodule(config)
    datamodule.setup()
    vocab_size = datamodule.tokenizer.vocab_size

    model = build_model(config, vocab_size=vocab_size).to(device)

    if bool(config["train"]["compile"]) and hasattr(torch, "compile"):
        model = torch.compile(model)

    if distributed:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config["train"]["mixed_precision"]) and device.type == "cuda")

    train_loader = datamodule.get_dataloader("train", distributed=distributed)
    val_loader = datamodule.get_dataloader("val", distributed=distributed)

    if is_main_process():
        table = Table(title="Q-BypassGPT Training Setup")
        table.add_column("Field")
        table.add_column("Value")
        table.add_row("Experiment", exp_name)
        table.add_row("Device", str(device))
        table.add_row("Distributed", str(distributed))
        table.add_row("World size", str(world_size))
        table.add_row("Parameters", f"{count_parameters(unwrap_model(model)):,}")
        table.add_row("Attention", config["model"]["attention_type"])
        table.add_row("Dataset", f'{config["data"]["dataset_name"]}/{config["data"]["dataset_config"]}')
        console.print(table)

    max_steps = int(config["train"]["max_steps"])
    grad_accum_steps = int(config["train"]["grad_accum_steps"])
    eval_interval = int(config["train"]["eval_interval"])
    log_interval = int(config["train"]["log_interval"])
    save_interval = int(config["train"]["save_interval"])
    max_eval_batches = int(config["eval"]["max_eval_batches"])
    grad_clip = float(config["train"]["grad_clip"])

    global_step = 0
    best_val_loss = float("inf")
    epoch = 0

    if bool(config["train"]["detect_anomaly"]):
        torch.autograd.set_detect_anomaly(True)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    start_time = time.time()

    while global_step < max_steps:
        if distributed and hasattr(train_loader, "sampler") and train_loader.sampler is not None:
            train_loader.sampler.set_epoch(epoch)

        progress = tqdm(train_loader, disable=not is_main_process(), desc=f"Epoch {epoch}")
        for step_idx, (x, y) in enumerate(progress):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=bool(config["train"]["mixed_precision"]) and device.type == "cuda"):
                out = model(x, y)
                loss = out["loss"] / grad_accum_steps

            if torch.isnan(loss):
                raise RuntimeError("NaN loss detected during training.")

            scaler.scale(loss).backward()

            if (step_idx + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                loss_to_log = reduce_mean(loss.detach() * grad_accum_steps).item()

                if is_main_process() and global_step % log_interval == 0:
                    peak_mem_mb = (
                        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                        if device.type == "cuda"
                        else 0.0
                    )
                    lr = scheduler.get_last_lr()[0]
                    tokens_seen = global_step * x.size(0) * x.size(1) * world_size
                    elapsed = max(time.time() - start_time, 1e-6)
                    tokens_per_sec = tokens_seen / elapsed

                    progress.set_postfix(
                        step=global_step,
                        loss=f"{loss_to_log:.4f}",
                        lr=f"{lr:.2e}",
                        mem_mb=f"{peak_mem_mb:.1f}",
                    )
                    logger.log(
                        {
                            "split": "train",
                            "step": global_step,
                            "epoch": epoch,
                            "loss": loss_to_log,
                            "perplexity": perplexity_from_loss(loss_to_log),
                            "lr": lr,
                            "peak_mem_mb": peak_mem_mb,
                            "tokens_per_sec": tokens_per_sec,
                        }
                    )

                if global_step % eval_interval == 0:
                    val_loss, eval_time = evaluate(model, val_loader, device, max_batches=max_eval_batches)
                    if is_main_process():
                        logger.log(
                            {
                                "split": "val",
                                "step": global_step,
                                "epoch": epoch,
                                "loss": val_loss,
                                "perplexity": perplexity_from_loss(val_loss),
                                "eval_time_sec": eval_time,
                            }
                        )

                        save_checkpoint(
                            ckpt_dir / "last.pt",
                            unwrap_model(model),
                            optimizer,
                            scheduler,
                            scaler,
                            global_step,
                            epoch,
                            best_val_loss,
                            config,
                        )
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            save_checkpoint(
                                ckpt_dir / "best.pt",
                                unwrap_model(model),
                                optimizer,
                                scheduler,
                                scaler,
                                global_step,
                                epoch,
                                best_val_loss,
                                config,
                            )

                if is_main_process() and global_step % save_interval == 0:
                    save_checkpoint(
                        ckpt_dir / f"step_{global_step}.pt",
                        unwrap_model(model),
                        optimizer,
                        scheduler,
                        scaler,
                        global_step,
                        epoch,
                        best_val_loss,
                        config,
                    )

                if global_step >= max_steps:
                    break

        epoch += 1

    if is_main_process():
        save_checkpoint(
            ckpt_dir / "final.pt",
            unwrap_model(model),
            optimizer,
            scheduler,
            scaler,
            global_step,
            epoch,
            best_val_loss,
            config,
        )
        final_peak_mem_mb = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            if device.type == "cuda"
            else 0.0
        )

        console.print(
            f"[green]Final peak GPU memory during training: "
            f"{final_peak_mem_mb:.2f} MB[/green]"
        )

        if logger is not None:
            logger.log(
                {
                    "split": "summary",
                    "step": global_step,
                    "epoch": epoch,
                    "peak_mem_mb": final_peak_mem_mb,
                }
            )
        console.print(f"[green]Training complete.[/green] Best val loss: {best_val_loss:.4f}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
