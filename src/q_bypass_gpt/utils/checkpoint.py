from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    model,
    optimizer,
    scheduler,
    scaler,
    step: int,
    epoch: int,
    best_val_loss: float,
    config: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "config": config,
    }
    torch.save(state, path)


def load_checkpoint(path: str | Path, model, optimizer=None, scheduler=None, scaler=None, map_location="cpu"):
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state["model"])
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    return state
