from __future__ import annotations

import os

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def setup_distributed() -> tuple[bool, int, int, int]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0, 1, 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return True, rank, world_size, local_rank


def cleanup_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()


def reduce_mean(value: torch.Tensor) -> torch.Tensor:
    if not is_distributed():
        return value
    value = value.clone()
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    value /= get_world_size()
    return value


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model
