#!/usr/bin/env python3
"""Strict four-rank entrypoint for Stage 2 Video Gate training."""

from __future__ import annotations

import os
import sys
from datetime import timedelta


_EXPECTED_WORLD_SIZE = 4


def _required_integer_environment(name: str) -> int:
    raw = os.environ.get(name)
    if raw is None:
        raise RuntimeError(f"{name} is required for distributed Gate training")
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from error
    return value


def _isolate_rank_cuda_device() -> tuple[int, int, int]:
    """Mask each torchrun child to its assigned GPU before importing torch."""

    if "torch" in sys.modules:
        raise RuntimeError(
            "PyTorch was imported before distributed Gate CUDA isolation"
        )
    world_size = _required_integer_environment("WORLD_SIZE")
    rank = _required_integer_environment("RANK")
    local_rank = _required_integer_environment("LOCAL_RANK")
    if world_size != _EXPECTED_WORLD_SIZE:
        raise RuntimeError(
            "distributed Gate training requires exactly four ranks; "
            f"got {world_size}"
        )
    if rank != local_rank or not 0 <= rank < world_size:
        raise RuntimeError(
            "distributed Gate training requires one node with RANK == "
            "LOCAL_RANK in [0, 3]"
        )
    local_world_size = os.environ.get("LOCAL_WORLD_SIZE")
    if local_world_size is not None:
        try:
            parsed_local_world_size = int(local_world_size)
        except ValueError as error:
            raise ValueError(
                "LOCAL_WORLD_SIZE must be an integer"
            ) from error
        if parsed_local_world_size != world_size:
            raise RuntimeError(
                "LOCAL_WORLD_SIZE must equal WORLD_SIZE for one-node Gate DDP"
            )

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must explicitly name four GPUs"
        )
    tokens = tuple(token.strip() for token in visible.split(","))
    if (
        len(tokens) != world_size
        or any(not token or token == "-1" for token in tokens)
        or len(set(tokens)) != world_size
    ):
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must contain exactly four unique GPUs"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = tokens[local_rank]
    return rank, local_rank, world_size


_RANK, _LOCAL_RANK, _WORLD_SIZE = _isolate_rank_cuda_device()

# These imports must remain below _isolate_rank_cuda_device.
import hydra  # noqa: E402
from omegaconf import DictConfig  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from fastwam.gating.distributed import DistributedGateContext  # noqa: E402
from train_video_gate import run_train_video_gate_distributed  # noqa: E402


def _initialize_process_group() -> DistributedGateContext:
    if not dist.is_available():
        raise RuntimeError("torch.distributed is unavailable")
    if dist.is_initialized():
        raise RuntimeError("Gate process group was initialized too early")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "each distributed Gate rank must see exactly one CUDA device"
        )
    torch.cuda.set_device(0)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(minutes=10),
    )
    context = DistributedGateContext(
        rank=dist.get_rank(),
        local_rank=_LOCAL_RANK,
        world_size=dist.get_world_size(),
        backend=str(dist.get_backend()),
    )
    if (
        context.rank != _RANK
        or context.world_size != _WORLD_SIZE
        or context.backend != "nccl"
    ):
        raise RuntimeError("initialized Gate process-group identity is invalid")
    return context


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="train_video_gate",
)
def main(config: DictConfig) -> None:
    context: DistributedGateContext | None = None
    try:
        context = _initialize_process_group()
        summary = run_train_video_gate_distributed(
            config,
            distributed_context=context,
        )
        if context.is_main:
            print(
                "Distributed Stage 2 Video Gate training complete:\n"
                f"  world_size: {context.world_size}\n"
                f"  epoch: {summary['final_epoch']}\n"
                f"  global_step: {summary['global_step']}\n"
                f"  best_val_bce: {summary['best_val_bce']}"
            )
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
