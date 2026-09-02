from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from scripts import train_video_gate as train_cli
from fastwam.gating.distributed import DistributedGateContext


REPO_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = REPO_ROOT / "scripts" / "train_video_gate_distributed.py"


def _contract(context=None):
    return train_cli.build_training_config_contract(
        data={"train": {"dataset_dirs": ["/data"]}, "val": None},
        gate={"proprio_dim": 7},
        training={"batch_size": 64, "seed": 42},
        runtime={
            "device": "cuda:0",
            "require_cuda": True,
            "deterministic_algorithms": True,
        },
        numerical_runtime={"device": {"name": "NVIDIA H100 80GB HBM3"}},
        distributed_context=context,
    )


def test_distributed_contract_binds_exact_topology_and_reduction_semantics():
    context = DistributedGateContext(
        rank=0,
        local_rank=0,
        world_size=4,
        backend="nccl",
    )

    contract = _contract(context)

    assert contract["schema_version"] == 2
    assert contract["training"]["batch_size"] == 64
    assert contract["dataloader_seed_algorithm"] == (
        "distributed_sampler_base_seed_plus_zero_based_epoch_v1"
    )
    assert contract["distributed"] == {
        "backend": "nccl",
        "world_size": 4,
        "global_batch_size": 64,
        "per_rank_batch_size": 16,
        "sampler_algorithm": (
            "torch_distributed_sampler_exact_divisible_v1"
        ),
        "objective_reduction_algorithm": (
            "global_weighted_mean_all_reduce_v1"
        ),
        "metric_reduction_algorithm": (
            "variable_length_all_gather_rank_order_v1"
        ),
        "device_visibility_algorithm": (
            "one_visible_cuda_device_per_rank_v1"
        ),
        "worker_seed_algorithm": (
            "base_plus_epoch_plus_rank_times_10000000_v1"
        ),
        "checkpoint_writer_rank": 0,
        "rng_state_algorithm": "distributed_per_rank_rng_v1",
    }
    serial = _contract()
    assert serial["schema_version"] == 1
    assert "distributed" not in serial
    assert serial["dataloader_seed_algorithm"] == (
        "base_seed_plus_zero_based_epoch_v1"
    )


def _entrypoint_probe(**updates: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(
        {
            "WORLD_SIZE": "4",
            "LOCAL_WORLD_SIZE": "4",
            "RANK": "2",
            "LOCAL_RANK": "2",
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            **updates,
        }
    )
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")]
    )
    probe = (
        "import json, os, runpy; "
        f"namespace=runpy.run_path({str(ENTRYPOINT)!r}, run_name='probe'); "
        "print(json.dumps({"
        "'rank': namespace['_RANK'], "
        "'local_rank': namespace['_LOCAL_RANK'], "
        "'world_size': namespace['_WORLD_SIZE'], "
        "'visible': os.environ['CUDA_VISIBLE_DEVICES']"
        "}, sort_keys=True))"
    )
    return subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_entrypoint_masks_to_local_gpu_before_importing_torch():
    result = _entrypoint_probe()

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.splitlines()[-1])
    assert payload == {
        "local_rank": 2,
        "rank": 2,
        "visible": "2",
        "world_size": 4,
    }


def test_entrypoint_rejects_non_four_rank_launch_before_torch_startup():
    result = _entrypoint_probe(WORLD_SIZE="2", LOCAL_WORLD_SIZE="2")

    assert result.returncode != 0
    assert "requires exactly four ranks" in result.stderr
