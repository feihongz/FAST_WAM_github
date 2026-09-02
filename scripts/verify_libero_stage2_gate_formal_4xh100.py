#!/usr/bin/env python3
"""Strict verifier for the topology-bound 4-H100 LIBERO Gate run."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import torch

try:
    from scripts.verify_libero_stage2_gate_formal import (
        _validate_training_contract as _validate_base_training_contract,
        _verify_with_profile,
    )
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from verify_libero_stage2_gate_formal import (  # type: ignore[no-redef]
        _validate_training_contract as _validate_base_training_contract,
        _verify_with_profile,
    )


EXPECTED_WORLD_SIZE = 4
EXPECTED_DISTRIBUTED_CONFIG = {
    "backend": "nccl",
    "world_size": EXPECTED_WORLD_SIZE,
    "global_batch_size": 64,
    "per_rank_batch_size": 16,
    "sampler_algorithm": "torch_distributed_sampler_exact_divisible_v1",
    "objective_reduction_algorithm": "global_weighted_mean_all_reduce_v1",
    "metric_reduction_algorithm": "variable_length_all_gather_rank_order_v1",
    "device_visibility_algorithm": "one_visible_cuda_device_per_rank_v1",
    "checkpoint_writer_rank": 0,
    "rng_state_algorithm": "distributed_per_rank_rng_v1",
    "worker_seed_algorithm": (
        "base_plus_epoch_plus_rank_times_10000000_v1"
    ),
}
EXPECTED_DATALOADER_SEED_ALGORITHM = (
    "distributed_sampler_base_seed_plus_zero_based_epoch_v1"
)


def _validate_training_contract(
    run_identity: Mapping[str, Any],
    training_identity: Mapping[str, Any],
) -> dict[str, Any]:
    contract = _validate_base_training_contract(
        run_identity,
        training_identity,
        expected_schema_version=2,
        expected_dataloader_seed_algorithm=(
            EXPECTED_DATALOADER_SEED_ALGORITHM
        ),
        expected_distributed=EXPECTED_DISTRIBUTED_CONFIG,
    )
    if type(contract["schema_version"]) is not int:  # noqa: E721
        raise ValueError("Gate distributed contract schema_version must be int")
    distributed = contract["distributed"]
    for field, expected in EXPECTED_DISTRIBUTED_CONFIG.items():
        actual = distributed[field]
        if type(actual) is not type(expected):  # noqa: E721
            raise ValueError(
                f"Gate formal distributed config {field} type mismatch"
            )
    return contract


def verify(
    *,
    output_dir: Path,
    expected_git_commit: str,
    receipt: Path,
    resume_device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Verify one exact 1x4 NCCL artifact from a one-H100 parent process."""

    return _verify_with_profile(
        output_dir=output_dir,
        expected_git_commit=expected_git_commit,
        receipt=receipt,
        resume_device=resume_device,
        training_contract_validator=_validate_training_contract,
        expected_distributed_rng_world_size=EXPECTED_WORLD_SIZE,
        allow_resumed_summary=True,
        verification_kind=(
            "libero_stage2_gate_formal_4xh100_verification"
        ),
        result_metadata={
            "topology": "1x4",
            "training_contract_schema_version": 2,
            "distributed": dict(EXPECTED_DISTRIBUTED_CONFIG),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = verify(
        output_dir=args.output_dir,
        expected_git_commit=args.expected_git_commit,
        receipt=args.receipt,
    )
    print(
        "[verify] LIBERO Stage 2 Gate 4xH100 formal run passed: "
        f"topology={result['topology']} epoch={result['final_epoch']} "
        f"step={result['global_step']} best_epoch={result['best_epoch']} "
        f"stopped_early={result['stopped_early']}"
    )
    print(f"[verify] receipt={args.receipt.expanduser().resolve()}")


if __name__ == "__main__":
    main()
