#!/usr/bin/env python3
"""Verify an uninterrupted Stage 3 step against a strict-resume replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

from fastwam.alignment.checkpointing import (
    STRICT_RESUME_PROVENANCE_KIND,
    STRICT_RESUME_PROVENANCE_SCHEMA_VERSION,
    TRAINING_STATE_SCHEMA_VERSION,
    sha256_file,
    validate_training_state,
)
if __package__:
    from .compare_stage3_adapter_exports import (
        compare_stage3_adapter_exports,
    )
else:
    from compare_stage3_adapter_exports import compare_stage3_adapter_exports


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_STATE_CONTRACT_KEYS = (
    "base_checkpoint",
    "base_checkpoint_sha256",
    "base_checkpoint_size_bytes",
    "alignment_config",
    "training_contract",
    "training_contract_sha256",
    "git_commit",
    "world_size",
    "distributed_type",
    "zero_stage",
    "mixed_precision",
    "device_type",
    "gradient_accumulation_steps",
    "batch_size_per_rank",
    "dataset_length",
    "micro_batches_per_epoch",
    "drop_last",
    "deepspeed_config_sha256",
    "asset_sha256",
    "data_manifest_sha256",
    "git_tracked_dirty",
    "git_untracked_source_files",
    "versions",
    "dataloader_contract",
)


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{field} must contain exactly 64 lowercase hex chars")
    return value


def _require_positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _require_nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _resolve_state_dir(path: str | Path, *, field: str) -> Path:
    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"{field} is not a directory: {resolved}")
    return resolved


def _require_distinct_directories(paths: dict[str, Path]) -> None:
    resolved_values = list(paths.values())
    if len(set(resolved_values)) != len(resolved_values):
        raise ValueError("Stage 3 resume state directories must be distinct")
    identities = [(path.stat().st_dev, path.stat().st_ino) for path in resolved_values]
    if len(set(identities)) != len(identities):
        raise ValueError("Stage 3 resume state directories share an inode")


def _state_identity(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    rng_files = {
        name: identity
        for name, identity in sorted(manifest["files"].items())
        if name.startswith("accelerator/random_states_")
        and name.endswith(".pkl")
    }
    return {
        "path": str(path),
        "manifest_sha256": sha256_file(path / "manifest.json"),
        "complete_sha256": sha256_file(path / "COMPLETE"),
        "global_step": manifest["global_step"],
        "epoch": manifest["epoch"],
        "batch_in_epoch": manifest["batch_in_epoch"],
        "scheduler_last_epoch": manifest["scheduler_last_epoch"],
        "world_size": manifest["world_size"],
        "distributed_type": manifest["distributed_type"],
        "zero_stage": manifest["zero_stage"],
        "batch_size_per_rank": manifest["batch_size_per_rank"],
        "gradient_accumulation_steps": manifest[
            "gradient_accumulation_steps"
        ],
        "rng_files": rng_files,
        "adapter_export_sha256": manifest["files"]["adapter_export.pt"][
            "sha256"
        ],
    }


def verify_stage3_resume_equivalence(
    uninterrupted_state_dir: str | Path,
    resume_source_state_dir: str | Path,
    resumed_state_dir: str | Path,
    uninterrupted_export: str | Path,
    resumed_export: str | Path,
    *,
    expected_final_step: int,
    expected_resume_step: int,
    expected_world_size: int,
    expected_zero_stage: int,
    expected_batch_size_per_rank: int,
    expected_gradient_accumulation_steps: int,
    expected_base_checkpoint_sha256: str,
    expected_data_manifest_sha256: str,
    expected_training_contract_sha256: str,
    expected_git_commit: str,
) -> dict[str, Any]:
    """Validate topology, lineage, state integrity, and exact Adapter equality."""

    expected_final_step = _require_nonnegative_int(
        expected_final_step,
        field="expected_final_step",
    )
    expected_resume_step = _require_nonnegative_int(
        expected_resume_step,
        field="expected_resume_step",
    )
    if expected_resume_step >= expected_final_step:
        raise ValueError("expected_resume_step must precede expected_final_step")
    expected_world_size = _require_positive_int(
        expected_world_size,
        field="expected_world_size",
    )
    expected_batch_size_per_rank = _require_positive_int(
        expected_batch_size_per_rank,
        field="expected_batch_size_per_rank",
    )
    expected_gradient_accumulation_steps = _require_positive_int(
        expected_gradient_accumulation_steps,
        field="expected_gradient_accumulation_steps",
    )
    if (
        isinstance(expected_zero_stage, bool)
        or not isinstance(expected_zero_stage, int)
        or expected_zero_stage not in {0, 1, 2}
    ):
        raise ValueError("expected_zero_stage must be 0, 1, or 2")
    expected_base_checkpoint_sha256 = _require_sha256(
        expected_base_checkpoint_sha256,
        field="expected_base_checkpoint_sha256",
    )
    expected_data_manifest_sha256 = _require_sha256(
        expected_data_manifest_sha256,
        field="expected_data_manifest_sha256",
    )
    expected_training_contract_sha256 = _require_sha256(
        expected_training_contract_sha256,
        field="expected_training_contract_sha256",
    )
    if (
        not isinstance(expected_git_commit, str)
        or re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", expected_git_commit)
        is None
    ):
        raise ValueError(
            "expected_git_commit must contain 40 or 64 lowercase hex characters"
        )

    state_paths = {
        "uninterrupted": _resolve_state_dir(
            uninterrupted_state_dir,
            field="uninterrupted_state_dir",
        ),
        "resume_source": _resolve_state_dir(
            resume_source_state_dir,
            field="resume_source_state_dir",
        ),
        "resumed": _resolve_state_dir(
            resumed_state_dir,
            field="resumed_state_dir",
        ),
    }
    _require_distinct_directories(state_paths)

    expected_contract = {
        "base_checkpoint_sha256": expected_base_checkpoint_sha256,
        "data_manifest_sha256": expected_data_manifest_sha256,
        "training_contract_sha256": expected_training_contract_sha256,
        "world_size": expected_world_size,
        "zero_stage": expected_zero_stage,
        "batch_size_per_rank": expected_batch_size_per_rank,
        "gradient_accumulation_steps": expected_gradient_accumulation_steps,
        "git_commit": expected_git_commit,
    }
    manifests = {
        name: validate_training_state(path, expected_contract=expected_contract)
        for name, path in state_paths.items()
    }
    for name, manifest in manifests.items():
        if manifest["schema_version"] != TRAINING_STATE_SCHEMA_VERSION:
            raise ValueError(f"formal {name} state must use training-state schema v2")
        expected_distributed_type = (
            "DistributedType.DEEPSPEED"
            if expected_zero_stage > 0
            else "DistributedType.NO"
        )
        if manifest.get("distributed_type") != expected_distributed_type:
            raise ValueError(
                f"formal {name} state distributed_type is not "
                f"{expected_distributed_type}"
            )
        if manifest.get("git_tracked_dirty") is not False:
            raise ValueError(f"formal {name} state has tracked Git changes")
        if manifest.get("git_untracked_source_files") != []:
            raise ValueError(f"formal {name} state has untracked source files")

    source = manifests["resume_source"]
    uninterrupted = manifests["uninterrupted"]
    resumed = manifests["resumed"]
    if source["global_step"] != expected_resume_step:
        raise ValueError("resume source state has the wrong global_step")
    for name, manifest in (
        ("uninterrupted", uninterrupted),
        ("resumed", resumed),
    ):
        if manifest["global_step"] != expected_final_step:
            raise ValueError(f"{name} state has the wrong global_step")
    cursor_keys = (
        "global_step",
        "epoch",
        "batch_in_epoch",
        "micro_step_in_accumulation",
        "scheduler_last_epoch",
    )
    if any(uninterrupted[key] != resumed[key] for key in cursor_keys):
        raise ValueError("uninterrupted and resumed state cursors differ")
    for key in _STATE_CONTRACT_KEYS:
        reference = source.get(key)
        if uninterrupted.get(key) != reference or resumed.get(key) != reference:
            raise ValueError(f"Stage 3 state contract differs for {key}")
    if source.get("strict_resume_provenance") is not None:
        raise ValueError("resume source must come from the fresh baseline run")
    if uninterrupted.get("strict_resume_provenance") is not None:
        raise ValueError("uninterrupted final state must have no resume provenance")

    state_identities = {
        name: _state_identity(state_paths[name], manifest)
        for name, manifest in manifests.items()
    }
    expected_provenance = {
        "schema_version": STRICT_RESUME_PROVENANCE_SCHEMA_VERSION,
        "kind": STRICT_RESUME_PROVENANCE_KIND,
        "source_manifest_sha256": state_identities["resume_source"][
            "manifest_sha256"
        ],
        "source_complete_sha256": state_identities["resume_source"][
            "complete_sha256"
        ],
        "source_global_step": source["global_step"],
        "source_epoch": source["epoch"],
        "source_batch_in_epoch": source["batch_in_epoch"],
        "source_scheduler_last_epoch": source["scheduler_last_epoch"],
        "source_training_contract_sha256": source[
            "training_contract_sha256"
        ],
        "source_world_size": source["world_size"],
        "source_zero_stage": source["zero_stage"],
    }
    if resumed.get("strict_resume_provenance") != expected_provenance:
        raise ValueError("resumed state provenance does not match the source state")

    comparison_kwargs = {
        "expected_step": expected_final_step,
        "expected_base_checkpoint_sha256": expected_base_checkpoint_sha256,
        "expected_data_manifest_sha256": expected_data_manifest_sha256,
        "expected_training_contract_sha256": expected_training_contract_sha256,
        "expected_git_commit": expected_git_commit,
    }
    external_equivalence = compare_stage3_adapter_exports(
        uninterrupted_export,
        resumed_export,
        **comparison_kwargs,
    )
    uninterrupted_binding = compare_stage3_adapter_exports(
        uninterrupted_export,
        state_paths["uninterrupted"] / "adapter_export.pt",
        **comparison_kwargs,
    )
    resumed_binding = compare_stage3_adapter_exports(
        resumed_export,
        state_paths["resumed"] / "adapter_export.pt",
        **comparison_kwargs,
    )
    for name, binding in (
        ("uninterrupted", uninterrupted_binding),
        ("resumed", resumed_binding),
    ):
        expected_export_sha256 = state_identities[name][
            "adapter_export_sha256"
        ]
        if (
            binding["left"]["sha256"] != expected_export_sha256
            or binding["right"]["sha256"] != expected_export_sha256
        ):
            raise ValueError(
                f"{name} external export SHA256 does not match state inventory"
            )

    return {
        "status": "ok",
        "expected_final_step": expected_final_step,
        "expected_resume_step": expected_resume_step,
        "training_contract_sha256": expected_training_contract_sha256,
        "data_manifest_sha256": expected_data_manifest_sha256,
        "states": state_identities,
        "verified_resume_provenance": expected_provenance,
        "adapter_equivalence": external_equivalence,
        "external_state_bindings": {
            "uninterrupted": uninterrupted_binding,
            "resumed": resumed_binding,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify Stage 3 uninterrupted step N against a replay from step N-1."
        )
    )
    parser.add_argument("uninterrupted_state_dir", type=Path)
    parser.add_argument("resume_source_state_dir", type=Path)
    parser.add_argument("resumed_state_dir", type=Path)
    parser.add_argument("uninterrupted_export", type=Path)
    parser.add_argument("resumed_export", type=Path)
    parser.add_argument("--expected-final-step", type=int, required=True)
    parser.add_argument("--expected-resume-step", type=int, required=True)
    parser.add_argument("--expected-world-size", type=int, required=True)
    parser.add_argument("--expected-zero-stage", type=int, required=True)
    parser.add_argument("--expected-batch-size-per-rank", type=int, required=True)
    parser.add_argument(
        "--expected-gradient-accumulation-steps",
        type=int,
        required=True,
    )
    parser.add_argument("--expected-base-checkpoint-sha256", required=True)
    parser.add_argument("--expected-data-manifest-sha256", required=True)
    parser.add_argument("--expected-training-contract-sha256", required=True)
    parser.add_argument("--expected-git-commit", required=True)
    args = parser.parse_args()
    receipt = verify_stage3_resume_equivalence(
        args.uninterrupted_state_dir,
        args.resume_source_state_dir,
        args.resumed_state_dir,
        args.uninterrupted_export,
        args.resumed_export,
        expected_final_step=args.expected_final_step,
        expected_resume_step=args.expected_resume_step,
        expected_world_size=args.expected_world_size,
        expected_zero_stage=args.expected_zero_stage,
        expected_batch_size_per_rank=args.expected_batch_size_per_rank,
        expected_gradient_accumulation_steps=(
            args.expected_gradient_accumulation_steps
        ),
        expected_base_checkpoint_sha256=args.expected_base_checkpoint_sha256,
        expected_data_manifest_sha256=args.expected_data_manifest_sha256,
        expected_training_contract_sha256=(
            args.expected_training_contract_sha256
        ),
        expected_git_commit=args.expected_git_commit,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
