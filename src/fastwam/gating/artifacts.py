"""Versioned, fail-closed artifacts for offline Stage 2 Gate labels.

The label job is intentionally a pure-CPU persistence layer.  Model inference
produces row values elsewhere; this module binds those values to the exact data,
checkpoints, split, seed rule, and resolved configuration that produced them.
Chunks are immutable JSON objects rather than append-only JSONL files, so a
resume can distinguish a complete chunk from an interrupted write.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from types import MappingProxyType
from typing import Any

from fastwam.alignment.checkpointing import (
    canonical_json_sha256,
    sha256_file,
    write_json_atomic,
    write_text_atomic,
)
from fastwam.gating.contracts import (
    EpisodeSplitLookup,
    build_episode_split_lookup,
    dataset_id_from_lookup,
    derive_pair_seeds,
    require_sha256,
    sample_id_from_lookup,
    split_for_identity,
    validate_episode_split,
    validate_sample_identity_with_lookup,
)
from fastwam.gating.inference import STAGE2_NUM_INFERENCE_STEPS


LABEL_CONTRACT_SCHEMA_VERSION = 2
LABEL_CONTRACT_KIND = "stage2_gate_label_contract"
LABEL_ROW_SCHEMA_VERSION = 1
LABEL_ROW_KIND = "stage2_gate_label_row"
LABEL_CHUNK_SCHEMA_VERSION = 1
LABEL_CHUNK_KIND = "stage2_gate_label_chunk"
LABEL_MANIFEST_SCHEMA_VERSION = 1
LABEL_MANIFEST_KIND = "stage2_gate_label_manifest"
SHARD_ALGORITHM = "sample_sha256_prefix_mod_v1"
SEED_ALGORITHM = "stage2_pair_seed_v1"
LABEL_RULE = "e10_lt_one_minus_margin_times_e0_v1"
CHUNK_PLAN_ALGORITHM = "sample_id_sorted_fixed_chunks_per_shard_v1"

_GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_CONTRACT_KEYS = {
    "schema_version",
    "kind",
    "data_manifest_sha256",
    "base_checkpoint_sha256",
    "adapter_checkpoint_sha256",
    "normalization_stats_sha256",
    "data_config_sha256",
    "vae_sha256",
    "label_runtime_config_sha256",
    "git_identity",
    "episode_split_sha256",
    "episode_assignment_sha256",
    "base_seed",
    "num_seed_pairs",
    "seed_algorithm",
    "relative_margin",
    "relative_gain_epsilon",
    "label_rule",
    "num_inference_steps",
    "sigma_shift",
    "rand_device",
    "tiled",
    "num_shards",
    "shard_algorithm",
    "chunk_size",
    "chunk_plan_algorithm",
    "contract_sha256",
}
_ROW_KEYS = {
    "schema_version",
    "kind",
    "contract_sha256",
    "sample_id",
    "dataset_id",
    "dataset_index",
    "episode_id",
    "frame_id",
    "global_sample_index",
    "dataset_frame_index",
    "split",
    "e0",
    "e10",
    "relative_gain",
    "label",
    "sample_weight",
    "seeds",
    "margin",
    "num_inference_steps",
    "num_video_frames",
    "shard_index",
}
_CHUNK_KEYS = {
    "schema_version",
    "kind",
    "contract_sha256",
    "shard_index",
    "chunk_index",
    "planned_row_count",
    "planned_sample_ids_sha256",
    "row_count",
    "rows_sha256",
    "rows",
    "chunk_sha256",
}
_MANIFEST_KEYS = {
    "schema_version",
    "kind",
    "contract",
    "contract_sha256",
    "expected_sample_ids_sha256",
    "row_count",
    "positive_count",
    "split_counts",
    "shard_counts",
    "chunks",
    "rows_file",
    "rows_file_sha256",
    "manifest_sha256",
}


@dataclass(frozen=True)
class LabelArtifactContext:
    """One-time-validated immutable state for O(1) per-row artifact work."""

    contract: Mapping[str, Any]
    data_manifest: Mapping[str, Any]
    episode_split: Mapping[str, Any]
    episode_lookup: EpisodeSplitLookup


@dataclass(frozen=True, slots=True)
class ValidatedMergedLabelArtifact:
    """Detached, recursively immutable merged manifest and label rows."""

    manifest: Mapping[str, Any]
    rows: tuple[Mapping[str, Any], ...]


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _exact_keys(payload: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{name} keys differ: missing={missing}, extra={extra}")


def _integer(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return int(value)


def _finite_float(value: Any, *, field: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field} must be finite")
    if minimum is not None and normalized < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return normalized


def _sigma_shift(value: Any) -> float | None:
    if value is None:
        return None
    normalized = _finite_float(value, field="sigma_shift", minimum=0.0)
    if normalized == 0.0:
        raise ValueError("sigma_shift must be positive when provided")
    return normalized


def _rand_device(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("rand_device must be a non-empty string")
    return value


def _tiled(value: Any) -> bool:
    if not isinstance(value, bool):
        raise TypeError("tiled must be bool")
    return value


def _self_sha256(payload: Mapping[str, Any], *, field: str) -> str:
    unhashed = dict(payload)
    unhashed.pop(field, None)
    return canonical_json_sha256(unhashed)


def _validated_git_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("git_identity must be a mapping")
    payload = dict(value)
    _exact_keys(
        payload,
        {"commit", "tracked_dirty", "untracked_source_files"},
        name="git_identity",
    )
    commit = payload["commit"]
    if not isinstance(commit, str) or not _GIT_COMMIT_PATTERN.fullmatch(commit):
        raise ValueError("git_identity commit must be a full lowercase Git SHA")
    if not isinstance(payload["tracked_dirty"], bool):
        raise TypeError("git_identity tracked_dirty must be bool")
    files = payload["untracked_source_files"]
    if not isinstance(files, list) or any(
        not isinstance(path, str) or not path for path in files
    ):
        raise TypeError("git_identity untracked_source_files must be strings")
    if files != sorted(set(files)):
        raise ValueError(
            "git_identity untracked_source_files must be sorted and unique"
        )
    return {
        "commit": commit,
        "tracked_dirty": payload["tracked_dirty"],
        "untracked_source_files": list(files),
    }


def _episode_split_sha256(split: Mapping[str, Any]) -> str:
    return canonical_json_sha256(dict(split))


def build_label_contract(
    *,
    data_manifest: Mapping[str, Any],
    episode_split: Mapping[str, Any],
    base_checkpoint_sha256: str,
    adapter_checkpoint_sha256: str,
    normalization_stats_sha256: str,
    data_config_sha256: str,
    vae_sha256: str,
    label_runtime_config_sha256: str,
    git_identity: Mapping[str, Any],
    base_seed: int,
    num_seed_pairs: int,
    relative_margin: float,
    num_shards: int,
    chunk_size: int,
    relative_gain_epsilon: float = 1.0e-12,
    num_inference_steps: int = STAGE2_NUM_INFERENCE_STEPS,
    sigma_shift: float | None = None,
    rand_device: str = "cpu",
    tiled: bool = False,
) -> dict[str, Any]:
    """Build the immutable identity shared by every row in one label job."""

    validated_split = validate_episode_split(episode_split, data_manifest)
    steps = _integer(
        num_inference_steps,
        field="num_inference_steps",
        minimum=1,
    )
    if steps != STAGE2_NUM_INFERENCE_STEPS:
        raise ValueError(
            f"formal Stage 2 labels require exactly "
            f"{STAGE2_NUM_INFERENCE_STEPS} inference steps"
        )
    pairs = _integer(num_seed_pairs, field="num_seed_pairs", minimum=1)
    if not 2 <= pairs <= 4:
        raise ValueError("formal Stage 2 labels require 2--4 seed pairs")
    margin = _finite_float(relative_margin, field="relative_margin")
    if not 0.0 <= margin < 1.0:
        raise ValueError("relative_margin must be in [0, 1)")
    epsilon = _finite_float(
        relative_gain_epsilon,
        field="relative_gain_epsilon",
        minimum=0.0,
    )
    if epsilon == 0.0:
        raise ValueError("relative_gain_epsilon must be positive")
    payload: dict[str, Any] = {
        "schema_version": LABEL_CONTRACT_SCHEMA_VERSION,
        "kind": LABEL_CONTRACT_KIND,
        "data_manifest_sha256": require_sha256(
            data_manifest.get("manifest_sha256"),
            field="data_manifest_sha256",
        ),
        "base_checkpoint_sha256": require_sha256(
            base_checkpoint_sha256,
            field="base_checkpoint_sha256",
        ),
        "adapter_checkpoint_sha256": require_sha256(
            adapter_checkpoint_sha256,
            field="adapter_checkpoint_sha256",
        ),
        "normalization_stats_sha256": require_sha256(
            normalization_stats_sha256,
            field="normalization_stats_sha256",
        ),
        "data_config_sha256": require_sha256(
            data_config_sha256,
            field="data_config_sha256",
        ),
        "vae_sha256": require_sha256(
            vae_sha256,
            field="vae_sha256",
        ),
        "label_runtime_config_sha256": require_sha256(
            label_runtime_config_sha256,
            field="label_runtime_config_sha256",
        ),
        "git_identity": _validated_git_identity(git_identity),
        "episode_split_sha256": _episode_split_sha256(validated_split),
        "episode_assignment_sha256": require_sha256(
            validated_split.get("assignment_sha256"),
            field="episode_assignment_sha256",
        ),
        "base_seed": _integer(base_seed, field="base_seed"),
        "num_seed_pairs": pairs,
        "seed_algorithm": SEED_ALGORITHM,
        "relative_margin": margin,
        "relative_gain_epsilon": epsilon,
        "label_rule": LABEL_RULE,
        "num_inference_steps": steps,
        "sigma_shift": _sigma_shift(sigma_shift),
        "rand_device": _rand_device(rand_device),
        "tiled": _tiled(tiled),
        "num_shards": _integer(num_shards, field="num_shards", minimum=1),
        "shard_algorithm": SHARD_ALGORITHM,
        "chunk_size": _integer(
            chunk_size,
            field="chunk_size",
            minimum=1,
        ),
        "chunk_plan_algorithm": CHUNK_PLAN_ALGORITHM,
    }
    payload["contract_sha256"] = canonical_json_sha256(payload)
    return payload


def validate_label_contract(
    contract: Mapping[str, Any],
    *,
    data_manifest: Mapping[str, Any] | None = None,
    episode_split: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(contract, Mapping):
        raise TypeError("label contract must be a mapping")
    payload = dict(contract)
    _exact_keys(payload, _CONTRACT_KEYS, name="label contract")
    if _integer(
        payload["schema_version"], field="label contract schema_version"
    ) != LABEL_CONTRACT_SCHEMA_VERSION:
        raise ValueError("unsupported label contract schema_version")
    if payload["kind"] != LABEL_CONTRACT_KIND:
        raise ValueError("unsupported label contract kind")
    recorded = require_sha256(
        payload["contract_sha256"], field="contract_sha256"
    )
    if _self_sha256(payload, field="contract_sha256") != recorded:
        raise ValueError("label contract SHA256 does not match its contents")

    # Rebuild all typed/fixed fields instead of trusting a self-consistent but
    # semantically invalid JSON object.
    require_sha256(payload["data_manifest_sha256"], field="data_manifest_sha256")
    for field in (
        "base_checkpoint_sha256",
        "adapter_checkpoint_sha256",
        "normalization_stats_sha256",
        "data_config_sha256",
        "vae_sha256",
        "label_runtime_config_sha256",
        "episode_split_sha256",
        "episode_assignment_sha256",
    ):
        require_sha256(payload[field], field=field)
    _validated_git_identity(payload["git_identity"])
    _integer(payload["base_seed"], field="base_seed")
    pairs = _integer(payload["num_seed_pairs"], field="num_seed_pairs", minimum=1)
    if not 2 <= pairs <= 4:
        raise ValueError("formal Stage 2 labels require 2--4 seed pairs")
    if payload["seed_algorithm"] != SEED_ALGORITHM:
        raise ValueError("unsupported label seed algorithm")
    margin = _finite_float(payload["relative_margin"], field="relative_margin")
    if not 0.0 <= margin < 1.0:
        raise ValueError("relative_margin must be in [0, 1)")
    epsilon = _finite_float(
        payload["relative_gain_epsilon"],
        field="relative_gain_epsilon",
        minimum=0.0,
    )
    if epsilon == 0.0:
        raise ValueError("relative_gain_epsilon must be positive")
    if payload["label_rule"] != LABEL_RULE:
        raise ValueError("unsupported Gate label rule")
    if _integer(
        payload["num_inference_steps"], field="num_inference_steps", minimum=1
    ) != STAGE2_NUM_INFERENCE_STEPS:
        raise ValueError("formal Stage 2 label contract must use 10 steps")
    _sigma_shift(payload["sigma_shift"])
    _rand_device(payload["rand_device"])
    _tiled(payload["tiled"])
    _integer(payload["num_shards"], field="num_shards", minimum=1)
    if payload["shard_algorithm"] != SHARD_ALGORITHM:
        raise ValueError("unsupported label shard algorithm")
    _integer(payload["chunk_size"], field="chunk_size", minimum=1)
    if payload["chunk_plan_algorithm"] != CHUNK_PLAN_ALGORITHM:
        raise ValueError("unsupported label chunk plan algorithm")

    if (data_manifest is None) != (episode_split is None):
        raise ValueError("data_manifest and episode_split must be provided together")
    if data_manifest is not None and episode_split is not None:
        validated_split = validate_episode_split(episode_split, data_manifest)
        if payload["data_manifest_sha256"] != data_manifest.get("manifest_sha256"):
            raise ValueError("label contract data manifest SHA256 mismatch")
        if payload["episode_split_sha256"] != _episode_split_sha256(
            validated_split
        ):
            raise ValueError("label contract episode split SHA256 mismatch")
        if payload["episode_assignment_sha256"] != validated_split.get(
            "assignment_sha256"
        ):
            raise ValueError("label contract episode assignment SHA256 mismatch")
    return payload


def build_label_artifact_context(
    *,
    contract: Mapping[str, Any],
    data_manifest: Mapping[str, Any],
    episode_split: Mapping[str, Any],
) -> LabelArtifactContext:
    """Fully validate job identity once, then freeze its O(1) row state."""

    validated_contract = validate_label_contract(contract)
    episode_lookup = build_episode_split_lookup(episode_split, data_manifest)
    if (
        validated_contract["data_manifest_sha256"]
        != episode_lookup.data_manifest_sha256
    ):
        raise ValueError("label contract data manifest SHA256 mismatch")
    if validated_contract["episode_split_sha256"] != _episode_split_sha256(
        episode_split
    ):
        raise ValueError("label contract episode split SHA256 mismatch")
    if (
        validated_contract["episode_assignment_sha256"]
        != episode_lookup.assignment_sha256
    ):
        raise ValueError("label contract episode assignment SHA256 mismatch")
    return LabelArtifactContext(
        contract=_freeze_json(validated_contract),
        data_manifest=_freeze_json(dict(data_manifest)),
        episode_split=_freeze_json(dict(episode_split)),
        episode_lookup=episode_lookup,
    )


def shard_for_sample_id(sample_id_sha256: str, *, num_shards: int) -> int:
    stable_id = require_sha256(sample_id_sha256, field="sample_id")
    count = _integer(num_shards, field="num_shards", minimum=1)
    return int(stable_id[:16], 16) % count


def build_label_row(
    *,
    contract: Mapping[str, Any],
    data_manifest: Mapping[str, Any],
    episode_split: Mapping[str, Any],
    identity: Mapping[str, Any],
    e0: float,
    e10: float,
    relative_gain: float,
    label: bool,
    sample_weight: float,
    num_video_frames: int,
) -> dict[str, Any]:
    context = build_label_artifact_context(
        contract=contract,
        data_manifest=data_manifest,
        episode_split=episode_split,
    )
    return build_label_row_from_context(
        context=context,
        identity=identity,
        e0=e0,
        e10=e10,
        relative_gain=relative_gain,
        label=label,
        sample_weight=sample_weight,
        num_video_frames=num_video_frames,
    )


def build_label_row_from_context(
    *,
    context: LabelArtifactContext,
    identity: Mapping[str, Any],
    e0: float,
    e10: float,
    relative_gain: float,
    label: bool,
    sample_weight: float,
    num_video_frames: int,
) -> dict[str, Any]:
    """Build one row using only prevalidated O(1) identity lookups."""

    if not isinstance(context, LabelArtifactContext):
        raise TypeError("context must be a LabelArtifactContext")
    contract = context.contract
    normalized_identity = validate_sample_identity_with_lookup(
        identity, context.episode_lookup
    )
    stable_id = sample_id_from_lookup(
        normalized_identity, context.episode_lookup
    )
    row = {
        "schema_version": LABEL_ROW_SCHEMA_VERSION,
        "kind": LABEL_ROW_KIND,
        "contract_sha256": contract["contract_sha256"],
        "sample_id": stable_id,
        "dataset_id": dataset_id_from_lookup(
            normalized_identity["dataset_index"], context.episode_lookup
        ),
        "dataset_index": normalized_identity["dataset_index"],
        "episode_id": normalized_identity["episode_index"],
        "frame_id": normalized_identity["frame_index"],
        "global_sample_index": normalized_identity["global_sample_index"],
        "dataset_frame_index": normalized_identity["dataset_frame_index"],
        "split": split_for_identity(
            context.episode_split,
            context.data_manifest,
            normalized_identity,
            lookup=context.episode_lookup,
        ),
        "e0": _finite_float(e0, field="e0", minimum=0.0),
        "e10": _finite_float(e10, field="e10", minimum=0.0),
        "relative_gain": _finite_float(relative_gain, field="relative_gain"),
        "label": label,
        "sample_weight": _finite_float(
            sample_weight, field="sample_weight", minimum=0.0
        ),
        "seeds": derive_pair_seeds(
            sample_id_sha256=stable_id,
            base_seed=contract["base_seed"],
            num_pairs=contract["num_seed_pairs"],
        ),
        "margin": contract["relative_margin"],
        "num_inference_steps": contract["num_inference_steps"],
        "num_video_frames": num_video_frames,
        "shard_index": shard_for_sample_id(
            stable_id, num_shards=contract["num_shards"]
        ),
    }
    return validate_label_row_from_context(row, context=context)


def validate_label_row(
    row: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    data_manifest: Mapping[str, Any],
    episode_split: Mapping[str, Any],
) -> dict[str, Any]:
    context = build_label_artifact_context(
        contract=contract,
        data_manifest=data_manifest,
        episode_split=episode_split,
    )
    return validate_label_row_from_context(row, context=context)


def validate_label_row_from_context(
    row: Mapping[str, Any],
    *,
    context: LabelArtifactContext,
) -> dict[str, Any]:
    """Validate one row without rescanning the data manifest or split."""

    if not isinstance(context, LabelArtifactContext):
        raise TypeError("context must be a LabelArtifactContext")
    if not isinstance(row, Mapping):
        raise TypeError("label row must be a mapping")
    payload = dict(row)
    _exact_keys(payload, _ROW_KEYS, name="label row")
    contract = context.contract
    if _integer(
        payload["schema_version"], field="label row schema_version"
    ) != LABEL_ROW_SCHEMA_VERSION:
        raise ValueError("unsupported label row schema_version")
    if payload["kind"] != LABEL_ROW_KIND:
        raise ValueError("unsupported label row kind")
    if payload["contract_sha256"] != contract["contract_sha256"]:
        raise ValueError("label row contract SHA256 mismatch")

    identity = {
        "global_sample_index": payload["global_sample_index"],
        "dataset_index": payload["dataset_index"],
        "episode_index": payload["episode_id"],
        "frame_index": payload["frame_id"],
        "dataset_frame_index": payload["dataset_frame_index"],
    }
    normalized_identity = validate_sample_identity_with_lookup(
        identity, context.episode_lookup
    )
    stable_id = sample_id_from_lookup(
        normalized_identity, context.episode_lookup
    )
    if payload["sample_id"] != stable_id:
        raise ValueError("label row sample_id mismatch")
    expected_dataset_id = dataset_id_from_lookup(
        normalized_identity["dataset_index"], context.episode_lookup
    )
    if payload["dataset_id"] != expected_dataset_id:
        raise ValueError("label row dataset_id mismatch")
    expected_split = split_for_identity(
        context.episode_split,
        context.data_manifest,
        normalized_identity,
        lookup=context.episode_lookup,
    )
    if payload["split"] != expected_split:
        raise ValueError("label row episode split mismatch")

    e0 = _finite_float(payload["e0"], field="label row e0", minimum=0.0)
    e10 = _finite_float(payload["e10"], field="label row e10", minimum=0.0)
    gain = _finite_float(payload["relative_gain"], field="label row relative_gain")
    epsilon = contract["relative_gain_epsilon"]
    expected_gain = (e0 - e10) / max(e0, epsilon) if e0 > epsilon else 0.0
    if not math.isclose(gain, expected_gain, rel_tol=1.0e-6, abs_tol=1.0e-7):
        raise ValueError("label row relative_gain disagrees with E0/E10")
    if not isinstance(payload["label"], bool):
        raise TypeError("label row label must be bool")
    expected_label = e10 < (1.0 - contract["relative_margin"]) * e0
    if payload["label"] != expected_label:
        raise ValueError("label row label disagrees with E0/E10 and margin")
    _finite_float(
        payload["sample_weight"],
        field="label row sample_weight",
        minimum=0.0,
    )
    if payload["sample_weight"] <= 0.0:
        raise ValueError("label row sample_weight must be positive")

    expected_seeds = derive_pair_seeds(
        sample_id_sha256=stable_id,
        base_seed=contract["base_seed"],
        num_pairs=contract["num_seed_pairs"],
    )
    if payload["seeds"] != expected_seeds:
        raise ValueError("label row seeds disagree with sample_id/base_seed")
    if payload["margin"] != contract["relative_margin"]:
        raise ValueError("label row margin disagrees with label contract")
    if _integer(
        payload["num_inference_steps"],
        field="label row num_inference_steps",
        minimum=1,
    ) != STAGE2_NUM_INFERENCE_STEPS:
        raise ValueError("label row must use exactly 10 inference steps")
    video_frames = _integer(
        payload["num_video_frames"], field="label row num_video_frames", minimum=2
    )
    if video_frames % 4 != 1:
        raise ValueError("label row num_video_frames must satisfy T % 4 == 1")
    expected_shard = shard_for_sample_id(
        stable_id, num_shards=contract["num_shards"]
    )
    if _integer(
        payload["shard_index"], field="label row shard_index"
    ) != expected_shard:
        raise ValueError("label row shard_index disagrees with sample_id")
    return payload


def _validated_planned_ids(
    sample_ids: Sequence[str], *, contract: Mapping[str, Any], shard_index: int
) -> list[str]:
    if isinstance(sample_ids, (str, bytes)) or not isinstance(sample_ids, Sequence):
        raise TypeError("planned sample IDs must be a sequence")
    normalized = [
        require_sha256(value, field="planned sample_id")
        for value in sample_ids
    ]
    if normalized != sorted(set(normalized)):
        raise ValueError("planned sample IDs must be sorted and unique")
    num_shards = contract["num_shards"]
    if any(
        shard_for_sample_id(value, num_shards=num_shards) != shard_index
        for value in normalized
    ):
        raise ValueError("planned sample ID maps to a different shard")
    return normalized


def build_label_chunk(
    *,
    contract: Mapping[str, Any],
    data_manifest: Mapping[str, Any],
    episode_split: Mapping[str, Any],
    shard_index: int,
    chunk_index: int,
    planned_sample_ids: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    context = build_label_artifact_context(
        contract=contract,
        data_manifest=data_manifest,
        episode_split=episode_split,
    )
    return build_label_chunk_from_context(
        context=context,
        shard_index=shard_index,
        chunk_index=chunk_index,
        planned_sample_ids=planned_sample_ids,
        rows=rows,
    )


def build_label_chunk_from_context(
    *,
    context: LabelArtifactContext,
    shard_index: int,
    chunk_index: int,
    planned_sample_ids: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not isinstance(context, LabelArtifactContext):
        raise TypeError("context must be a LabelArtifactContext")
    contract = context.contract
    shard = _integer(shard_index, field="shard_index")
    if shard >= contract["num_shards"]:
        raise ValueError("shard_index is out of range")
    chunk = _integer(chunk_index, field="chunk_index")
    planned = _validated_planned_ids(
        planned_sample_ids, contract=contract, shard_index=shard
    )
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise TypeError("chunk rows must be a sequence")
    validated_rows = [
        validate_label_row_from_context(row, context=context)
        for row in rows
    ]
    validated_rows.sort(key=lambda row: row["sample_id"])
    row_ids = [row["sample_id"] for row in validated_rows]
    if row_ids != planned:
        raise ValueError("chunk rows do not exactly cover planned sample IDs")
    if any(row["shard_index"] != shard for row in validated_rows):
        raise ValueError("chunk contains a row assigned to another shard")
    payload: dict[str, Any] = {
        "schema_version": LABEL_CHUNK_SCHEMA_VERSION,
        "kind": LABEL_CHUNK_KIND,
        "contract_sha256": contract["contract_sha256"],
        "shard_index": shard,
        "chunk_index": chunk,
        "planned_row_count": len(planned),
        "planned_sample_ids_sha256": canonical_json_sha256(planned),
        "row_count": len(validated_rows),
        "rows_sha256": canonical_json_sha256(validated_rows),
        "rows": validated_rows,
    }
    payload["chunk_sha256"] = canonical_json_sha256(payload)
    return payload


def validate_label_chunk(
    chunk: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    data_manifest: Mapping[str, Any],
    episode_split: Mapping[str, Any],
    planned_sample_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    context = build_label_artifact_context(
        contract=contract,
        data_manifest=data_manifest,
        episode_split=episode_split,
    )
    return validate_label_chunk_from_context(
        chunk,
        context=context,
        planned_sample_ids=planned_sample_ids,
    )


def validate_label_chunk_from_context(
    chunk: Mapping[str, Any],
    *,
    context: LabelArtifactContext,
    planned_sample_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(context, LabelArtifactContext):
        raise TypeError("context must be a LabelArtifactContext")
    if not isinstance(chunk, Mapping):
        raise TypeError("label chunk must be a mapping")
    payload = dict(chunk)
    _exact_keys(payload, _CHUNK_KEYS, name="label chunk")
    contract = context.contract
    if _integer(
        payload["schema_version"], field="label chunk schema_version"
    ) != LABEL_CHUNK_SCHEMA_VERSION:
        raise ValueError("unsupported label chunk schema_version")
    if payload["kind"] != LABEL_CHUNK_KIND:
        raise ValueError("unsupported label chunk kind")
    if payload["contract_sha256"] != contract["contract_sha256"]:
        raise ValueError("label chunk uses a mixed label contract")
    recorded = require_sha256(payload["chunk_sha256"], field="chunk_sha256")
    if _self_sha256(payload, field="chunk_sha256") != recorded:
        raise ValueError("label chunk SHA256 does not match its contents")
    shard = _integer(payload["shard_index"], field="shard_index")
    if shard >= contract["num_shards"]:
        raise ValueError("label chunk shard_index is out of range")
    _integer(payload["chunk_index"], field="chunk_index")
    rows = payload["rows"]
    if not isinstance(rows, list):
        raise TypeError("label chunk rows must be a list")
    validated_rows = [
        validate_label_row_from_context(row, context=context)
        for row in rows
    ]
    if validated_rows != sorted(validated_rows, key=lambda row: row["sample_id"]):
        raise ValueError("label chunk rows must be sorted by sample_id")
    row_ids = [row["sample_id"] for row in validated_rows]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("label chunk contains duplicate sample IDs")
    if _integer(payload["row_count"], field="label chunk row_count") != len(
        validated_rows
    ):
        raise ValueError("label chunk row_count mismatch")
    if _integer(
        payload["planned_row_count"], field="label chunk planned_row_count"
    ) != len(validated_rows):
        raise ValueError("label chunk is incomplete")
    if payload["rows_sha256"] != canonical_json_sha256(validated_rows):
        raise ValueError("label chunk rows SHA256 mismatch")
    if payload["planned_sample_ids_sha256"] != canonical_json_sha256(row_ids):
        raise ValueError("label chunk planned sample IDs SHA256 mismatch")
    if any(row["shard_index"] != shard for row in validated_rows):
        raise ValueError("label chunk contains a row assigned to another shard")
    if planned_sample_ids is not None:
        planned = _validated_planned_ids(
            planned_sample_ids, contract=contract, shard_index=shard
        )
        if row_ids != planned:
            raise ValueError("label chunk does not match the external chunk plan")
    return payload


def write_label_chunk_atomic(
    path: str | Path,
    *,
    contract: Mapping[str, Any],
    data_manifest: Mapping[str, Any],
    episode_split: Mapping[str, Any],
    shard_index: int,
    chunk_index: int,
    planned_sample_ids: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    context = build_label_artifact_context(
        contract=contract,
        data_manifest=data_manifest,
        episode_split=episode_split,
    )
    return write_label_chunk_atomic_from_context(
        path,
        context=context,
        shard_index=shard_index,
        chunk_index=chunk_index,
        planned_sample_ids=planned_sample_ids,
        rows=rows,
    )


def write_label_chunk_atomic_from_context(
    path: str | Path,
    *,
    context: LabelArtifactContext,
    shard_index: int,
    chunk_index: int,
    planned_sample_ids: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    payload = build_label_chunk_from_context(
        context=context,
        shard_index=shard_index,
        chunk_index=chunk_index,
        planned_sample_ids=planned_sample_ids,
        rows=rows,
    )
    return write_json_atomic(path, payload)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_json_atomic_no_clobber(
    path: str | Path,
    payload: Mapping[str, Any],
) -> bool:
    """Durably publish JSON with atomic create-if-absent semantics.

    The payload is fully serialized into a unique file in the destination
    directory, flushed, and fsynced before a hard link atomically publishes the
    final name. Returns False when that name already exists; the existing inode
    is never opened for writing or replaced.
    """

    serialized = (
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    descriptor_owned = True
    try:
        stream = os.fdopen(descriptor, "wb")
        descriptor_owned = False
        with stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # A hard link is an atomic create-if-absent operation. Because the
            # temporary file is in the same directory, it cannot cross devices.
            os.link(temporary, output)
        except FileExistsError:
            return False
        return True
    finally:
        if descriptor_owned:
            os.close(descriptor)
        try:
            temporary.unlink(missing_ok=True)
        finally:
            # Persist both the final link (when won) and temporary-name cleanup.
            _fsync_directory(output.parent)


def publish_label_chunk_atomic(
    path: str | Path,
    *,
    contract: Mapping[str, Any],
    data_manifest: Mapping[str, Any],
    episode_split: Mapping[str, Any],
    shard_index: int,
    chunk_index: int,
    planned_sample_ids: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> bool:
    """Atomically publish a complete chunk without replacing an existing one.

    Returns ``True`` only when this invocation created ``path``.  ``False``
    means another artifact already occupied the destination and was left
    byte-for-byte untouched; callers must validate that artifact separately.
    """

    context = build_label_artifact_context(
        contract=contract,
        data_manifest=data_manifest,
        episode_split=episode_split,
    )
    return publish_label_chunk_atomic_from_context(
        path,
        context=context,
        shard_index=shard_index,
        chunk_index=chunk_index,
        planned_sample_ids=planned_sample_ids,
        rows=rows,
    )


def publish_label_chunk_atomic_from_context(
    path: str | Path,
    *,
    context: LabelArtifactContext,
    shard_index: int,
    chunk_index: int,
    planned_sample_ids: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> bool:
    """Durably publish a chunk with an atomic POSIX no-clobber operation."""

    payload = build_label_chunk_from_context(
        context=context,
        shard_index=shard_index,
        chunk_index=chunk_index,
        planned_sample_ids=planned_sample_ids,
        rows=rows,
    )
    return publish_json_atomic_no_clobber(path, payload)


def load_complete_label_chunk(
    path: str | Path,
    *,
    contract: Mapping[str, Any],
    data_manifest: Mapping[str, Any],
    episode_split: Mapping[str, Any],
    planned_sample_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    context = build_label_artifact_context(
        contract=contract,
        data_manifest=data_manifest,
        episode_split=episode_split,
    )
    return load_complete_label_chunk_from_context(
        path,
        context=context,
        planned_sample_ids=planned_sample_ids,
    )


def load_complete_label_chunk_from_context(
    path: str | Path,
    *,
    context: LabelArtifactContext,
    planned_sample_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"label chunk is unreadable or incomplete: {source}"
        ) from error
    return validate_label_chunk_from_context(
        payload,
        context=context,
        planned_sample_ids=planned_sample_ids,
    )


def merge_label_chunks(
    chunk_paths: Sequence[str | Path],
    *,
    contract: Mapping[str, Any],
    data_manifest: Mapping[str, Any],
    episode_split: Mapping[str, Any],
    expected_sample_ids: Sequence[str],
    rows_output: str | Path,
    manifest_output: str | Path,
) -> dict[str, Any]:
    """Validate and deterministically merge a complete label job."""

    validated_contract = validate_label_contract(
        contract,
        data_manifest=data_manifest,
        episode_split=episode_split,
    )
    if isinstance(chunk_paths, (str, bytes)) or not isinstance(
        chunk_paths, Sequence
    ):
        raise TypeError("chunk_paths must be a sequence")
    if not chunk_paths:
        raise ValueError("at least one label chunk is required")
    if isinstance(expected_sample_ids, (str, bytes)) or not isinstance(
        expected_sample_ids, Sequence
    ):
        raise TypeError("expected_sample_ids must be a sequence")
    expected = [
        require_sha256(value, field="expected sample_id")
        for value in expected_sample_ids
    ]
    if expected != sorted(set(expected)):
        raise ValueError("expected sample IDs must be sorted and unique")

    chunks: list[dict[str, Any]] = []
    chunk_keys: set[tuple[int, int]] = set()
    rows_by_id: dict[str, dict[str, Any]] = {}
    for raw_path in chunk_paths:
        chunk = load_complete_label_chunk(
            raw_path,
            contract=validated_contract,
            data_manifest=data_manifest,
            episode_split=episode_split,
        )
        key = (chunk["shard_index"], chunk["chunk_index"])
        if key in chunk_keys:
            raise ValueError("duplicate label chunk shard/chunk index")
        chunk_keys.add(key)
        chunks.append(chunk)
        for row in chunk["rows"]:
            stable_id = row["sample_id"]
            if stable_id in rows_by_id:
                raise ValueError("duplicate sample ID across label chunks")
            rows_by_id[stable_id] = row

    actual_ids = sorted(rows_by_id)
    missing = sorted(set(expected) - set(actual_ids))
    extra = sorted(set(actual_ids) - set(expected))
    if missing or extra:
        raise ValueError(
            "merged label coverage differs from expected samples: "
            f"missing={missing}, extra={extra}"
        )
    ordered_rows = [rows_by_id[stable_id] for stable_id in expected]
    rows_text = "".join(
        json.dumps(
            row,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
        for row in ordered_rows
    )
    rows_path = Path(rows_output)
    manifest_path = Path(manifest_output)
    if rows_path.parent.resolve() != manifest_path.parent.resolve():
        raise ValueError("merged rows and manifest must share one directory")
    if rows_path.resolve() == manifest_path.resolve():
        raise ValueError("merged rows and manifest paths must differ")
    rows_path = write_text_atomic(rows_path, rows_text)

    split_counts = {"train": 0, "validation": 0}
    shard_counts = [0 for _ in range(validated_contract["num_shards"])]
    positive_count = 0
    for row in ordered_rows:
        split_counts[row["split"]] += 1
        shard_counts[row["shard_index"]] += 1
        positive_count += int(row["label"])
    chunk_records = [
        {
            "shard_index": chunk["shard_index"],
            "chunk_index": chunk["chunk_index"],
            "row_count": chunk["row_count"],
            "chunk_sha256": chunk["chunk_sha256"],
        }
        for chunk in sorted(
            chunks, key=lambda value: (value["shard_index"], value["chunk_index"])
        )
    ]
    manifest: dict[str, Any] = {
        "schema_version": LABEL_MANIFEST_SCHEMA_VERSION,
        "kind": LABEL_MANIFEST_KIND,
        "contract": validated_contract,
        "contract_sha256": validated_contract["contract_sha256"],
        "expected_sample_ids_sha256": canonical_json_sha256(expected),
        "row_count": len(ordered_rows),
        "positive_count": positive_count,
        "split_counts": split_counts,
        "shard_counts": shard_counts,
        "chunks": chunk_records,
        "rows_file": rows_path.name,
        "rows_file_sha256": sha256_file(rows_path),
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    write_json_atomic(manifest_path, manifest)
    return manifest


def load_validated_merged_label_artifact(
    manifest_path: str | Path,
    *,
    contract: Mapping[str, Any],
    data_manifest: Mapping[str, Any],
    episode_split: Mapping[str, Any],
) -> ValidatedMergedLabelArtifact:
    """Load and validate one immutable snapshot of a merged label artifact.

    The JSONL file is read exactly once. Its SHA256 and parsed rows therefore
    refer to the same byte snapshot even if the path changes during loading.
    """

    source = Path(manifest_path)
    try:
        manifest_snapshot = source.read_bytes()
        manifest = json.loads(manifest_snapshot.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"label manifest is unreadable: {source}") from error
    if not isinstance(manifest, Mapping):
        raise TypeError("label manifest must be a mapping")
    payload = dict(manifest)
    _exact_keys(payload, _MANIFEST_KEYS, name="label manifest")
    if _integer(
        payload["schema_version"], field="label manifest schema_version"
    ) != LABEL_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported label manifest schema_version")
    if payload["kind"] != LABEL_MANIFEST_KIND:
        raise ValueError("unsupported label manifest kind")
    validated_contract = validate_label_contract(
        contract,
        data_manifest=data_manifest,
        episode_split=episode_split,
    )
    if payload["contract"] != validated_contract:
        raise ValueError("label manifest embeds a different label contract")
    if payload["contract_sha256"] != validated_contract["contract_sha256"]:
        raise ValueError("label manifest contract SHA256 mismatch")
    recorded = require_sha256(payload["manifest_sha256"], field="manifest_sha256")
    if _self_sha256(payload, field="manifest_sha256") != recorded:
        raise ValueError("label manifest SHA256 does not match its contents")
    rows_file = payload["rows_file"]
    if (
        not isinstance(rows_file, str)
        or rows_file in {"", ".", ".."}
        or Path(rows_file).name != rows_file
    ):
        raise ValueError("label manifest rows_file must be a local basename")
    rows_path = source.parent / rows_file
    expected_rows_sha256 = require_sha256(
        payload["rows_file_sha256"], field="rows_file_sha256"
    )
    try:
        rows_snapshot = rows_path.read_bytes()
    except OSError as error:
        raise ValueError(f"merged label JSONL is unreadable: {rows_path}") from error
    if hashlib.sha256(rows_snapshot).hexdigest() != expected_rows_sha256:
        raise ValueError("merged label JSONL SHA256 mismatch")
    try:
        lines = rows_snapshot.decode("utf-8").splitlines()
        rows = [json.loads(line) for line in lines]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("merged label JSONL is corrupt") from error
    context = build_label_artifact_context(
        contract=validated_contract,
        data_manifest=data_manifest,
        episode_split=episode_split,
    )
    validated_rows = [
        validate_label_row_from_context(row, context=context)
        for row in rows
    ]
    ids = [row["sample_id"] for row in validated_rows]
    if ids != sorted(set(ids)):
        raise ValueError("merged label JSONL rows must be sorted and unique")
    if _integer(payload["row_count"], field="manifest row_count") != len(
        validated_rows
    ):
        raise ValueError("label manifest row_count mismatch")
    if payload["expected_sample_ids_sha256"] != canonical_json_sha256(ids):
        raise ValueError("label manifest expected sample set mismatch")
    expected_positive = sum(int(row["label"]) for row in validated_rows)
    if _integer(
        payload["positive_count"], field="manifest positive_count"
    ) != expected_positive:
        raise ValueError("label manifest positive_count mismatch")
    split_counts = {"train": 0, "validation": 0}
    shard_counts = [0 for _ in range(validated_contract["num_shards"])]
    for row in validated_rows:
        split_counts[row["split"]] += 1
        shard_counts[row["shard_index"]] += 1
    if payload["split_counts"] != split_counts:
        raise ValueError("label manifest split_counts mismatch")
    if payload["shard_counts"] != shard_counts:
        raise ValueError("label manifest shard_counts mismatch")
    chunks = payload["chunks"]
    if not isinstance(chunks, list):
        raise TypeError("label manifest chunks must be a list")
    for chunk in chunks:
        if not isinstance(chunk, Mapping):
            raise TypeError("label manifest chunk record must be a mapping")
        if set(chunk) != {
            "shard_index",
            "chunk_index",
            "row_count",
            "chunk_sha256",
        }:
            raise ValueError("label manifest chunk record keys differ")
        shard_index = _integer(
            chunk["shard_index"], field="manifest chunk shard_index"
        )
        if shard_index >= validated_contract["num_shards"]:
            raise ValueError("label manifest chunk shard_index is out of range")
        _integer(chunk["chunk_index"], field="manifest chunk chunk_index")
        _integer(chunk["row_count"], field="manifest chunk row_count")
        require_sha256(chunk["chunk_sha256"], field="manifest chunk SHA256")
    expected_chunk_keys = sorted(
        (chunk["shard_index"], chunk["chunk_index"]) for chunk in chunks
    )
    if expected_chunk_keys != [
        (chunk["shard_index"], chunk["chunk_index"]) for chunk in chunks
    ] or len(expected_chunk_keys) != len(set(expected_chunk_keys)):
        raise ValueError("label manifest chunks must be sorted and unique")
    if sum(chunk.get("row_count", -1) for chunk in chunks) != len(validated_rows):
        raise ValueError("label manifest chunk row counts mismatch")
    return ValidatedMergedLabelArtifact(
        manifest=_freeze_json(payload),
        rows=_freeze_json(validated_rows),
    )


def validate_merged_label_artifact(
    manifest_path: str | Path,
    *,
    contract: Mapping[str, Any],
    data_manifest: Mapping[str, Any],
    episode_split: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a merged artifact and return its legacy mutable manifest."""

    artifact = load_validated_merged_label_artifact(
        manifest_path,
        contract=contract,
        data_manifest=data_manifest,
        episode_split=episode_split,
    )
    return _thaw_json(artifact.manifest)


__all__ = [
    "CHUNK_PLAN_ALGORITHM",
    "LABEL_CHUNK_KIND",
    "LABEL_CHUNK_SCHEMA_VERSION",
    "LABEL_CONTRACT_KIND",
    "LABEL_CONTRACT_SCHEMA_VERSION",
    "LABEL_MANIFEST_KIND",
    "LABEL_MANIFEST_SCHEMA_VERSION",
    "LABEL_ROW_KIND",
    "LABEL_ROW_SCHEMA_VERSION",
    "LabelArtifactContext",
    "ValidatedMergedLabelArtifact",
    "build_label_artifact_context",
    "build_label_chunk",
    "build_label_chunk_from_context",
    "build_label_contract",
    "build_label_row",
    "build_label_row_from_context",
    "load_complete_label_chunk",
    "load_complete_label_chunk_from_context",
    "load_validated_merged_label_artifact",
    "merge_label_chunks",
    "publish_json_atomic_no_clobber",
    "publish_label_chunk_atomic",
    "publish_label_chunk_atomic_from_context",
    "shard_for_sample_id",
    "validate_label_chunk",
    "validate_label_chunk_from_context",
    "validate_label_contract",
    "validate_label_row",
    "validate_label_row_from_context",
    "validate_merged_label_artifact",
    "write_label_chunk_atomic",
    "write_label_chunk_atomic_from_context",
]
