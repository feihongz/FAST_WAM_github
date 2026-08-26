"""Identity, seed, and episode-split contracts for Stage 2 labels."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any

from fastwam.alignment.checkpointing import canonical_json_sha256
from fastwam.alignment.data_identity import (
    canonical_data_manifest_sha256,
    require_supported_data_manifest_header,
)


EPISODE_SPLIT_SCHEMA_VERSION = 1
EPISODE_SPLIT_KIND = "stage2_gate_episode_split"
_IDENTITY_KEYS = {
    "global_sample_index",
    "dataset_index",
    "episode_index",
    "frame_index",
    "dataset_frame_index",
}


@dataclass(frozen=True)
class EpisodeSplitLookup:
    """Prevalidated, constant-time Stage 2 episode routing metadata."""

    data_manifest_sha256: str
    assignment_sha256: str
    dataset_count: int
    assignments: Mapping[tuple[int, int], str]
    frame_boundaries: Mapping[tuple[int, int], tuple[int, int]]
    dataset_frame_starts: Mapping[tuple[int, int], int]


def require_sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must contain exactly 64 lowercase hex chars")
    return value


def _nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value < 0:
        raise ValueError(f"{field} must be non-negative")
    return int(value)


def _validated_data_manifest(manifest: Mapping[str, Any]) -> tuple[dict, str]:
    payload = dict(manifest)
    require_supported_data_manifest_header(payload)
    recorded = require_sha256(
        payload.get("manifest_sha256"),
        field="data manifest manifest_sha256",
    )
    actual = canonical_data_manifest_sha256(payload)
    if actual != recorded:
        raise ValueError("data manifest SHA256 does not match its contents")
    roots = payload.get("dataset_roots")
    if not isinstance(roots, list) or not roots:
        raise ValueError("data manifest dataset_roots must be a non-empty list")
    manifest_num_frames = _nonnegative_int(
        payload.get("num_frames"),
        field="data manifest num_frames",
    )
    total_frames = 0
    for dataset_index, root in enumerate(roots):
        if not isinstance(root, Mapping):
            raise TypeError("data manifest dataset root must be a mapping")
        recorded_index = _nonnegative_int(
            root.get("dataset_index"),
            field="data manifest dataset_index",
        )
        if recorded_index != dataset_index:
            raise ValueError("data manifest dataset indices must be contiguous")
        root_num_frames = _nonnegative_int(
            root.get("num_frames"),
            field="dataset num_frames",
        )
        selected = root.get("selected_episodes")
        if not isinstance(selected, list) or not selected:
            raise ValueError("data manifest selected_episodes must be non-empty")
        selected_episodes = [
            _nonnegative_int(value, field="selected episode")
            for value in selected
        ]
        if len(set(selected_episodes)) != len(selected_episodes):
            raise ValueError("selected episodes must be unique")

        boundaries = root.get("episode_boundaries")
        if not isinstance(boundaries, list) or len(boundaries) != len(
            selected_episodes
        ):
            raise ValueError(
                "data manifest episode_boundaries must match selected_episodes"
            )
        expected_start = 0
        for position, (episode_index, boundary) in enumerate(
            zip(selected_episodes, boundaries, strict=True)
        ):
            if not isinstance(boundary, Mapping):
                raise TypeError("data manifest episode boundary must be a mapping")
            boundary_episode = _nonnegative_int(
                boundary.get("episode_index"),
                field=f"episode boundary episode_index[{position}]",
            )
            start = _nonnegative_int(
                boundary.get("from"),
                field=f"episode boundary from[{position}]",
            )
            end = _nonnegative_int(
                boundary.get("to"),
                field=f"episode boundary to[{position}]",
            )
            length = _nonnegative_int(
                boundary.get("length"),
                field=f"episode boundary length[{position}]",
            )
            if boundary_episode != episode_index:
                raise ValueError(
                    "episode boundaries must follow selected_episodes in order"
                )
            if start != expected_start or length <= 0 or end - start != length:
                raise ValueError(
                    "episode boundaries must be contiguous and non-empty"
                )
            expected_start = end
        if expected_start != root_num_frames:
            raise ValueError(
                "episode boundaries must exactly cover dataset num_frames"
            )
        total_frames += root_num_frames
    if total_frames != manifest_num_frames:
        raise ValueError(
            "dataset root num_frames do not sum to data manifest num_frames"
        )
    return payload, recorded


def validate_sample_identity(
    data_manifest: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> dict[str, int]:
    """Validate a returned current-frame identity against manifest boundaries."""

    manifest, _ = _validated_data_manifest(data_manifest)
    if set(identity) != _IDENTITY_KEYS:
        raise ValueError(
            "sample identity keys must be exactly "
            f"{sorted(_IDENTITY_KEYS)}"
        )
    normalized = {
        key: _nonnegative_int(identity[key], field=f"sample identity {key}")
        for key in sorted(_IDENTITY_KEYS)
    }
    roots = manifest["dataset_roots"]
    dataset_index = normalized["dataset_index"]
    if dataset_index >= len(roots):
        raise ValueError("sample identity dataset_index is out of range")
    root = roots[dataset_index]
    boundaries = root.get("episode_boundaries")
    if not isinstance(boundaries, list) or not boundaries:
        raise ValueError("data manifest episode_boundaries must be non-empty")
    matching = [
        row
        for row in boundaries
        if row.get("episode_index") == normalized["episode_index"]
    ]
    if len(matching) != 1:
        raise ValueError("sample identity episode is absent from data manifest")
    boundary = matching[0]
    length = _nonnegative_int(boundary.get("length"), field="episode length")
    start = _nonnegative_int(boundary.get("from"), field="episode start")
    end = _nonnegative_int(boundary.get("to"), field="episode end")
    if length <= 0 or end - start != length:
        raise ValueError("data manifest episode boundary is invalid")
    if normalized["frame_index"] >= length:
        raise ValueError("sample identity frame_index is outside its episode")
    root_offset = sum(
        _nonnegative_int(row.get("num_frames"), field="dataset num_frames")
        for row in roots[:dataset_index]
    )
    expected_global = root_offset + start + normalized["frame_index"]
    if normalized["global_sample_index"] != expected_global:
        raise ValueError(
            "sample identity global index disagrees with manifest boundary"
        )
    expected_dataset_frame = start + normalized["frame_index"]
    if normalized["dataset_frame_index"] != expected_dataset_frame:
        raise ValueError(
            "sample identity dataset frame index disagrees with manifest boundary"
        )
    return normalized


def sample_id(
    data_manifest: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> str:
    normalized = validate_sample_identity(data_manifest, identity)
    _, manifest_sha256 = _validated_data_manifest(data_manifest)
    return canonical_json_sha256(
        {
            "data_manifest_sha256": manifest_sha256,
            "dataset_index": normalized["dataset_index"],
            "episode_index": normalized["episode_index"],
            "frame_index": normalized["frame_index"],
        }
    )


def dataset_id(data_manifest: Mapping[str, Any], dataset_index: int) -> str:
    manifest, manifest_sha256 = _validated_data_manifest(data_manifest)
    index = _nonnegative_int(dataset_index, field="dataset_index")
    if index >= len(manifest["dataset_roots"]):
        raise ValueError("dataset_index is out of range")
    return canonical_json_sha256(
        {
            "data_manifest_sha256": manifest_sha256,
            "dataset_index": index,
        }
    )


def derive_pair_seeds(
    *,
    sample_id_sha256: str,
    base_seed: int,
    num_pairs: int,
) -> list[int]:
    stable_id = require_sha256(sample_id_sha256, field="sample_id")
    seed = _nonnegative_int(base_seed, field="base_seed")
    pairs = _nonnegative_int(num_pairs, field="num_pairs")
    if pairs < 1:
        raise ValueError("num_pairs must be positive")
    return [
        int(
            canonical_json_sha256(
                {
                    "algorithm": "stage2_pair_seed_v1",
                    "sample_id": stable_id,
                    "base_seed": seed,
                    "pair_index": pair_index,
                }
            )[:16],
            16,
        )
        % (2**63)
        for pair_index in range(pairs)
    ]


def build_episode_split(
    data_manifest: Mapping[str, Any],
    *,
    validation_fraction: float,
    split_seed: int,
) -> dict[str, Any]:
    manifest, manifest_sha256 = _validated_data_manifest(data_manifest)
    fraction = float(validation_fraction)
    seed = _nonnegative_int(split_seed, field="split_seed")
    if not 0.0 < fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")

    assignments: list[dict[str, Any]] = []
    root_counts: list[dict[str, int]] = []
    for dataset_index, root in enumerate(manifest["dataset_roots"]):
        if root.get("dataset_index") != dataset_index:
            raise ValueError("data manifest dataset indices must be contiguous")
        episodes = root.get("selected_episodes")
        if not isinstance(episodes, list) or len(episodes) < 2:
            raise ValueError(
                "every dataset root needs at least two episodes for splitting"
            )
        normalized = [
            _nonnegative_int(value, field="selected episode")
            for value in episodes
        ]
        if len(set(normalized)) != len(normalized):
            raise ValueError("selected episodes must be unique")
        ranked = sorted(
            normalized,
            key=lambda episode_index: canonical_json_sha256(
                {
                    "algorithm": "stage2_episode_split_v1",
                    "data_manifest_sha256": manifest_sha256,
                    "split_seed": seed,
                    "dataset_index": dataset_index,
                    "episode_index": episode_index,
                }
            ),
        )
        requested = int(math.floor(len(ranked) * fraction + 0.5))
        num_validation = min(max(requested, 1), len(ranked) - 1)
        validation = set(ranked[:num_validation])
        for episode_index in sorted(normalized):
            assignments.append(
                {
                    "dataset_index": dataset_index,
                    "episode_index": episode_index,
                    "split": (
                        "validation"
                        if episode_index in validation
                        else "train"
                    ),
                }
            )
        root_counts.append(
            {
                "dataset_index": dataset_index,
                "train_episodes": len(ranked) - num_validation,
                "validation_episodes": num_validation,
            }
        )

    assignment_sha256 = canonical_json_sha256(assignments)
    return {
        "schema_version": EPISODE_SPLIT_SCHEMA_VERSION,
        "kind": EPISODE_SPLIT_KIND,
        "algorithm": "sha256_rank_v1",
        "data_manifest_sha256": manifest_sha256,
        "split_seed": seed,
        "validation_fraction": fraction,
        "assignments": assignments,
        "assignment_sha256": assignment_sha256,
        "root_counts": root_counts,
    }


def validate_episode_split(
    split: Mapping[str, Any],
    data_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    payload = dict(split)
    expected = build_episode_split(
        data_manifest,
        validation_fraction=payload.get("validation_fraction"),
        split_seed=payload.get("split_seed"),
    )
    if payload != expected:
        raise ValueError("Stage 2 episode split differs from its contract")
    return expected


def build_episode_split_lookup(
    split: Mapping[str, Any],
    data_manifest: Mapping[str, Any],
) -> EpisodeSplitLookup:
    """Validate once and build O(1) sample-to-split lookup tables."""

    validated_split = validate_episode_split(split, data_manifest)
    manifest, manifest_sha256 = _validated_data_manifest(data_manifest)
    assignments: dict[tuple[int, int], str] = {}
    for row in validated_split["assignments"]:
        key = (row["dataset_index"], row["episode_index"])
        if key in assignments:
            raise ValueError("Stage 2 episode split contains duplicate assignments")
        assignments[key] = row["split"]

    frame_boundaries: dict[tuple[int, int], tuple[int, int]] = {}
    root_offset = 0
    dataset_frame_starts: dict[tuple[int, int], int] = {}
    for dataset_index, root in enumerate(manifest["dataset_roots"]):
        for boundary in root["episode_boundaries"]:
            key = (dataset_index, int(boundary["episode_index"]))
            if key in frame_boundaries:
                raise ValueError("data manifest contains duplicate episode boundaries")
            frame_boundaries[key] = (
                root_offset + int(boundary["from"]),
                int(boundary["length"]),
            )
            dataset_frame_starts[key] = int(boundary["from"])
        root_offset += int(root["num_frames"])
    if assignments.keys() != frame_boundaries.keys():
        raise ValueError(
            "Stage 2 split assignments do not cover manifest episodes exactly"
        )
    return EpisodeSplitLookup(
        data_manifest_sha256=manifest_sha256,
        assignment_sha256=validated_split["assignment_sha256"],
        dataset_count=len(manifest["dataset_roots"]),
        assignments=MappingProxyType(assignments),
        frame_boundaries=MappingProxyType(frame_boundaries),
        dataset_frame_starts=MappingProxyType(dataset_frame_starts),
    )


def validate_sample_identity_with_lookup(
    identity: Mapping[str, Any],
    lookup: EpisodeSplitLookup,
) -> dict[str, int]:
    """Validate one identity against an already-validated immutable lookup."""

    if not isinstance(lookup, EpisodeSplitLookup):
        raise TypeError("lookup must be an EpisodeSplitLookup")
    if set(identity) != _IDENTITY_KEYS:
        raise ValueError(
            "sample identity keys must be exactly "
            f"{sorted(_IDENTITY_KEYS)}"
        )
    normalized = {
        key: _nonnegative_int(identity[key], field=f"sample identity {key}")
        for key in sorted(_IDENTITY_KEYS)
    }
    key = (normalized["dataset_index"], normalized["episode_index"])
    boundary = lookup.frame_boundaries.get(key)
    if boundary is None or key not in lookup.assignments:
        raise ValueError("sample identity has no unique episode split")
    global_start, length = boundary
    frame_index = normalized["frame_index"]
    if frame_index >= length:
        raise ValueError("sample identity frame_index is outside its episode")
    if normalized["global_sample_index"] != global_start + frame_index:
        raise ValueError(
            "sample identity global index disagrees with manifest boundary"
        )
    dataset_frame_start = lookup.dataset_frame_starts.get(key)
    if dataset_frame_start is None:
        raise ValueError("sample identity has no dataset frame boundary")
    if normalized["dataset_frame_index"] != dataset_frame_start + frame_index:
        raise ValueError(
            "sample identity dataset frame index disagrees with manifest boundary"
        )
    return normalized


def sample_id_from_lookup(
    identity: Mapping[str, Any],
    lookup: EpisodeSplitLookup,
) -> str:
    """Build a stable semantic frame ID without rehashing the full manifest."""

    normalized = validate_sample_identity_with_lookup(identity, lookup)
    return canonical_json_sha256(
        {
            "data_manifest_sha256": lookup.data_manifest_sha256,
            "dataset_index": normalized["dataset_index"],
            "episode_index": normalized["episode_index"],
            "frame_index": normalized["frame_index"],
        }
    )


def dataset_id_from_lookup(
    dataset_index: int,
    lookup: EpisodeSplitLookup,
) -> str:
    """Build a stable dataset ID from prevalidated manifest identity."""

    if not isinstance(lookup, EpisodeSplitLookup):
        raise TypeError("lookup must be an EpisodeSplitLookup")
    index = _nonnegative_int(dataset_index, field="dataset_index")
    if index >= lookup.dataset_count:
        raise ValueError("dataset_index is out of range")
    return canonical_json_sha256(
        {
            "data_manifest_sha256": lookup.data_manifest_sha256,
            "dataset_index": index,
        }
    )


def _split_from_prevalidated_lookup(
    *,
    split: Mapping[str, Any],
    data_manifest: Mapping[str, Any],
    identity: Mapping[str, Any],
    lookup: EpisodeSplitLookup,
) -> str:
    if not isinstance(lookup, EpisodeSplitLookup):
        raise TypeError("lookup must be an EpisodeSplitLookup")
    if data_manifest.get("manifest_sha256") != lookup.data_manifest_sha256:
        raise ValueError("episode split lookup data manifest SHA256 mismatch")
    if split.get("data_manifest_sha256") != lookup.data_manifest_sha256:
        raise ValueError("episode split lookup split manifest SHA256 mismatch")
    if split.get("assignment_sha256") != lookup.assignment_sha256:
        raise ValueError("episode split lookup assignment SHA256 mismatch")
    normalized = validate_sample_identity_with_lookup(identity, lookup)
    key = (normalized["dataset_index"], normalized["episode_index"])
    route = lookup.assignments.get(key)
    if route is None:
        raise ValueError("sample identity has no unique episode split")
    return route


def split_for_identity(
    split: Mapping[str, Any],
    data_manifest: Mapping[str, Any],
    identity: Mapping[str, Any],
    *,
    lookup: EpisodeSplitLookup | None = None,
) -> str:
    if lookup is not None:
        return _split_from_prevalidated_lookup(
            split=split,
            data_manifest=data_manifest,
            identity=identity,
            lookup=lookup,
        )
    validated_split = validate_episode_split(split, data_manifest)
    normalized = validate_sample_identity(data_manifest, identity)
    matches = [
        row["split"]
        for row in validated_split["assignments"]
        if row["dataset_index"] == normalized["dataset_index"]
        and row["episode_index"] == normalized["episode_index"]
    ]
    if len(matches) != 1:
        raise ValueError("sample identity has no unique episode split")
    return str(matches[0])


__all__ = [
    "EPISODE_SPLIT_KIND",
    "EPISODE_SPLIT_SCHEMA_VERSION",
    "EpisodeSplitLookup",
    "build_episode_split",
    "build_episode_split_lookup",
    "dataset_id",
    "dataset_id_from_lookup",
    "derive_pair_seeds",
    "require_sha256",
    "sample_id",
    "sample_id_from_lookup",
    "split_for_identity",
    "validate_episode_split",
    "validate_sample_identity",
    "validate_sample_identity_with_lookup",
]
