"""Deterministic, nested Stage 2 label-selection artifacts.

The selection contract is deliberately independent from the numerical label
contract.  A label computed for a sample therefore remains reusable when a
later coverage tier activates another immutable cohort.

This module only plans semantic frame identities.  It never opens parquet or
video files and it never constructs a robot dataset.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from fastwam.alignment.checkpointing import (
    canonical_json_sha256,
    write_json_atomic,
    write_text_atomic,
)
from fastwam.alignment.data_identity import (
    canonical_data_manifest_sha256,
    require_supported_data_manifest_header,
)
from fastwam.gating.contracts import require_sha256


STRATIFIED_EPISODE_SPLIT_SCHEMA_VERSION = 2
STRATIFIED_EPISODE_SPLIT_KIND = "stage2_gate_episode_split"
STRATIFIED_EPISODE_SPLIT_ALGORITHM = (
    "stratified_sha256_rank_largest_remainder_v2"
)
EXPLICIT_STRATUM_RESOLVER = "explicit_episode_strata_v1"

LABEL_SELECTION_SCHEMA_VERSION = 1
LABEL_SELECTION_KIND = "stage2_gate_label_selection"
LABEL_SELECTION_ALGORITHM = "nested_dyadic_temporal_bins_v1"
LABEL_SELECTION_ROW_SCHEMA_VERSION = 1
LABEL_SELECTION_ROW_KIND = "stage2_gate_label_selection_row"
LABEL_COVERAGE_SCHEMA_VERSION = 1
LABEL_COVERAGE_KIND = "stage2_gate_label_coverage"

DEFAULT_MAX_TEMPORAL_BINS = 64
DEFAULT_TRAIN_TARGETS = (8, 16, 32, 64)
DEFAULT_VALIDATION_TARGET = 32
DEFAULT_COVERAGE_NAMES = ("pilot", "medium", "formal", "cap")

EPISODE_SPLIT_FILENAME = "episode_split.json"
SELECTION_DESCRIPTOR_FILENAME = "label_selection.json"
SELECTION_ROWS_FILENAME = "label_selection_rows.jsonl"

_TIER_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_SPLIT_KEYS = {
    "schema_version",
    "kind",
    "algorithm",
    "stratum_resolver",
    "data_manifest_sha256",
    "split_seed",
    "validation_fraction",
    "target_validation_episodes",
    "strata_sha256",
    "assignments",
    "assignment_sha256",
    "stratum_counts",
    "root_counts",
}
_STRATUM_ROW_KEYS = {"dataset_index", "episode_index", "stratum_id"}
_ASSIGNMENT_KEYS = _STRATUM_ROW_KEYS | {"split"}
_STRATUM_COUNT_KEYS = {
    "stratum_id",
    "episodes",
    "train_episodes",
    "validation_episodes",
}
_ROOT_COUNT_KEYS = {
    "dataset_index",
    "train_episodes",
    "validation_episodes",
}
_DESCRIPTOR_KEYS = {
    "schema_version",
    "kind",
    "algorithm",
    "data_manifest_sha256",
    "episode_split_file",
    "episode_split_sha256",
    "episode_assignment_sha256",
    "strata_sha256",
    "selection_seed",
    "max_temporal_bins",
    "train_targets",
    "validation_target",
    "cohorts",
    "coverage_tiers",
    "rows_file",
    "row_count",
    "rows_file_sha256",
    "sample_ids_sha256",
    "selection_sha256",
}
_COHORT_KEYS = {
    "cohort_index",
    "cohort_id",
    "split",
    "rank_from",
    "rank_to",
    "row_count",
    "sample_ids_sha256",
}
_TIER_KEYS = {"tier", "train_max_rank", "coverage_file"}
_ROW_KEYS = {
    "schema_version",
    "kind",
    "sample_id",
    "global_sample_index",
    "dataset_index",
    "episode_index",
    "frame_index",
    "dataset_frame_index",
    "split",
    "stratum_id",
    "episode_selection_rank",
    "temporal_bin",
    "episode_length",
    "distance_to_episode_end",
    "cohort_index",
    "cohort_id",
}
_COVERAGE_KEYS = {
    "schema_version",
    "kind",
    "selection_sha256",
    "tier",
    "train_max_rank",
    "active_cohort_indices",
    "active_cohort_ids",
    "sample_count",
    "sample_ids_sha256",
    "split_counts",
    "coverage_sha256",
}


@dataclass(frozen=True, slots=True)
class _Episode:
    dataset_index: int
    episode_index: int
    dataset_frame_start: int
    global_frame_start: int
    length: int


@dataclass(frozen=True)
class SelectionArtifacts:
    """In-memory representation of one complete master selection campaign."""

    episode_split: Mapping[str, Any]
    descriptor: Mapping[str, Any]
    rows: tuple[Mapping[str, Any], ...]
    coverages: Mapping[str, Mapping[str, Any]]


def _exact_keys(payload: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    actual = set(payload)
    if actual != expected:
        raise ValueError(
            f"{name} keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _integer(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return int(value)


def _finite_fraction(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError(f"{field} must be finite and in (0, 1)")
    return result


def _nonempty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _self_sha256(payload: Mapping[str, Any], *, field: str) -> str:
    unhashed = dict(payload)
    unhashed.pop(field, None)
    return canonical_json_sha256(unhashed)


def _canonical_jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    try:
        return "".join(
            json.dumps(
                dict(row),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
            for row in rows
        )
    except (TypeError, ValueError) as error:
        raise ValueError("selection rows must be canonical-JSON serializable") from error


def _jsonl_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(_canonical_jsonl(rows).encode("utf-8")).hexdigest()


def _sample_ids_sha256(sample_ids: Sequence[str]) -> str:
    normalized = sorted(
        require_sha256(value, field="sample_id") for value in sample_ids
    )
    if len(normalized) != len(set(normalized)):
        raise ValueError("sample IDs must be unique")
    return canonical_json_sha256(normalized)


def _prepare_manifest(
    data_manifest: Mapping[str, Any],
) -> tuple[str, tuple[_Episode, ...], int]:
    if not isinstance(data_manifest, Mapping):
        raise TypeError("data_manifest must be a mapping")
    payload = dict(data_manifest)
    require_supported_data_manifest_header(payload)
    recorded = require_sha256(
        payload.get("manifest_sha256"), field="data manifest manifest_sha256"
    )
    if canonical_data_manifest_sha256(payload) != recorded:
        raise ValueError("data manifest SHA256 does not match its contents")
    roots = payload.get("dataset_roots")
    if not isinstance(roots, list) or not roots:
        raise ValueError("data manifest dataset_roots must be a non-empty list")

    episodes: list[_Episode] = []
    global_root_start = 0
    for dataset_index, root_value in enumerate(roots):
        if not isinstance(root_value, Mapping):
            raise TypeError("data manifest dataset root must be a mapping")
        root = dict(root_value)
        if _integer(
            root.get("dataset_index"), field="data manifest dataset_index"
        ) != dataset_index:
            raise ValueError("data manifest dataset indices must be contiguous")
        root_frames = _integer(root.get("num_frames"), field="dataset num_frames")
        selected = root.get("selected_episodes")
        boundaries = root.get("episode_boundaries")
        if not isinstance(selected, list) or not selected:
            raise ValueError("data manifest selected_episodes must be non-empty")
        if not isinstance(boundaries, list) or len(boundaries) != len(selected):
            raise ValueError(
                "data manifest episode_boundaries must match selected_episodes"
            )
        normalized_selected = [
            _integer(value, field="selected episode") for value in selected
        ]
        if len(normalized_selected) != len(set(normalized_selected)):
            raise ValueError("selected episodes must be unique")
        expected_start = 0
        for position, (episode_index, boundary_value) in enumerate(
            zip(normalized_selected, boundaries, strict=True)
        ):
            if not isinstance(boundary_value, Mapping):
                raise TypeError("data manifest episode boundary must be a mapping")
            boundary = dict(boundary_value)
            recorded_episode = _integer(
                boundary.get("episode_index"),
                field=f"episode boundary episode_index[{position}]",
            )
            start = _integer(
                boundary.get("from"), field=f"episode boundary from[{position}]"
            )
            end = _integer(
                boundary.get("to"), field=f"episode boundary to[{position}]"
            )
            length = _integer(
                boundary.get("length"),
                field=f"episode boundary length[{position}]",
                minimum=1,
            )
            if recorded_episode != episode_index:
                raise ValueError(
                    "episode boundaries must follow selected_episodes in order"
                )
            if start != expected_start or end - start != length:
                raise ValueError("episode boundaries must be contiguous and non-empty")
            episodes.append(
                _Episode(
                    dataset_index=dataset_index,
                    episode_index=episode_index,
                    dataset_frame_start=start,
                    global_frame_start=global_root_start + start,
                    length=length,
                )
            )
            expected_start = end
        if expected_start != root_frames:
            raise ValueError("episode boundaries must exactly cover dataset num_frames")
        global_root_start += root_frames
    if _integer(payload.get("num_frames"), field="data manifest num_frames") != (
        global_root_start
    ):
        raise ValueError("dataset root num_frames do not sum to data manifest num_frames")
    return recorded, tuple(episodes), len(roots)


def _normalize_episode_strata(
    episode_strata: Sequence[Mapping[str, Any]],
    *,
    episodes: Sequence[_Episode],
) -> list[dict[str, Any]]:
    if isinstance(episode_strata, (str, bytes)) or not isinstance(
        episode_strata, Sequence
    ):
        raise TypeError("episode_strata must be a sequence")
    expected = {(row.dataset_index, row.episode_index) for row in episodes}
    normalized: list[dict[str, Any]] = []
    observed: set[tuple[int, int]] = set()
    for position, value in enumerate(episode_strata):
        if not isinstance(value, Mapping):
            raise TypeError(f"episode_strata[{position}] must be a mapping")
        row = dict(value)
        _exact_keys(row, _STRATUM_ROW_KEYS, name=f"episode_strata[{position}]")
        dataset_index = _integer(
            row["dataset_index"], field=f"episode_strata[{position}].dataset_index"
        )
        episode_index = _integer(
            row["episode_index"], field=f"episode_strata[{position}].episode_index"
        )
        stratum_id = _nonempty_string(
            row["stratum_id"], field=f"episode_strata[{position}].stratum_id"
        )
        key = (dataset_index, episode_index)
        if key in observed:
            raise ValueError("episode_strata contains duplicate episodes")
        observed.add(key)
        normalized.append(
            {
                "dataset_index": dataset_index,
                "episode_index": episode_index,
                "stratum_id": stratum_id,
            }
        )
    if observed != expected:
        missing = sorted(expected - observed)[:5]
        extra = sorted(observed - expected)[:5]
        raise ValueError(
            "episode_strata must cover manifest episodes exactly: "
            f"missing={missing}, extra={extra}"
        )
    return sorted(
        normalized,
        key=lambda row: (row["dataset_index"], row["episode_index"]),
    )


def build_libero_episode_strata(
    data_manifest: Mapping[str, Any],
    *,
    episode_task_indices: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Resolve LIBERO strata from an explicit episode-to-task artifact.

    The Stage 3 data manifest binds ``meta/episodes.jsonl`` by content hash but
    does not embed each episode's local task index.  Inferring a task from the
    episode number would therefore be unsafe.  Callers must read the pinned
    metadata and supply rows with exactly ``dataset_index``, ``episode_index``,
    and ``local_task_index``.
    """

    _, episodes, _ = _prepare_manifest(data_manifest)
    if isinstance(episode_task_indices, (str, bytes)) or not isinstance(
        episode_task_indices, Sequence
    ):
        raise TypeError("episode_task_indices must be a sequence")
    generic: list[dict[str, Any]] = []
    for position, value in enumerate(episode_task_indices):
        if not isinstance(value, Mapping):
            raise TypeError(f"episode_task_indices[{position}] must be a mapping")
        row = dict(value)
        _exact_keys(
            row,
            {"dataset_index", "episode_index", "local_task_index"},
            name=f"episode_task_indices[{position}]",
        )
        dataset_index = _integer(
            row["dataset_index"],
            field=f"episode_task_indices[{position}].dataset_index",
        )
        episode_index = _integer(
            row["episode_index"],
            field=f"episode_task_indices[{position}].episode_index",
        )
        local_task_index = _integer(
            row["local_task_index"],
            field=f"episode_task_indices[{position}].local_task_index",
        )
        generic.append(
            {
                "dataset_index": dataset_index,
                "episode_index": episode_index,
                "stratum_id": (
                    f"libero/dataset-{dataset_index:04d}/task-{local_task_index:06d}"
                ),
            }
        )
    return _normalize_episode_strata(generic, episodes=episodes)


def _validation_quotas(
    grouped: Mapping[str, Sequence[dict[str, Any]]],
    *,
    validation_fraction: float,
    target: int,
    manifest_sha256: str,
    split_seed: int,
) -> dict[str, int]:
    if any(len(rows) < 2 for rows in grouped.values()):
        small = sorted(key for key, rows in grouped.items() if len(rows) < 2)
        raise ValueError(
            "every stratum needs at least two episodes for episode-disjoint "
            f"train/validation coverage: {small[:5]}"
        )
    minimum = len(grouped)
    maximum = sum(len(rows) - 1 for rows in grouped.values())
    if not minimum <= target <= maximum:
        raise ValueError(
            "global validation target cannot give every stratum both splits: "
            f"target={target}, feasible=[{minimum}, {maximum}]"
        )

    remainders: dict[str, float] = {}
    quotas: dict[str, int] = {}
    for stratum_id, rows in grouped.items():
        ideal = len(rows) * validation_fraction
        floor_value = math.floor(ideal)
        quotas[stratum_id] = min(max(floor_value, 1), len(rows) - 1)
        remainders[stratum_id] = ideal - floor_value

    def tie_hash(stratum_id: str) -> str:
        return canonical_json_sha256(
            {
                "algorithm": STRATIFIED_EPISODE_SPLIT_ALGORITHM,
                "purpose": "largest_remainder_tie_break",
                "data_manifest_sha256": manifest_sha256,
                "split_seed": split_seed,
                "stratum_id": stratum_id,
            }
        )

    current = sum(quotas.values())
    while current < target:
        candidates = [
            key for key, rows in grouped.items() if quotas[key] < len(rows) - 1
        ]
        if not candidates:
            raise RuntimeError("validation quota apportionment exhausted candidates")
        chosen = min(candidates, key=lambda key: (-remainders[key], tie_hash(key)))
        quotas[chosen] += 1
        current += 1
    while current > target:
        candidates = [key for key in grouped if quotas[key] > 1]
        if not candidates:
            raise RuntimeError("validation quota apportionment exhausted candidates")
        chosen = min(candidates, key=lambda key: (remainders[key], tie_hash(key)))
        quotas[chosen] -= 1
        current -= 1
    return quotas


def build_stratified_episode_split(
    data_manifest: Mapping[str, Any],
    *,
    episode_strata: Sequence[Mapping[str, Any]],
    validation_fraction: float,
    split_seed: int,
) -> dict[str, Any]:
    """Build an exact-size, episode-disjoint split within explicit strata."""

    manifest_sha256, episodes, dataset_count = _prepare_manifest(data_manifest)
    fraction = _finite_fraction(validation_fraction, field="validation_fraction")
    seed = _integer(split_seed, field="split_seed")
    strata = _normalize_episode_strata(episode_strata, episodes=episodes)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in strata:
        grouped.setdefault(row["stratum_id"], []).append(row)
    target = int(math.floor(len(episodes) * fraction + 0.5))
    quotas = _validation_quotas(
        grouped,
        validation_fraction=fraction,
        target=target,
        manifest_sha256=manifest_sha256,
        split_seed=seed,
    )

    validation_keys: set[tuple[int, int]] = set()
    for stratum_id, rows in grouped.items():
        ranked = sorted(
            rows,
            key=lambda row: canonical_json_sha256(
                {
                    "algorithm": STRATIFIED_EPISODE_SPLIT_ALGORITHM,
                    "purpose": "episode_rank",
                    "data_manifest_sha256": manifest_sha256,
                    "split_seed": seed,
                    "stratum_id": stratum_id,
                    "dataset_index": row["dataset_index"],
                    "episode_index": row["episode_index"],
                }
            ),
        )
        validation_keys.update(
            (row["dataset_index"], row["episode_index"])
            for row in ranked[: quotas[stratum_id]]
        )

    assignments = [
        {
            **row,
            "split": (
                "validation"
                if (row["dataset_index"], row["episode_index"])
                in validation_keys
                else "train"
            ),
        }
        for row in strata
    ]
    stratum_counts = []
    for stratum_id in sorted(grouped):
        validation_count = quotas[stratum_id]
        stratum_counts.append(
            {
                "stratum_id": stratum_id,
                "episodes": len(grouped[stratum_id]),
                "train_episodes": len(grouped[stratum_id]) - validation_count,
                "validation_episodes": validation_count,
            }
        )
    root_counts = []
    for dataset_index in range(dataset_count):
        root_rows = [
            row for row in assignments if row["dataset_index"] == dataset_index
        ]
        root_counts.append(
            {
                "dataset_index": dataset_index,
                "train_episodes": sum(row["split"] == "train" for row in root_rows),
                "validation_episodes": sum(
                    row["split"] == "validation" for row in root_rows
                ),
            }
        )
    return {
        "schema_version": STRATIFIED_EPISODE_SPLIT_SCHEMA_VERSION,
        "kind": STRATIFIED_EPISODE_SPLIT_KIND,
        "algorithm": STRATIFIED_EPISODE_SPLIT_ALGORITHM,
        "stratum_resolver": EXPLICIT_STRATUM_RESOLVER,
        "data_manifest_sha256": manifest_sha256,
        "split_seed": seed,
        "validation_fraction": fraction,
        "target_validation_episodes": target,
        "strata_sha256": canonical_json_sha256(strata),
        "assignments": assignments,
        "assignment_sha256": canonical_json_sha256(assignments),
        "stratum_counts": stratum_counts,
        "root_counts": root_counts,
    }


def validate_stratified_episode_split(
    episode_split: Mapping[str, Any],
    data_manifest: Mapping[str, Any],
    *,
    episode_strata: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if not isinstance(episode_split, Mapping):
        raise TypeError("episode_split must be a mapping")
    payload = dict(episode_split)
    _exact_keys(payload, _SPLIT_KEYS, name="stratified episode split")
    if payload["schema_version"] != STRATIFIED_EPISODE_SPLIT_SCHEMA_VERSION:
        raise ValueError("unsupported stratified episode split schema_version")
    if payload["kind"] != STRATIFIED_EPISODE_SPLIT_KIND:
        raise ValueError("unsupported stratified episode split kind")
    if payload["algorithm"] != STRATIFIED_EPISODE_SPLIT_ALGORITHM:
        raise ValueError("unsupported stratified episode split algorithm")
    if payload["stratum_resolver"] != EXPLICIT_STRATUM_RESOLVER:
        raise ValueError("unsupported episode stratum resolver")
    assignments_value = payload["assignments"]
    if not isinstance(assignments_value, list):
        raise TypeError("episode split assignments must be a list")
    derived_strata: list[dict[str, Any]] = []
    for position, value in enumerate(assignments_value):
        if not isinstance(value, Mapping):
            raise TypeError(f"episode split assignments[{position}] must be a mapping")
        row = dict(value)
        _exact_keys(row, _ASSIGNMENT_KEYS, name=f"assignment[{position}]")
        if row["split"] not in {"train", "validation"}:
            raise ValueError("episode split route must be train or validation")
        derived_strata.append({key: row[key] for key in _STRATUM_ROW_KEYS})
    source_strata = episode_strata if episode_strata is not None else derived_strata
    expected = build_stratified_episode_split(
        data_manifest,
        episode_strata=source_strata,
        validation_fraction=payload.get("validation_fraction"),
        split_seed=payload.get("split_seed"),
    )
    if payload != expected:
        raise ValueError("stratified episode split differs from its contract")
    return expected


def _positive_power_of_two(value: Any, *, field: str) -> int:
    result = _integer(value, field=field, minimum=1)
    if result & (result - 1):
        raise ValueError(f"{field} must be a power of two")
    return result


def _normalize_train_targets(
    values: Sequence[int], *, max_temporal_bins: int
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("train_targets must be a sequence")
    targets = tuple(
        _integer(value, field=f"train_targets[{index}]", minimum=1)
        for index, value in enumerate(values)
    )
    if not targets:
        raise ValueError("train_targets must not be empty")
    if tuple(sorted(set(targets))) != targets:
        raise ValueError("train_targets must be strictly increasing")
    if any(value & (value - 1) for value in targets):
        raise ValueError("train_targets must contain powers of two")
    if targets[-1] != max_temporal_bins:
        raise ValueError("last train target must equal max_temporal_bins")
    return targets


def _normalize_coverage_names(
    values: Sequence[str], *, target_count: int
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("coverage_names must be a sequence")
    names = tuple(values)
    if len(names) != target_count:
        raise ValueError("coverage_names must match train_targets")
    if any(not isinstance(name, str) or not _TIER_PATTERN.fullmatch(name) for name in names):
        raise ValueError("coverage names must be canonical lowercase identifiers")
    if len(set(names)) != len(names):
        raise ValueError("coverage names must be unique")
    return names


def _bit_reverse(value: int, bits: int) -> int:
    result = 0
    for _ in range(bits):
        result = (result << 1) | (value & 1)
        value >>= 1
    return result


def _temporal_candidates(
    episode: _Episode,
    *,
    manifest_sha256: str,
    selection_seed: int,
    max_temporal_bins: int,
) -> list[tuple[int, int]]:
    bits = max_temporal_bins.bit_length() - 1
    rotation = int(
        canonical_json_sha256(
            {
                "algorithm": LABEL_SELECTION_ALGORITHM,
                "purpose": "episode_bin_rotation",
                "data_manifest_sha256": manifest_sha256,
                "selection_seed": selection_seed,
                "dataset_index": episode.dataset_index,
                "episode_index": episode.episode_index,
            }
        )[:16],
        16,
    ) % max_temporal_bins
    result: list[tuple[int, int]] = []
    observed_frames: set[int] = set()
    for rank in range(max_temporal_bins):
        temporal_bin = _bit_reverse(rank, bits) ^ rotation
        start = (temporal_bin * episode.length) // max_temporal_bins
        end = ((temporal_bin + 1) * episode.length) // max_temporal_bins
        if end <= start:
            continue
        width = end - start
        if width % 2:
            frame_index = start + width // 2
        else:
            centers = (start + width // 2 - 1, start + width // 2)
            frame_index = min(
                centers,
                key=lambda candidate: canonical_json_sha256(
                    {
                        "algorithm": LABEL_SELECTION_ALGORITHM,
                        "purpose": "bin_center_tie_break",
                        "data_manifest_sha256": manifest_sha256,
                        "selection_seed": selection_seed,
                        "dataset_index": episode.dataset_index,
                        "episode_index": episode.episode_index,
                        "temporal_bin": temporal_bin,
                        "frame_index": candidate,
                    }
                ),
            )
        if frame_index in observed_frames:
            raise RuntimeError("temporal bin selection produced a duplicate frame")
        observed_frames.add(frame_index)
        result.append((temporal_bin, frame_index))
    expected_count = min(episode.length, max_temporal_bins)
    if len(result) != expected_count:
        raise RuntimeError(
            "temporal bin selection did not saturate the episode without duplicates"
        )
    return result


def _cohort_specs(
    *, train_targets: tuple[int, ...], validation_target: int
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    lower = 0
    for target in train_targets:
        specs.append(
            {
                "cohort_index": len(specs),
                "cohort_id": f"train_rank_{lower:03d}_{target:03d}",
                "split": "train",
                "rank_from": lower,
                "rank_to": target,
            }
        )
        lower = target
    specs.append(
        {
            "cohort_index": len(specs),
            "cohort_id": f"validation_rank_000_{validation_target:03d}",
            "split": "validation",
            "rank_from": 0,
            "rank_to": validation_target,
        }
    )
    return specs


def _selection_rows(
    *,
    manifest_sha256: str,
    episodes: Sequence[_Episode],
    episode_split: Mapping[str, Any],
    selection_seed: int,
    max_temporal_bins: int,
    cohort_specs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    assignments = {
        (row["dataset_index"], row["episode_index"]): row
        for row in episode_split["assignments"]
    }
    specs_by_split = {
        split: [row for row in cohort_specs if row["split"] == split]
        for split in ("train", "validation")
    }
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        assignment = assignments[(episode.dataset_index, episode.episode_index)]
        route = assignment["split"]
        candidates = _temporal_candidates(
            episode,
            manifest_sha256=manifest_sha256,
            selection_seed=selection_seed,
            max_temporal_bins=max_temporal_bins,
        )
        for episode_rank, (temporal_bin, frame_index) in enumerate(candidates):
            matches = [
                spec
                for spec in specs_by_split[route]
                if spec["rank_from"] <= episode_rank < spec["rank_to"]
            ]
            if not matches:
                continue
            if len(matches) != 1:
                raise RuntimeError("selection cohort rank ranges overlap")
            cohort = matches[0]
            identity = {
                "global_sample_index": episode.global_frame_start + frame_index,
                "dataset_index": episode.dataset_index,
                "episode_index": episode.episode_index,
                "frame_index": frame_index,
                "dataset_frame_index": episode.dataset_frame_start + frame_index,
            }
            semantic_id = canonical_json_sha256(
                {
                    "data_manifest_sha256": manifest_sha256,
                    "dataset_index": episode.dataset_index,
                    "episode_index": episode.episode_index,
                    "frame_index": frame_index,
                }
            )
            rows.append(
                {
                    "schema_version": LABEL_SELECTION_ROW_SCHEMA_VERSION,
                    "kind": LABEL_SELECTION_ROW_KIND,
                    "sample_id": semantic_id,
                    **identity,
                    "split": route,
                    "stratum_id": assignment["stratum_id"],
                    "episode_selection_rank": episode_rank,
                    "temporal_bin": temporal_bin,
                    "episode_length": episode.length,
                    "distance_to_episode_end": episode.length - 1 - frame_index,
                    "cohort_index": cohort["cohort_index"],
                    "cohort_id": cohort["cohort_id"],
                }
            )
    return sorted(rows, key=lambda row: (row["cohort_index"], row["sample_id"]))


def _assemble_selection(
    data_manifest: Mapping[str, Any],
    episode_split: Mapping[str, Any],
    *,
    selection_seed: int,
    max_temporal_bins: int,
    train_targets: tuple[int, ...],
    validation_target: int,
    coverage_names: tuple[str, ...],
) -> SelectionArtifacts:
    manifest_sha256, episodes, _ = _prepare_manifest(data_manifest)
    validated_split = validate_stratified_episode_split(
        episode_split, data_manifest
    )
    split_sha256 = canonical_json_sha256(validated_split)
    specs = _cohort_specs(
        train_targets=train_targets, validation_target=validation_target
    )
    rows = _selection_rows(
        manifest_sha256=manifest_sha256,
        episodes=episodes,
        episode_split=validated_split,
        selection_seed=selection_seed,
        max_temporal_bins=max_temporal_bins,
        cohort_specs=specs,
    )
    cohort_descriptors: list[dict[str, Any]] = []
    for spec in specs:
        cohort_rows = [
            row for row in rows if row["cohort_index"] == spec["cohort_index"]
        ]
        cohort_descriptors.append(
            {
                **dict(spec),
                "row_count": len(cohort_rows),
                "sample_ids_sha256": _sample_ids_sha256(
                    [row["sample_id"] for row in cohort_rows]
                ),
            }
        )
    coverage_tiers = [
        {
            "tier": name,
            "train_max_rank": target,
            "coverage_file": f"label_coverage_{name}.json",
        }
        for name, target in zip(coverage_names, train_targets, strict=True)
    ]
    descriptor = {
        "schema_version": LABEL_SELECTION_SCHEMA_VERSION,
        "kind": LABEL_SELECTION_KIND,
        "algorithm": LABEL_SELECTION_ALGORITHM,
        "data_manifest_sha256": manifest_sha256,
        "episode_split_file": EPISODE_SPLIT_FILENAME,
        "episode_split_sha256": split_sha256,
        "episode_assignment_sha256": validated_split["assignment_sha256"],
        "strata_sha256": validated_split["strata_sha256"],
        "selection_seed": selection_seed,
        "max_temporal_bins": max_temporal_bins,
        "train_targets": list(train_targets),
        "validation_target": validation_target,
        "cohorts": cohort_descriptors,
        "coverage_tiers": coverage_tiers,
        "rows_file": SELECTION_ROWS_FILENAME,
        "row_count": len(rows),
        "rows_file_sha256": _jsonl_sha256(rows),
        "sample_ids_sha256": _sample_ids_sha256(
            [row["sample_id"] for row in rows]
        ),
    }
    descriptor["selection_sha256"] = canonical_json_sha256(descriptor)

    coverages: dict[str, dict[str, Any]] = {}
    validation_cohort = cohort_descriptors[-1]
    for tier, train_max_rank in zip(coverage_names, train_targets, strict=True):
        active = [
            cohort
            for cohort in cohort_descriptors[:-1]
            if cohort["rank_to"] <= train_max_rank
        ] + [validation_cohort]
        active_indices = {cohort["cohort_index"] for cohort in active}
        active_rows = [
            row for row in rows if row["cohort_index"] in active_indices
        ]
        coverage = {
            "schema_version": LABEL_COVERAGE_SCHEMA_VERSION,
            "kind": LABEL_COVERAGE_KIND,
            "selection_sha256": descriptor["selection_sha256"],
            "tier": tier,
            "train_max_rank": train_max_rank,
            "active_cohort_indices": [
                cohort["cohort_index"] for cohort in active
            ],
            "active_cohort_ids": [cohort["cohort_id"] for cohort in active],
            "sample_count": len(active_rows),
            "sample_ids_sha256": _sample_ids_sha256(
                [row["sample_id"] for row in active_rows]
            ),
            "split_counts": {
                "train": sum(row["split"] == "train" for row in active_rows),
                "validation": sum(
                    row["split"] == "validation" for row in active_rows
                ),
            },
        }
        coverage["coverage_sha256"] = canonical_json_sha256(coverage)
        coverages[tier] = coverage
    return SelectionArtifacts(
        episode_split=validated_split,
        descriptor=descriptor,
        rows=tuple(rows),
        coverages=coverages,
    )


def build_selection_artifacts(
    data_manifest: Mapping[str, Any],
    *,
    episode_strata: Sequence[Mapping[str, Any]],
    validation_fraction: float = 0.1,
    split_seed: int = 42,
    selection_seed: int = 42,
    max_temporal_bins: int = DEFAULT_MAX_TEMPORAL_BINS,
    train_targets: Sequence[int] = DEFAULT_TRAIN_TARGETS,
    validation_target: int = DEFAULT_VALIDATION_TARGET,
    coverage_names: Sequence[str] = DEFAULT_COVERAGE_NAMES,
) -> SelectionArtifacts:
    """Build the split, immutable master rows, and all nested coverage tiers."""

    bins = _positive_power_of_two(
        max_temporal_bins, field="max_temporal_bins"
    )
    targets = _normalize_train_targets(train_targets, max_temporal_bins=bins)
    val_target = _integer(
        validation_target, field="validation_target", minimum=1
    )
    if val_target > bins:
        raise ValueError("validation_target must not exceed max_temporal_bins")
    names = _normalize_coverage_names(coverage_names, target_count=len(targets))
    seed = _integer(selection_seed, field="selection_seed")
    split = build_stratified_episode_split(
        data_manifest,
        episode_strata=episode_strata,
        validation_fraction=validation_fraction,
        split_seed=split_seed,
    )
    return _assemble_selection(
        data_manifest,
        split,
        selection_seed=seed,
        max_temporal_bins=bins,
        train_targets=targets,
        validation_target=val_target,
        coverage_names=names,
    )


def build_selection_from_split(
    data_manifest: Mapping[str, Any],
    episode_split: Mapping[str, Any],
    *,
    selection_seed: int = 42,
    max_temporal_bins: int = DEFAULT_MAX_TEMPORAL_BINS,
    train_targets: Sequence[int] = DEFAULT_TRAIN_TARGETS,
    validation_target: int = DEFAULT_VALIDATION_TARGET,
    coverage_names: Sequence[str] = DEFAULT_COVERAGE_NAMES,
) -> SelectionArtifacts:
    """Build selection rows from an already-pinned stratified split."""

    bins = _positive_power_of_two(
        max_temporal_bins, field="max_temporal_bins"
    )
    targets = _normalize_train_targets(train_targets, max_temporal_bins=bins)
    val_target = _integer(
        validation_target, field="validation_target", minimum=1
    )
    if val_target > bins:
        raise ValueError("validation_target must not exceed max_temporal_bins")
    names = _normalize_coverage_names(coverage_names, target_count=len(targets))
    seed = _integer(selection_seed, field="selection_seed")
    return _assemble_selection(
        data_manifest,
        episode_split,
        selection_seed=seed,
        max_temporal_bins=bins,
        train_targets=targets,
        validation_target=val_target,
        coverage_names=names,
    )


def _prevalidate_descriptor(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(descriptor, Mapping):
        raise TypeError("selection descriptor must be a mapping")
    payload = dict(descriptor)
    _exact_keys(payload, _DESCRIPTOR_KEYS, name="selection descriptor")
    if payload["schema_version"] != LABEL_SELECTION_SCHEMA_VERSION:
        raise ValueError("unsupported selection descriptor schema_version")
    if payload["kind"] != LABEL_SELECTION_KIND:
        raise ValueError("unsupported selection descriptor kind")
    if payload["algorithm"] != LABEL_SELECTION_ALGORITHM:
        raise ValueError("unsupported selection algorithm")
    recorded = require_sha256(
        payload["selection_sha256"], field="selection_sha256"
    )
    if _self_sha256(payload, field="selection_sha256") != recorded:
        raise ValueError("selection descriptor SHA256 does not match its contents")
    require_sha256(payload["rows_file_sha256"], field="rows_file_sha256")
    require_sha256(payload["sample_ids_sha256"], field="sample_ids_sha256")
    if payload["rows_file"] != SELECTION_ROWS_FILENAME:
        raise ValueError("selection rows_file is not canonical")
    if payload["episode_split_file"] != EPISODE_SPLIT_FILENAME:
        raise ValueError("selection episode_split_file is not canonical")
    return payload


def _prevalidate_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise TypeError("selection rows must be a sequence")
    normalized: list[dict[str, Any]] = []
    for position, value in enumerate(rows):
        if not isinstance(value, Mapping):
            raise TypeError(f"selection rows[{position}] must be a mapping")
        row = dict(value)
        _exact_keys(row, _ROW_KEYS, name=f"selection row[{position}]")
        normalized.append(row)
    return normalized


def _prevalidate_coverages(
    coverages: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not isinstance(coverages, Mapping):
        raise TypeError("coverages must be a mapping")
    normalized: dict[str, dict[str, Any]] = {}
    for key, value in coverages.items():
        if not isinstance(key, str) or not isinstance(value, Mapping):
            raise TypeError("coverage entries must map string tiers to mappings")
        payload = dict(value)
        _exact_keys(payload, _COVERAGE_KEYS, name=f"coverage[{key}]")
        if payload["schema_version"] != LABEL_COVERAGE_SCHEMA_VERSION:
            raise ValueError("unsupported label coverage schema_version")
        if payload["kind"] != LABEL_COVERAGE_KIND:
            raise ValueError("unsupported label coverage kind")
        if payload["tier"] != key:
            raise ValueError("coverage mapping key and tier differ")
        recorded = require_sha256(
            payload["coverage_sha256"], field=f"coverage[{key}].coverage_sha256"
        )
        if _self_sha256(payload, field="coverage_sha256") != recorded:
            raise ValueError(f"coverage {key} SHA256 does not match its contents")
        normalized[key] = payload
    return normalized


def validate_selection_artifacts(
    artifacts: SelectionArtifacts,
    *,
    data_manifest: Mapping[str, Any],
    episode_strata: Sequence[Mapping[str, Any]] | None = None,
) -> SelectionArtifacts:
    """Rebuild every derived byte and reject any incomplete or altered artifact."""

    if not isinstance(artifacts, SelectionArtifacts):
        raise TypeError("artifacts must be SelectionArtifacts")
    split = validate_stratified_episode_split(
        artifacts.episode_split,
        data_manifest,
        episode_strata=episode_strata,
    )
    descriptor = _prevalidate_descriptor(artifacts.descriptor)
    rows = _prevalidate_rows(artifacts.rows)
    coverages = _prevalidate_coverages(artifacts.coverages)
    if descriptor["rows_file_sha256"] != _jsonl_sha256(rows):
        raise ValueError("selection rows SHA256 does not match descriptor")
    if descriptor["row_count"] != len(rows):
        raise ValueError("selection row_count does not match rows")

    tiers_value = descriptor["coverage_tiers"]
    if not isinstance(tiers_value, list):
        raise TypeError("coverage_tiers must be a list")
    tier_names: list[str] = []
    tier_targets: list[int] = []
    for position, value in enumerate(tiers_value):
        if not isinstance(value, Mapping):
            raise TypeError(f"coverage_tiers[{position}] must be a mapping")
        tier = dict(value)
        _exact_keys(tier, _TIER_KEYS, name=f"coverage_tiers[{position}]")
        tier_names.append(tier["tier"])
        tier_targets.append(tier["train_max_rank"])
        if tier["coverage_file"] != f"label_coverage_{tier['tier']}.json":
            raise ValueError("coverage filename is not canonical")
    if tier_targets != descriptor["train_targets"]:
        raise ValueError("coverage tiers and train_targets differ")
    if set(coverages) != set(tier_names):
        raise ValueError("coverage files do not match descriptor tiers")

    expected = build_selection_from_split(
        data_manifest,
        split,
        selection_seed=descriptor["selection_seed"],
        max_temporal_bins=descriptor["max_temporal_bins"],
        train_targets=descriptor["train_targets"],
        validation_target=descriptor["validation_target"],
        coverage_names=tier_names,
    )
    if descriptor != expected.descriptor:
        raise ValueError("selection descriptor differs from deterministic contract")
    if rows != list(expected.rows):
        raise ValueError("selection rows differ from deterministic contract")
    if coverages != expected.coverages:
        raise ValueError("selection coverages differ from deterministic contract")
    return expected


def write_selection_artifacts(
    output_dir: str | Path,
    artifacts: SelectionArtifacts,
    *,
    data_manifest: Mapping[str, Any],
    episode_strata: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Path]:
    """Atomically write rows first and the selection descriptor last."""

    validated = validate_selection_artifacts(
        artifacts, data_manifest=data_manifest, episode_strata=episode_strata
    )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    paths["episode_split"] = write_json_atomic(
        root / EPISODE_SPLIT_FILENAME, validated.episode_split
    )
    paths["rows"] = write_text_atomic(
        root / SELECTION_ROWS_FILENAME, _canonical_jsonl(validated.rows)
    )
    for tier in validated.descriptor["coverage_tiers"]:
        name = tier["tier"]
        paths[f"coverage:{name}"] = write_json_atomic(
            root / tier["coverage_file"], validated.coverages[name]
        )
    # This is the commit marker: all objects it names already exist.
    paths["descriptor"] = write_json_atomic(
        root / SELECTION_DESCRIPTOR_FILENAME, validated.descriptor
    )
    return paths


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object contains duplicate key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"selection artifact is unreadable: {path}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"selection artifact must contain a JSON object: {path}")
    return payload


def _load_rows(path: Path) -> list[dict[str, Any]]:
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"selection rows are unreadable: {path}") from error
    if raw and not raw.endswith("\n"):
        raise ValueError("selection rows JSONL must end with a newline")
    rows: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(raw.splitlines(), start=1):
            if not line:
                raise ValueError("selection rows JSONL contains a blank line")
            value = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
            if not isinstance(value, dict):
                raise TypeError(
                    f"selection row {line_number} must contain a JSON object"
                )
            rows.append(value)
    except json.JSONDecodeError as error:
        raise ValueError("selection rows JSONL is invalid") from error
    if raw != _canonical_jsonl(rows):
        raise ValueError("selection rows JSONL is not canonically serialized")
    return rows


def load_selection_artifacts(
    output_dir: str | Path,
    *,
    data_manifest: Mapping[str, Any],
    episode_strata: Sequence[Mapping[str, Any]] | None = None,
) -> SelectionArtifacts:
    """Load a committed artifact directory and validate it fail-closed."""

    root = Path(output_dir)
    descriptor = _load_json(root / SELECTION_DESCRIPTOR_FILENAME)
    prevalidated = _prevalidate_descriptor(descriptor)
    split = _load_json(root / prevalidated["episode_split_file"])
    rows = _load_rows(root / prevalidated["rows_file"])
    coverages: dict[str, dict[str, Any]] = {}
    tiers = prevalidated["coverage_tiers"]
    if not isinstance(tiers, list):
        raise TypeError("coverage_tiers must be a list")
    for value in tiers:
        if not isinstance(value, Mapping):
            raise TypeError("coverage tier must be a mapping")
        tier = dict(value)
        _exact_keys(tier, _TIER_KEYS, name="coverage tier")
        coverages[tier["tier"]] = _load_json(root / tier["coverage_file"])
    artifacts = SelectionArtifacts(
        episode_split=split,
        descriptor=descriptor,
        rows=tuple(rows),
        coverages=coverages,
    )
    return validate_selection_artifacts(
        artifacts, data_manifest=data_manifest, episode_strata=episode_strata
    )


def selected_rows_for_coverage(
    artifacts: SelectionArtifacts,
    *,
    tier: str,
) -> tuple[Mapping[str, Any], ...]:
    """Return the exact canonical rows activated by one validated coverage."""

    if not isinstance(tier, str) or tier not in artifacts.coverages:
        raise ValueError(f"unknown selection coverage tier: {tier}")
    coverage = artifacts.coverages[tier]
    indices = set(coverage["active_cohort_indices"])
    rows = tuple(row for row in artifacts.rows if row["cohort_index"] in indices)
    if len(rows) != coverage["sample_count"]:
        raise ValueError("coverage sample_count does not match selection rows")
    if _sample_ids_sha256([row["sample_id"] for row in rows]) != coverage[
        "sample_ids_sha256"
    ]:
        raise ValueError("coverage sample IDs do not match selection rows")
    return rows


__all__ = [
    "DEFAULT_COVERAGE_NAMES",
    "DEFAULT_MAX_TEMPORAL_BINS",
    "DEFAULT_TRAIN_TARGETS",
    "DEFAULT_VALIDATION_TARGET",
    "EPISODE_SPLIT_FILENAME",
    "EXPLICIT_STRATUM_RESOLVER",
    "LABEL_COVERAGE_KIND",
    "LABEL_COVERAGE_SCHEMA_VERSION",
    "LABEL_SELECTION_ALGORITHM",
    "LABEL_SELECTION_KIND",
    "LABEL_SELECTION_ROW_KIND",
    "LABEL_SELECTION_ROW_SCHEMA_VERSION",
    "LABEL_SELECTION_SCHEMA_VERSION",
    "SELECTION_DESCRIPTOR_FILENAME",
    "SELECTION_ROWS_FILENAME",
    "STRATIFIED_EPISODE_SPLIT_ALGORITHM",
    "STRATIFIED_EPISODE_SPLIT_KIND",
    "STRATIFIED_EPISODE_SPLIT_SCHEMA_VERSION",
    "SelectionArtifacts",
    "build_libero_episode_strata",
    "build_selection_artifacts",
    "build_selection_from_split",
    "build_stratified_episode_split",
    "load_selection_artifacts",
    "selected_rows_for_coverage",
    "validate_selection_artifacts",
    "validate_stratified_episode_split",
    "write_selection_artifacts",
]
