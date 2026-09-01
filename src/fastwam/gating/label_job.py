"""Deterministic, resumable orchestration for Stage 2 Gate labels.

This module deliberately does not construct models or datasets.  It consumes
an already validated :class:`LabelArtifactContext`, a strict dataset, and a
frozen inference model.  Planning is metadata-only; video decoding happens
only for rows in chunks that do not already have a complete artifact.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import sqlite3
import tempfile
from types import MappingProxyType
from typing import Any

import torch

from .artifacts import (
    CHUNK_PLAN_ALGORITHM,
    COHORT_CHUNK_PLAN_ALGORITHM,
    LabelArtifactContext,
    build_label_row_from_context,
    load_complete_label_chunk_from_context,
    publish_label_chunk_atomic_from_context,
    shard_for_sample_id,
)
from .contracts import (
    derive_pair_seeds,
    sample_id_from_lookup,
    validate_sample_identity_with_lookup,
)
from .inference import PairedActionRollouts, run_paired_action_rollouts
from .labels import GateLabelStatistics, paired_gate_label_statistics
from .selection import SelectionArtifacts, selected_rows_for_coverage


_IDENTITY_KEYS = frozenset(
    {
        "global_sample_index",
        "dataset_index",
        "episode_index",
        "frame_index",
        "dataset_frame_index",
    }
)

_PLAN_INDEX_INSERT_BATCH_SIZE = 4096
_PLAN_INDEX_CACHE_KIB = 32 * 1024


@dataclass(frozen=True, slots=True)
class PlannedLabelSample:
    """One immutable semantic frame in a Stage 2 label plan."""

    sample_id: str
    identity: Mapping[str, int]
    shard_index: int
    cohort_index: int | None = None

    def __post_init__(self) -> None:
        normalized = dict(self.identity)
        if set(normalized) != _IDENTITY_KEYS:
            raise ValueError("planned sample identity has unexpected keys")
        for field, value in normalized.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"planned sample identity {field} must be an integer")
            if value < 0:
                raise ValueError(
                    f"planned sample identity {field} must be non-negative"
                )
        object.__setattr__(self, "identity", MappingProxyType(normalized))
        if self.cohort_index is not None:
            if isinstance(self.cohort_index, bool) or not isinstance(
                self.cohort_index, int
            ):
                raise TypeError("planned sample cohort_index must be an integer")
            if self.cohort_index < 0:
                raise ValueError(
                    "planned sample cohort_index must be non-negative"
                )

    @property
    def global_sample_index(self) -> int:
        return self.identity["global_sample_index"]


@dataclass(frozen=True, slots=True)
class LabelChunkPlan:
    """A fixed, self-contained chunk assignment within one stable shard."""

    path: Path
    shard_index: int
    chunk_index: int
    samples: tuple[PlannedLabelSample, ...]
    cohort_index: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        if not self.samples:
            raise ValueError("a label chunk plan must contain at least one sample")
        if any(sample.shard_index != self.shard_index for sample in self.samples):
            raise ValueError("a label chunk plan cannot mix shards")
        if any(
            sample.cohort_index != self.cohort_index for sample in self.samples
        ):
            raise ValueError("a label chunk plan cannot mix cohorts")
        if self.cohort_index is not None:
            if isinstance(self.cohort_index, bool) or not isinstance(
                self.cohort_index, int
            ):
                raise TypeError("label chunk cohort_index must be an integer")
            if self.cohort_index < 0:
                raise ValueError("label chunk cohort_index must be non-negative")
        sample_ids = tuple(sample.sample_id for sample in self.samples)
        if sample_ids != tuple(sorted(set(sample_ids))):
            raise ValueError("chunk plan sample IDs must be sorted and unique")

    @property
    def planned_sample_ids(self) -> tuple[str, ...]:
        return tuple(sample.sample_id for sample in self.samples)


@dataclass(frozen=True, slots=True)
class LabelJobDependencies:
    """Injectable numerical kernels; persistence and validation stay fixed."""

    run_rollouts: Callable[..., PairedActionRollouts] = run_paired_action_rollouts
    compute_statistics: Callable[..., GateLabelStatistics] = (
        paired_gate_label_statistics
    )


@dataclass(frozen=True, slots=True)
class LabelJobResult:
    """Deterministic summary of one label-job invocation."""

    planned_chunk_count: int
    planned_sample_count: int
    written_chunk_count: int
    resumed_chunk_count: int
    inferred_sample_count: int
    chunk_paths: tuple[Path, ...]


def _require_context(context: LabelArtifactContext) -> LabelArtifactContext:
    if not isinstance(context, LabelArtifactContext):
        raise TypeError("context must be a LabelArtifactContext")
    return context


def _positive_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value < 1:
        raise ValueError(f"{field} must be positive")
    return int(value)


def _selected_shards(
    shard_indices: Sequence[int] | None,
    *,
    num_shards: int,
) -> tuple[int, ...]:
    if shard_indices is None:
        return tuple(range(num_shards))
    if isinstance(shard_indices, (str, bytes)) or not isinstance(
        shard_indices, Sequence
    ):
        raise TypeError("shard_indices must be a sequence of integers")
    normalized: list[int] = []
    for value in shard_indices:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("every shard index must be an integer")
        if not 0 <= value < num_shards:
            raise ValueError("shard index is out of range")
        normalized.append(int(value))
    if not normalized:
        raise ValueError("shard_indices must not be empty")
    if len(normalized) != len(set(normalized)):
        raise ValueError("shard_indices must be unique")
    return tuple(sorted(normalized))


def _validated_selection_request(
    context: LabelArtifactContext,
    *,
    selection_artifacts: SelectionArtifacts | None,
    coverage_tier: str | None,
) -> tuple[SelectionArtifacts | None, str | None]:
    has_artifacts = selection_artifacts is not None
    has_tier = coverage_tier is not None
    if has_artifacts != has_tier:
        raise ValueError(
            "selection_artifacts and coverage_tier must be provided together"
        )
    if not has_artifacts:
        return None, None
    if not isinstance(selection_artifacts, SelectionArtifacts):
        raise TypeError("selection_artifacts must be SelectionArtifacts")
    if not isinstance(coverage_tier, str) or not coverage_tier:
        raise TypeError("coverage_tier must be a non-empty string")
    if (
        context.contract["chunk_plan_algorithm"]
        != COHORT_CHUNK_PLAN_ALGORITHM
    ):
        raise ValueError("unsupported label chunk plan algorithm")
    descriptor = selection_artifacts.descriptor
    if not isinstance(descriptor, Mapping):
        raise TypeError("selection artifact descriptor must be a mapping")
    for descriptor_field, contract_field in (
        ("data_manifest_sha256", "data_manifest_sha256"),
        ("episode_split_sha256", "episode_split_sha256"),
        ("episode_assignment_sha256", "episode_assignment_sha256"),
    ):
        if descriptor.get(descriptor_field) != context.contract[contract_field]:
            raise ValueError(
                "selection artifacts disagree with the label context: "
                f"{descriptor_field}"
            )
    selected_rows_for_coverage(selection_artifacts, tier=coverage_tier)
    return selection_artifacts, coverage_tier


def iter_label_samples(
    context: LabelArtifactContext,
    *,
    selection_artifacts: SelectionArtifacts | None = None,
    coverage_tier: str | None = None,
) -> Iterator[PlannedLabelSample]:
    """Stream exact semantic identities without materializing the whole dataset."""

    context = _require_context(context)
    selection_artifacts, coverage_tier = _validated_selection_request(
        context,
        selection_artifacts=selection_artifacts,
        coverage_tier=coverage_tier,
    )
    num_shards = int(context.contract["num_shards"])
    if selection_artifacts is not None:
        assert coverage_tier is not None
        for row in selected_rows_for_coverage(
            selection_artifacts, tier=coverage_tier
        ):
            identity = {key: row[key] for key in _IDENTITY_KEYS}
            normalized = validate_sample_identity_with_lookup(
                identity, context.episode_lookup
            )
            stable_id = sample_id_from_lookup(normalized, context.episode_lookup)
            if row.get("sample_id") != stable_id:
                raise ValueError(
                    "selection row sample_id disagrees with its semantic identity"
                )
            cohort_index = row.get("cohort_index")
            if isinstance(cohort_index, bool) or not isinstance(cohort_index, int):
                raise TypeError("selection row cohort_index must be an integer")
            yield PlannedLabelSample(
                sample_id=stable_id,
                identity=normalized,
                shard_index=shard_for_sample_id(
                    stable_id, num_shards=num_shards
                ),
                cohort_index=cohort_index,
            )
        return

    root_offset = 0
    emitted = 0
    for dataset_index, root in enumerate(context.data_manifest["dataset_roots"]):
        for boundary in root["episode_boundaries"]:
            episode_index = int(boundary["episode_index"])
            dataset_start = int(boundary["from"])
            length = int(boundary["length"])
            for frame_index in range(length):
                identity = {
                    "global_sample_index": (
                        root_offset + dataset_start + frame_index
                    ),
                    "dataset_index": dataset_index,
                    "episode_index": episode_index,
                    "frame_index": frame_index,
                    "dataset_frame_index": dataset_start + frame_index,
                }
                normalized = validate_sample_identity_with_lookup(
                    identity, context.episode_lookup
                )
                stable_id = sample_id_from_lookup(
                    normalized, context.episode_lookup
                )
                if normalized["global_sample_index"] != emitted:
                    raise RuntimeError(
                        "label sample enumeration is not globally contiguous"
                    )
                yield PlannedLabelSample(
                    sample_id=stable_id,
                    identity=normalized,
                    shard_index=shard_for_sample_id(
                        stable_id, num_shards=num_shards
                    ),
                )
                emitted += 1
        root_offset += int(root["num_frames"])

    manifest_count = int(context.data_manifest["num_frames"])
    if emitted != manifest_count:
        raise RuntimeError(
            "label sample enumeration disagrees with data manifest num_frames"
        )


def enumerate_label_samples(
    context: LabelArtifactContext,
    *,
    selection_artifacts: SelectionArtifacts | None = None,
    coverage_tier: str | None = None,
) -> tuple[PlannedLabelSample, ...]:
    """Materialize the streamed label plan for compatibility and diagnostics."""

    samples = tuple(
        iter_label_samples(
            context,
            selection_artifacts=selection_artifacts,
            coverage_tier=coverage_tier,
        )
    )
    sample_ids = [sample.sample_id for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("label sample enumeration produced duplicate sample IDs")
    return samples


def _validated_plan_request(
    *,
    context: LabelArtifactContext,
    output_dir: str | Path,
    chunk_size: int,
    shard_indices: Sequence[int] | None = None,
    selection_artifacts: SelectionArtifacts | None = None,
    coverage_tier: str | None = None,
) -> tuple[
    LabelArtifactContext,
    Path,
    int,
    tuple[int, ...],
    SelectionArtifacts | None,
    str | None,
]:
    context = _require_context(context)
    selection_artifacts, coverage_tier = _validated_selection_request(
        context,
        selection_artifacts=selection_artifacts,
        coverage_tier=coverage_tier,
    )
    size = _positive_integer(chunk_size, field="chunk_size")
    contract_size = int(context.contract["chunk_size"])
    if size != contract_size:
        raise ValueError(
            "chunk_size disagrees with the immutable label contract: "
            f"requested={size}, contract={contract_size}"
        )
    expected_algorithm = (
        CHUNK_PLAN_ALGORITHM
        if selection_artifacts is None
        else COHORT_CHUNK_PLAN_ALGORITHM
    )
    if context.contract["chunk_plan_algorithm"] != expected_algorithm:
        raise ValueError("unsupported label chunk plan algorithm")
    num_shards = int(context.contract["num_shards"])
    selected = _selected_shards(shard_indices, num_shards=num_shards)
    root = Path(output_dir).expanduser().resolve()
    return (
        context,
        root,
        size,
        selected,
        selection_artifacts,
        coverage_tier,
    )


def _insert_plan_index_batch(
    connection: sqlite3.Connection,
    rows: list[tuple[Any, ...]],
) -> None:
    if not rows:
        return
    try:
        connection.executemany(
            """
            INSERT INTO planned_samples (
                sample_id,
                cohort_index,
                shard_index,
                global_sample_index,
                dataset_index,
                episode_index,
                frame_index,
                dataset_frame_index
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    except sqlite3.IntegrityError as error:
        raise RuntimeError(
            "label sample enumeration produced duplicate sample IDs"
        ) from error
    rows.clear()


def _iter_indexed_label_chunks(
    *,
    context: LabelArtifactContext,
    root: Path,
    chunk_size: int,
    selected: tuple[int, ...],
    selection_artifacts: SelectionArtifacts | None,
    coverage_tier: str | None,
) -> Iterator[LabelChunkPlan]:
    """External-sort selected samples while keeping Python memory bounded."""

    selected_set = set(selected)
    with tempfile.TemporaryDirectory(prefix="fastwam-stage2-plan-") as temp_dir:
        index_path = Path(temp_dir) / "planned-samples.sqlite3"
        connection = sqlite3.connect(index_path)
        try:
            # The B-tree is the external sort. Keep both SQLite temporary state
            # and its page cache explicitly bounded/on disk; Python retains at
            # most one insertion batch until chunk iteration starts.
            connection.execute("PRAGMA temp_store = FILE")
            connection.execute(f"PRAGMA cache_size = -{_PLAN_INDEX_CACHE_KIB}")
            connection.execute("PRAGMA journal_mode = OFF")
            connection.execute("PRAGMA synchronous = OFF")
            connection.execute(
                """
                CREATE TABLE planned_samples (
                    sample_id BLOB PRIMARY KEY,
                    cohort_index INTEGER NOT NULL,
                    shard_index INTEGER NOT NULL,
                    global_sample_index INTEGER NOT NULL,
                    dataset_index INTEGER NOT NULL,
                    episode_index INTEGER NOT NULL,
                    frame_index INTEGER NOT NULL,
                    dataset_frame_index INTEGER NOT NULL
                ) WITHOUT ROWID
                """
            )
            connection.execute(
                """
                CREATE INDEX planned_samples_plan_order
                ON planned_samples (cohort_index, shard_index, sample_id)
                """
            )

            batch: list[tuple[Any, ...]] = []
            if selection_artifacts is None:
                # Preserve the legacy call shape for existing injected and
                # monkeypatched sample iterators.
                sample_iterator = iter_label_samples(context)
            else:
                sample_iterator = iter_label_samples(
                    context,
                    selection_artifacts=selection_artifacts,
                    coverage_tier=coverage_tier,
                )
            for sample in sample_iterator:
                if (selection_artifacts is None) != (
                    sample.cohort_index is None
                ):
                    raise RuntimeError(
                        "label sample cohort binding disagrees with the plan"
                    )
                if sample.shard_index not in selected_set:
                    continue
                identity = sample.identity
                batch.append(
                    (
                        bytes.fromhex(sample.sample_id),
                        -1 if sample.cohort_index is None else sample.cohort_index,
                        sample.shard_index,
                        identity["global_sample_index"],
                        identity["dataset_index"],
                        identity["episode_index"],
                        identity["frame_index"],
                        identity["dataset_frame_index"],
                    )
                )
                if len(batch) == _PLAN_INDEX_INSERT_BATCH_SIZE:
                    _insert_plan_index_batch(connection, batch)
            _insert_plan_index_batch(connection, batch)
            connection.commit()

            cohort_rows = connection.execute(
                "SELECT DISTINCT cohort_index FROM planned_samples "
                "ORDER BY cohort_index"
            )
            cohort_indices = tuple(int(row[0]) for row in cohort_rows)
            for indexed_cohort in cohort_indices:
                cohort_index = None if indexed_cohort == -1 else indexed_cohort
                cohort_root = (
                    root
                    if cohort_index is None
                    else root / f"cohort-{cohort_index:05d}"
                )
                for shard_index in selected:
                    cursor = connection.execute(
                        """
                        SELECT
                            sample_id,
                            global_sample_index,
                            dataset_index,
                            episode_index,
                            frame_index,
                            dataset_frame_index
                        FROM planned_samples
                        WHERE cohort_index = ? AND shard_index = ?
                        ORDER BY sample_id
                        """,
                        (indexed_cohort, shard_index),
                    )
                    chunk_index = 0
                    chunk_samples: list[PlannedLabelSample] = []
                    for row in cursor:
                        sample_id_bytes = row[0]
                        if not isinstance(sample_id_bytes, bytes):
                            raise RuntimeError(
                                "label plan index sample ID is corrupt"
                            )
                        chunk_samples.append(
                            PlannedLabelSample(
                                sample_id=sample_id_bytes.hex(),
                                identity={
                                    "global_sample_index": int(row[1]),
                                    "dataset_index": int(row[2]),
                                    "episode_index": int(row[3]),
                                    "frame_index": int(row[4]),
                                    "dataset_frame_index": int(row[5]),
                                },
                                shard_index=shard_index,
                                cohort_index=cohort_index,
                            )
                        )
                        if len(chunk_samples) != chunk_size:
                            continue
                        yield LabelChunkPlan(
                            path=(
                                cohort_root
                                / f"shard-{shard_index:05d}"
                                / f"chunk-{chunk_index:08d}.json"
                            ),
                            shard_index=shard_index,
                            chunk_index=chunk_index,
                            samples=tuple(chunk_samples),
                            cohort_index=cohort_index,
                        )
                        chunk_index += 1
                        chunk_samples.clear()
                    if chunk_samples:
                        yield LabelChunkPlan(
                            path=(
                                cohort_root
                                / f"shard-{shard_index:05d}"
                                / f"chunk-{chunk_index:08d}.json"
                            ),
                            shard_index=shard_index,
                            chunk_index=chunk_index,
                            samples=tuple(chunk_samples),
                            cohort_index=cohort_index,
                        )
        finally:
            connection.close()


def iter_label_chunks(
    *,
    context: LabelArtifactContext,
    output_dir: str | Path,
    chunk_size: int,
    shard_indices: Sequence[int] | None = None,
    selection_artifacts: SelectionArtifacts | None = None,
    coverage_tier: str | None = None,
) -> Iterator[LabelChunkPlan]:
    """Stream deterministic fixed-size plans through a bounded disk sort.

    Construction is lazy. On first iteration, all manifest identities are
    validated and selected samples are indexed on local temporary storage.
    Plans are then yielded in the immutable canonical order: ascending shard,
    ascending sample ID, with at most ``chunk_size`` samples resident in a
    plan. The temporary index is removed when iteration completes or closes.
    """

    (
        validated,
        root,
        size,
        selected,
        selection_artifacts,
        coverage_tier,
    ) = _validated_plan_request(
        context=context,
        output_dir=output_dir,
        chunk_size=chunk_size,
        shard_indices=shard_indices,
        selection_artifacts=selection_artifacts,
        coverage_tier=coverage_tier,
    )
    return _iter_indexed_label_chunks(
        context=validated,
        root=root,
        chunk_size=size,
        selected=selected,
        selection_artifacts=selection_artifacts,
        coverage_tier=coverage_tier,
    )


def plan_label_chunks(
    *,
    context: LabelArtifactContext,
    output_dir: str | Path,
    chunk_size: int,
    shard_indices: Sequence[int] | None = None,
    selection_artifacts: SelectionArtifacts | None = None,
    coverage_tier: str | None = None,
) -> tuple[LabelChunkPlan, ...]:
    """Materialize chunk plans for compatibility and small diagnostics.

    Formal label generation uses :func:`iter_label_chunks` directly so it
    never retains every selected sample or plan at once.
    """

    return tuple(
        iter_label_chunks(
            context=context,
            output_dir=output_dir,
            chunk_size=chunk_size,
            shard_indices=shard_indices,
            selection_artifacts=selection_artifacts,
            coverage_tier=coverage_tier,
        )
    )


def _path_present(path: Path) -> bool:
    """Treat broken symlinks as existing artifacts, too."""

    return os.path.lexists(path)


def _load_existing_chunk(
    plan: LabelChunkPlan,
    *,
    context: LabelArtifactContext,
    selection_sha256: str | None = None,
) -> dict[str, Any]:
    artifact_binding: dict[str, Any] = {}
    if plan.cohort_index is not None:
        if selection_sha256 is None:
            raise ValueError("cohort chunk requires selection_sha256")
        artifact_binding = {
            "selection_sha256": selection_sha256,
            "cohort_index": plan.cohort_index,
        }
    payload = load_complete_label_chunk_from_context(
        plan.path,
        context=context,
        planned_sample_ids=plan.planned_sample_ids,
        **artifact_binding,
    )
    if (
        payload["shard_index"] != plan.shard_index
        or payload["chunk_index"] != plan.chunk_index
    ):
        raise ValueError(
            "label chunk coordinates do not match its deterministic output path"
        )
    return payload


def _required_tensor(sample: Mapping[str, Any], key: str) -> torch.Tensor:
    value = sample.get(key)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Stage 2 label sample {key!r} must be a torch.Tensor")
    return value


def _validate_rollouts(
    rollouts: Any,
    *,
    sample: Mapping[str, Any],
    seeds: tuple[int, ...],
    num_inference_steps: int,
) -> PairedActionRollouts:
    if not isinstance(rollouts, PairedActionRollouts):
        raise TypeError("run_rollouts must return PairedActionRollouts")
    action = _required_tensor(sample, "action")
    video = _required_tensor(sample, "video")
    if action.ndim != 2 or video.ndim != 4:
        raise ValueError("Stage 2 label sample action/video rank is invalid")
    expected_shape = (len(seeds), 1, int(action.shape[0]), int(action.shape[1]))
    if tuple(rollouts.action_wo.shape) != expected_shape:
        raise ValueError("paired wo rollout shape disagrees with the planned sample")
    if tuple(rollouts.action_w.shape) != expected_shape:
        raise ValueError("paired w rollout shape disagrees with the planned sample")
    if rollouts.seeds != seeds:
        raise ValueError("paired rollout seeds disagree with the label plan")
    if rollouts.action_horizon != action.shape[0]:
        raise ValueError("paired rollout action horizon disagrees with the sample")
    if rollouts.num_video_frames != video.shape[1]:
        raise ValueError("paired rollout video frame count disagrees with the sample")
    if rollouts.num_inference_steps != num_inference_steps:
        raise ValueError("paired rollout inference steps disagree with the contract")
    return rollouts


def _statistics_scalar(value: Any, *, field: str) -> Any:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Gate label statistics {field} must be a tensor")
    if value.numel() != 1:
        raise ValueError(f"Gate label statistics {field} must contain one value")
    return value.detach().to(device="cpu").reshape(()).item()


def _row_for_sample(
    planned: PlannedLabelSample,
    sample: Mapping[str, Any],
    *,
    model: Any,
    context: LabelArtifactContext,
    dependencies: LabelJobDependencies,
) -> dict[str, Any]:
    raw_identity = sample.get("sample_identity")
    if not isinstance(raw_identity, Mapping):
        raise ValueError("Stage 2 label sample has no sample_identity mapping")
    actual_identity = validate_sample_identity_with_lookup(
        raw_identity, context.episode_lookup
    )
    if actual_identity != dict(planned.identity):
        raise ValueError(
            "dataset sample_identity disagrees with the deterministic label plan "
            f"at global index {planned.global_sample_index}"
        )
    actual_sample_id = sample_id_from_lookup(
        actual_identity, context.episode_lookup
    )
    if actual_sample_id != planned.sample_id:
        raise ValueError("dataset sample_id disagrees with the deterministic plan")

    seeds = tuple(
        derive_pair_seeds(
            sample_id_sha256=planned.sample_id,
            base_seed=context.contract["base_seed"],
            num_pairs=context.contract["num_seed_pairs"],
        )
    )
    rollouts = dependencies.run_rollouts(
        model,
        sample,
        seeds=seeds,
        num_inference_steps=context.contract["num_inference_steps"],
        sigma_shift=context.contract["sigma_shift"],
        rand_device=context.contract["rand_device"],
        tiled=context.contract["tiled"],
    )
    rollouts = _validate_rollouts(
        rollouts,
        sample=sample,
        seeds=seeds,
        num_inference_steps=context.contract["num_inference_steps"],
    )

    action = _required_tensor(sample, "action")
    action_is_pad = _required_tensor(sample, "action_is_pad")
    action_dim_is_pad = _required_tensor(sample, "action_dim_is_pad")
    statistics = dependencies.compute_statistics(
        action_wo=rollouts.action_wo,
        action_w=rollouts.action_w,
        target_action=action.unsqueeze(0),
        action_is_pad=action_is_pad.unsqueeze(0),
        action_dim_is_pad=action_dim_is_pad,
        relative_margin=context.contract["relative_margin"],
        relative_gain_epsilon=context.contract["relative_gain_epsilon"],
    )
    if not isinstance(statistics, GateLabelStatistics):
        raise TypeError("compute_statistics must return GateLabelStatistics")

    return build_label_row_from_context(
        context=context,
        identity=actual_identity,
        e0=float(_statistics_scalar(statistics.e0, field="e0")),
        e10=float(_statistics_scalar(statistics.e10, field="e10")),
        relative_gain=float(
            _statistics_scalar(statistics.relative_gain, field="relative_gain")
        ),
        label=bool(_statistics_scalar(statistics.label, field="label")),
        sample_weight=float(
            _statistics_scalar(statistics.sample_weight, field="sample_weight")
        ),
        num_video_frames=rollouts.num_video_frames,
    )


def _check_source_guard_for_plan(
    source_guard: Callable[[], None] | None,
    plan: LabelChunkPlan,
) -> None:
    if source_guard is None:
        return
    scoped_check = getattr(source_guard, "check_sample_identities", None)
    if scoped_check is None:
        # Backward compatibility for injected no-argument callbacks.
        source_guard()
        return
    if not callable(scoped_check):
        raise TypeError("source_guard.check_sample_identities must be callable")
    scoped_check(sample.identity for sample in plan.samples)


def run_label_job(
    model: Any,
    dataset: Any,
    *,
    context: LabelArtifactContext,
    output_dir: str | Path,
    chunk_size: int,
    shard_indices: Sequence[int] | None = None,
    selection_artifacts: SelectionArtifacts | None = None,
    coverage_tier: str | None = None,
    dependencies: LabelJobDependencies | None = None,
    source_guard: Callable[[], None] | None = None,
) -> LabelJobResult:
    """Generate every missing chunk and strictly resume complete chunks.

    Any pre-existing destination is treated as immutable: it must validate
    against the exact planned sample IDs and path coordinates or the job fails
    before reading that chunk's samples.  Rows are accumulated in memory and
    published only through the shared atomic chunk writer.
    """

    context = _require_context(context)
    selection_artifacts, coverage_tier = _validated_selection_request(
        context,
        selection_artifacts=selection_artifacts,
        coverage_tier=coverage_tier,
    )
    selection_sha256 = (
        None
        if selection_artifacts is None
        else selection_artifacts.descriptor["selection_sha256"]
    )
    operations = dependencies or LabelJobDependencies()
    if not isinstance(operations, LabelJobDependencies):
        raise TypeError("dependencies must be a LabelJobDependencies")
    if not callable(operations.run_rollouts) or not callable(
        operations.compute_statistics
    ):
        raise TypeError("label job dependencies must be callable")
    if source_guard is not None and not callable(source_guard):
        raise TypeError("source_guard must be callable")
    try:
        dataset_length = len(dataset)
    except Exception as error:
        raise TypeError("dataset must provide len()") from error
    manifest_length = int(context.data_manifest["num_frames"])
    if dataset_length != manifest_length:
        raise ValueError(
            "Stage 2 label dataset length disagrees with the data manifest: "
            f"dataset={dataset_length}, manifest={manifest_length}"
        )

    plans = iter_label_chunks(
        context=context,
        output_dir=output_dir,
        chunk_size=chunk_size,
        shard_indices=shard_indices,
        selection_artifacts=selection_artifacts,
        coverage_tier=coverage_tier,
    )
    written = 0
    resumed = 0
    inferred = 0
    planned_chunks = 0
    planned_samples = 0
    chunk_paths: list[Path] = []
    try:
        for plan in plans:
            planned_chunks += 1
            planned_samples += len(plan.samples)
            chunk_paths.append(plan.path)
            # The formal SourceStatGuard checks globals plus only the episodes
            # in this chunk. Injected legacy callbacks retain their no-argument
            # API.
            _check_source_guard_for_plan(source_guard, plan)

            if _path_present(plan.path):
                _load_existing_chunk(
                    plan,
                    context=context,
                    selection_sha256=selection_sha256,
                )
                _check_source_guard_for_plan(source_guard, plan)
                resumed += 1
                continue

            rows: list[dict[str, Any]] = []
            for planned in plan.samples:
                sample = dataset[planned.global_sample_index]
                if not isinstance(sample, Mapping):
                    raise TypeError(
                        "Stage 2 label dataset sample must be a mapping"
                    )
                rows.append(
                    _row_for_sample(
                        planned,
                        sample,
                        model=model,
                        context=context,
                        dependencies=operations,
                    )
                )
                inferred += 1

            # Inference can be long-running. Never publish rows if a selected
            # source changed while this chunk was being computed.
            _check_source_guard_for_plan(source_guard, plan)

            artifact_binding: dict[str, Any] = {}
            if plan.cohort_index is not None:
                artifact_binding = {
                    "selection_sha256": selection_sha256,
                    "cohort_index": plan.cohort_index,
                }
            published = publish_label_chunk_atomic_from_context(
                plan.path,
                context=context,
                shard_index=plan.shard_index,
                chunk_index=plan.chunk_index,
                planned_sample_ids=plan.planned_sample_ids,
                rows=rows,
                **artifact_binding,
            )
            # Publication is create-if-absent. A losing worker must validate
            # the winner just as strictly as a chunk found during initial
            # resume.
            _load_existing_chunk(
                plan,
                context=context,
                selection_sha256=selection_sha256,
            )
            if published:
                written += 1
            else:
                resumed += 1
    finally:
        close_plans = getattr(plans, "close", None)
        if callable(close_plans):
            close_plans()

    return LabelJobResult(
        planned_chunk_count=planned_chunks,
        planned_sample_count=planned_samples,
        written_chunk_count=written,
        resumed_chunk_count=resumed,
        inferred_sample_count=inferred,
        chunk_paths=tuple(chunk_paths),
    )


__all__ = [
    "LabelChunkPlan",
    "LabelJobDependencies",
    "LabelJobResult",
    "PlannedLabelSample",
    "enumerate_label_samples",
    "iter_label_chunks",
    "iter_label_samples",
    "plan_label_chunks",
    "run_label_job",
]
