"""Deterministic, resumable orchestration for Stage 2 Gate labels.

This module deliberately does not construct models or datasets.  It consumes
an already validated :class:`LabelArtifactContext`, a strict dataset, and a
frozen inference model.  Planning is metadata-only; video decoding happens
only for rows in chunks that do not already have a complete artifact.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch

from .artifacts import (
    CHUNK_PLAN_ALGORITHM,
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


_IDENTITY_KEYS = frozenset(
    {
        "global_sample_index",
        "dataset_index",
        "episode_index",
        "frame_index",
        "dataset_frame_index",
    }
)


@dataclass(frozen=True, slots=True)
class PlannedLabelSample:
    """One immutable semantic frame in a Stage 2 label plan."""

    sample_id: str
    identity: Mapping[str, int]
    shard_index: int

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

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        if not self.samples:
            raise ValueError("a label chunk plan must contain at least one sample")
        if any(sample.shard_index != self.shard_index for sample in self.samples):
            raise ValueError("a label chunk plan cannot mix shards")
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


def enumerate_label_samples(
    context: LabelArtifactContext,
) -> tuple[PlannedLabelSample, ...]:
    """Expand validated manifest boundaries into exact semantic identities."""

    context = _require_context(context)
    num_shards = int(context.contract["num_shards"])
    samples: list[PlannedLabelSample] = []
    root_offset = 0
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
                samples.append(
                    PlannedLabelSample(
                        sample_id=stable_id,
                        identity=normalized,
                        shard_index=shard_for_sample_id(
                            stable_id, num_shards=num_shards
                        ),
                    )
                )
        root_offset += int(root["num_frames"])

    manifest_count = int(context.data_manifest["num_frames"])
    if len(samples) != manifest_count:
        raise RuntimeError(
            "label sample enumeration disagrees with data manifest num_frames"
        )
    sample_ids = [sample.sample_id for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("label sample enumeration produced duplicate sample IDs")
    global_indices = [sample.global_sample_index for sample in samples]
    if global_indices != list(range(manifest_count)):
        raise RuntimeError("label sample enumeration is not globally contiguous")
    return tuple(samples)


def plan_label_chunks(
    *,
    context: LabelArtifactContext,
    output_dir: str | Path,
    chunk_size: int,
    shard_indices: Sequence[int] | None = None,
) -> tuple[LabelChunkPlan, ...]:
    """Build deterministic fixed-size chunks without reading the dataset."""

    context = _require_context(context)
    size = _positive_integer(chunk_size, field="chunk_size")
    contract_size = int(context.contract["chunk_size"])
    if size != contract_size:
        raise ValueError(
            "chunk_size disagrees with the immutable label contract: "
            f"requested={size}, contract={contract_size}"
        )
    if context.contract["chunk_plan_algorithm"] != CHUNK_PLAN_ALGORITHM:
        raise ValueError("unsupported label chunk plan algorithm")
    num_shards = int(context.contract["num_shards"])
    selected = _selected_shards(shard_indices, num_shards=num_shards)
    selected_set = set(selected)
    grouped: dict[int, list[PlannedLabelSample]] = {
        shard_index: [] for shard_index in selected
    }
    for sample in enumerate_label_samples(context):
        if sample.shard_index in selected_set:
            grouped[sample.shard_index].append(sample)

    root = Path(output_dir).expanduser().resolve()
    plans: list[LabelChunkPlan] = []
    for shard_index in selected:
        shard_samples = sorted(
            grouped[shard_index], key=lambda sample: sample.sample_id
        )
        for chunk_index, start in enumerate(range(0, len(shard_samples), size)):
            plans.append(
                LabelChunkPlan(
                    path=(
                        root
                        / f"shard-{shard_index:05d}"
                        / f"chunk-{chunk_index:08d}.json"
                    ),
                    shard_index=shard_index,
                    chunk_index=chunk_index,
                    samples=tuple(shard_samples[start : start + size]),
                )
            )
    return tuple(plans)


def _path_present(path: Path) -> bool:
    """Treat broken symlinks as existing artifacts, too."""

    return os.path.lexists(path)


def _load_existing_chunk(
    plan: LabelChunkPlan,
    *,
    context: LabelArtifactContext,
) -> dict[str, Any]:
    payload = load_complete_label_chunk_from_context(
        plan.path,
        context=context,
        planned_sample_ids=plan.planned_sample_ids,
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


def run_label_job(
    model: Any,
    dataset: Any,
    *,
    context: LabelArtifactContext,
    output_dir: str | Path,
    chunk_size: int,
    shard_indices: Sequence[int] | None = None,
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

    plans = plan_label_chunks(
        context=context,
        output_dir=output_dir,
        chunk_size=chunk_size,
        shard_indices=shard_indices,
    )
    written = 0
    resumed = 0
    inferred = 0
    for plan in plans:
        # Check every chunk boundary, including chunks resumed from disk.
        # The fixed callback is activated after full manifest verification.
        if source_guard is not None:
            source_guard()

        if _path_present(plan.path):
            _load_existing_chunk(plan, context=context)
            resumed += 1
            continue

        rows: list[dict[str, Any]] = []
        for planned in plan.samples:
            sample = dataset[planned.global_sample_index]
            if not isinstance(sample, Mapping):
                raise TypeError("Stage 2 label dataset sample must be a mapping")
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
        if source_guard is not None:
            source_guard()

        published = publish_label_chunk_atomic_from_context(
            plan.path,
            context=context,
            shard_index=plan.shard_index,
            chunk_index=plan.chunk_index,
            planned_sample_ids=plan.planned_sample_ids,
            rows=rows,
        )
        # Publication is create-if-absent. A losing worker must validate the
        # winner just as strictly as a chunk found during initial resume.
        _load_existing_chunk(plan, context=context)
        if published:
            written += 1
        else:
            resumed += 1

    return LabelJobResult(
        planned_chunk_count=len(plans),
        planned_sample_count=sum(len(plan.samples) for plan in plans),
        written_chunk_count=written,
        resumed_chunk_count=resumed,
        inferred_sample_count=inferred,
        chunk_paths=tuple(plan.path for plan in plans),
    )


__all__ = [
    "LabelChunkPlan",
    "LabelJobDependencies",
    "LabelJobResult",
    "PlannedLabelSample",
    "enumerate_label_samples",
    "plan_label_chunks",
    "run_label_job",
]
