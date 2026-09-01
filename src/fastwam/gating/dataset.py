"""Fail-closed, current-only dataset for Stage 2 Gate training."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any, Protocol, runtime_checkable

import torch

from .artifacts import LABEL_ROW_KIND, LABEL_ROW_SCHEMA_VERSION
from .contracts import (
    EpisodeSplitLookup,
    build_episode_split_lookup,
    require_sha256,
    sample_id_from_lookup,
    split_for_identity,
    validate_sample_identity_with_lookup,
)


_SPLITS = frozenset({"train", "validation"})


_GATE_INPUT_SCHEMA_VERSION = 1
_GATE_INPUT_KEYS = frozenset(
    {"input_image", "context", "context_mask", "proprio", "sample_identity"}
)


@runtime_checkable
class GateInputDataset(Protocol):
    """Minimal strict current-query source accepted by Stage2GateDataset."""

    gate_input_schema_version: int
    strict_data_mode: bool

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class _GateRow:
    """The Gate-safe subset of one already-validated label artifact row."""

    global_sample_index: int
    dataset_index: int
    episode_index: int
    frame_index: int
    dataset_frame_index: int
    sample_id: str
    split: str
    label: bool
    sample_weight: float
    contract_sha256: str

    def identity(self) -> dict[str, int]:
        return {
            "global_sample_index": self.global_sample_index,
            "dataset_index": self.dataset_index,
            "episode_index": self.episode_index,
            "frame_index": self.frame_index,
            "dataset_frame_index": self.dataset_frame_index,
        }


def _manifest_identities(
    data_manifest: Mapping[str, Any],
) -> Iterator[dict[str, int]]:
    """Expand validated manifest boundaries without reading video samples."""

    root_offset = 0
    for dataset_index, root in enumerate(data_manifest["dataset_roots"]):
        for boundary in root["episode_boundaries"]:
            episode_index = int(boundary["episode_index"])
            dataset_start = int(boundary["from"])
            length = int(boundary["length"])
            for frame_index in range(length):
                yield {
                    "global_sample_index": (
                        root_offset + dataset_start + frame_index
                    ),
                    "dataset_index": dataset_index,
                    "episode_index": episode_index,
                    "frame_index": frame_index,
                    "dataset_frame_index": dataset_start + frame_index,
                }
        root_offset += int(root["num_frames"])


def _gate_row(
    row: Mapping[str, Any],
    *,
    data_manifest: Mapping[str, Any],
    episode_split: Mapping[str, Any],
    lookup: EpisodeSplitLookup,
) -> _GateRow:
    if not isinstance(row, Mapping):
        raise TypeError("label row must be a mapping")
    if row.get("schema_version") != LABEL_ROW_SCHEMA_VERSION:
        raise ValueError("unsupported label row schema_version")
    if row.get("kind") != LABEL_ROW_KIND:
        raise ValueError("unsupported label row kind")

    try:
        identity = {
            "global_sample_index": row["global_sample_index"],
            "dataset_index": row["dataset_index"],
            "episode_index": row["episode_id"],
            "frame_index": row["frame_id"],
            "dataset_frame_index": row["dataset_frame_index"],
        }
        normalized = validate_sample_identity_with_lookup(identity, lookup)
        stable_id = require_sha256(row["sample_id"], field="label row sample_id")
        contract_sha256 = require_sha256(
            row["contract_sha256"], field="label row contract_sha256"
        )
        row_split = row["split"]
        label = row["label"]
        raw_weight = row["sample_weight"]
    except KeyError as error:
        raise ValueError(
            f"label row is missing required field {error.args[0]!r}"
        ) from error

    expected_id = sample_id_from_lookup(normalized, lookup)
    if stable_id != expected_id:
        raise ValueError("label row sample_id disagrees with its sample identity")
    expected_split = split_for_identity(
        episode_split,
        data_manifest,
        normalized,
        lookup=lookup,
    )
    if row_split != expected_split:
        raise ValueError("label row split disagrees with its episode assignment")
    if not isinstance(label, bool):
        raise TypeError("label row label must be bool")
    if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
        raise TypeError("label row sample_weight must be a number")
    sample_weight = float(raw_weight)
    if not math.isfinite(sample_weight) or sample_weight <= 0.0:
        raise ValueError("label row sample_weight must be finite and positive")

    return _GateRow(
        global_sample_index=normalized["global_sample_index"],
        dataset_index=normalized["dataset_index"],
        episode_index=normalized["episode_index"],
        frame_index=normalized["frame_index"],
        dataset_frame_index=normalized["dataset_frame_index"],
        sample_id=stable_id,
        split=expected_split,
        label=label,
        sample_weight=sample_weight,
        contract_sha256=contract_sha256,
    )


def _required_tensor(sample: Mapping[str, Any], name: str) -> torch.Tensor:
    try:
        value = sample[name]
    except KeyError as error:
        raise ValueError(f"Gate input sample is missing {name!r}") from error
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Gate input sample {name!r} must be a tensor")
    return value


class Stage2GateDataset(torch.utils.data.Dataset):
    """Join strict current-only Gate inputs to validated Stage 2 labels.

    ``label_rows`` must come from the full artifact validation path (for
    example :func:`fastwam.gating.artifacts.validate_label_row`). This class
    rechecks the join-critical metadata and strips the E0/E10 generation
    evidence; it does not replace validation of the complete label artifact.

    Construction performs an O(N) metadata-only coverage check. Video samples
    remain lazy and their identities are revalidated whenever they are read.
    Only current-query inputs are retained or returned; label-only E0/E10 data
    and all action/future tensors are deliberately absent from this dataset.
    """

    def __init__(
        self,
        robot_video_dataset: GateInputDataset,
        *,
        label_rows: Sequence[Mapping[str, Any]],
        data_manifest: Mapping[str, Any],
        episode_split: Mapping[str, Any],
        split: str,
        expected_sample_ids: Sequence[str] | None = None,
    ) -> None:
        if not isinstance(robot_video_dataset, GateInputDataset):
            raise TypeError("robot_video_dataset must implement GateInputDataset")
        if robot_video_dataset.gate_input_schema_version != _GATE_INPUT_SCHEMA_VERSION:
            raise ValueError("unsupported Gate input schema version")
        if not bool(getattr(robot_video_dataset, "strict_data_mode", False)):
            raise ValueError("Gate input dataset must enable strict_data_mode")
        if split not in _SPLITS:
            raise ValueError("split must be 'train' or 'validation'")
        if isinstance(label_rows, (str, bytes)) or not isinstance(
            label_rows, Sequence
        ):
            raise TypeError("label_rows must be a sequence of validated rows")

        normalized_expected_ids: tuple[str, ...] | None = None
        if expected_sample_ids is not None:
            if isinstance(expected_sample_ids, (str, bytes)) or not isinstance(
                expected_sample_ids, Sequence
            ):
                raise TypeError("expected_sample_ids must be a sequence or None")
            if not expected_sample_ids:
                raise ValueError("expected_sample_ids must be non-empty")
            normalized_expected_ids = tuple(
                require_sha256(value, field="expected_sample_ids entry")
                for value in expected_sample_ids
            )
            if normalized_expected_ids != tuple(
                sorted(set(normalized_expected_ids))
            ):
                raise ValueError(
                    "expected_sample_ids must be sorted and unique"
                )

        lookup = build_episode_split_lookup(episode_split, data_manifest)
        manifest_size = int(data_manifest["num_frames"])
        if len(robot_video_dataset) != manifest_size:
            raise ValueError(
                "Gate input dataset length disagrees with the data manifest: "
                f"dataset={len(robot_video_dataset)}, manifest={manifest_size}"
            )

        rows: list[_GateRow]
        if normalized_expected_ids is None:
            if len(label_rows) != manifest_size:
                raise ValueError(
                    "label row coverage must be complete: "
                    f"expected {manifest_size}, got {len(label_rows)}"
                )

            ordered_rows: list[_GateRow | None] = [None] * manifest_size
            sample_ids: set[str] = set()
            contract_hashes: set[str] = set()
            for artifact_row in label_rows:
                row = _gate_row(
                    artifact_row,
                    data_manifest=data_manifest,
                    episode_split=episode_split,
                    lookup=lookup,
                )
                if ordered_rows[row.global_sample_index] is not None:
                    raise ValueError(
                        "label rows contain duplicate global_sample_index values"
                    )
                if row.sample_id in sample_ids:
                    raise ValueError(
                        "label rows contain duplicate sample_id values"
                    )
                ordered_rows[row.global_sample_index] = row
                sample_ids.add(row.sample_id)
                contract_hashes.add(row.contract_sha256)
            if len(contract_hashes) != 1:
                raise ValueError("label rows mix multiple label contracts")

            rows = []
            for expected, row in zip(
                _manifest_identities(data_manifest), ordered_rows, strict=True
            ):
                if row is None:
                    raise ValueError(
                        "label row coverage has a missing sample identity at "
                        f"global index {expected['global_sample_index']}"
                    )
                if row.identity() != expected:
                    raise ValueError(
                        "label row coverage has a missing or drifted sample identity "
                        f"at global index {expected['global_sample_index']}"
                    )
                rows.append(row)
        else:
            subset_rows: list[_GateRow] = []
            global_indices: set[int] = set()
            sample_ids = set()
            contract_hashes = set()
            for artifact_row in label_rows:
                row = _gate_row(
                    artifact_row,
                    data_manifest=data_manifest,
                    episode_split=episode_split,
                    lookup=lookup,
                )
                if row.global_sample_index in global_indices:
                    raise ValueError(
                        "label rows contain duplicate global_sample_index values"
                    )
                if row.sample_id in sample_ids:
                    raise ValueError(
                        "label rows contain duplicate sample_id values"
                    )
                global_indices.add(row.global_sample_index)
                sample_ids.add(row.sample_id)
                contract_hashes.add(row.contract_sha256)
                subset_rows.append(row)

            expected_id_set = set(normalized_expected_ids)
            if sample_ids != expected_id_set:
                missing = sorted(expected_id_set - sample_ids)
                extra = sorted(sample_ids - expected_id_set)
                raise ValueError(
                    "label row sample IDs must exactly match "
                    "expected_sample_ids: "
                    f"missing={missing}, extra={extra}"
                )
            if len(contract_hashes) != 1:
                raise ValueError("label rows mix multiple label contracts")
            rows = sorted(
                subset_rows, key=lambda row: row.global_sample_index
            )

        selected_rows = tuple(row for row in rows if row.split == split)
        if not selected_rows:
            raise ValueError(f"label rows contain no samples for split {split!r}")

        self.robot_video_dataset = robot_video_dataset
        self.split = split
        self.contract_sha256 = next(iter(contract_hashes))
        self._rows = selected_rows
        self._lookup = lookup
        self._data_manifest_identity = {
            "manifest_sha256": lookup.data_manifest_sha256
        }
        self._episode_split_identity = {
            "data_manifest_sha256": lookup.data_manifest_sha256,
            "assignment_sha256": lookup.assignment_sha256,
        }

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def labels(self) -> tuple[bool, ...]:
        """Return fixed-order binary labels for GateTrainer construction."""

        return tuple(row.label for row in self._rows)

    @property
    def sample_ids(self) -> tuple[str, ...]:
        """Return fixed-order semantic IDs without exposing label evidence."""

        return tuple(row.sample_id for row in self._rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("Stage2GateDataset index must be an integer")
        if index < 0 or index >= len(self):
            raise IndexError(
                f"Stage2GateDataset index {index} is outside [0, {len(self)})"
            )

        row = self._rows[index]
        sample = self.robot_video_dataset[row.global_sample_index]
        if not isinstance(sample, Mapping):
            raise TypeError("Gate input sample must be a mapping")
        if set(sample) != _GATE_INPUT_KEYS:
            unexpected = sorted(map(str, set(sample) - _GATE_INPUT_KEYS))
            missing = sorted(_GATE_INPUT_KEYS - set(sample))
            raise ValueError(
                "Gate input sample fields must match the current-only schema; "
                f"missing={missing}, unexpected={unexpected}"
            )
        raw_identity = sample.get("sample_identity")
        if not isinstance(raw_identity, Mapping):
            raise ValueError("Gate input sample has no sample_identity mapping")
        actual_identity = validate_sample_identity_with_lookup(
            raw_identity, self._lookup
        )
        if actual_identity != row.identity():
            raise ValueError(
                "Gate input sample_identity drifted from its label row "
                f"at global index {row.global_sample_index}"
            )
        actual_sample_id = sample_id_from_lookup(actual_identity, self._lookup)
        if actual_sample_id != row.sample_id:
            raise ValueError("Gate input sample_id disagrees with its label row")
        actual_split = split_for_identity(
            self._episode_split_identity,
            self._data_manifest_identity,
            actual_identity,
            lookup=self._lookup,
        )
        if actual_split != row.split or actual_split != self.split:
            raise ValueError("Gate input sample split disagrees with its label row")

        input_image = _required_tensor(sample, "input_image")
        context = _required_tensor(sample, "context")
        context_mask = _required_tensor(sample, "context_mask")
        proprio = _required_tensor(sample, "proprio")
        if input_image.ndim != 3 or input_image.shape[0] != 3:
            raise ValueError("input_image must have shape [3,H,W]")
        if context.ndim != 2 or context.shape[0] < 1:
            raise ValueError("context must have non-empty shape [L,D]")
        if (
            context_mask.ndim != 1
            or context_mask.shape[0] != context.shape[0]
            or context_mask.dtype != torch.bool
            or not bool(context_mask.any().item())
        ):
            raise ValueError(
                "context_mask must be a non-empty bool [L] mask for context"
            )
        if proprio.ndim != 1 or proprio.shape[0] < 1:
            raise ValueError("proprio must have non-empty shape [D]")

        return {
            "input_image": input_image,
            "context": context,
            "context_mask": context_mask,
            "proprio": proprio,
            "label": row.label,
            "sample_weight": row.sample_weight,
            "sample_id": row.sample_id,
        }


__all__ = ["GateInputDataset", "Stage2GateDataset"]
