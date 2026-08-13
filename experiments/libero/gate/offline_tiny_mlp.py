"""Fail-closed CPU core for the preregistered LIBERO Gate feasibility run.

This module deliberately has no Validation4 input.  It can read only the
Target5 labels (seeds 42--46) and the sealed current-state feature cache.  The
independent seeds 47--50 are consumed later by the analyzer, after the OOF
predictions produced here have been sealed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors
from torch import nn

from experiments.libero.gate.demo_utility_target_v2 import (
    canonical_json,
    load_target_bundle,
    sha256_file,
    sha256_json,
)


FEATURE_CACHE_KIND = "libero_gate_current_state_feature_cache"
FEATURE_COMPLETION_KIND = "libero_gate_current_state_feature_cache_completion"
FOLD_PLAN_KIND = "libero_gate_offline_tiny_mlp_fold_plan"
RUN_KIND = "libero_gate_offline_tiny_mlp_run"
PREDICTION_KIND = "libero_gate_oof_prediction"
COMPLETION_KIND = "libero_gate_offline_tiny_mlp_completion"
SCHEMA_VERSION = 1
TASK_FOLD_NAMESPACE = "libero_gate_mlp_v1"
INNER_FOLD_NAMESPACE = "libero_gate_mlp_inner_v1"
RANDOM_NAMESPACE = "libero_gate_random_v1"
INIT_SEEDS = (101, 202, 303, 404, 505)
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")

TASK_MODELS: tuple[tuple[str, str, str], ...] = (
    ("full_hybrid", "full", "hybrid"),
    ("full_huber", "full", "huber"),
    ("visual_proprio_hybrid", "visual_proprio", "hybrid"),
    ("instruction_proprio_hybrid", "instruction_proprio", "hybrid"),
    ("instruction_only_hybrid", "instruction", "hybrid"),
    ("constant_train_mean", "none", "baseline"),
    ("suite_mean_fallback", "suite_id", "baseline"),
    ("task_lookup_fallback", "task_id", "baseline"),
)
SUITE_MODELS: tuple[tuple[str, str, str], ...] = (
    ("full_hybrid", "full", "hybrid"),
)
FEATURE_DIMS = {"full": 137, "visual": 64, "instruction": 65, "proprio": 8}
PREDICTION_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "prediction_id",
        "prediction_sha256",
        "selection_order",
        "sample_id",
        "source_index",
        "suite",
        "task_index",
        "task",
        "target_id",
        "target_sha256",
        "input_combined_sha256",
        "feature_id",
        "feature_record_sha256",
        "outer_scheme",
        "fold_id",
        "test_group",
        "model_name",
        "feature_view",
        "loss_name",
        "init_seeds",
        "init_predictions",
        "prediction",
        "target5_utility_mean",
        "target5_utility_sem",
        "target5_high_confidence",
    }
)


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _require_sha(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _load_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"blank line in {label} at line {line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"malformed JSON in {label} at line {line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"{label} line {line_number} is not an object")
            rows.append(row)
    return rows


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(f"{canonical_json(row)}\n" for row in rows).encode("utf-8")


def _payload_sha(record: Mapping[str, Any], hash_field: str) -> str:
    return sha256_json({key: value for key, value in record.items() if key != hash_field})


def _completion_sha(record: Mapping[str, Any]) -> str:
    return _payload_sha(record, "completion_sha256")


def _stable_identity_sha(namespace: str, value: Any) -> str:
    payload = f"{namespace}\0{canonical_json(value)}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _group_dict(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "suite": str(row["suite"]),
        "task_index": int(row["task_index"]),
        "task": str(row["task"]),
    }


def group_id(row: Mapping[str, Any]) -> str:
    return canonical_json(_group_dict(row))


def _groups(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for row in rows:
        result.setdefault(group_id(row), []).append(int(row["selection_order"]))
    return result


def _validate_target_identities(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("Target5 rows cannot be empty")
    orders = [int(row["selection_order"]) for row in rows]
    if orders != list(range(len(rows))):
        raise ValueError("Target5 selection_order must be exact file order 0..N-1")
    if len({str(row["sample_id"]) for row in rows}) != len(rows):
        raise ValueError("Target5 contains duplicate sample_id")
    if len({int(row["source_index"]) for row in rows}) != len(rows):
        raise ValueError("Target5 contains duplicate global source_index")
    for row in rows:
        if str(row["suite"]) not in SUITES:
            raise ValueError(f"unsupported LIBERO suite {row['suite']!r}")


def build_fold_plan(
    targets: Sequence[Mapping[str, Any]], *, strict_v1: bool = True
) -> dict[str, Any]:
    """Build identity-only task and suite held-out folds without reading labels."""

    _validate_target_identities(targets)
    by_group = _groups(targets)
    group_meta = {
        key: {
            **_group_dict(targets[orders[0]]),
            "group_id": key,
            "selection_orders": sorted(orders),
            "state_count": len(orders),
        }
        for key, orders in by_group.items()
    }
    if strict_v1 and len(targets) != 100:
        raise ValueError(f"formal V1 requires exactly 100 states, got {len(targets)}")

    task_fold_groups: list[list[str]] = [[] for _ in range(5)]
    for suite in SUITES:
        suite_groups = [value for value in group_meta.values() if value["suite"] == suite]
        two = sorted(
            (value for value in suite_groups if value["state_count"] == 2),
            key=lambda value: _stable_identity_sha(TASK_FOLD_NAMESPACE, value["group_id"]),
        )
        three = sorted(
            (value for value in suite_groups if value["state_count"] == 3),
            key=lambda value: _stable_identity_sha(TASK_FOLD_NAMESPACE, value["group_id"]),
        )
        if len(two) != 5 or len(three) != 5:
            raise ValueError(
                f"suite {suite} must contain five 2-state and five 3-state tasks; "
                f"got {len(two)} and {len(three)}"
            )
        pairs = [(two[index]["group_id"], three[index]["group_id"]) for index in range(5)]
        pairs.sort(key=lambda pair: _stable_identity_sha(TASK_FOLD_NAMESPACE, list(pair)))
        for fold_id, pair in enumerate(pairs):
            task_fold_groups[fold_id].extend(pair)

    def make_fold(
        *, scheme: str, fold_id: int, test_groups: Sequence[str], eligible_groups: Sequence[str]
    ) -> dict[str, Any]:
        test_set = set(test_groups)
        candidates = [key for key in eligible_groups if key not in test_set]
        inner_groups: list[str] = []
        for suite in SUITES:
            available = [key for key in candidates if group_meta[key]["suite"] == suite]
            if not available:
                continue
            available.sort(
                key=lambda key: _stable_identity_sha(INNER_FOLD_NAMESPACE, key)
            )
            inner_groups.append(available[0])
        train_groups = sorted(set(candidates) - set(inner_groups))
        train_orders = sorted(order for key in train_groups for order in by_group[key])
        val_orders = sorted(order for key in inner_groups for order in by_group[key])
        test_orders = sorted(order for key in test_groups for order in by_group[key])
        if set(train_orders) & set(val_orders) or set(train_orders) & set(test_orders):
            raise AssertionError("fold state leakage")
        if set(val_orders) & set(test_orders):
            raise AssertionError("fold state leakage")
        return {
            "scheme": scheme,
            "fold_id": int(fold_id),
            "train_groups": train_groups,
            "inner_validation_groups": sorted(inner_groups),
            "test_groups": sorted(test_groups),
            "train_selection_orders": train_orders,
            "inner_validation_selection_orders": val_orders,
            "test_selection_orders": test_orders,
        }

    all_groups = sorted(by_group)
    task_folds = [
        make_fold(
            scheme="task_heldout",
            fold_id=fold_id,
            test_groups=groups,
            eligible_groups=all_groups,
        )
        for fold_id, groups in enumerate(task_fold_groups)
    ]
    suite_folds = []
    for fold_id, suite in enumerate(SUITES):
        test_groups = [key for key in all_groups if group_meta[key]["suite"] == suite]
        fold = make_fold(
            scheme="suite_heldout",
            fold_id=fold_id,
            test_groups=test_groups,
            eligible_groups=all_groups,
        )
        fold["heldout_suite"] = suite
        suite_folds.append(fold)

    if strict_v1:
        for fold in task_folds:
            if len(fold["test_groups"]) != 8 or len(fold["test_selection_orders"]) != 20:
                raise ValueError("formal task fold must hold out exactly 8 tasks / 20 states")
            suite_counts = {
                suite: sum(
                    str(targets[index]["suite"]) == suite
                    for index in fold["test_selection_orders"]
                )
                for suite in SUITES
            }
            if set(suite_counts.values()) != {5}:
                raise ValueError(f"formal task fold suite counts invalid: {suite_counts}")
        if sorted(order for fold in task_folds for order in fold["test_selection_orders"]) != list(
            range(100)
        ):
            raise ValueError("task folds do not form exact OOF partition")
        if sorted(order for fold in suite_folds for order in fold["test_selection_orders"]) != list(
            range(100)
        ):
            raise ValueError("suite folds do not form exact OOF partition")

    identity_rows = [
        {
            "selection_order": int(row["selection_order"]),
            "sample_id": str(row["sample_id"]),
            "source_index": int(row["source_index"]),
            **_group_dict(row),
        }
        for row in targets
    ]
    plan = {
        "schema_version": SCHEMA_VERSION,
        "kind": FOLD_PLAN_KIND,
        "task_fold_namespace": TASK_FOLD_NAMESPACE,
        "inner_fold_namespace": INNER_FOLD_NAMESPACE,
        "num_states": len(targets),
        "identity_plan_sha256": sha256_json(identity_rows),
        "groups": [group_meta[key] for key in sorted(group_meta)],
        "task_heldout_folds": task_folds,
        "suite_heldout_folds": suite_folds,
    }
    plan["fold_membership_sha256"] = sha256_json(
        {
            "task_heldout_folds": task_folds,
            "suite_heldout_folds": suite_folds,
        }
    )
    return plan


def validate_fold_plan(plan: Mapping[str, Any], targets: Sequence[Mapping[str, Any]]) -> None:
    if plan.get("kind") != FOLD_PLAN_KIND or int(plan.get("schema_version", -1)) != 1:
        raise ValueError("invalid fold plan kind/schema")
    expected = build_fold_plan(targets, strict_v1=len(targets) == 100)
    if canonical_json(plan) != canonical_json(expected):
        raise ValueError("fold plan differs from deterministic identity-only plan")


def tensor_content_sha256(tensor: torch.Tensor) -> str:
    """Digest tensor semantic content using dtype, shape and C-order bytes."""

    value = tensor.detach().to(device="cpu").contiguous()
    header = canonical_json(
        {
            "schema_version": 1,
            "dtype": str(value.dtype),
            "shape": [int(dimension) for dimension in value.shape],
        }
    ).encode("utf-8")
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
    return hashlib.sha256(header + b"\0" + raw).hexdigest()


@dataclass(frozen=True)
class LoadedFeatureBundle:
    root: Path
    manifest: dict[str, Any]
    index: tuple[dict[str, Any], ...]
    tensors: dict[str, torch.Tensor]
    completion: dict[str, Any]
    manifest_sha256: str
    index_sha256: str
    features_sha256: str
    completion_sha256: str


def load_feature_bundle(
    feature_dir: str | Path, *, expected_completion_sha256: str
) -> LoadedFeatureBundle:
    root = Path(feature_dir).resolve()
    paths = {
        "manifest": root / "manifest.json",
        "index": root / "feature_index.jsonl",
        "features": root / "features.safetensors",
        "completion": root / "completion.json",
    }
    expected_completion = _require_sha(
        expected_completion_sha256, field="expected_completion_sha256"
    )
    completion = _load_json(paths["completion"], label="feature completion")
    completion_file_sha = sha256_file(paths["completion"])
    if expected_completion not in {
        completion_file_sha,
        str(completion.get("completion_sha256", "")),
    }:
        raise ValueError(
            "feature completion trust anchor matches neither sealed payload nor file bytes"
        )
    if completion.get("kind") != FEATURE_COMPLETION_KIND or int(
        completion.get("schema_version", -1)
    ) != 1:
        raise ValueError("invalid feature completion kind/schema")
    if completion.get("completion_sha256") != _completion_sha(completion):
        raise ValueError("feature completion payload hash is invalid")
    actual_file_shas = {
        "manifest_sha256": sha256_file(paths["manifest"]),
        "feature_index_sha256": sha256_file(paths["index"]),
        "features_sha256": sha256_file(paths["features"]),
    }
    for field, actual in actual_file_shas.items():
        if completion.get(field) != actual:
            raise ValueError(f"feature completion {field} binding mismatch")

    manifest = _load_json(paths["manifest"], label="feature manifest")
    if manifest.get("kind") != FEATURE_CACHE_KIND or int(manifest.get("schema_version", -1)) != 1:
        raise ValueError("invalid feature manifest kind/schema")
    compatibility = _require_mapping(manifest.get("compatibility"), field="feature compatibility")
    if manifest.get("compatibility_fingerprint") != sha256_json(compatibility):
        raise ValueError("feature compatibility fingerprint mismatch")
    if completion.get("manifest_compatibility_fingerprint") != manifest.get(
        "compatibility_fingerprint"
    ):
        raise ValueError("feature completion is not bound to manifest compatibility")

    rows = _load_jsonl(paths["index"], label="feature index")
    count = int(completion.get("num_states", -1))
    if count <= 0 or len(rows) != count or int(manifest.get("num_states", count)) != count:
        raise ValueError("feature cache state counts are inconsistent")
    if [int(row.get("selection_order", -1)) for row in rows] != list(range(count)):
        raise ValueError("feature index order must be exact 0..N-1")
    if len({str(row.get("sample_id")) for row in rows}) != count:
        raise ValueError("feature index contains duplicate sample_id")
    if len({int(row.get("source_index", -1)) for row in rows}) != count:
        raise ValueError("feature index contains duplicate global source_index")
    for row in rows:
        if row.get("kind") != FEATURE_CACHE_KIND or int(
            row.get("feature_record_schema_version", -1)
        ) != 1:
            raise ValueError("invalid feature index record kind/schema")
        if row.get("feature_record_sha256") != _payload_sha(row, "feature_record_sha256"):
            raise ValueError("feature index record hash mismatch")

    tensors = load_safetensors(str(paths["features"]), device="cpu")
    if set(tensors) != set(FEATURE_DIMS):
        raise ValueError(f"unexpected feature tensor keys: {sorted(tensors)}")
    tensor_contract = _require_mapping(completion.get("tensors"), field="completion.tensors")
    for key, width in FEATURE_DIMS.items():
        tensor = tensors[key]
        if tensor.dtype != torch.float32 or tuple(tensor.shape) != (count, width):
            raise ValueError(
                f"feature tensor {key} must be float32 [{count},{width}], got "
                f"{tensor.dtype} {list(tensor.shape)}"
            )
        if not tensor.is_contiguous() or not torch.isfinite(tensor).all():
            raise ValueError(f"feature tensor {key} must be contiguous and finite")
        declared = _require_mapping(tensor_contract.get(key), field=f"tensors.{key}")
        if list(declared.get("shape", [])) != [count, width] or declared.get("dtype") not in (
            "float32",
            "torch.float32",
        ):
            raise ValueError(f"feature tensor {key} metadata mismatch")
        semantic_sha = tensor_content_sha256(tensor)
        if declared.get("content_sha256") != semantic_sha:
            raise ValueError(f"feature tensor {key} content hash mismatch")
        for order, row in enumerate(rows):
            hashes = _require_mapping(row.get("feature_hashes"), field="feature_hashes")
            if hashes.get(key) != tensor_content_sha256(tensor[order]):
                raise ValueError(f"feature row hash mismatch for {key} at order {order}")
    if not torch.equal(
        tensors["full"],
        torch.cat(
            (tensors["visual"], tensors["instruction"], tensors["proprio"]), dim=1
        ),
    ):
        raise ValueError("full feature tensor is not exact visual+instruction+proprio")

    return LoadedFeatureBundle(
        root=root,
        manifest=manifest,
        index=tuple(rows),
        tensors={key: value.contiguous() for key, value in tensors.items()},
        completion=completion,
        manifest_sha256=actual_file_shas["manifest_sha256"],
        index_sha256=actual_file_shas["feature_index_sha256"],
        features_sha256=actual_file_shas["features_sha256"],
        completion_sha256=str(completion["completion_sha256"]),
    )


@dataclass(frozen=True)
class TrainingInputs:
    target_dir: Path
    target_manifest: dict[str, Any]
    targets: tuple[dict[str, Any], ...]
    target_manifest_sha256: str
    target_targets_sha256: str
    features: LoadedFeatureBundle


def load_training_inputs(
    *,
    target_dir: str | Path,
    target_manifest_sha256: str,
    target_targets_sha256: str,
    feature_dir: str | Path,
    feature_completion_sha256: str,
    expected_num_states: int = 100,
) -> TrainingInputs:
    """Load the only two inputs accepted by training: Target5 and features."""

    target_manifest, targets = load_target_bundle(
        target_dir,
        expected_manifest_sha256=_require_sha(
            target_manifest_sha256, field="target_manifest_sha256"
        ),
        expected_targets_sha256=_require_sha(
            target_targets_sha256, field="target_targets_sha256"
        ),
        expected_num_states=expected_num_states,
    )
    features = load_feature_bundle(
        feature_dir, expected_completion_sha256=feature_completion_sha256
    )
    if len(features.index) != len(targets):
        raise ValueError("Target5 and feature cache state counts differ")
    for order, (target, feature) in enumerate(zip(targets, features.index, strict=True)):
        exact = {
            "selection_order": int(target["selection_order"]),
            "sample_id": str(target["sample_id"]),
            "source_index": int(target["source_index"]),
            "suite": str(target["suite"]),
            "task_index": int(target["task_index"]),
            "task": str(target["task"]),
            "target_id": str(target["target_id"]),
            "target_sha256": str(target["target_sha256"]),
            "input_combined_sha256": str(target["input_hashes"]["combined"]),
        }
        for field, expected in exact.items():
            actual = feature.get(field)
            if actual != expected:
                raise ValueError(
                    f"feature/Target5 mismatch at order {order}: {field} "
                    f"actual={actual!r}, expected={expected!r}"
                )
    compatibility = features.manifest.get("compatibility", {})
    target_file_bindings = {
        "target_manifest_sha256": target_manifest_sha256,
        "target_records_sha256": target_targets_sha256,
    }
    for field, expected in target_file_bindings.items():
        if field in compatibility and compatibility[field] != expected:
            raise ValueError(f"feature manifest {field} differs from Target5 input")
    return TrainingInputs(
        target_dir=Path(target_dir).resolve(),
        target_manifest=target_manifest,
        targets=tuple(targets),
        target_manifest_sha256=target_manifest_sha256,
        target_targets_sha256=target_targets_sha256,
        features=features,
    )


@dataclass(frozen=True)
class Standardizer:
    mean: torch.Tensor
    std: torch.Tensor

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / self.std


def fit_standardizer(values: torch.Tensor) -> Standardizer:
    if values.ndim != 2 or values.shape[0] < 1 or not torch.isfinite(values).all():
        raise ValueError("standardizer requires a finite non-empty rank-2 tensor")
    source = values.detach().to(dtype=torch.float32, device="cpu")
    mean = source.mean(dim=0)
    std = source.std(dim=0, correction=0).clamp_min(1e-6)
    return Standardizer(mean=mean, std=std)


@dataclass(frozen=True)
class RobustTargetScale:
    median: float
    scale: float

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.median) / self.scale

    def inverse(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.scale + self.median


def fit_robust_target_scale(values: torch.Tensor) -> RobustTargetScale:
    if values.ndim != 1 or values.numel() < 1 or not torch.isfinite(values).all():
        raise ValueError("target scaling requires a finite non-empty vector")
    source = values.detach().to(dtype=torch.float32, device="cpu")
    median = float(torch.quantile(source, 0.5, interpolation="linear").item())
    mad = float(
        torch.quantile((source - median).abs(), 0.5, interpolation="linear").item()
    )
    return RobustTargetScale(median=median, scale=max(1.4826 * mad, 1e-6))


def uncertainty_weights(sem: torch.Tensor, robust_scale: float) -> torch.Tensor:
    if sem.ndim != 1 or not torch.isfinite(sem).all() or torch.any(sem < 0):
        raise ValueError("SEM must be a finite non-negative vector")
    raw = 1.0 / (1.0 + (sem / float(robust_scale)).square())
    weights = (raw / raw.mean()).clamp(0.25, 2.0)
    return weights / weights.mean()


def build_ranking_pairs(
    rows: Sequence[Mapping[str, Any]], orders: Sequence[int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    local_rows = [rows[index] for index in orders]
    left: list[int] = []
    right: list[int] = []
    weights: list[float] = []
    for first in range(len(local_rows)):
        for second in range(first + 1, len(local_rows)):
            a, b = local_rows[first], local_rows[second]
            if float(a["t95_ci_low"]) > float(b["t95_ci_high"]):
                high, low = first, second
            elif float(b["t95_ci_low"]) > float(a["t95_ci_high"]):
                high, low = second, first
            else:
                continue
            left.append(high)
            right.append(low)
            weights.append(2.0 if group_id(a) == group_id(b) else 1.0)
    return (
        torch.tensor(left, dtype=torch.long),
        torch.tensor(right, dtype=torch.long),
        torch.tensor(weights, dtype=torch.float32),
    )


class TinyUtilityMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    gradient_clip_norm: float = 1.0
    max_epochs: int = 1000
    min_epochs: int = 100
    early_stop_patience: int = 100
    init_seeds: tuple[int, ...] = INIT_SEEDS
    ranking_coefficient: float = 0.25
    cpu_threads: int = 1

    def validate(self, *, formal: bool) -> None:
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer parameters")
        if self.gradient_clip_norm <= 0 or self.max_epochs < 1:
            raise ValueError("invalid training duration/gradient clip")
        if not (1 <= self.min_epochs <= self.max_epochs):
            raise ValueError("min_epochs must be within 1..max_epochs")
        if self.early_stop_patience < 1 or not self.init_seeds:
            raise ValueError("patience and init seeds must be non-empty/positive")
        if isinstance(self.cpu_threads, bool) or int(self.cpu_threads) < 1:
            raise ValueError("cpu_threads must be a positive integer")
        if formal and self != TrainingConfig():
            raise ValueError("formal run requires the exact preregistered TrainingConfig")


@dataclass(frozen=True)
class EnsembleResult:
    init_predictions: tuple[tuple[float, ...], ...]
    ensemble_prediction: tuple[float, ...]
    best_epochs: tuple[int, ...]
    train_transform: dict[str, Any]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def train_fold_ensemble(
    *,
    features: torch.Tensor,
    targets: Sequence[Mapping[str, Any]],
    train_orders: Sequence[int],
    validation_orders: Sequence[int],
    test_orders: Sequence[int],
    loss_name: str,
    config: TrainingConfig,
) -> EnsembleResult:
    if loss_name not in {"hybrid", "huber"}:
        raise ValueError(f"unsupported learned loss {loss_name!r}")
    if not train_orders or not validation_orders or not test_orders:
        raise ValueError("train/inner-validation/test splits must all be non-empty")
    config.validate(formal=False)
    values = features.detach().to(dtype=torch.float32, device="cpu")
    if values.ndim != 2 or values.shape[0] != len(targets):
        raise ValueError("feature tensor and target row counts differ")
    train_index = torch.tensor(train_orders, dtype=torch.long)
    val_index = torch.tensor(validation_orders, dtype=torch.long)
    test_index = torch.tensor(test_orders, dtype=torch.long)
    feature_scale = fit_standardizer(values[train_index])
    x_train = feature_scale.transform(values[train_index])
    x_val = feature_scale.transform(values[val_index])
    x_test = feature_scale.transform(values[test_index])
    y_all = torch.tensor([float(row["utility_mean"]) for row in targets], dtype=torch.float32)
    sem_all = torch.tensor([float(row["utility_sem"]) for row in targets], dtype=torch.float32)
    target_scale = fit_robust_target_scale(y_all[train_index])
    y_train = target_scale.transform(y_all[train_index])
    y_val = target_scale.transform(y_all[val_index])
    weights = uncertainty_weights(sem_all[train_index], target_scale.scale)
    pair_high, pair_low, pair_weights = build_ranking_pairs(targets, train_orders)

    all_predictions: list[tuple[float, ...]] = []
    best_epochs: list[int] = []
    for seed in config.init_seeds:
        _seed_everything(int(seed))
        model = TinyUtilityMLP(values.shape[1]).to(device="cpu", dtype=torch.float32)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        best_loss = math.inf
        best_state: dict[str, torch.Tensor] | None = None
        best_epoch = -1
        stale = 0
        for epoch in range(config.max_epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x_train)
            point = F.smooth_l1_loss(prediction, y_train, beta=1.0, reduction="none")
            loss = (point * weights).mean()
            if loss_name == "hybrid" and pair_high.numel() > 0:
                rank_terms = F.softplus(-(prediction[pair_high] - prediction[pair_low]))
                rank_loss = (rank_terms * pair_weights).sum() / pair_weights.sum()
                loss = loss + config.ranking_coefficient * rank_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()

            model.eval()
            with torch.no_grad():
                validation_loss = float(
                    F.smooth_l1_loss(model(x_val), y_val, beta=1.0, reduction="mean").item()
                )
            if validation_loss < best_loss - 1e-12:
                best_loss = validation_loss
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
                best_epoch = epoch + 1
                stale = 0
            else:
                stale += 1
            if epoch + 1 >= config.min_epochs and stale >= config.early_stop_patience:
                break
        if best_state is None:
            raise RuntimeError("training did not produce a finite checkpoint")
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            raw = target_scale.inverse(model(x_test)).to(torch.float64)
        if not torch.isfinite(raw).all():
            raise FloatingPointError("non-finite OOF prediction")
        all_predictions.append(tuple(float(value) for value in raw.tolist()))
        best_epochs.append(best_epoch)
    stacked = torch.tensor(all_predictions, dtype=torch.float64)
    ensemble = tuple(float(value) for value in stacked.mean(dim=0).tolist())
    return EnsembleResult(
        init_predictions=tuple(all_predictions),
        ensemble_prediction=ensemble,
        best_epochs=tuple(best_epochs),
        train_transform={
            "feature_mean_sha256": tensor_content_sha256(feature_scale.mean),
            "feature_std_sha256": tensor_content_sha256(feature_scale.std),
            "target_median": target_scale.median,
            "target_scale": target_scale.scale,
            "uncertainty_weights_sha256": tensor_content_sha256(weights),
            "ranking_pair_count": int(pair_high.numel()),
        },
    )


def feature_view(tensors: Mapping[str, torch.Tensor], name: str) -> torch.Tensor:
    if name == "full":
        return tensors["full"]
    if name == "visual_proprio":
        return torch.cat((tensors["visual"], tensors["proprio"]), dim=1)
    if name == "instruction_proprio":
        return torch.cat((tensors["instruction"], tensors["proprio"]), dim=1)
    if name == "instruction":
        return tensors["instruction"]
    raise ValueError(f"feature view {name!r} is not learned")


def _baseline_predictions(
    *,
    model_name: str,
    targets: Sequence[Mapping[str, Any]],
    train_orders: Sequence[int],
    test_orders: Sequence[int],
) -> tuple[float, ...]:
    train = [targets[index] for index in train_orders]
    global_mean = float(np.mean([float(row["utility_mean"]) for row in train]))
    if model_name == "constant_train_mean":
        return tuple(global_mean for _ in test_orders)
    if model_name == "suite_mean_fallback":
        suite_means: dict[str, float] = {}
        for suite in SUITES:
            values = [float(row["utility_mean"]) for row in train if row["suite"] == suite]
            if values:
                suite_means[suite] = float(np.mean(values))
        return tuple(suite_means.get(str(targets[index]["suite"]), global_mean) for index in test_orders)
    if model_name == "task_lookup_fallback":
        lookups: dict[str, float] = {}
        for key in sorted({group_id(row) for row in train}):
            values = [float(row["utility_mean"]) for row in train if group_id(row) == key]
            lookups[key] = float(np.mean(values))
        return tuple(lookups.get(group_id(targets[index]), global_mean) for index in test_orders)
    raise ValueError(f"unknown baseline {model_name!r}")


def _prediction_record(
    *,
    target: Mapping[str, Any],
    feature: Mapping[str, Any],
    scheme: str,
    fold_id: int,
    model_name: str,
    view: str,
    loss_name: str,
    init_predictions: Sequence[float],
    prediction: float,
) -> dict[str, Any]:
    if len(init_predictions) != len(INIT_SEEDS):
        raise ValueError("every prediction must carry the five preregistered init values")
    record = {
        "schema_version": SCHEMA_VERSION,
        "kind": PREDICTION_KIND,
        "prediction_id": (
            f"{scheme}/fold_{fold_id}/{model_name}/{target['sample_id']}"
        ),
        "selection_order": int(target["selection_order"]),
        "sample_id": str(target["sample_id"]),
        "source_index": int(target["source_index"]),
        "suite": str(target["suite"]),
        "task_index": int(target["task_index"]),
        "task": str(target["task"]),
        "target_id": str(target["target_id"]),
        "target_sha256": str(target["target_sha256"]),
        "input_combined_sha256": str(target["input_hashes"]["combined"]),
        "feature_id": str(feature["feature_id"]),
        "feature_record_sha256": str(feature["feature_record_sha256"]),
        "outer_scheme": scheme,
        "fold_id": int(fold_id),
        "test_group": group_id(target) if scheme == "task_heldout" else str(target["suite"]),
        "model_name": model_name,
        "feature_view": view,
        "loss_name": loss_name,
        "init_seeds": list(INIT_SEEDS),
        "init_predictions": [float(value) for value in init_predictions],
        "prediction": float(prediction),
        "target5_utility_mean": float(target["utility_mean"]),
        "target5_utility_sem": float(target["utility_sem"]),
        "target5_high_confidence": bool(target["high_confidence"]),
    }
    record["prediction_sha256"] = _payload_sha(record, "prediction_sha256")
    return record


def _train_oof_predictions_impl(
    inputs: TrainingInputs,
    fold_plan: Mapping[str, Any],
    *,
    config: TrainingConfig = TrainingConfig(),
    formal: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config.validate(formal=formal)
    targets = inputs.targets
    validate_fold_plan(fold_plan, targets)
    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {"task_heldout": {}, "suite_heldout": {}}

    for scheme, folds, models in (
        ("task_heldout", fold_plan["task_heldout_folds"], TASK_MODELS),
        ("suite_heldout", fold_plan["suite_heldout_folds"], SUITE_MODELS),
    ):
        for fold in folds:
            fold_id = int(fold["fold_id"])
            train_orders = list(fold["train_selection_orders"])
            val_orders = list(fold["inner_validation_selection_orders"])
            test_orders = list(fold["test_selection_orders"])
            outer_train_orders = sorted(train_orders + val_orders)
            for model_name, view, loss_name in models:
                if loss_name == "baseline":
                    ensemble = _baseline_predictions(
                        model_name=model_name,
                        targets=targets,
                        train_orders=outer_train_orders,
                        test_orders=test_orders,
                    )
                    init_matrix = tuple(tuple(value for value in ensemble) for _ in INIT_SEEDS)
                    model_diag = {"baseline": True}
                else:
                    result = train_fold_ensemble(
                        features=feature_view(inputs.features.tensors, view),
                        targets=targets,
                        train_orders=train_orders,
                        validation_orders=val_orders,
                        test_orders=test_orders,
                        loss_name=loss_name,
                        config=config,
                    )
                    ensemble = result.ensemble_prediction
                    init_matrix = result.init_predictions
                    model_diag = {
                        "best_epochs": list(result.best_epochs),
                        "train_transform": result.train_transform,
                    }
                diagnostics[scheme][f"fold_{fold_id}/{model_name}"] = model_diag
                for local_index, selection_order in enumerate(test_orders):
                    init_values = [values[local_index] for values in init_matrix]
                    rows.append(
                        _prediction_record(
                            target=targets[selection_order],
                            feature=inputs.features.index[selection_order],
                            scheme=scheme,
                            fold_id=fold_id,
                            model_name=model_name,
                            view=view,
                            loss_name=loss_name,
                            init_predictions=init_values,
                            prediction=ensemble[local_index],
                        )
                    )
    rows.sort(
        key=lambda row: (
            0 if row["outer_scheme"] == "task_heldout" else 1,
            row["model_name"],
            row["selection_order"],
        )
    )
    expected = 900 if formal else sum(
        len(fold["test_selection_orders"]) * len(models)
        for folds, models in (
            (fold_plan["task_heldout_folds"], TASK_MODELS),
            (fold_plan["suite_heldout_folds"], SUITE_MODELS),
        )
        for fold in folds
    )
    if len(rows) != expected:
        raise AssertionError(f"OOF record count {len(rows)} differs from expected {expected}")
    return rows, diagnostics


def train_oof_predictions(
    inputs: TrainingInputs,
    fold_plan: Mapping[str, Any],
    *,
    config: TrainingConfig = TrainingConfig(),
    formal: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Train under the frozen CPU thread count and restore process state."""

    config.validate(formal=formal)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(int(config.cpu_threads))
    try:
        return _train_oof_predictions_impl(
            inputs, fold_plan, config=config, formal=formal
        )
    finally:
        torch.set_num_threads(previous_threads)


def _source_sha256() -> str:
    return sha256_file(Path(__file__).resolve())


def build_run_manifest(
    *,
    inputs: TrainingInputs,
    fold_plan_sha256: str,
    rows: Sequence[Mapping[str, Any]],
    config: TrainingConfig,
    diagnostics: Mapping[str, Any],
    formal: bool,
) -> dict[str, Any]:
    task_count = sum(row["outer_scheme"] == "task_heldout" for row in rows)
    suite_count = sum(row["outer_scheme"] == "suite_heldout" for row in rows)
    feature_completion = inputs.features.completion
    compatibility = {
        "schema_version": SCHEMA_VERSION,
        "kind": RUN_KIND,
        "target_manifest_sha256": inputs.target_manifest_sha256,
        "target_targets_sha256": inputs.target_targets_sha256,
        "target_compatibility_fingerprint": inputs.target_manifest[
            "compatibility_fingerprint"
        ],
        "target_records_sha256": inputs.target_manifest["targets"][
            "canonical_records_sha256"
        ],
        "feature_manifest_sha256": inputs.features.manifest_sha256,
        "feature_index_sha256": inputs.features.index_sha256,
        "features_sha256": inputs.features.features_sha256,
        "feature_completion_sha256": inputs.features.completion_sha256,
        "feature_manifest_compatibility_fingerprint": inputs.features.manifest[
            "compatibility_fingerprint"
        ],
        "feature_index_records_sha256": hashlib.sha256(
            _jsonl_bytes(inputs.features.index)
        ).hexdigest(),
        "feature_tensor_content_sha256": {
            key: feature_completion["tensors"][key]["content_sha256"]
            for key in sorted(FEATURE_DIMS)
        },
        "fold_plan_sha256": fold_plan_sha256,
        "trainer_source_sha256": _source_sha256(),
        "config_sha256": sha256_json(asdict(config)),
        "random_namespace": RANDOM_NAMESPACE,
        "random_salts_count": 1000,
        "num_states": len(inputs.targets),
        "task_oof_rows": task_count,
        "suite_oof_rows": suite_count,
        "total_oof_rows": len(rows),
        "formal_protocol": bool(formal),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": RUN_KIND,
        "compatibility": compatibility,
        "compatibility_fingerprint": sha256_json(compatibility),
        "training_config": asdict(config),
        "models": {
            "task_heldout": [model for model, _, _ in TASK_MODELS],
            "suite_heldout": [model for model, _, _ in SUITE_MODELS],
            "best_nonvisual": "instruction_proprio_hybrid",
        },
        "predictions": {
            "filename": "oof_predictions.jsonl",
            "count": len(rows),
            "ordered_prediction_ids": [str(row["prediction_id"]) for row in rows],
            "ordered_prediction_sha256": [str(row["prediction_sha256"]) for row in rows],
            "canonical_records_sha256": hashlib.sha256(_jsonl_bytes(rows)).hexdigest(),
        },
        "diagnostics": diagnostics,
    }
    return manifest


def validate_formal_run_contract(
    manifest: Mapping[str, Any],
    fold_plan: Mapping[str, Any],
    targets: Sequence[Mapping[str, Any]],
) -> None:
    """Re-prove that a sealed run is the exact preregistered formal experiment.

    File seals establish internal consistency.  This validation is intentionally
    separate: it rejects a self-consistently rehashed run whose split, config,
    model set, counts, random baseline, or diagnostics were changed.
    """

    # This is first by design: no manifest claim can override identity-only folds.
    validate_fold_plan(fold_plan, targets)
    if len(targets) != 100:
        raise ValueError("formal run requires exactly 100 Target5 states")
    if manifest.get("kind") != RUN_KIND or int(manifest.get("schema_version", -1)) != 1:
        raise ValueError("formal run manifest kind/schema is invalid")
    compatibility = _require_mapping(
        manifest.get("compatibility"), field="formal compatibility"
    )
    if manifest.get("compatibility_fingerprint") != sha256_json(compatibility):
        raise ValueError("formal run compatibility fingerprint is invalid")
    if compatibility.get("kind") != RUN_KIND or int(
        compatibility.get("schema_version", -1)
    ) != SCHEMA_VERSION:
        raise ValueError("formal compatibility kind/schema is invalid")
    if compatibility.get("formal_protocol") is not True:
        raise ValueError("formal_protocol must be exactly true")

    exact_compatibility = {
        "num_states": 100,
        "task_oof_rows": 800,
        "suite_oof_rows": 100,
        "total_oof_rows": 900,
        "random_namespace": RANDOM_NAMESPACE,
        "random_salts_count": 1000,
    }
    for field, expected in exact_compatibility.items():
        if compatibility.get(field) != expected:
            raise ValueError(
                f"formal compatibility {field}={compatibility.get(field)!r}, "
                f"expected {expected!r}"
            )

    sha_fields = (
        "target_manifest_sha256",
        "target_targets_sha256",
        "target_compatibility_fingerprint",
        "target_records_sha256",
        "feature_manifest_sha256",
        "feature_index_sha256",
        "features_sha256",
        "feature_completion_sha256",
        "feature_manifest_compatibility_fingerprint",
        "feature_index_records_sha256",
        "fold_plan_sha256",
        "trainer_source_sha256",
        "config_sha256",
    )
    for field in sha_fields:
        _require_sha(compatibility.get(field), field=f"compatibility.{field}")
    tensor_hashes = _require_mapping(
        compatibility.get("feature_tensor_content_sha256"),
        field="feature_tensor_content_sha256",
    )
    if set(tensor_hashes) != set(FEATURE_DIMS):
        raise ValueError("formal feature tensor hash keys are invalid")
    for key in FEATURE_DIMS:
        _require_sha(tensor_hashes[key], field=f"feature_tensor_content_sha256.{key}")

    expected_config = asdict(TrainingConfig())
    training_config = _require_mapping(
        manifest.get("training_config"), field="training_config"
    )
    if canonical_json(training_config) != canonical_json(expected_config):
        raise ValueError("formal training_config differs from the preregistration")
    if compatibility.get("config_sha256") != sha256_json(expected_config):
        raise ValueError("formal config_sha256 differs from the preregistration")

    expected_models = {
        "task_heldout": [name for name, _, _ in TASK_MODELS],
        "suite_heldout": [name for name, _, _ in SUITE_MODELS],
        "best_nonvisual": "instruction_proprio_hybrid",
    }
    if canonical_json(manifest.get("models")) != canonical_json(expected_models):
        raise ValueError("formal model set/order differs from the preregistration")

    predictions = _require_mapping(manifest.get("predictions"), field="predictions")
    if predictions.get("filename") != "oof_predictions.jsonl" or predictions.get(
        "count"
    ) != 900:
        raise ValueError("formal prediction filename/count must be 900-row OOF")
    prediction_ids = predictions.get("ordered_prediction_ids")
    prediction_hashes = predictions.get("ordered_prediction_sha256")
    if (
        not isinstance(prediction_ids, list)
        or not isinstance(prediction_hashes, list)
        or len(prediction_ids) != 900
        or len(prediction_hashes) != 900
        or len(set(map(str, prediction_ids))) != 900
    ):
        raise ValueError("formal ordered prediction IDs/hashes must each contain 900 rows")
    for index, value in enumerate(prediction_hashes):
        _require_sha(value, field=f"ordered_prediction_sha256[{index}]")
    _require_sha(
        predictions.get("canonical_records_sha256"),
        field="predictions.canonical_records_sha256",
    )

    diagnostics = _require_mapping(manifest.get("diagnostics"), field="diagnostics")
    if set(diagnostics) != {"task_heldout", "suite_heldout"}:
        raise ValueError("formal diagnostics schemes are invalid")
    expected_keys = {
        "task_heldout": {
            f"fold_{fold_id}/{name}"
            for fold_id in range(5)
            for name, _, _ in TASK_MODELS
        },
        "suite_heldout": {
            f"fold_{fold_id}/{name}"
            for fold_id in range(4)
            for name, _, _ in SUITE_MODELS
        },
    }
    model_losses = {name: loss for name, _, loss in TASK_MODELS + SUITE_MODELS}
    for scheme, keys in expected_keys.items():
        entries = _require_mapping(diagnostics.get(scheme), field=f"diagnostics.{scheme}")
        if set(entries) != keys:
            raise ValueError(f"formal diagnostics key contract differs for {scheme}")
        for key, value in entries.items():
            entry = _require_mapping(value, field=f"diagnostics.{scheme}.{key}")
            model_name = key.split("/", 1)[1]
            if model_losses[model_name] == "baseline":
                if dict(entry) != {"baseline": True}:
                    raise ValueError(f"formal baseline diagnostic is invalid: {key}")
                continue
            if set(entry) != {"best_epochs", "train_transform"}:
                raise ValueError(f"formal learned diagnostic fields are invalid: {key}")
            epochs = entry["best_epochs"]
            if (
                not isinstance(epochs, list)
                or len(epochs) != len(INIT_SEEDS)
                or any(
                    isinstance(epoch, bool)
                    or not isinstance(epoch, int)
                    or not 1 <= epoch <= TrainingConfig().max_epochs
                    for epoch in epochs
                )
            ):
                raise ValueError(f"formal best_epochs are invalid: {key}")
            transform = _require_mapping(
                entry["train_transform"], field=f"diagnostics.{key}.train_transform"
            )
            expected_transform_fields = {
                "feature_mean_sha256",
                "feature_std_sha256",
                "target_median",
                "target_scale",
                "uncertainty_weights_sha256",
                "ranking_pair_count",
            }
            if set(transform) != expected_transform_fields:
                raise ValueError(f"formal train transform fields are invalid: {key}")
            for field in (
                "feature_mean_sha256",
                "feature_std_sha256",
                "uncertainty_weights_sha256",
            ):
                _require_sha(transform[field], field=f"diagnostics.{key}.{field}")
            if not math.isfinite(float(transform["target_median"])):
                raise ValueError(f"formal target median is invalid: {key}")
            if not math.isfinite(float(transform["target_scale"])) or float(
                transform["target_scale"]
            ) <= 0:
                raise ValueError(f"formal target scale is invalid: {key}")
            pair_count = transform["ranking_pair_count"]
            if isinstance(pair_count, bool) or not isinstance(pair_count, int) or pair_count < 0:
                raise ValueError(f"formal ranking pair count is invalid: {key}")


def validate_prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    fold_plan: Mapping[str, Any],
) -> None:
    predictions = _require_mapping(manifest.get("predictions"), field="predictions")
    if len(rows) != int(predictions.get("count", -1)):
        raise ValueError("OOF prediction count differs from manifest")
    if [row.get("prediction_id") for row in rows] != list(
        predictions.get("ordered_prediction_ids", [])
    ):
        raise ValueError("OOF prediction IDs differ from manifest order")
    if [row.get("prediction_sha256") for row in rows] != list(
        predictions.get("ordered_prediction_sha256", [])
    ):
        raise ValueError("OOF prediction hashes differ from manifest order")
    if hashlib.sha256(_jsonl_bytes(rows)).hexdigest() != predictions.get(
        "canonical_records_sha256"
    ):
        raise ValueError("OOF prediction canonical bytes digest mismatch")
    fold_lookup: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    for key, scheme in (
        ("task_heldout_folds", "task_heldout"),
        ("suite_heldout_folds", "suite_heldout"),
    ):
        for fold in fold_plan[key]:
            for order in fold["test_selection_orders"]:
                fold_lookup[(scheme, int(fold["fold_id"]), int(order))] = fold
    model_contract = {
        "task_heldout": {name: (view, loss) for name, view, loss in TASK_MODELS},
        "suite_heldout": {name: (view, loss) for name, view, loss in SUITE_MODELS},
    }
    seen: set[tuple[str, str, int]] = set()
    prediction_ids: set[str] = set()
    for row in rows:
        if set(row) != PREDICTION_FIELDS:
            raise ValueError("OOF prediction fields differ from the frozen schema")
        if row.get("kind") != PREDICTION_KIND or int(row.get("schema_version", -1)) != 1:
            raise ValueError("invalid OOF prediction kind/schema")
        if row.get("prediction_sha256") != _payload_sha(row, "prediction_sha256"):
            raise ValueError("OOF prediction row hash mismatch")
        prediction_id = str(row.get("prediction_id"))
        if prediction_id in prediction_ids:
            raise ValueError("duplicate OOF prediction_id")
        prediction_ids.add(prediction_id)
        if tuple(row.get("init_seeds", ())) != INIT_SEEDS or len(
            row.get("init_predictions", [])
        ) != len(INIT_SEEDS):
            raise ValueError("OOF init prediction contract mismatch")
        numeric = [row.get("prediction"), *row.get("init_predictions", [])]
        if any(not math.isfinite(float(value)) for value in numeric):
            raise ValueError("OOF prediction contains non-finite value")
        init_mean = float(np.mean([float(value) for value in row["init_predictions"]]))
        if not math.isclose(
            float(row["prediction"]), init_mean, rel_tol=1e-7, abs_tol=1e-9
        ):
            raise ValueError("OOF prediction is not the five-init ensemble mean")
        scheme = str(row.get("outer_scheme"))
        model_name = str(row.get("model_name"))
        if scheme not in model_contract or model_name not in model_contract[scheme]:
            raise ValueError("OOF scheme/model is outside the preregistered contract")
        if (row.get("feature_view"), row.get("loss_name")) != model_contract[scheme][
            model_name
        ]:
            raise ValueError("OOF feature/loss contract differs from model_name")
        key = (
            scheme,
            int(row.get("fold_id", -1)),
            int(row.get("selection_order", -1)),
        )
        if key not in fold_lookup:
            raise ValueError("OOF row is not a member of its declared test fold")
        expected_group = (
            group_id(row) if scheme == "task_heldout" else str(row["suite"])
        )
        if row.get("test_group") != expected_group:
            raise ValueError("OOF test_group differs from the row identity")
        uniqueness = (scheme, model_name, int(row["selection_order"]))
        if uniqueness in seen:
            raise ValueError("duplicate OOF scheme/model/state prediction")
        seen.add(uniqueness)


def validate_prediction_bindings(
    rows: Sequence[Mapping[str, Any]], inputs: TrainingInputs
) -> None:
    for row in rows:
        order = int(row["selection_order"])
        if order < 0 or order >= len(inputs.targets):
            raise ValueError("OOF selection_order is outside Target5")
        target = inputs.targets[order]
        feature = inputs.features.index[order]
        exact = {
            "sample_id": str(target["sample_id"]),
            "source_index": int(target["source_index"]),
            "suite": str(target["suite"]),
            "task_index": int(target["task_index"]),
            "task": str(target["task"]),
            "target_id": str(target["target_id"]),
            "target_sha256": str(target["target_sha256"]),
            "input_combined_sha256": str(target["input_hashes"]["combined"]),
            "feature_id": str(feature["feature_id"]),
            "feature_record_sha256": str(feature["feature_record_sha256"]),
            "target5_utility_mean": float(target["utility_mean"]),
            "target5_utility_sem": float(target["utility_sem"]),
            "target5_high_confidence": bool(target["high_confidence"]),
        }
        for field, expected in exact.items():
            if row.get(field) != expected:
                raise ValueError(f"OOF {field} differs from Target5/feature cache")


def prepare_fold_plan_staging(
    output_dir: str | Path, fold_plan: Mapping[str, Any]
) -> tuple[Path, Path, str]:
    """Write and hash-freeze the complete split before any model fitting."""

    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"immutable run output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    try:
        fold_path = staging / "fold_plan.json"
        with fold_path.open("x", encoding="utf-8") as stream:
            json.dump(dict(fold_plan), stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        directory_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        frozen_sha = sha256_file(fold_path)
        if _load_json(fold_path, label="pre-fit fold plan") != dict(fold_plan):
            raise ValueError("pre-fit fold plan bytes do not reproduce the split")
        return output, staging, frozen_sha
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def write_sealed_run(
    output_dir: str | Path,
    *,
    inputs: TrainingInputs,
    fold_plan: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    diagnostics: Mapping[str, Any],
    config: TrainingConfig,
    formal: bool,
    _pretraining_staging: Path | None = None,
    _pretraining_fold_plan_sha256: str | None = None,
) -> Path:
    validate_fold_plan(fold_plan, inputs.targets)
    validate_prediction_bindings(rows, inputs)
    output = Path(output_dir).resolve()
    if _pretraining_staging is None:
        if _pretraining_fold_plan_sha256 is not None:
            raise ValueError("pretraining fold SHA supplied without staging directory")
        output, staging, fold_sha = prepare_fold_plan_staging(output, fold_plan)
    else:
        if _pretraining_fold_plan_sha256 is None:
            raise ValueError("pretraining staging requires its frozen fold SHA")
        staging = Path(_pretraining_staging).resolve()
        if output.exists():
            raise FileExistsError(f"immutable run output already exists: {output}")
        if staging.parent != output.parent or not staging.is_dir():
            raise ValueError("pretraining staging directory is invalid")
        fold_path = staging / "fold_plan.json"
        fold_sha = sha256_file(fold_path)
        if fold_sha != _require_sha(
            _pretraining_fold_plan_sha256,
            field="pretraining_fold_plan_sha256",
        ):
            raise ValueError("pre-fit fold plan changed after its training seal")
        if _load_json(fold_path, label="pre-fit fold plan") != dict(fold_plan):
            raise ValueError("pre-fit fold plan differs from in-memory training split")
    try:
        manifest = build_run_manifest(
            inputs=inputs,
            fold_plan_sha256=fold_sha,
            rows=rows,
            config=config,
            diagnostics=diagnostics,
            formal=formal,
        )
        validate_prediction_rows(rows, manifest, fold_plan)
        manifest_path = staging / "run_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        prediction_path = staging / "oof_predictions.jsonl"
        prediction_path.write_bytes(_jsonl_bytes(rows))
        completion = {
            "schema_version": SCHEMA_VERSION,
            "kind": COMPLETION_KIND,
            "complete": True,
            "run_manifest_sha256": sha256_file(manifest_path),
            "fold_plan_sha256": fold_sha,
            "oof_predictions_sha256": sha256_file(prediction_path),
            "num_predictions": len(rows),
            "compatibility_fingerprint": manifest["compatibility_fingerprint"],
        }
        completion["completion_sha256"] = _completion_sha(completion)
        (staging / "completion.json").write_text(
            json.dumps(completion, indent=2, ensure_ascii=False) + "\n"
        )
        for path in staging.iterdir():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        os.replace(staging, output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return output


def load_sealed_run(
    run_dir: str | Path, *, expected_completion_sha256: str | None = None
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Load and fully revalidate a sealed trainer output for the analyzer."""

    root = Path(run_dir).resolve()
    paths = {
        "manifest": root / "run_manifest.json",
        "fold": root / "fold_plan.json",
        "predictions": root / "oof_predictions.jsonl",
        "completion": root / "completion.json",
    }
    if expected_completion_sha256 is not None:
        if sha256_file(paths["completion"]) != _require_sha(
            expected_completion_sha256, field="expected_completion_sha256"
        ):
            raise ValueError("run completion SHA-256 differs from expected")
    completion = _load_json(paths["completion"], label="run completion")
    if completion.get("kind") != COMPLETION_KIND or completion.get("complete") is not True:
        raise ValueError("run is not sealed complete")
    if completion.get("completion_sha256") != _completion_sha(completion):
        raise ValueError("run completion payload hash mismatch")
    actual = {
        "run_manifest_sha256": sha256_file(paths["manifest"]),
        "fold_plan_sha256": sha256_file(paths["fold"]),
        "oof_predictions_sha256": sha256_file(paths["predictions"]),
    }
    for field, digest in actual.items():
        if completion.get(field) != digest:
            raise ValueError(f"run completion {field} binding mismatch")
    manifest = _load_json(paths["manifest"], label="run manifest")
    fold_plan = _load_json(paths["fold"], label="fold plan")
    rows = _load_jsonl(paths["predictions"], label="OOF predictions")
    if manifest.get("kind") != RUN_KIND or int(manifest.get("schema_version", -1)) != 1:
        raise ValueError("invalid run manifest kind/schema")
    compatibility = _require_mapping(manifest.get("compatibility"), field="compatibility")
    if manifest.get("compatibility_fingerprint") != sha256_json(compatibility):
        raise ValueError("run manifest compatibility fingerprint mismatch")
    if completion.get("compatibility_fingerprint") != manifest.get(
        "compatibility_fingerprint"
    ):
        raise ValueError("completion is not bound to run compatibility")
    if compatibility.get("fold_plan_sha256") != actual["fold_plan_sha256"]:
        raise ValueError("run manifest is not bound to fold plan bytes")
    if int(completion.get("num_predictions", -1)) != len(rows):
        raise ValueError("completion prediction count mismatch")
    validate_prediction_rows(rows, manifest, fold_plan)
    return manifest, fold_plan, rows


def run_offline_feasibility(
    *,
    target_dir: str | Path,
    target_manifest_sha256: str,
    target_targets_sha256: str,
    feature_dir: str | Path,
    feature_completion_sha256: str,
    output_dir: str | Path,
) -> Path:
    inputs = load_training_inputs(
        target_dir=target_dir,
        target_manifest_sha256=target_manifest_sha256,
        target_targets_sha256=target_targets_sha256,
        feature_dir=feature_dir,
        feature_completion_sha256=feature_completion_sha256,
        expected_num_states=100,
    )
    plan = build_fold_plan(inputs.targets, strict_v1=True)
    validate_fold_plan(plan, inputs.targets)
    output, staging, fold_sha = prepare_fold_plan_staging(output_dir, plan)
    config = TrainingConfig()
    try:
        rows, diagnostics = train_oof_predictions(
            inputs, plan, config=config, formal=True
        )
        return write_sealed_run(
            output,
            inputs=inputs,
            fold_plan=plan,
            rows=rows,
            diagnostics=diagnostics,
            config=config,
            formal=True,
            _pretraining_staging=staging,
            _pretraining_fold_plan_sha256=fold_sha,
        )
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
