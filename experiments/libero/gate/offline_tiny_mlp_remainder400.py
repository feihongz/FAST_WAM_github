"""Sealed Target5-only remainder-400 external-test trainer.

The training boundary is intentionally asymmetric: labels are loaded only for
the Pilot-500 remainder states, while the locked original-100 panel contributes
identity and current-state features only.  In particular this module has no
Validation4 input and never parses the original Phase-3 prediction file.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from experiments.libero.gate import offline_tiny_mlp as v1


SCHEMA_VERSION = 1
FOLD_PLAN_KIND = "libero_gate_remainder400_fold_plan"
RUN_KIND = "libero_gate_remainder400_run"
PREDICTION_KIND = "libero_gate_remainder400_external_prediction"
COMPLETION_KIND = "libero_gate_remainder400_completion"
REPRESENTATION = "exact_v1_137"
CURVE_NAMESPACE = "libero_gate_remainder400_curve_v1"
CURVES: tuple[tuple[str, float], ...] = (
    ("q25", 0.25),
    ("q50", 0.50),
    ("q75", 0.75),
    ("q100", 1.00),
)
EXPECTED_EXTRACTOR_FINGERPRINT = (
    "975726ec657e117f2d0c0554e3aaf3a1e31eb343f5f97b9635f7fb4538987d7c"
)
EXPECTED_REMAINDER_STATES = 400
EXPECTED_ORIGINAL_STATES = 100
EXPECTED_REMAINDER_EPISODES = 346
EXPECTED_REMAINDER_TASKS = 39
EXPECTED_PREDICTIONS = 1200
FORMAL_COMPATIBILITY_SHA_FIELDS = frozenset(
    {
        "remainder_target_manifest_sha256",
        "remainder_target_targets_sha256",
        "remainder_target_compatibility_fingerprint",
        "remainder_feature_manifest_sha256",
        "remainder_feature_index_sha256",
        "remainder_features_sha256",
        "remainder_feature_completion_sha256",
        "remainder_feature_completion_payload_sha256",
        "remainder_feature_compatibility_fingerprint",
        "original_feature_manifest_sha256",
        "original_feature_index_sha256",
        "original_features_sha256",
        "original_feature_completion_sha256",
        "original_feature_completion_payload_sha256",
        "original_feature_compatibility_fingerprint",
        "original_fold_source_completion_sha256",
        "original_fold_source_completion_payload_sha256",
        "original_fold_plan_sha256",
        "followup_fold_plan_sha256",
        "protocol_doc_sha256",
        "trainer_source_sha256",
        "exact_v1_core_source_sha256",
        "training_config_sha256",
    }
)


PREDICTION_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "prediction_id",
        "prediction_sha256",
        "prediction_order",
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
        "representation",
        "curve_label",
        "curve_fraction",
        "outer_scheme",
        "fold_id",
        "test_group",
        "model_name",
        "feature_view",
        "loss_name",
        "init_seeds",
        "init_predictions",
        "prediction",
    }
)
FORBIDDEN_PREDICTION_FIELDS = frozenset(
    {
        "utility",
        "utility_mean",
        "utility_sem",
        "target5_utility_mean",
        "target5_utility_sem",
        "target5_high_confidence",
        "validation4_utility",
        "e0",
        "e10",
        "efull",
    }
)


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
                raise ValueError(f"malformed {label} line {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{label} line {line_number} is not an object")
            rows.append(row)
    return rows


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(f"{v1.canonical_json(row)}\n" for row in rows).encode("utf-8")


def _payload_sha(record: Mapping[str, Any], field: str) -> str:
    return v1.sha256_json({key: value for key, value in record.items() if key != field})


def _require_sha(value: Any, *, field: str) -> str:
    return v1._require_sha(value, field=field)


def _group_id(row: Mapping[str, Any]) -> str:
    return v1.group_id(row)


def _episode_id(row: Mapping[str, Any]) -> str:
    return v1.canonical_json(
        {
            "dataset_name": str(row["dataset_name"]),
            "episode_index": int(row["episode_index"]),
        }
    )


def _episode_sha(row: Mapping[str, Any]) -> str:
    payload = (
        f"{CURVE_NAMESPACE}\0{row['dataset_name']}\0{int(row['episode_index'])}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class OriginalFoldSource:
    root: Path
    completion: dict[str, Any]
    fold_plan: dict[str, Any]
    completion_file_sha256: str
    fold_plan_sha256: str


@dataclass(frozen=True)
class FollowupInputs:
    remainder: v1.TrainingInputs
    original_features: v1.LoadedFeatureBundle
    original_fold_source: OriginalFoldSource
    protocol_doc: Path
    protocol_doc_sha256: str
    remainder_feature_completion_file_sha256: str
    original_feature_completion_file_sha256: str


def load_original_fold_source(
    run_dir: str | Path, *, expected_completion_sha256: str
) -> OriginalFoldSource:
    """Verify the original seal but parse only completion and identity fold plan."""

    root = Path(run_dir).resolve()
    completion_path = root / "completion.json"
    fold_path = root / "fold_plan.json"
    manifest_path = root / "run_manifest.json"
    predictions_path = root / "oof_predictions.jsonl"
    expected = _require_sha(
        expected_completion_sha256, field="original_fold_source_completion_sha256"
    )
    completion_file_sha = v1.sha256_file(completion_path)
    if completion_file_sha != expected:
        raise ValueError("original fold-source completion file SHA-256 mismatch")
    completion = _load_json(completion_path, label="original run completion")
    if completion.get("kind") != v1.COMPLETION_KIND or completion.get("complete") is not True:
        raise ValueError("original fold source is not a sealed Phase-3 run")
    if completion.get("completion_sha256") != v1._completion_sha(completion):
        raise ValueError("original fold-source completion payload hash mismatch")
    actual = {
        "run_manifest_sha256": v1.sha256_file(manifest_path),
        "fold_plan_sha256": v1.sha256_file(fold_path),
        "oof_predictions_sha256": v1.sha256_file(predictions_path),
    }
    for field, digest in actual.items():
        if completion.get(field) != digest:
            raise ValueError(f"original fold-source {field} binding mismatch")
    # Deliberately do not JSON-parse run_manifest.json or oof_predictions.jsonl.
    plan = _load_json(fold_path, label="original Phase-3 fold plan")
    return OriginalFoldSource(
        root=root,
        completion=completion,
        fold_plan=plan,
        completion_file_sha256=completion_file_sha,
        fold_plan_sha256=actual["fold_plan_sha256"],
    )


def _extractor_contract(bundle: v1.LoadedFeatureBundle) -> Mapping[str, Any]:
    compatibility = v1._require_mapping(
        bundle.manifest.get("compatibility"), field="feature compatibility"
    )
    if compatibility.get("extractor_fingerprint") != EXPECTED_EXTRACTOR_FINGERPRINT:
        raise ValueError("feature bundle is not the frozen exact-V1 extractor")
    if compatibility.get("feature_dimensions") != v1.FEATURE_DIMS:
        raise ValueError("feature bundle dimensions differ from exact-V1")
    extractor = v1._require_mapping(
        compatibility.get("extractor"), field="feature compatibility.extractor"
    )
    if extractor.get("extractor_fingerprint") != EXPECTED_EXTRACTOR_FINGERPRINT:
        raise ValueError("nested feature extractor fingerprint mismatch")
    return extractor


def load_followup_inputs(
    *,
    remainder_target_dir: str | Path,
    remainder_target_manifest_sha256: str,
    remainder_target_targets_sha256: str,
    remainder_feature_dir: str | Path,
    remainder_feature_completion_sha256: str,
    original_feature_dir: str | Path,
    original_feature_completion_sha256: str,
    original_fold_source_dir: str | Path,
    original_fold_source_completion_sha256: str,
    protocol_doc_path: str | Path,
    protocol_doc_sha256: str,
) -> FollowupInputs:
    """Load Target5 training inputs and label-free locked-test inputs."""

    remainder = v1.load_training_inputs(
        target_dir=remainder_target_dir,
        target_manifest_sha256=remainder_target_manifest_sha256,
        target_targets_sha256=remainder_target_targets_sha256,
        feature_dir=remainder_feature_dir,
        feature_completion_sha256=remainder_feature_completion_sha256,
        expected_num_states=EXPECTED_REMAINDER_STATES,
    )
    original_features = v1.load_feature_bundle(
        original_feature_dir,
        expected_completion_sha256=original_feature_completion_sha256,
    )
    if len(original_features.index) != EXPECTED_ORIGINAL_STATES:
        raise ValueError("locked original feature panel must contain exactly 100 states")
    remainder_extractor = _extractor_contract(remainder.features)
    original_extractor = _extractor_contract(original_features)
    if v1.canonical_json(remainder_extractor) != v1.canonical_json(original_extractor):
        raise ValueError("remainder/original numerical extractor contracts differ")

    fold_source = load_original_fold_source(
        original_fold_source_dir,
        expected_completion_sha256=original_fold_source_completion_sha256,
    )
    v1.validate_fold_plan(fold_source.fold_plan, original_features.index)

    protocol = Path(protocol_doc_path).resolve()
    expected_protocol_sha = _require_sha(protocol_doc_sha256, field="protocol_doc_sha256")
    if v1.sha256_file(protocol) != expected_protocol_sha:
        raise ValueError("follow-up preregistration document SHA-256 mismatch")
    remainder_feature_compatibility = v1._require_mapping(
        remainder.features.manifest.get("compatibility"),
        field="remainder feature compatibility",
    )
    if remainder_feature_compatibility.get("followup_protocol_sha256") != (
        expected_protocol_sha
    ):
        raise ValueError("remainder feature bundle is not bound to this preregistration")
    return FollowupInputs(
        remainder=remainder,
        original_features=original_features,
        original_fold_source=fold_source,
        protocol_doc=protocol,
        protocol_doc_sha256=expected_protocol_sha,
        remainder_feature_completion_file_sha256=v1.sha256_file(
            remainder.features.root / "completion.json"
        ),
        original_feature_completion_file_sha256=v1.sha256_file(
            original_features.root / "completion.json"
        ),
    )


def _orders_by_group(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for order, row in enumerate(rows):
        if int(row["selection_order"]) != order:
            raise ValueError("selection_order must equal exact row order")
        result.setdefault(_group_id(row), []).append(order)
    return result


def _episode_plan_for_group(
    rows: Sequence[Mapping[str, Any]], orders: Sequence[int]
) -> list[dict[str, Any]]:
    episodes: dict[str, list[int]] = {}
    identity: dict[str, Mapping[str, Any]] = {}
    for order in orders:
        row = rows[order]
        key = _episode_id(row)
        episodes.setdefault(key, []).append(order)
        identity[key] = row
    planned = [
        {
            "episode_group": key,
            "dataset_name": str(identity[key]["dataset_name"]),
            "episode_index": int(identity[key]["episode_index"]),
            "episode_sha256": _episode_sha(identity[key]),
            "selection_orders": sorted(episode_orders),
        }
        for key, episode_orders in episodes.items()
    ]
    planned.sort(key=lambda item: (item["episode_sha256"], item["episode_group"]))
    return planned


def _curve_memberships(
    episode_plans: Sequence[Mapping[str, Any]],
) -> dict[str, list[int]]:
    count = len(episode_plans)
    if count < 1:
        raise ValueError("every eligible train task must contain at least one episode")
    memberships: dict[str, list[int]] = {}
    previous: set[int] = set()
    for label, fraction in CURVES:
        take = max(1, math.ceil(fraction * count))
        orders = sorted(
            int(order)
            for episode in episode_plans[:take]
            for order in episode["selection_orders"]
        )
        current = set(orders)
        if not previous.issubset(current):
            raise AssertionError("learning-curve memberships are not nested")
        memberships[label] = orders
        previous = current
    return memberships


def build_followup_fold_plan(
    remainder_targets: Sequence[Mapping[str, Any]],
    original_features: Sequence[Mapping[str, Any]],
    original_fold_plan: Mapping[str, Any],
    *,
    formal: bool = True,
) -> dict[str, Any]:
    """Derive all label-independent splits and nested episode subsets."""

    if formal and len(remainder_targets) != EXPECTED_REMAINDER_STATES:
        raise ValueError("formal remainder panel must contain exactly 400 states")
    if formal and len(original_features) != EXPECTED_ORIGINAL_STATES:
        raise ValueError("formal locked test panel must contain exactly 100 states")
    v1.validate_fold_plan(original_fold_plan, original_features)
    remainder_by_group = _orders_by_group(remainder_targets)
    original_by_group = _orders_by_group(original_features)
    unknown = set(remainder_by_group) - set(original_by_group)
    if unknown:
        raise ValueError("remainder contains task groups outside the locked original panel")
    if formal and len(remainder_by_group) != EXPECTED_REMAINDER_TASKS:
        raise ValueError("formal remainder must contain exactly 39 available task groups")
    episode_ids = {_episode_id(row) for row in remainder_targets}
    episode_task_groups: dict[str, set[str]] = {}
    for row in remainder_targets:
        episode_task_groups.setdefault(_episode_id(row), set()).add(_group_id(row))
    if any(len(groups) != 1 for groups in episode_task_groups.values()):
        raise ValueError("an episode group spans multiple task groups")
    if formal and len(episode_ids) != EXPECTED_REMAINDER_EPISODES:
        raise ValueError("formal remainder must contain exactly 346 episode groups")

    def make_fold(source: Mapping[str, Any], *, scheme: str) -> dict[str, Any]:
        fold_id = int(source["fold_id"])
        heldout = list(source["test_groups"])
        inner = list(source["inner_validation_groups"])
        missing_inner = set(inner) - set(remainder_by_group)
        if missing_inner:
            raise ValueError(
                f"sealed inner-validation groups missing from remainder fold {fold_id}"
            )
        candidates = sorted(set(remainder_by_group) - set(heldout) - set(inner))
        val_orders = sorted(order for key in inner for order in remainder_by_group[key])
        per_task: list[dict[str, Any]] = []
        curve_orders = {label: [] for label, _ in CURVES}
        for key in candidates:
            episodes = _episode_plan_for_group(remainder_targets, remainder_by_group[key])
            memberships = _curve_memberships(episodes)
            per_task.append(
                {
                    "group_id": key,
                    "ordered_episodes": episodes,
                    "curve_selection_orders": memberships,
                }
            )
            for label, _ in CURVES:
                curve_orders[label].extend(memberships[label])
        for label, _ in CURVES:
            curve_orders[label] = sorted(curve_orders[label])
            if set(curve_orders[label]) & set(val_orders):
                raise AssertionError("inner-train/inner-validation leakage")
            present = {_group_id(remainder_targets[order]) for order in curve_orders[label]}
            if present != set(candidates):
                raise ValueError(f"{scheme} fold {fold_id} {label} drops an eligible task")
        return {
            "scheme": scheme,
            "fold_id": fold_id,
            "heldout_groups": heldout,
            "missing_heldout_groups": sorted(set(heldout) - set(remainder_by_group)),
            "original_test_selection_orders": list(source["test_selection_orders"]),
            "original_test_groups": heldout,
            "remainder_inner_validation_groups": inner,
            "remainder_inner_validation_selection_orders": val_orders,
            "remainder_train_groups": candidates,
            "ordered_episode_groups_by_task": per_task,
            "curve_train_selection_orders": curve_orders,
            "membership_sha256": v1.sha256_json(
                {
                    "test": list(source["test_selection_orders"]),
                    "inner_validation": val_orders,
                    "curves": curve_orders,
                }
            ),
        }

    task_folds = [
        make_fold(source, scheme="task_heldout")
        for source in original_fold_plan["task_heldout_folds"]
    ]
    suite_folds = [
        make_fold(source, scheme="suite_heldout")
        for source in original_fold_plan["suite_heldout_folds"]
    ]
    identity_remainder = [
        {
            "selection_order": int(row["selection_order"]),
            "sample_id": str(row["sample_id"]),
            "source_index": int(row["source_index"]),
            "suite": str(row["suite"]),
            "task_index": int(row["task_index"]),
            "task": str(row["task"]),
            "dataset_name": str(row["dataset_name"]),
            "episode_index": int(row["episode_index"]),
        }
        for row in remainder_targets
    ]
    identity_original = [
        {
            "selection_order": int(row["selection_order"]),
            "sample_id": str(row["sample_id"]),
            "source_index": int(row["source_index"]),
            "suite": str(row["suite"]),
            "task_index": int(row["task_index"]),
            "task": str(row["task"]),
        }
        for row in original_features
    ]
    original_feature_bindings = [
        {
            "selection_order": int(row["selection_order"]),
            "sample_id": str(row["sample_id"]),
            "source_index": int(row["source_index"]),
            "suite": str(row["suite"]),
            "task_index": int(row["task_index"]),
            "task": str(row["task"]),
            "target_id": str(row["target_id"]),
            "target_sha256": str(row["target_sha256"]),
            "input_combined_sha256": str(row["input_combined_sha256"]),
            "feature_id": str(row["feature_id"]),
            "feature_record_sha256": str(row["feature_record_sha256"]),
        }
        for row in original_features
    ]
    plan = {
        "schema_version": SCHEMA_VERSION,
        "kind": FOLD_PLAN_KIND,
        "inner_fold_namespace": v1.INNER_FOLD_NAMESPACE,
        "curve_namespace": CURVE_NAMESPACE,
        "curve_fractions": {label: fraction for label, fraction in CURVES},
        "episode_group_fields": ["dataset_name", "episode_index"],
        "episode_sort_rule": "sha256(namespace+NUL+dataset_name+NUL+episode_index)",
        "num_remainder_states": len(remainder_targets),
        "num_original_test_states": len(original_features),
        "num_remainder_task_groups": len(remainder_by_group),
        "num_remainder_episode_groups": len(episode_ids),
        "remainder_identity_sha256": v1.sha256_json(identity_remainder),
        "original_identity_sha256": v1.sha256_json(identity_original),
        "original_feature_bindings_sha256": v1.sha256_json(
            original_feature_bindings
        ),
        "original_fold_membership_sha256": str(
            original_fold_plan["fold_membership_sha256"]
        ),
        "original_task_heldout_folds": original_fold_plan["task_heldout_folds"],
        "original_suite_heldout_folds": original_fold_plan["suite_heldout_folds"],
        "task_heldout_folds": task_folds,
        "suite_heldout_folds": suite_folds,
    }
    plan["followup_membership_sha256"] = v1.sha256_json(
        {"task_heldout_folds": task_folds, "suite_heldout_folds": suite_folds}
    )
    return plan


def validate_followup_fold_plan(
    plan: Mapping[str, Any],
    remainder_targets: Sequence[Mapping[str, Any]],
    original_features: Sequence[Mapping[str, Any]],
    original_fold_plan: Mapping[str, Any],
    *,
    formal: bool = True,
) -> None:
    if plan.get("kind") != FOLD_PLAN_KIND or int(plan.get("schema_version", -1)) != 1:
        raise ValueError("invalid follow-up fold plan kind/schema")
    expected = build_followup_fold_plan(
        remainder_targets, original_features, original_fold_plan, formal=formal
    )
    if v1.canonical_json(plan) != v1.canonical_json(expected):
        raise ValueError("follow-up fold plan differs from deterministic identity plan")


def _feature_view(tensors: Mapping[str, torch.Tensor], name: str) -> torch.Tensor:
    return v1.feature_view(tensors, name)


def train_external_ensemble(
    *,
    train_features: torch.Tensor,
    test_features: torch.Tensor,
    targets: Sequence[Mapping[str, Any]],
    train_orders: Sequence[int],
    validation_orders: Sequence[int],
    loss_name: str,
    config: v1.TrainingConfig,
) -> v1.EnsembleResult:
    """Exact V1 fit with a feature-only external test matrix."""

    if loss_name not in {"hybrid", "huber"}:
        raise ValueError(f"unsupported learned loss {loss_name!r}")
    if not train_orders or not validation_orders or test_features.shape[0] < 1:
        raise ValueError("train/inner-validation/external-test must be non-empty")
    config.validate(formal=False)
    values = train_features.detach().to(dtype=torch.float32, device="cpu")
    external = test_features.detach().to(dtype=torch.float32, device="cpu")
    if values.ndim != 2 or values.shape[0] != len(targets):
        raise ValueError("remainder feature/target rows differ")
    if external.ndim != 2 or external.shape[1] != values.shape[1]:
        raise ValueError("external feature width differs from remainder feature width")
    train_index = torch.tensor(train_orders, dtype=torch.long)
    val_index = torch.tensor(validation_orders, dtype=torch.long)
    feature_scale = v1.fit_standardizer(values[train_index])
    x_train = feature_scale.transform(values[train_index])
    x_val = feature_scale.transform(values[val_index])
    x_test = feature_scale.transform(external)
    y_all = torch.tensor(
        [float(row["utility_mean"]) for row in targets], dtype=torch.float32
    )
    sem_all = torch.tensor(
        [float(row["utility_sem"]) for row in targets], dtype=torch.float32
    )
    target_scale = v1.fit_robust_target_scale(y_all[train_index])
    y_train = target_scale.transform(y_all[train_index])
    y_val = target_scale.transform(y_all[val_index])
    weights = v1.uncertainty_weights(sem_all[train_index], target_scale.scale)
    pair_high, pair_low, pair_weights = v1.build_ranking_pairs(targets, train_orders)

    all_predictions: list[tuple[float, ...]] = []
    best_epochs: list[int] = []
    for seed in config.init_seeds:
        random.seed(int(seed))
        np.random.seed(int(seed) % (2**32))
        torch.manual_seed(int(seed))
        model = v1.TinyUtilityMLP(values.shape[1]).to(dtype=torch.float32, device="cpu")
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
                val_loss = float(
                    F.smooth_l1_loss(model(x_val), y_val, beta=1.0).item()
                )
            if val_loss < best_loss - 1e-12:
                best_loss = val_loss
                best_state = {
                    key: value.detach().clone() for key, value in model.state_dict().items()
                }
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
            raise FloatingPointError("non-finite external prediction")
        all_predictions.append(tuple(float(value) for value in raw.tolist()))
        best_epochs.append(best_epoch)
    ensemble = tuple(
        float(value)
        for value in torch.tensor(all_predictions, dtype=torch.float64).mean(dim=0).tolist()
    )
    return v1.EnsembleResult(
        init_predictions=tuple(all_predictions),
        ensemble_prediction=ensemble,
        best_epochs=tuple(best_epochs),
        train_transform={
            "feature_mean_sha256": v1.tensor_content_sha256(feature_scale.mean),
            "feature_std_sha256": v1.tensor_content_sha256(feature_scale.std),
            "target_median": target_scale.median,
            "target_scale": target_scale.scale,
            "uncertainty_weights_sha256": v1.tensor_content_sha256(weights),
            "ranking_pair_count": int(pair_high.numel()),
        },
    )


def _baseline_external_predictions(
    *,
    model_name: str,
    targets: Sequence[Mapping[str, Any]],
    train_orders: Sequence[int],
    original_rows: Sequence[Mapping[str, Any]],
) -> tuple[float, ...]:
    train = [targets[index] for index in train_orders]
    global_mean = float(np.mean([float(row["utility_mean"]) for row in train]))
    if model_name == "constant_train_mean":
        return tuple(global_mean for _ in original_rows)
    if model_name == "suite_mean_fallback":
        means = {
            suite: float(
                np.mean(
                    [float(row["utility_mean"]) for row in train if row["suite"] == suite]
                )
            )
            for suite in v1.SUITES
            if any(row["suite"] == suite for row in train)
        }
        return tuple(means.get(str(row["suite"]), global_mean) for row in original_rows)
    if model_name == "task_lookup_fallback":
        means = {
            key: float(
                np.mean(
                    [float(row["utility_mean"]) for row in train if _group_id(row) == key]
                )
            )
            for key in {_group_id(row) for row in train}
        }
        return tuple(means.get(_group_id(row), global_mean) for row in original_rows)
    raise ValueError(f"unknown baseline {model_name!r}")


def _prediction_record(
    *,
    original: Mapping[str, Any],
    curve_label: str,
    curve_fraction: float,
    scheme: str,
    fold_id: int,
    model_name: str,
    feature_view: str,
    loss_name: str,
    init_seeds: Sequence[int],
    init_predictions: Sequence[float],
    prediction: float,
) -> dict[str, Any]:
    record = {
        "schema_version": SCHEMA_VERSION,
        "kind": PREDICTION_KIND,
        "prediction_id": (
            f"{REPRESENTATION}/{curve_label}/{scheme}/fold_{fold_id}/"
            f"{model_name}/{original['sample_id']}"
        ),
        "prediction_order": -1,
        "selection_order": int(original["selection_order"]),
        "sample_id": str(original["sample_id"]),
        "source_index": int(original["source_index"]),
        "suite": str(original["suite"]),
        "task_index": int(original["task_index"]),
        "task": str(original["task"]),
        "target_id": str(original["target_id"]),
        "target_sha256": str(original["target_sha256"]),
        "input_combined_sha256": str(original["input_combined_sha256"]),
        "feature_id": str(original["feature_id"]),
        "feature_record_sha256": str(original["feature_record_sha256"]),
        "representation": REPRESENTATION,
        "curve_label": curve_label,
        "curve_fraction": float(curve_fraction),
        "outer_scheme": scheme,
        "fold_id": int(fold_id),
        "test_group": _group_id(original) if scheme == "task_heldout" else str(original["suite"]),
        "model_name": model_name,
        "feature_view": feature_view,
        "loss_name": loss_name,
        "init_seeds": [int(seed) for seed in init_seeds],
        "init_predictions": [float(value) for value in init_predictions],
        "prediction": float(prediction),
    }
    return record


def train_external_predictions(
    inputs: FollowupInputs,
    plan: Mapping[str, Any],
    *,
    config: v1.TrainingConfig = v1.TrainingConfig(),
    formal: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config.validate(formal=formal)
    validate_followup_fold_plan(
        plan,
        inputs.remainder.targets,
        inputs.original_features.index,
        inputs.original_fold_source.fold_plan,
        formal=formal,
    )
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(int(config.cpu_threads))
    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {"fits": {}}
    try:
        curve_map = dict(CURVES)
        for fold in plan["task_heldout_folds"]:
            fold_id = int(fold["fold_id"])
            test_orders = list(fold["original_test_selection_orders"])
            original_rows = [inputs.original_features.index[index] for index in test_orders]
            val_orders = list(fold["remainder_inner_validation_selection_orders"])
            for curve_label, fraction in CURVES:
                train_orders = list(fold["curve_train_selection_orders"][curve_label])
                models = v1.TASK_MODELS if curve_label == "q100" else (
                    ("full_hybrid", "full", "hybrid"),
                )
                for model_name, view, loss_name in models:
                    if loss_name == "baseline":
                        ensemble = _baseline_external_predictions(
                            model_name=model_name,
                            targets=inputs.remainder.targets,
                            train_orders=sorted(train_orders + val_orders),
                            original_rows=original_rows,
                        )
                        init_matrix = tuple(tuple(ensemble) for _ in config.init_seeds)
                        diagnostic: dict[str, Any] = {"baseline": True}
                    else:
                        result = train_external_ensemble(
                            train_features=_feature_view(inputs.remainder.features.tensors, view),
                            test_features=_feature_view(inputs.original_features.tensors, view)[test_orders],
                            targets=inputs.remainder.targets,
                            train_orders=train_orders,
                            validation_orders=val_orders,
                            loss_name=loss_name,
                            config=config,
                        )
                        ensemble = result.ensemble_prediction
                        init_matrix = result.init_predictions
                        diagnostic = {
                            "best_epochs": list(result.best_epochs),
                            "train_transform": result.train_transform,
                        }
                    diagnostic.update(
                        {
                            "train_state_count": len(train_orders),
                            "inner_validation_state_count": len(val_orders),
                            "test_state_count": len(test_orders),
                            "train_membership_sha256": v1.sha256_json(train_orders),
                        }
                    )
                    key = f"{curve_label}/task_heldout/fold_{fold_id}/{model_name}"
                    diagnostics["fits"][key] = diagnostic
                    for local, original in enumerate(original_rows):
                        rows.append(
                            _prediction_record(
                                original=original,
                                curve_label=curve_label,
                                curve_fraction=curve_map[curve_label],
                                scheme="task_heldout",
                                fold_id=fold_id,
                                model_name=model_name,
                                feature_view=view,
                                loss_name=loss_name,
                                init_seeds=config.init_seeds,
                                init_predictions=[values[local] for values in init_matrix],
                                prediction=ensemble[local],
                            )
                        )

        for fold in plan["suite_heldout_folds"]:
            fold_id = int(fold["fold_id"])
            test_orders = list(fold["original_test_selection_orders"])
            train_orders = list(fold["curve_train_selection_orders"]["q100"])
            val_orders = list(fold["remainder_inner_validation_selection_orders"])
            result = train_external_ensemble(
                train_features=inputs.remainder.features.tensors["full"],
                test_features=inputs.original_features.tensors["full"][test_orders],
                targets=inputs.remainder.targets,
                train_orders=train_orders,
                validation_orders=val_orders,
                loss_name="hybrid",
                config=config,
            )
            key = f"q100/suite_heldout/fold_{fold_id}/full_hybrid"
            diagnostics["fits"][key] = {
                "best_epochs": list(result.best_epochs),
                "train_transform": result.train_transform,
                "train_state_count": len(train_orders),
                "inner_validation_state_count": len(val_orders),
                "test_state_count": len(test_orders),
                "train_membership_sha256": v1.sha256_json(train_orders),
            }
            for local, order in enumerate(test_orders):
                rows.append(
                    _prediction_record(
                        original=inputs.original_features.index[order],
                        curve_label="q100",
                        curve_fraction=1.0,
                        scheme="suite_heldout",
                        fold_id=fold_id,
                        model_name="full_hybrid",
                        feature_view="full",
                        loss_name="hybrid",
                        init_seeds=config.init_seeds,
                        init_predictions=[values[local] for values in result.init_predictions],
                        prediction=result.ensemble_prediction[local],
                    )
                )
    finally:
        torch.set_num_threads(previous_threads)

    curve_rank = {label: index for index, (label, _) in enumerate(CURVES)}
    scheme_rank = {"task_heldout": 0, "suite_heldout": 1}
    model_rank = {name: index for index, (name, _, _) in enumerate(v1.TASK_MODELS)}
    rows.sort(
        key=lambda row: (
            curve_rank[row["curve_label"]],
            scheme_rank[row["outer_scheme"]],
            model_rank[row["model_name"]],
            int(row["selection_order"]),
        )
    )
    for order, row in enumerate(rows):
        row["prediction_order"] = order
        row["prediction_sha256"] = _payload_sha(row, "prediction_sha256")
    if formal and len(rows) != EXPECTED_PREDICTIONS:
        raise AssertionError(f"formal run produced {len(rows)} predictions, expected 1200")
    return rows, diagnostics


def _feature_file_bindings(
    prefix: str,
    bundle: v1.LoadedFeatureBundle,
    completion_file_sha256: str,
) -> dict[str, Any]:
    return {
        f"{prefix}_feature_manifest_sha256": bundle.manifest_sha256,
        f"{prefix}_feature_index_sha256": bundle.index_sha256,
        f"{prefix}_features_sha256": bundle.features_sha256,
        f"{prefix}_feature_completion_sha256": completion_file_sha256,
        f"{prefix}_feature_completion_payload_sha256": bundle.completion_sha256,
        f"{prefix}_feature_compatibility_fingerprint": bundle.manifest[
            "compatibility_fingerprint"
        ],
    }


def _source_sha256() -> str:
    return v1.sha256_file(Path(__file__).resolve())


def build_run_manifest(
    *,
    inputs: FollowupInputs,
    fold_plan_sha256: str,
    rows: Sequence[Mapping[str, Any]],
    diagnostics: Mapping[str, Any],
    config: v1.TrainingConfig,
    formal: bool,
) -> dict[str, Any]:
    compatibility = {
        "schema_version": SCHEMA_VERSION,
        "kind": RUN_KIND,
        "formal_protocol": bool(formal),
        "representation": REPRESENTATION,
        "curve_namespace": CURVE_NAMESPACE,
        "curve_fractions": {label: fraction for label, fraction in CURVES},
        "remainder_target_manifest_sha256": inputs.remainder.target_manifest_sha256,
        "remainder_target_targets_sha256": inputs.remainder.target_targets_sha256,
        "remainder_target_compatibility_fingerprint": inputs.remainder.target_manifest[
            "compatibility_fingerprint"
        ],
        **_feature_file_bindings(
            "remainder",
            inputs.remainder.features,
            inputs.remainder_feature_completion_file_sha256,
        ),
        **_feature_file_bindings(
            "original",
            inputs.original_features,
            inputs.original_feature_completion_file_sha256,
        ),
        "extractor_fingerprint": EXPECTED_EXTRACTOR_FINGERPRINT,
        "original_fold_source_completion_sha256": (
            inputs.original_fold_source.completion_file_sha256
        ),
        "original_fold_plan_sha256": inputs.original_fold_source.fold_plan_sha256,
        "original_fold_source_completion_payload_sha256": str(
            inputs.original_fold_source.completion["completion_sha256"]
        ),
        "followup_fold_plan_sha256": fold_plan_sha256,
        "protocol_doc_sha256": inputs.protocol_doc_sha256,
        "trainer_source_sha256": _source_sha256(),
        "exact_v1_core_source_sha256": v1.sha256_file(Path(v1.__file__).resolve()),
        "training_config_sha256": v1.sha256_json(asdict(config)),
        "num_remainder_states": len(inputs.remainder.targets),
        "num_original_test_states": len(inputs.original_features.index),
        "num_predictions": len(rows),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": RUN_KIND,
        "compatibility": compatibility,
        "compatibility_fingerprint": v1.sha256_json(compatibility),
        "training_config": asdict(config),
        "models": {
            "q25_q50_q75_task_heldout": ["full_hybrid"],
            "q100_task_heldout": [name for name, _, _ in v1.TASK_MODELS],
            "q100_suite_heldout": [name for name, _, _ in v1.SUITE_MODELS],
            "primary": "q100/task_heldout/full_hybrid",
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


def validate_prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    fold_plan: Mapping[str, Any],
    *,
    formal: bool = True,
) -> None:
    predictions = v1._require_mapping(manifest.get("predictions"), field="predictions")
    if len(rows) != int(predictions.get("count", -1)):
        raise ValueError("prediction count differs from manifest")
    if formal and len(rows) != EXPECTED_PREDICTIONS:
        raise ValueError("formal prediction file must contain exactly 1200 rows")
    if [row.get("prediction_id") for row in rows] != predictions.get(
        "ordered_prediction_ids"
    ):
        raise ValueError("prediction IDs differ from manifest order")
    if [row.get("prediction_sha256") for row in rows] != predictions.get(
        "ordered_prediction_sha256"
    ):
        raise ValueError("prediction hashes differ from manifest order")
    if hashlib.sha256(_jsonl_bytes(rows)).hexdigest() != predictions.get(
        "canonical_records_sha256"
    ):
        raise ValueError("prediction canonical digest mismatch")

    expected_original_fold: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    for scheme, key in (
        ("task_heldout", "task_heldout_folds"),
        ("suite_heldout", "suite_heldout_folds"),
    ):
        for fold in fold_plan[key]:
            for order in fold["original_test_selection_orders"]:
                expected_original_fold[(scheme, int(fold["fold_id"]), int(order))] = fold
    curve_map = dict(CURVES)
    model_contract = {name: (view, loss) for name, view, loss in v1.TASK_MODELS}
    seen: set[tuple[str, str, str, int]] = set()
    identity_by_order: dict[int, dict[str, Any]] = {}
    for expected_order, row in enumerate(rows):
        if set(row) != PREDICTION_FIELDS or set(row) & FORBIDDEN_PREDICTION_FIELDS:
            raise ValueError("prediction fields differ from label-free frozen schema")
        if row.get("kind") != PREDICTION_KIND or int(row.get("schema_version", -1)) != 1:
            raise ValueError("invalid prediction kind/schema")
        if int(row.get("prediction_order", -1)) != expected_order:
            raise ValueError("prediction_order must equal exact file order")
        if row.get("prediction_sha256") != _payload_sha(row, "prediction_sha256"):
            raise ValueError("prediction row payload hash mismatch")
        label = str(row.get("curve_label"))
        if label not in curve_map or float(row.get("curve_fraction")) != curve_map[label]:
            raise ValueError("invalid curve label/fraction")
        if row.get("representation") != REPRESENTATION:
            raise ValueError("prediction representation is not exact-V1 137")
        scheme = str(row.get("outer_scheme"))
        model = str(row.get("model_name"))
        if label != "q100" and (scheme != "task_heldout" or model != "full_hybrid"):
            raise ValueError("q25/q50/q75 may only contain task-heldout full_hybrid")
        if scheme == "suite_heldout" and (label != "q100" or model != "full_hybrid"):
            raise ValueError("suite-heldout is only q100 full_hybrid")
        if model not in model_contract:
            raise ValueError("prediction model is outside frozen V1 panel")
        if (row.get("feature_view"), row.get("loss_name")) != model_contract[model]:
            raise ValueError("prediction feature/loss differs from model name")
        key = (scheme, int(row["fold_id"]), int(row["selection_order"]))
        if key not in expected_original_fold:
            raise ValueError("prediction is outside its locked external fold")
        expected_group = (
            _group_id(row) if scheme == "task_heldout" else str(row["suite"])
        )
        if row.get("test_group") != expected_group:
            raise ValueError("prediction test_group differs from locked identity")
        expected_prediction_id = (
            f"{REPRESENTATION}/{label}/{scheme}/fold_{int(row['fold_id'])}/"
            f"{model}/{row['sample_id']}"
        )
        if row.get("prediction_id") != expected_prediction_id:
            raise ValueError("prediction_id differs from frozen naming contract")
        unique = (label, scheme, model, int(row["selection_order"]))
        if unique in seen:
            raise ValueError("duplicate candidate/model/state prediction")
        seen.add(unique)
        identity = {
            field: row[field]
            for field in (
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
            )
        }
        selection_order = int(row["selection_order"])
        if selection_order in identity_by_order and identity_by_order[selection_order] != identity:
            raise ValueError("candidate rows disagree on locked original feature identity")
        identity_by_order[selection_order] = identity
        seeds = tuple(row.get("init_seeds", ()))
        if formal and seeds != v1.INIT_SEEDS:
            raise ValueError("formal prediction init seeds differ from exact V1")
        values = list(row.get("init_predictions", ()))
        if len(values) != len(seeds) or not values:
            raise ValueError("init prediction count differs from init seeds")
        if any(not math.isfinite(float(value)) for value in [*values, row["prediction"]]):
            raise ValueError("prediction contains non-finite value")
        if not math.isclose(
            float(row["prediction"]),
            float(np.mean([float(value) for value in values])),
            rel_tol=1e-7,
            abs_tol=1e-9,
        ):
            raise ValueError("prediction is not the init ensemble mean")

    if formal:
        expected_counts = {
            ("q25", "task_heldout", "full_hybrid"): 100,
            ("q50", "task_heldout", "full_hybrid"): 100,
            ("q75", "task_heldout", "full_hybrid"): 100,
            ("q100", "suite_heldout", "full_hybrid"): 100,
        }
        for name, _, _ in v1.TASK_MODELS:
            expected_counts[("q100", "task_heldout", name)] = 100
        actual: dict[tuple[str, str, str], int] = {}
        for row in rows:
            key = (row["curve_label"], row["outer_scheme"], row["model_name"])
            actual[key] = actual.get(key, 0) + 1
        if actual != expected_counts:
            raise ValueError("formal 1200-row candidate/model panel differs from protocol")
        if sorted(identity_by_order) != list(range(EXPECTED_ORIGINAL_STATES)):
            raise ValueError("formal rows do not cover exact original selection order 0..99")
        ordered_bindings = [identity_by_order[index] for index in range(100)]
        if v1.sha256_json(ordered_bindings) != fold_plan.get(
            "original_feature_bindings_sha256"
        ):
            raise ValueError(
                "formal rows differ from frozen original feature bindings"
            )


def validate_formal_run_contract(
    manifest: Mapping[str, Any],
    fold_plan: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    if manifest.get("kind") != RUN_KIND or int(manifest.get("schema_version", -1)) != 1:
        raise ValueError("invalid formal run kind/schema")
    compatibility = v1._require_mapping(manifest.get("compatibility"), field="compatibility")
    if manifest.get("compatibility_fingerprint") != v1.sha256_json(compatibility):
        raise ValueError("formal compatibility fingerprint mismatch")
    exact = {
        "schema_version": SCHEMA_VERSION,
        "kind": RUN_KIND,
        "formal_protocol": True,
        "representation": REPRESENTATION,
        "curve_namespace": CURVE_NAMESPACE,
        "curve_fractions": {label: fraction for label, fraction in CURVES},
        "extractor_fingerprint": EXPECTED_EXTRACTOR_FINGERPRINT,
        "num_remainder_states": EXPECTED_REMAINDER_STATES,
        "num_original_test_states": EXPECTED_ORIGINAL_STATES,
        "num_predictions": EXPECTED_PREDICTIONS,
    }
    for field, value in exact.items():
        if compatibility.get(field) != value:
            raise ValueError(f"formal compatibility {field} differs from protocol")
    expected_fields = set(exact) | set(FORMAL_COMPATIBILITY_SHA_FIELDS)
    if set(compatibility) != expected_fields:
        raise ValueError("formal compatibility fields differ from frozen schema")
    for field in FORMAL_COMPATIBILITY_SHA_FIELDS:
        _require_sha(compatibility[field], field=f"compatibility.{field}")
    if v1.canonical_json(manifest.get("training_config")) != v1.canonical_json(
        asdict(v1.TrainingConfig())
    ):
        raise ValueError("formal TrainingConfig differs from exact V1")
    expected_models = {
        "q25_q50_q75_task_heldout": ["full_hybrid"],
        "q100_task_heldout": [name for name, _, _ in v1.TASK_MODELS],
        "q100_suite_heldout": [name for name, _, _ in v1.SUITE_MODELS],
        "primary": "q100/task_heldout/full_hybrid",
    }
    if manifest.get("models") != expected_models:
        raise ValueError("formal model panel differs from protocol")
    if fold_plan.get("kind") != FOLD_PLAN_KIND:
        raise ValueError("formal fold plan kind is invalid")
    if int(fold_plan.get("num_remainder_states", -1)) != EXPECTED_REMAINDER_STATES:
        raise ValueError("formal fold plan remainder count mismatch")
    if int(fold_plan.get("num_original_test_states", -1)) != EXPECTED_ORIGINAL_STATES:
        raise ValueError("formal fold plan original count mismatch")
    if int(fold_plan.get("num_remainder_task_groups", -1)) != EXPECTED_REMAINDER_TASKS:
        raise ValueError("formal fold plan task count mismatch")
    if int(fold_plan.get("num_remainder_episode_groups", -1)) != EXPECTED_REMAINDER_EPISODES:
        raise ValueError("formal fold plan episode count mismatch")
    validate_prediction_rows(rows, manifest, fold_plan, formal=True)


def _assert_inputs_unchanged(inputs: FollowupInputs) -> None:
    checks = {
        inputs.remainder.target_dir / "manifest.json": inputs.remainder.target_manifest_sha256,
        inputs.remainder.target_dir / "targets.jsonl": inputs.remainder.target_targets_sha256,
        inputs.remainder.features.root / "manifest.json": inputs.remainder.features.manifest_sha256,
        inputs.remainder.features.root / "feature_index.jsonl": inputs.remainder.features.index_sha256,
        inputs.remainder.features.root / "features.safetensors": inputs.remainder.features.features_sha256,
        inputs.original_features.root / "manifest.json": inputs.original_features.manifest_sha256,
        inputs.original_features.root / "feature_index.jsonl": inputs.original_features.index_sha256,
        inputs.original_features.root / "features.safetensors": inputs.original_features.features_sha256,
        inputs.original_fold_source.root / "completion.json": (
            inputs.original_fold_source.completion_file_sha256
        ),
        inputs.original_fold_source.root / "fold_plan.json": (
            inputs.original_fold_source.fold_plan_sha256
        ),
        inputs.original_fold_source.root / "run_manifest.json": str(
            inputs.original_fold_source.completion["run_manifest_sha256"]
        ),
        inputs.original_fold_source.root / "oof_predictions.jsonl": str(
            inputs.original_fold_source.completion["oof_predictions_sha256"]
        ),
        inputs.protocol_doc: inputs.protocol_doc_sha256,
        inputs.remainder.features.root / "completion.json": (
            inputs.remainder_feature_completion_file_sha256
        ),
        inputs.original_features.root / "completion.json": (
            inputs.original_feature_completion_file_sha256
        ),
    }
    for path, expected in checks.items():
        if v1.sha256_file(path) != expected:
            raise ValueError(f"sealed input changed during fit: {path}")


def write_sealed_run(
    output_dir: str | Path,
    *,
    inputs: FollowupInputs,
    fold_plan: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    diagnostics: Mapping[str, Any],
    config: v1.TrainingConfig,
    formal: bool,
    staging: Path,
    frozen_fold_plan_sha256: str,
) -> Path:
    output = Path(output_dir).resolve()
    fold_path = staging / "fold_plan.json"
    if output.exists() or not staging.is_dir():
        raise FileExistsError("immutable output exists or staging is missing")
    if v1.sha256_file(fold_path) != frozen_fold_plan_sha256:
        raise ValueError("pre-fit fold plan changed during training")
    validate_followup_fold_plan(
        fold_plan,
        inputs.remainder.targets,
        inputs.original_features.index,
        inputs.original_fold_source.fold_plan,
        formal=formal,
    )
    _assert_inputs_unchanged(inputs)
    try:
        manifest = build_run_manifest(
            inputs=inputs,
            fold_plan_sha256=frozen_fold_plan_sha256,
            rows=rows,
            diagnostics=diagnostics,
            config=config,
            formal=formal,
        )
        validate_prediction_rows(rows, manifest, fold_plan, formal=formal)
        if formal:
            validate_formal_run_contract(manifest, fold_plan, rows)
        manifest_path = staging / "run_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        predictions_path = staging / "oof_predictions.jsonl"
        predictions_path.write_bytes(_jsonl_bytes(rows))
        completion = {
            "schema_version": SCHEMA_VERSION,
            "kind": COMPLETION_KIND,
            "complete": True,
            "run_manifest_sha256": v1.sha256_file(manifest_path),
            "fold_plan_sha256": frozen_fold_plan_sha256,
            "oof_predictions_sha256": v1.sha256_file(predictions_path),
            "num_predictions": len(rows),
            "compatibility_fingerprint": manifest["compatibility_fingerprint"],
        }
        completion["completion_sha256"] = _payload_sha(completion, "completion_sha256")
        (staging / "completion.json").write_text(
            json.dumps(completion, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
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
    run_dir: str | Path, *, expected_completion_sha256: str
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    root = Path(run_dir).resolve()
    paths = {
        "manifest": root / "run_manifest.json",
        "fold": root / "fold_plan.json",
        "predictions": root / "oof_predictions.jsonl",
        "completion": root / "completion.json",
    }
    expected = _require_sha(expected_completion_sha256, field="expected_completion_sha256")
    if v1.sha256_file(paths["completion"]) != expected:
        raise ValueError("run completion FILE SHA-256 differs from expected")
    completion = _load_json(paths["completion"], label="run completion")
    if completion.get("kind") != COMPLETION_KIND or completion.get("complete") is not True:
        raise ValueError("follow-up run is not sealed complete")
    if completion.get("completion_sha256") != _payload_sha(completion, "completion_sha256"):
        raise ValueError("run completion payload hash mismatch")
    actual = {
        "run_manifest_sha256": v1.sha256_file(paths["manifest"]),
        "fold_plan_sha256": v1.sha256_file(paths["fold"]),
        "oof_predictions_sha256": v1.sha256_file(paths["predictions"]),
    }
    for field, digest in actual.items():
        if completion.get(field) != digest:
            raise ValueError(f"run completion {field} binding mismatch")
    manifest = _load_json(paths["manifest"], label="run manifest")
    plan = _load_json(paths["fold"], label="fold plan")
    rows = _load_jsonl(paths["predictions"], label="external predictions")
    if int(completion.get("num_predictions", -1)) != len(rows):
        raise ValueError("completion prediction count mismatch")
    if completion.get("compatibility_fingerprint") != manifest.get(
        "compatibility_fingerprint"
    ):
        raise ValueError("completion is not bound to run compatibility")
    compatibility = v1._require_mapping(manifest.get("compatibility"), field="compatibility")
    if compatibility.get("followup_fold_plan_sha256") != actual["fold_plan_sha256"]:
        raise ValueError("run manifest is not bound to follow-up fold plan bytes")
    validate_formal_run_contract(manifest, plan, rows)
    return manifest, plan, rows


def run_remainder400_followup(
    *,
    remainder_target_dir: str | Path,
    remainder_target_manifest_sha256: str,
    remainder_target_targets_sha256: str,
    remainder_feature_dir: str | Path,
    remainder_feature_completion_sha256: str,
    original_feature_dir: str | Path,
    original_feature_completion_sha256: str,
    original_fold_source_dir: str | Path,
    original_fold_source_completion_sha256: str,
    protocol_doc_path: str | Path,
    protocol_doc_sha256: str,
    output_dir: str | Path,
) -> Path:
    inputs = load_followup_inputs(
        remainder_target_dir=remainder_target_dir,
        remainder_target_manifest_sha256=remainder_target_manifest_sha256,
        remainder_target_targets_sha256=remainder_target_targets_sha256,
        remainder_feature_dir=remainder_feature_dir,
        remainder_feature_completion_sha256=remainder_feature_completion_sha256,
        original_feature_dir=original_feature_dir,
        original_feature_completion_sha256=original_feature_completion_sha256,
        original_fold_source_dir=original_fold_source_dir,
        original_fold_source_completion_sha256=original_fold_source_completion_sha256,
        protocol_doc_path=protocol_doc_path,
        protocol_doc_sha256=protocol_doc_sha256,
    )
    plan = build_followup_fold_plan(
        inputs.remainder.targets,
        inputs.original_features.index,
        inputs.original_fold_source.fold_plan,
        formal=True,
    )
    output, staging, frozen_sha = v1.prepare_fold_plan_staging(output_dir, plan)
    config = v1.TrainingConfig()
    try:
        rows, diagnostics = train_external_predictions(
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
            staging=staging,
            frozen_fold_plan_sha256=frozen_sha,
        )
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
