#!/usr/bin/env python3
"""Export held-out Gate probabilities and deterministic compute thresholds."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


# Executing ``python scripts/calibrate_video_gate.py`` places only ``scripts``
# on sys.path.  Add the checkout root so this entrypoint can reuse the exact
# training-contract validators without copying their semantics.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import train_video_gate as gate_training  # noqa: E402
from fastwam.alignment.checkpointing import (  # noqa: E402
    canonical_json_sha256,
    read_git_identity,
    resolve_base_checkpoint,
    sha256_file,
    write_json_atomic,
    write_text_atomic,
)
from fastwam.gating.artifacts import (  # noqa: E402
    load_validated_merged_label_artifact,
)
from fastwam.gating.calibration import (  # noqa: E402
    CALIBRATION_ALGORITHM,
    calibrate_probability_thresholds,
)
from fastwam.gating.contracts import require_sha256  # noqa: E402
from fastwam.gating.dataset import Stage2GateDataset  # noqa: E402
from fastwam.gating.eval_runtime import load_gate_for_evaluation  # noqa: E402
from fastwam.gating.metrics import compute_gate_binary_metrics  # noqa: E402
from fastwam.gating.runtime_identity import (  # noqa: E402
    collect_numerical_runtime_environment,
)
from fastwam.gating.source_guard import (  # noqa: E402
    capture_selected_source_snapshot,
)
from fastwam.models.video_gate import BinaryVideoGate  # noqa: E402
from fastwam.utils.config_resolvers import register_default_resolvers  # noqa: E402
from fastwam.utils.pytorch_utils import set_global_seed  # noqa: E402


register_default_resolvers()

_CALIBRATION_KEYS = {
    "gate_checkpoint",
    "gate_checkpoint_sha256",
    "gate_run_identity",
    "gate_run_identity_sha256",
    "output_dir",
    "source_split",
    "target_with_rates",
    "configured_video_steps",
    "expected_validation_samples",
    "validation_batch_size",
    "validation_num_workers",
    "validation_pin_memory",
    "metric_abs_tolerance",
    "require_exact_training_numerical_runtime",
    "progress_every_batches",
    "records_file",
    "thresholds_file",
    "manifest_file",
    "complete_file",
}
_RUN_IDENTITY_KEYS = {
    "schema_version",
    "kind",
    "training_config",
    "training_config_sha256",
    "training_identity",
}
_TRAINING_IDENTITY_KEYS = {
    "label_manifest_sha256",
    "adapter_checkpoint_sha256",
    "base_checkpoint_sha256",
    "data_manifest_sha256",
    "episode_split_assignment_sha256",
    "training_config_sha256",
    "git_identity",
}
_BATCH_KEYS = {
    "input_image",
    "context",
    "context_mask",
    "proprio",
    "label",
    "sample_weight",
    "sample_id",
}
_METRIC_FIELDS = (
    "objective_bce",
    "bce",
    "auroc",
    "auprc",
    "positive_rate",
    "predicted_positive_rate",
    "expected_calibration_error",
    "num_examples",
    "num_batches",
)


def _resolved_configs(
    config: DictConfig | Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not OmegaConf.is_config(config):
        config = OmegaConf.create(config)
    payload = OmegaConf.to_container(config, resolve=True)
    if not isinstance(payload, dict):
        raise TypeError("Gate calibration config must resolve to a mapping")
    calibration = payload.pop("calibration", None)
    if not isinstance(calibration, Mapping):
        raise TypeError("calibration must be a mapping")
    calibration = dict(calibration)
    if set(calibration) != _CALIBRATION_KEYS:
        raise ValueError(
            "calibration fields do not match schema; "
            f"missing={sorted(_CALIBRATION_KEYS - set(calibration))}, "
            f"unexpected={sorted(set(calibration) - _CALIBRATION_KEYS)}"
        )
    return gate_training._resolved_config(payload), calibration


def _positive_int(value: Any, *, field: str, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return int(value)


def _finite_nonnegative(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field} must be finite and non-negative")
    return result


def _safe_basename(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ValueError(f"{field} must be a non-empty local basename")
    if value in {".", ".."}:
        raise ValueError(f"{field} must be a non-empty local basename")
    return value


def _load_json_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must contain a JSON mapping")
    return dict(payload)


def _load_gate_run_identity(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> tuple[Path, dict[str, Any], str]:
    source = Path(path).expanduser().resolve(strict=True)
    expected_file_sha256 = require_sha256(
        expected_file_sha256, field="expected Gate run identity file SHA256"
    )
    actual_file_sha256 = sha256_file(source)
    if actual_file_sha256 != expected_file_sha256:
        raise ValueError(
            "Gate run identity file SHA256 mismatch: "
            f"expected={expected_file_sha256}, actual={actual_file_sha256}"
        )
    payload = _load_json_mapping(source, label="Gate run identity")
    if set(payload) != _RUN_IDENTITY_KEYS:
        raise ValueError("Gate run identity fields do not match schema")
    if (
        payload.get("schema_version") != 1
        or payload.get("kind") != "stage2_binary_video_gate_run_identity"
    ):
        raise ValueError("unsupported Gate run identity")
    training_config = payload.get("training_config")
    training_identity = payload.get("training_identity")
    if not isinstance(training_config, Mapping):
        raise TypeError("Gate run identity training_config must be a mapping")
    if not isinstance(training_identity, Mapping):
        raise TypeError("Gate run identity training_identity must be a mapping")
    training_identity = dict(training_identity)
    if set(training_identity) != _TRAINING_IDENTITY_KEYS:
        raise ValueError("Gate run training_identity fields do not match schema")
    recorded_config_sha = require_sha256(
        payload.get("training_config_sha256"),
        field="Gate run identity training_config_sha256",
    )
    if canonical_json_sha256(dict(training_config)) != recorded_config_sha:
        raise ValueError("Gate run identity training_config self-hash mismatch")
    if training_identity.get("training_config_sha256") != recorded_config_sha:
        raise ValueError("Gate run training_identity config SHA256 mismatch")
    payload["training_config"] = dict(training_config)
    payload["training_identity"] = training_identity
    return source, payload, actual_file_sha256


def _validate_recorded_training_contract(
    recorded: Mapping[str, Any],
    *,
    resolved: Mapping[str, Any],
    gate_config: Mapping[str, Any],
    training: Mapping[str, Any],
) -> None:
    if recorded.get("kind") != "stage2_binary_video_gate_training_contract":
        raise ValueError("Gate run has an unsupported training contract kind")
    if recorded.get("schema_version") not in {1, 2}:
        raise ValueError("Gate run has an unsupported training contract version")
    if recorded.get("data") != resolved["data"]:
        raise ValueError("current data config differs from Gate training contract")
    if recorded.get("gate") != dict(gate_config):
        raise ValueError("current Gate config differs from Gate training contract")
    if recorded.get("training") != dict(training):
        raise ValueError("current training config differs from Gate training contract")
    runtime = recorded.get("runtime")
    if not isinstance(runtime, Mapping):
        raise TypeError("Gate training contract runtime must be a mapping")
    current_runtime = resolved["runtime"]
    for field in ("device", "require_cuda", "deterministic_algorithms"):
        if runtime.get(field) != current_runtime[field]:
            raise ValueError(
                f"current runtime.{field} differs from Gate training contract"
            )


def _validate_batch(
    batch: Mapping[str, Any],
    *,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, tuple[str, ...]]:
    if not isinstance(batch, Mapping) or set(batch) != _BATCH_KEYS:
        raise ValueError("Gate calibration batch fields do not match schema")
    inputs: dict[str, torch.Tensor] = {}
    for field in ("input_image", "context", "context_mask", "proprio"):
        value = batch[field]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Gate calibration batch {field} must be a tensor")
        inputs[field] = value.to(device=device)
    batch_size = int(inputs["input_image"].shape[0])

    labels = batch["label"]
    if not isinstance(labels, torch.Tensor) or labels.shape != (batch_size,):
        raise ValueError("Gate calibration labels must have shape [B]")
    labels = labels.detach().to(device="cpu")
    if labels.dtype == torch.bool:
        labels = labels.to(dtype=torch.int64)
    elif labels.is_floating_point() or labels.dtype in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        if not bool(((labels == 0) | (labels == 1)).all().item()):
            raise ValueError("Gate calibration labels must be binary")
        labels = labels.to(dtype=torch.int64)
    else:
        raise TypeError("Gate calibration labels must be bool or numeric binary")

    weights = batch["sample_weight"]
    if (
        not isinstance(weights, torch.Tensor)
        or weights.shape != (batch_size,)
        or not weights.is_floating_point()
    ):
        raise ValueError("Gate calibration sample_weight must be floating [B]")
    weights = weights.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(weights).all() or bool((weights < 0).any().item()):
        raise ValueError(
            "Gate calibration sample_weight must be finite and non-negative"
        )
    if not float(weights.sum().item()) > 0.0:
        raise ValueError("Gate calibration sample_weight sum must be positive")

    sample_ids_value = batch["sample_id"]
    if isinstance(sample_ids_value, str):
        sample_ids = (sample_ids_value,)
    elif isinstance(sample_ids_value, Sequence):
        sample_ids = tuple(sample_ids_value)
    else:
        raise TypeError("Gate calibration sample_id must be a sequence")
    if len(sample_ids) != batch_size or any(
        not isinstance(sample_id, str) or not sample_id for sample_id in sample_ids
    ):
        raise ValueError("Gate calibration sample_id must contain B strings")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Gate calibration batch contains duplicate sample_id values")
    return inputs, labels, weights, sample_ids


def _gate_output_views(
    raw_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return metric and runtime views without changing Router numerics.

    Historical Gate metrics promote the raw float32 logits to CPU float64 and
    apply sigmoid inside ``compute_gate_binary_metrics``.  Closed-loop routing
    instead applies float32 sigmoid on the Gate device.  Keep both paths
    explicit: thresholds and exported records must use the latter exactly.
    """

    if (
        not isinstance(raw_logits, torch.Tensor)
        or raw_logits.ndim != 1
        or not raw_logits.is_floating_point()
        or not torch.isfinite(raw_logits).all()
    ):
        raise ValueError("Gate calibration produced invalid logits")
    runtime_probabilities = torch.sigmoid(raw_logits.float())
    metric_logits = raw_logits.detach().to(device="cpu", dtype=torch.float64)
    runtime_logits = raw_logits.detach().to(device="cpu", dtype=torch.float32)
    runtime_probabilities = runtime_probabilities.detach().to(device="cpu")
    if runtime_probabilities.dtype != torch.float32:
        raise RuntimeError("Gate runtime probabilities must be float32")
    if not torch.isfinite(runtime_probabilities).all():
        raise RuntimeError("Gate calibration produced invalid probabilities")
    return metric_logits, runtime_logits, runtime_probabilities


def _validation_objective_batch(
    *,
    raw_logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    pos_weight: float,
) -> tuple[float, float]:
    """Replay one validation batch using ``GateTrainer._forward_loss`` math."""

    if raw_logits.ndim != 1 or raw_logits.shape != labels.shape:
        raise ValueError("objective logits and labels must have matching shape [B]")
    if weights.shape != raw_logits.shape:
        raise ValueError("objective weights must have shape [B]")
    if not math.isfinite(pos_weight) or pos_weight <= 0.0:
        raise ValueError("objective pos_weight must be finite and positive")
    labels_device = labels.to(device=raw_logits.device, dtype=raw_logits.dtype)
    weights_device = weights.to(device=raw_logits.device, dtype=raw_logits.dtype)
    elementwise = F.binary_cross_entropy_with_logits(
        raw_logits,
        labels_device,
        pos_weight=torch.as_tensor(
            pos_weight,
            dtype=raw_logits.dtype,
            device=raw_logits.device,
        ),
        reduction="none",
    )
    loss = (elementwise * weights_device).sum() / weights_device.sum()
    if not torch.isfinite(loss):
        raise RuntimeError("Gate validation objective became non-finite")
    weight_sum = float(weights_device.detach().double().sum().cpu().item())
    return float(loss.detach().cpu().item()), weight_sum


_MISSING = object()


def _json_differences(
    expected: Any,
    observed: Any,
    *,
    path: str = "",
) -> list[dict[str, Any]]:
    """Return deterministic, JSON-safe leaf differences for two identities."""

    if isinstance(expected, Mapping) and isinstance(observed, Mapping):
        differences: list[dict[str, Any]] = []
        for key in sorted(set(expected) | set(observed)):
            child_path = f"{path}.{key}" if path else str(key)
            expected_value = expected.get(key, _MISSING)
            observed_value = observed.get(key, _MISSING)
            if expected_value is _MISSING:
                differences.append(
                    {
                        "path": child_path,
                        "kind": "unexpected",
                        "expected": None,
                        "observed": observed_value,
                    }
                )
            elif observed_value is _MISSING:
                differences.append(
                    {
                        "path": child_path,
                        "kind": "missing",
                        "expected": expected_value,
                        "observed": None,
                    }
                )
            else:
                differences.extend(
                    _json_differences(
                        expected_value,
                        observed_value,
                        path=child_path,
                    )
                )
        return differences
    if isinstance(expected, list) and isinstance(observed, list):
        if len(expected) != len(observed):
            return [
                {
                    "path": path,
                    "kind": "value",
                    "expected": expected,
                    "observed": observed,
                }
            ]
        differences = []
        for index, (expected_value, observed_value) in enumerate(
            zip(expected, observed, strict=True)
        ):
            differences.extend(
                _json_differences(
                    expected_value,
                    observed_value,
                    path=f"{path}[{index}]",
                )
            )
        return differences
    if expected == observed and type(expected) is type(observed):
        return []
    return [
        {
            "path": path,
            "kind": "value",
            "expected": expected,
            "observed": observed,
        }
    ]


def _numerical_runtime_comparison(
    *,
    training_runtime: Mapping[str, Any],
    current_runtime: Mapping[str, Any],
    require_exact: bool,
) -> dict[str, Any]:
    if not isinstance(require_exact, bool):
        raise TypeError("require_exact_training_numerical_runtime must be bool")
    if not isinstance(training_runtime, Mapping) or not isinstance(
        current_runtime, Mapping
    ):
        raise TypeError("numerical runtime identities must be mappings")
    expected = dict(training_runtime)
    observed = dict(current_runtime)
    differences = _json_differences(expected, observed)
    result = {
        "policy": (
            "exact_match_required"
            if require_exact
            else "metric_reproduction_guarded"
        ),
        "require_exact": require_exact,
        "exact_match": not differences,
        "training_runtime_sha256": canonical_json_sha256(expected),
        "current_runtime_sha256": canonical_json_sha256(observed),
        "num_differences": len(differences),
        "differences": differences,
    }
    if differences and require_exact:
        raise RuntimeError(
            "current numerical runtime differs from Gate training runtime: "
            f"{differences}"
        )
    return result


def _threshold_runtime_replay(
    *,
    probabilities: Sequence[float],
    calibration_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove every exported threshold replays inclusive Router comparisons."""

    calibrations = calibration_report.get("calibrations")
    if not isinstance(calibrations, list) or not calibrations:
        raise ValueError("calibration report contains no threshold points")
    points: list[dict[str, Any]] = []
    for index, calibration in enumerate(calibrations):
        if not isinstance(calibration, Mapping):
            raise TypeError(f"calibrations[{index}] must be a mapping")
        threshold = float(calibration["threshold"])
        selected_count = sum(
            float(probability) >= threshold for probability in probabilities
        )
        expected_count = int(calibration["selected_count"])
        if selected_count != expected_count:
            raise RuntimeError(
                "calibrated threshold does not replay Router semantics: "
                f"index={index}, expected={expected_count}, actual={selected_count}"
            )
        points.append(
            {
                "threshold": threshold,
                "selected_count": selected_count,
                "actual_with_rate": selected_count / len(probabilities),
            }
        )
    return {
        "passed": True,
        "probability_dtype": "float32",
        "sigmoid_location": "gate_device_before_cpu_transfer",
        "comparison": "python_float_probability >= python_float_threshold",
        "points": points,
    }


def _metric_reproduction(
    *,
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
    tolerance: float,
) -> dict[str, Any]:
    deltas: dict[str, float | None] = {}
    for field in _METRIC_FIELDS:
        if field not in expected or field not in observed:
            raise ValueError(f"Gate best metrics are missing {field}")
        actual_value = observed[field]
        expected_value = expected[field]
        if actual_value is None or expected_value is None:
            if actual_value is not None or expected_value is not None:
                raise ValueError(f"Gate validation metric {field} null mismatch")
            deltas[field] = None
            continue
        if field in {"num_examples", "num_batches"}:
            if actual_value != expected_value:
                raise ValueError(
                    f"Gate validation metric {field} mismatch: "
                    f"expected={expected_value}, actual={actual_value}"
                )
            deltas[field] = 0.0
            continue
        delta = abs(float(actual_value) - float(expected_value))
        if not math.isfinite(delta) or delta > tolerance:
            raise ValueError(
                f"Gate validation metric {field} failed reproduction: "
                f"expected={expected_value}, actual={actual_value}, "
                f"abs_delta={delta}, tolerance={tolerance}"
            )
        deltas[field] = delta
    return {
        "metric_abs_tolerance": tolerance,
        "expected_checkpoint_best_metrics": dict(expected),
        "observed": dict(observed),
        "absolute_deltas": deltas,
        "passed": True,
    }


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    dict(row),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                )
            )
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _artifact_file(path: Path, *, semantic_sha256: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "file_sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if semantic_sha256 is not None:
        result["semantic_sha256"] = require_sha256(
            semantic_sha256, field=f"semantic SHA256 for {path.name}"
        )
    return result


def run_gate_validation_calibration(
    config: DictConfig | Mapping[str, Any],
) -> dict[str, Any]:
    """Replay the immutable validation split once and publish reusable scores."""

    started = time.monotonic()
    resolved, calibration = _resolved_configs(config)
    gate_training._require_single_process_environment()
    if calibration["source_split"] != "validation":
        raise ValueError("Gate threshold calibration is fixed to validation split")
    target_rates = calibration["target_with_rates"]
    if not isinstance(target_rates, list) or not target_rates:
        raise ValueError("calibration.target_with_rates must be a non-empty list")
    configured_video_steps = _positive_int(
        calibration["configured_video_steps"],
        field="calibration.configured_video_steps",
    )
    if configured_video_steps != 10:
        raise ValueError("formal Stage 2 Gate calibration is bound to N=10")
    expected_validation_samples = _positive_int(
        calibration["expected_validation_samples"],
        field="calibration.expected_validation_samples",
    )
    batch_size = _positive_int(
        calibration["validation_batch_size"],
        field="calibration.validation_batch_size",
    )
    num_workers = _positive_int(
        calibration["validation_num_workers"],
        field="calibration.validation_num_workers",
        allow_zero=True,
    )
    if not isinstance(calibration["validation_pin_memory"], bool):
        raise TypeError("calibration.validation_pin_memory must be bool")
    progress_every = _positive_int(
        calibration["progress_every_batches"],
        field="calibration.progress_every_batches",
    )
    metric_tolerance = _finite_nonnegative(
        calibration["metric_abs_tolerance"],
        field="calibration.metric_abs_tolerance",
    )
    require_exact_runtime = calibration[
        "require_exact_training_numerical_runtime"
    ]
    if not isinstance(require_exact_runtime, bool):
        raise TypeError(
            "calibration.require_exact_training_numerical_runtime must be bool"
        )
    output_dir = Path(str(calibration["output_dir"])).expanduser().resolve()
    if os.path.lexists(output_dir):
        raise RuntimeError(f"refusing to reuse calibration output_dir: {output_dir}")
    filenames = {
        field: _safe_basename(calibration[field], field=f"calibration.{field}")
        for field in (
            "records_file",
            "thresholds_file",
            "manifest_file",
            "complete_file",
        )
    }
    if len(set(filenames.values())) != len(filenames):
        raise ValueError("calibration output basenames must be distinct")

    runtime = gate_training._exact_section(
        resolved,
        "runtime",
        {
            "repo_dir",
            "require_clean_git",
            "device",
            "require_cuda",
            "deterministic_algorithms",
        },
    )
    current_git_identity = gate_training._validated_git_identity(runtime)
    repo_dir = gate_training._repo_dir(runtime)
    resolved["data"] = gate_training._canonicalize_data_paths(
        resolved["data"], repo_dir=repo_dir
    )

    run_identity_path, run_identity, run_identity_file_sha = (
        _load_gate_run_identity(
            calibration["gate_run_identity"],
            expected_file_sha256=calibration["gate_run_identity_sha256"],
        )
    )
    gate_checkpoint_path = Path(
        str(calibration["gate_checkpoint"])
    ).expanduser().resolve(strict=True)
    configured_gate_run = Path(str(resolved["output_dir"])).expanduser().resolve()
    if run_identity_path.parent != configured_gate_run:
        raise ValueError("Gate run identity is outside configured Gate output_dir")
    if gate_checkpoint_path.parent != configured_gate_run:
        raise ValueError("Gate checkpoint is outside configured Gate output_dir")
    checkpoint_config = gate_training._exact_section(
        resolved,
        "checkpoint",
        {
            "strict_resume",
            "resume",
            "run_identity_file",
            "state_file",
            "best_file",
            "last_file",
            "summary_file",
        },
    )
    if gate_checkpoint_path.name != checkpoint_config["best_file"]:
        raise ValueError("calibration must load the configured Gate best export")
    if run_identity_path.name != checkpoint_config["run_identity_file"]:
        raise ValueError("Gate run identity basename differs from training config")

    data_manifest, episode_split, label_contract = (
        gate_training._load_identity_inputs(resolved)
    )
    selection, coverage, expected_sample_ids = (
        gate_training._load_optional_selection(
            resolved,
            data_manifest=data_manifest,
            episode_split=episode_split,
        )
    )
    source_identities = gate_training._exact_section(
        resolved,
        "source_identities",
        {"base_checkpoint_sha256", "adapter_checkpoint_sha256"},
    )
    for field in ("base_checkpoint_sha256", "adapter_checkpoint_sha256"):
        expected = require_sha256(
            source_identities[field], field=f"source_identities.{field}"
        )
        if label_contract.get(field) != expected:
            raise ValueError(f"label contract {field} mismatch")
    if label_contract.get("data_config_sha256") != canonical_json_sha256(
        resolved["data"]
    ):
        raise ValueError("Gate data config differs from label-generation config")

    label_manifest_spec = gate_training._exact_section(
        resolved, "label_manifest", {"path", "expected_sha256"}
    )
    merged_binding: dict[str, Any] = {}
    if selection is not None:
        assert coverage is not None
        merged_binding = {
            "selection_sha256": selection.descriptor["selection_sha256"],
            "coverage_sha256": coverage["coverage_sha256"],
            "active_cohort_indices": coverage["active_cohort_indices"],
        }
    merged = load_validated_merged_label_artifact(
        label_manifest_spec["path"],
        contract=label_contract,
        data_manifest=data_manifest,
        episode_split=episode_split,
        **merged_binding,
    )
    expected_label_manifest_sha = require_sha256(
        label_manifest_spec["expected_sha256"],
        field="label manifest expected_sha256",
    )
    if merged.manifest["manifest_sha256"] != expected_label_manifest_sha:
        raise ValueError("merged label manifest SHA256 mismatch")

    assets = gate_training._exact_section(
        resolved, "assets", {"normalization_stats"}
    )
    stats_spec = assets["normalization_stats"]
    if not isinstance(stats_spec, Mapping) or set(stats_spec) != {
        "path",
        "expected_sha256",
    }:
        raise ValueError("normalization stats config fields do not match schema")
    stats_identity = resolve_base_checkpoint(
        stats_spec["path"], expected_sha256=str(stats_spec["expected_sha256"])
    )
    if label_contract.get("normalization_stats_sha256") != stats_identity.sha256:
        raise ValueError("normalization stats differ from label contract")

    gate_config = gate_training._exact_section(
        resolved,
        "gate",
        {
            "proprio_dim",
            "context_dim",
            "cnn_channels",
            "context_feature_dim",
            "proprio_hidden_dim",
            "proprio_feature_dim",
            "fusion_hidden_dim",
        },
    )
    training = gate_training._exact_section(
        resolved,
        "training",
        {
            "seed",
            "batch_size",
            "num_workers",
            "pin_memory",
            "shuffle",
            "learning_rate",
            "weight_decay",
            "max_grad_norm",
            "num_epochs",
            "early_stop_patience",
            "min_delta",
            "threshold",
            "num_calibration_bins",
        },
    )
    if batch_size != int(training["batch_size"]):
        raise ValueError("validation batch size must reproduce Gate training batch size")
    if num_workers != int(training["num_workers"]):
        raise ValueError("validation workers must reproduce Gate training workers")
    if calibration["validation_pin_memory"] != training["pin_memory"]:
        raise ValueError("validation pin_memory must reproduce Gate training setting")
    _validate_recorded_training_contract(
        run_identity["training_config"],
        resolved=resolved,
        gate_config=gate_config,
        training=training,
    )
    recorded_config_sha = run_identity["training_config_sha256"]
    training_identity = run_identity["training_identity"]
    rebuilt_training_identity = gate_training.build_training_identity(
        label_manifest_sha256=expected_label_manifest_sha,
        contract=label_contract,
        training_config_sha256=recorded_config_sha,
        git_identity=training_identity["git_identity"],
    )
    if rebuilt_training_identity != training_identity:
        raise ValueError("Gate run training_identity differs from validated sources")

    set_global_seed(int(training["seed"]))
    torch.use_deterministic_algorithms(bool(runtime["deterministic_algorithms"]))
    device = gate_training._device(runtime)
    current_numerical_runtime = collect_numerical_runtime_environment(device)
    recorded_runtime = run_identity["training_config"].get("runtime")
    if not isinstance(recorded_runtime, Mapping) or not isinstance(
        recorded_runtime.get("numerical_runtime"), Mapping
    ):
        raise TypeError("Gate run is missing its training numerical runtime")
    numerical_runtime_comparison = _numerical_runtime_comparison(
        training_runtime=recorded_runtime["numerical_runtime"],
        current_runtime=current_numerical_runtime,
        require_exact=require_exact_runtime,
    )
    print(
        "[calibration] numerical_runtime "
        f"policy={numerical_runtime_comparison['policy']} "
        f"exact_match={numerical_runtime_comparison['exact_match']} "
        f"differences={numerical_runtime_comparison['num_differences']}",
        flush=True,
    )
    loaded_gate = load_gate_for_evaluation(
        gate_checkpoint_path,
        expected_checkpoint_sha256=calibration["gate_checkpoint_sha256"],
        expected_label_manifest_sha256=training_identity[
            "label_manifest_sha256"
        ],
        expected_adapter_checkpoint_sha256=training_identity[
            "adapter_checkpoint_sha256"
        ],
        expected_base_checkpoint_sha256=training_identity[
            "base_checkpoint_sha256"
        ],
        expected_data_manifest_sha256=training_identity[
            "data_manifest_sha256"
        ],
        expected_episode_split_assignment_sha256=training_identity[
            "episode_split_assignment_sha256"
        ],
        expected_training_config_sha256=training_identity[
            "training_config_sha256"
        ],
        expected_git_identity=training_identity["git_identity"],
        device=device,
    )
    expected_gate_config = BinaryVideoGate(**gate_config).config()
    if loaded_gate.gate.config() != expected_gate_config:
        raise ValueError("Gate checkpoint architecture differs from training contract")

    source_snapshot = capture_selected_source_snapshot(data_manifest)
    raw_dataset = instantiate(OmegaConf.create(resolved["data"]["train"]))
    gate_training._validate_formal_label_dataset(
        raw_dataset,
        data_manifest,
        normalization_stats_path=stats_identity.path,
        expected_data_manifest_sha256=data_manifest["manifest_sha256"],
    )
    source_snapshot.check_stats()
    gate_source = raw_dataset.current_only()
    del raw_dataset
    dataset_binding: dict[str, Any] = {}
    if expected_sample_ids is not None:
        dataset_binding["expected_sample_ids"] = expected_sample_ids
    validation_dataset = Stage2GateDataset(
        gate_source,
        label_rows=merged.rows,
        data_manifest=data_manifest,
        episode_split=episode_split,
        split="validation",
        **dataset_binding,
    )
    if len(validation_dataset) != expected_validation_samples:
        raise ValueError(
            "validation sample count mismatch: "
            f"expected={expected_validation_samples}, "
            f"actual={len(validation_dataset)}"
        )
    expected_order = validation_dataset.sample_ids
    identity_by_sample_id = dict(
        zip(
            expected_order,
            validation_dataset.sample_identities,
            strict=True,
        )
    )
    if len(identity_by_sample_id) != len(validation_dataset):
        raise RuntimeError("validation dataset contains duplicate sample IDs")
    validation_source_keys = source_snapshot.keys_for_sample_identities(
        validation_dataset.sample_identities
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        int(training["seed"])
        + 1_000_000_000
        + max(0, int(loaded_gate.identity["epoch"]) - 1)
    )
    loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(calibration["validation_pin_memory"]),
        drop_last=False,
        worker_init_fn=gate_training._seed_data_worker,
        persistent_workers=False,
        generator=generator,
    )
    total_batches = math.ceil(len(validation_dataset) / batch_size)
    train_labels = [
        bool(row["label"]) for row in merged.rows if row["split"] == "train"
    ]
    train_positive_count = sum(train_labels)
    train_negative_count = len(train_labels) - train_positive_count
    if train_positive_count <= 0 or train_negative_count <= 0:
        raise ValueError("Gate training labels must contain both classes")
    pos_weight = float(train_negative_count / train_positive_count)
    print(
        "[calibration] "
        f"split=validation samples={len(validation_dataset)} "
        f"batches={total_batches} batch_size={batch_size} device={device}",
        flush=True,
    )

    records: list[dict[str, Any]] = []
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    cursor = 0
    objective_numerator = 0.0
    objective_denominator = 0.0
    inference_started = time.monotonic()
    loaded_gate.gate.eval()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            inputs, labels, weights, sample_ids = _validate_batch(
                batch, device=device
            )
            if sample_ids != expected_order[cursor : cursor + len(sample_ids)]:
                raise RuntimeError("validation DataLoader order drifted")
            raw_logits = loaded_gate.gate(**inputs)
            metric_logits, runtime_logits, runtime_probabilities = (
                _gate_output_views(raw_logits)
            )
            if metric_logits.shape != labels.shape:
                raise RuntimeError("Gate calibration logit shape mismatch")
            objective_loss, objective_weight = _validation_objective_batch(
                raw_logits=raw_logits,
                labels=labels,
                weights=weights,
                pos_weight=pos_weight,
            )
            objective_numerator += objective_loss * objective_weight
            objective_denominator += objective_weight
            for row_index, sample_id in enumerate(sample_ids):
                identity = identity_by_sample_id[sample_id]
                records.append(
                    {
                        "sample_id": sample_id,
                        "split": "validation",
                        "global_sample_index": identity["global_sample_index"],
                        "dataset_index": identity["dataset_index"],
                        "episode_index": identity["episode_index"],
                        "frame_index": identity["frame_index"],
                        "dataset_frame_index": identity[
                            "dataset_frame_index"
                        ],
                        "label": bool(labels[row_index].item()),
                        "sample_weight": float(weights[row_index].item()),
                        "logit": float(runtime_logits[row_index].item()),
                        "probability": float(
                            runtime_probabilities[row_index].item()
                        ),
                    }
                )
            all_logits.append(metric_logits)
            all_labels.append(labels.to(dtype=torch.float64))
            cursor += len(sample_ids)
            if batch_index % progress_every == 0 or batch_index == total_batches:
                elapsed = time.monotonic() - inference_started
                print(
                    "[calibration] "
                    f"batch={batch_index}/{total_batches} "
                    f"samples={cursor}/{len(validation_dataset)} "
                    f"elapsed_s={elapsed:.1f}",
                    flush=True,
                )
    if cursor != len(validation_dataset) or len(records) != len(validation_dataset):
        raise RuntimeError("Gate calibration did not visit every validation sample")

    logits_tensor = torch.cat(all_logits)
    labels_tensor = torch.cat(all_labels)
    metrics = compute_gate_binary_metrics(
        logits=logits_tensor,
        labels=labels_tensor,
        threshold=float(training["threshold"]),
        num_calibration_bins=int(training["num_calibration_bins"]),
    ).to_dict()
    if not objective_denominator > 0.0:
        raise RuntimeError("Gate validation objective denominator is invalid")
    observed_metrics = {
        "objective_bce": objective_numerator / objective_denominator,
        **metrics,
        "num_batches": total_batches,
    }
    reproduction = _metric_reproduction(
        observed=observed_metrics,
        expected=loaded_gate.identity["best_metrics"],
        tolerance=metric_tolerance,
    )

    records.sort(key=lambda row: row["sample_id"])
    if [row["sample_id"] for row in records] != sorted(expected_order):
        raise RuntimeError("sorted validation prediction coverage mismatch")
    calibration_report = calibrate_probability_thresholds(
        (row["probability"] for row in records),
        (row["label"] for row in records),
        target_rates,
        configured_video_steps=configured_video_steps,
    )
    threshold_runtime_replay = _threshold_runtime_replay(
        probabilities=[row["probability"] for row in records],
        calibration_report=calibration_report,
    )

    # Revalidate all mutable sources before publishing any durable artifact.
    source_snapshot.check_content(keys=validation_source_keys)
    if sha256_file(gate_checkpoint_path) != require_sha256(
        calibration["gate_checkpoint_sha256"],
        field="expected Gate checkpoint SHA256",
    ):
        raise RuntimeError("Gate checkpoint changed during calibration")
    if sha256_file(run_identity_path) != run_identity_file_sha:
        raise RuntimeError("Gate run identity changed during calibration")
    if sha256_file(stats_identity.path) != stats_identity.sha256:
        raise RuntimeError("normalization stats changed during calibration")
    refreshed_manifest, refreshed_split, refreshed_contract = (
        gate_training._load_identity_inputs(resolved)
    )
    if (
        refreshed_manifest != data_manifest
        or refreshed_split != episode_split
        or refreshed_contract != label_contract
    ):
        raise RuntimeError("Stage 2 source identities changed during calibration")
    refreshed_selection, refreshed_coverage, refreshed_expected_ids = (
        gate_training._load_optional_selection(
            resolved,
            data_manifest=refreshed_manifest,
            episode_split=refreshed_split,
        )
    )
    if refreshed_expected_ids != expected_sample_ids:
        raise RuntimeError("Stage 2 selected coverage changed during calibration")
    if (refreshed_selection is None) != (selection is None) or (
        refreshed_coverage is None
    ) != (coverage is None):
        raise RuntimeError("Stage 2 selection binding changed during calibration")
    refreshed_merged = load_validated_merged_label_artifact(
        label_manifest_spec["path"],
        contract=refreshed_contract,
        data_manifest=refreshed_manifest,
        episode_split=refreshed_split,
        **merged_binding,
    )
    if refreshed_merged.manifest != merged.manifest:
        raise RuntimeError("merged Gate label artifact changed during calibration")
    if read_git_identity(repo_dir).as_dict() != current_git_identity:
        raise RuntimeError("evaluation Git identity changed during calibration")

    if os.path.lexists(output_dir):
        raise RuntimeError(f"refusing to reuse calibration output_dir: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    records_path = output_dir / filenames["records_file"]
    thresholds_path = output_dir / filenames["thresholds_file"]
    manifest_path = output_dir / filenames["manifest_file"]
    complete_path = output_dir / filenames["complete_file"]

    _write_jsonl_atomic(records_path, records)
    records_identity = _artifact_file(records_path)
    thresholds_payload = {
        "schema_version": 1,
        "kind": "stage2_gate_validation_thresholds",
        "source_split": "validation",
        "validation_predictions_sha256": records_identity["file_sha256"],
        "probability_semantics": {
            "logit_dtype": "float32",
            "sigmoid_dtype": "float32",
            "sigmoid_location": "gate_device_before_cpu_transfer",
            "threshold_scalar": "python_float_json_number",
        },
        "runtime_replay": threshold_runtime_replay,
        **calibration_report,
    }
    write_json_atomic(thresholds_path, thresholds_payload)
    thresholds_identity = _artifact_file(thresholds_path)

    data_manifest_path = Path(
        str(resolved["data_manifest"]["path"])
    ).expanduser().resolve(strict=True)
    split_path = Path(str(resolved["episode_split"]["path"])).expanduser().resolve(
        strict=True
    )
    contract_path = Path(
        str(resolved["label_contract"]["path"])
    ).expanduser().resolve(strict=True)
    merged_manifest_path = Path(
        str(label_manifest_spec["path"])
    ).expanduser().resolve(strict=True)
    merged_rows_path = (
        merged_manifest_path.parent / str(merged.manifest["rows_file"])
    ).resolve(strict=True)
    export_config = {
        "source_split": "validation",
        "target_with_rates": list(target_rates),
        "configured_video_steps": configured_video_steps,
        "expected_validation_samples": expected_validation_samples,
        "validation_batch_size": batch_size,
        "validation_num_workers": num_workers,
        "validation_pin_memory": calibration["validation_pin_memory"],
        "metric_abs_tolerance": metric_tolerance,
        "require_exact_training_numerical_runtime": require_exact_runtime,
        "progress_every_batches": progress_every,
        "calibration_algorithm": CALIBRATION_ALGORITHM,
    }
    source_files: dict[str, Any] = {
        "data_manifest": _artifact_file(
            data_manifest_path,
            semantic_sha256=data_manifest["manifest_sha256"],
        ),
        "episode_split": _artifact_file(
            split_path,
            semantic_sha256=episode_split["assignment_sha256"],
        ),
        "label_contract": _artifact_file(
            contract_path,
            semantic_sha256=label_contract["contract_sha256"],
        ),
        "merged_label_manifest": _artifact_file(
            merged_manifest_path,
            semantic_sha256=merged.manifest["manifest_sha256"],
        ),
        "merged_label_rows": _artifact_file(
            merged_rows_path,
            semantic_sha256=merged.manifest["rows_file_sha256"],
        ),
        "normalization_stats": stats_identity.as_dict(),
    }
    selection_binding: dict[str, Any] | None = None
    if selection is not None:
        assert coverage is not None
        selection_dir = Path(
            str(resolved["label_selection"]["directory"])
        ).expanduser().resolve(strict=True)
        selection_binding = {
            "directory": str(selection_dir),
            "selection_sha256": selection.descriptor["selection_sha256"],
            "coverage_tier": coverage["tier"],
            "coverage_sha256": coverage["coverage_sha256"],
            "active_cohort_indices": list(coverage["active_cohort_indices"]),
            "sample_count": coverage["sample_count"],
        }

    duration = time.monotonic() - started
    manifest = {
        "schema_version": 1,
        "kind": "stage2_gate_validation_calibration",
        "source_split": "validation",
        "calibration_algorithm": CALIBRATION_ALGORITHM,
        "export_config": export_config,
        "export_config_sha256": canonical_json_sha256(export_config),
        "gate_checkpoint": loaded_gate.identity,
        "gate_run_identity": {
            "path": str(run_identity_path),
            "file_sha256": run_identity_file_sha,
            "training_config_sha256": recorded_config_sha,
            "training_identity": training_identity,
        },
        "source_files": source_files,
        "selection_binding": selection_binding,
        "validation": {
            "num_samples": len(records),
            "num_batches": total_batches,
            "ordered_sample_ids_sha256": canonical_json_sha256(
                [row["sample_id"] for row in records]
            ),
            "training_label_statistics": {
                "num_samples": len(train_labels),
                "positive_count": train_positive_count,
                "negative_count": train_negative_count,
                "pos_weight": pos_weight,
            },
            "metrics_at_training_threshold": observed_metrics,
            "metric_reproduction": reproduction,
            "threshold_runtime_replay": threshold_runtime_replay,
            "records_file": records_identity,
        },
        "thresholds_file": thresholds_identity,
        "calibration_summary": {
            key: value
            for key, value in calibration_report.items()
            if key != "score_blocks"
        },
        "runtime": {
            "git_identity": current_git_identity,
            "numerical_environment": current_numerical_runtime,
            "training_numerical_environment_comparison": (
                numerical_runtime_comparison
            ),
            "duration_seconds": duration,
        },
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    write_json_atomic(manifest_path, manifest)
    manifest_file_sha = sha256_file(manifest_path)
    complete = {
        "schema_version": 1,
        "kind": "stage2_gate_validation_calibration_complete",
        "manifest_file": manifest_path.name,
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_file_sha256": manifest_file_sha,
        "records_file_sha256": records_identity["file_sha256"],
        "thresholds_file_sha256": thresholds_identity["file_sha256"],
    }
    write_text_atomic(
        complete_path,
        json.dumps(complete, sort_keys=True, indent=2, ensure_ascii=True) + "\n",
    )
    print(
        "[calibration] complete "
        f"samples={len(records)} duration_s={duration:.1f} "
        f"output_dir={output_dir}",
        flush=True,
    )
    return manifest


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="calibrate_video_gate",
)
def main(config: DictConfig) -> None:
    run_gate_validation_calibration(config)


if __name__ == "__main__":
    main()
