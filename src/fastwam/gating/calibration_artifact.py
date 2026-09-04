"""Strict loading of a published Stage 2 Gate calibration artifact.

The ``COMPLETE`` receipt is the single immutable root supplied by an
evaluation launcher.  It binds the calibration manifest, validation
predictions, and threshold table by physical SHA256.  The manifest in turn
binds the Gate checkpoint and every Stage 2/3 source identity used during
calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

from fastwam.alignment.checkpointing import canonical_json_sha256, sha256_file

from .calibration import CALIBRATION_ALGORITHM, DECISION_RULE
from .contracts import require_sha256


_COMPLETE_KEYS = {
    "schema_version",
    "kind",
    "manifest_file",
    "manifest_sha256",
    "manifest_file_sha256",
    "records_file_sha256",
    "thresholds_file_sha256",
}
_MANIFEST_KEYS = {
    "schema_version",
    "kind",
    "source_split",
    "calibration_algorithm",
    "export_config",
    "export_config_sha256",
    "gate_checkpoint",
    "gate_run_identity",
    "source_files",
    "selection_binding",
    "validation",
    "thresholds_file",
    "calibration_summary",
    "runtime",
    "manifest_sha256",
}
_THRESHOLDS_KEYS = {
    "schema_version",
    "kind",
    "source_split",
    "validation_predictions_sha256",
    "probability_semantics",
    "runtime_replay",
    "algorithm",
    "decision_rule",
    "num_examples",
    "configured_video_steps",
    "label_statistics",
    "probability_block_diagnostics",
    "score_blocks",
    "calibrations",
}
_CALIBRATION_SUMMARY_KEYS = {
    "algorithm",
    "decision_rule",
    "num_examples",
    "configured_video_steps",
    "label_statistics",
    "probability_block_diagnostics",
    "calibrations",
}
_PROBABILITY_SEMANTICS = {
    "logit_dtype": "float32",
    "sigmoid_dtype": "float32",
    "sigmoid_location": "gate_device_before_cpu_transfer",
    "threshold_scalar": "python_float_json_number",
}


@dataclass(frozen=True, slots=True)
class LoadedGateCalibration:
    """One verified calibration point and its immutable source manifest."""

    threshold: float
    target_with_rate: float
    manifest: dict[str, Any]
    receipt: dict[str, Any]


def _json_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {path}") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain a mapping")
    return dict(value)


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return dict(value)


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, field: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{field} schema mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return int(value)


def _probability(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a real number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be finite and in [0, 1]")
    return 0.0 if result == 0.0 else result


def _safe_member(root: Path, name: Any, *, field: str) -> Path:
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError(f"{field} must be a safe basename")
    path = (root / name).resolve(strict=True)
    if path.parent != root or not path.is_file():
        raise ValueError(f"{field} must resolve to a regular file in {root}")
    return path


def _verified_recorded_file(
    value: Any,
    *,
    root: Path,
    field: str,
) -> dict[str, Any]:
    identity = _mapping(value, field=field)
    _exact_keys(identity, {"path", "file_sha256", "size_bytes"}, field=field)
    raw_path = identity["path"]
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise ValueError(f"{field}.path must be absolute")
    path = Path(raw_path).expanduser().resolve(strict=True)
    if path.parent != root or not path.is_file():
        raise ValueError(f"{field}.path must be a regular file in {root}")
    expected_sha = require_sha256(
        identity["file_sha256"], field=f"{field}.file_sha256"
    )
    expected_size = _positive_int(identity["size_bytes"], field=f"{field}.size_bytes")
    actual_size = path.stat().st_size
    actual_sha = sha256_file(path)
    if actual_size != expected_size or actual_sha != expected_sha:
        raise ValueError(
            f"{field} identity mismatch: expected_size={expected_size}, "
            f"actual_size={actual_size}, expected_sha256={expected_sha}, "
            f"actual_sha256={actual_sha}"
        )
    return {"path": str(path), "sha256": actual_sha, "size_bytes": actual_size}


def _manifest_self_sha256(manifest: Mapping[str, Any]) -> str:
    unhashed = dict(manifest)
    unhashed.pop("manifest_sha256", None)
    return canonical_json_sha256(unhashed)


def _verify_gate_checkpoint_identity(value: Any) -> dict[str, Any]:
    identity = _mapping(value, field="calibration manifest gate_checkpoint")
    for field in (
        "sha256",
        "label_manifest_sha256",
        "adapter_checkpoint_sha256",
        "base_checkpoint_sha256",
        "data_manifest_sha256",
        "episode_split_assignment_sha256",
        "training_config_sha256",
    ):
        require_sha256(identity.get(field), field=f"gate_checkpoint.{field}")
    path_value = identity.get("path")
    if not isinstance(path_value, str) or not Path(path_value).is_absolute():
        raise ValueError("gate_checkpoint.path must be absolute")
    path = Path(path_value).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError("gate_checkpoint.path must be a regular file")
    expected_size = _positive_int(
        identity.get("size_bytes"), field="gate_checkpoint.size_bytes"
    )
    actual_sha = sha256_file(path)
    if path.stat().st_size != expected_size or actual_sha != identity["sha256"]:
        raise ValueError("calibrated Gate checkpoint byte identity mismatch")
    git_identity = _mapping(
        identity.get("git_identity"), field="gate_checkpoint.git_identity"
    )
    if set(git_identity) != {"commit", "tracked_dirty", "untracked_source_files"}:
        raise ValueError("gate_checkpoint.git_identity schema mismatch")
    return identity


def _verify_manifest_source_bindings(manifest: Mapping[str, Any]) -> None:
    source_files = _mapping(
        manifest.get("source_files"), field="calibration manifest source_files"
    )
    for field in (
        "data_manifest",
        "episode_split",
        "label_contract",
        "merged_label_manifest",
        "merged_label_rows",
        "normalization_stats",
    ):
        if field not in source_files:
            raise ValueError(f"calibration manifest source_files is missing {field}")

    gate = _mapping(
        manifest.get("gate_checkpoint"), field="calibration manifest gate_checkpoint"
    )
    expected = {
        "data_manifest": ("semantic_sha256", gate["data_manifest_sha256"]),
        "episode_split": (
            "semantic_sha256",
            gate["episode_split_assignment_sha256"],
        ),
        "merged_label_manifest": (
            "semantic_sha256",
            gate["label_manifest_sha256"],
        ),
    }
    for source_name, (key, expected_sha) in expected.items():
        source = _mapping(
            source_files[source_name], field=f"source_files.{source_name}"
        )
        actual_sha = require_sha256(
            source.get(key), field=f"source_files.{source_name}.{key}"
        )
        if actual_sha != expected_sha:
            raise ValueError(
                f"calibration source {source_name} disagrees with Gate identity"
            )

    require_sha256(
        _mapping(
            source_files["normalization_stats"],
            field="source_files.normalization_stats",
        ).get("sha256"),
        field="source_files.normalization_stats.sha256",
    )


def _select_point(
    calibrations: Any,
    *,
    target_with_rate: float,
    runtime_replay: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(calibrations, list) or not calibrations:
        raise ValueError("thresholds.calibrations must be a non-empty list")
    matches: list[dict[str, Any]] = []
    for index, raw_point in enumerate(calibrations):
        point = _mapping(raw_point, field=f"thresholds.calibrations[{index}]")
        point_target = _probability(
            point.get("target_with_rate"),
            field=f"thresholds.calibrations[{index}].target_with_rate",
        )
        if point_target == target_with_rate:
            matches.append(point)
    if len(matches) != 1:
        raise ValueError(
            "calibration target must identify exactly one published point: "
            f"target={target_with_rate}, matches={len(matches)}"
        )
    point = matches[0]
    threshold = _probability(point.get("threshold"), field="calibration threshold")
    selected_count = _positive_int(
        point.get("selected_count"), field="calibration selected_count"
    )
    actual_with_rate = _probability(
        point.get("actual_with_rate"), field="calibration actual_with_rate"
    )
    if point.get("exact_target") is not True or point.get("count_error") != 0:
        raise ValueError("formal calibration point must be an exact target")

    replay_points = runtime_replay.get("points")
    if not isinstance(replay_points, list):
        raise ValueError("threshold runtime replay points must be a list")
    replay_matches = [
        candidate
        for candidate in replay_points
        if isinstance(candidate, Mapping)
        and candidate.get("threshold") == threshold
        and candidate.get("selected_count") == selected_count
        and candidate.get("actual_with_rate") == actual_with_rate
    ]
    if len(replay_matches) != 1:
        raise ValueError("selected threshold is absent from exact runtime replay")
    return point


def load_gate_calibration_selection(
    complete_path: str | Path,
    *,
    expected_complete_sha256: str,
    target_with_rate: float,
    configured_video_steps: int = 10,
) -> LoadedGateCalibration:
    """Verify a complete artifact and resolve one published Gate threshold.

    No threshold or Gate identity is accepted from a second source.  Callers
    pin only the ``COMPLETE`` file SHA and the desired published target rate.
    """

    root_receipt = Path(complete_path).expanduser().resolve(strict=True)
    if not root_receipt.is_file():
        raise ValueError("Gate calibration COMPLETE path must be a regular file")
    root = root_receipt.parent
    expected_complete_sha = require_sha256(
        expected_complete_sha256,
        field="expected Gate calibration COMPLETE SHA256",
    )
    complete_size = root_receipt.stat().st_size
    actual_complete_sha = sha256_file(root_receipt)
    if actual_complete_sha != expected_complete_sha:
        raise ValueError(
            "Gate calibration COMPLETE SHA256 mismatch: "
            f"expected={expected_complete_sha}, actual={actual_complete_sha}"
        )
    complete = _json_mapping(root_receipt, label="Gate calibration COMPLETE")
    _exact_keys(complete, _COMPLETE_KEYS, field="Gate calibration COMPLETE")
    if complete["schema_version"] != 1 or complete["kind"] != (
        "stage2_gate_validation_calibration_complete"
    ):
        raise ValueError("unsupported Gate calibration COMPLETE schema")

    manifest_path = _safe_member(
        root,
        complete["manifest_file"],
        field="Gate calibration COMPLETE manifest_file",
    )
    manifest_file_sha = require_sha256(
        complete["manifest_file_sha256"],
        field="Gate calibration manifest file SHA256",
    )
    if sha256_file(manifest_path) != manifest_file_sha:
        raise ValueError("Gate calibration manifest file SHA256 mismatch")
    manifest = _json_mapping(manifest_path, label="Gate calibration manifest")
    _exact_keys(manifest, _MANIFEST_KEYS, field="Gate calibration manifest")
    if (
        manifest["schema_version"] != 1
        or manifest["kind"] != "stage2_gate_validation_calibration"
        or manifest["source_split"] != "validation"
        or manifest["calibration_algorithm"] != CALIBRATION_ALGORITHM
    ):
        raise ValueError("unsupported Gate calibration manifest schema")
    manifest_semantic_sha = require_sha256(
        manifest["manifest_sha256"], field="Gate calibration manifest_sha256"
    )
    if manifest_semantic_sha != require_sha256(
        complete["manifest_sha256"], field="COMPLETE manifest_sha256"
    ) or _manifest_self_sha256(manifest) != manifest_semantic_sha:
        raise ValueError("Gate calibration manifest semantic SHA256 mismatch")

    export_config = _mapping(
        manifest["export_config"], field="calibration manifest export_config"
    )
    export_config_sha = require_sha256(
        manifest["export_config_sha256"], field="export_config_sha256"
    )
    if canonical_json_sha256(export_config) != export_config_sha:
        raise ValueError("Gate calibration export_config SHA256 mismatch")

    threshold_identity = _verified_recorded_file(
        manifest["thresholds_file"], root=root, field="thresholds_file"
    )
    records_identity = _verified_recorded_file(
        _mapping(manifest["validation"], field="calibration validation").get(
            "records_file"
        ),
        root=root,
        field="validation.records_file",
    )
    if threshold_identity["sha256"] != require_sha256(
        complete["thresholds_file_sha256"],
        field="COMPLETE thresholds_file_sha256",
    ):
        raise ValueError("COMPLETE disagrees with thresholds file identity")
    if records_identity["sha256"] != require_sha256(
        complete["records_file_sha256"], field="COMPLETE records_file_sha256"
    ):
        raise ValueError("COMPLETE disagrees with validation records identity")

    thresholds_path = Path(threshold_identity["path"])
    thresholds = _json_mapping(thresholds_path, label="Gate calibration thresholds")
    _exact_keys(thresholds, _THRESHOLDS_KEYS, field="Gate calibration thresholds")
    if (
        thresholds["schema_version"] != 1
        or thresholds["kind"] != "stage2_gate_validation_thresholds"
        or thresholds["source_split"] != "validation"
        or thresholds["algorithm"] != CALIBRATION_ALGORITHM
        or thresholds["decision_rule"] != DECISION_RULE
        or thresholds["validation_predictions_sha256"]
        != records_identity["sha256"]
        or thresholds["probability_semantics"] != _PROBABILITY_SEMANTICS
    ):
        raise ValueError("unsupported or inconsistent Gate thresholds schema")

    video_steps = _positive_int(
        configured_video_steps, field="configured_video_steps"
    )
    if video_steps != 10:
        raise ValueError("formal Stage 2 Gate calibration is bound to N=10")
    if (
        thresholds["configured_video_steps"] != video_steps
        or export_config.get("configured_video_steps") != video_steps
    ):
        raise ValueError("Gate calibration video-step contract mismatch")

    summary = _mapping(
        manifest["calibration_summary"], field="calibration_summary"
    )
    _exact_keys(summary, _CALIBRATION_SUMMARY_KEYS, field="calibration_summary")
    threshold_summary = {key: thresholds[key] for key in _CALIBRATION_SUMMARY_KEYS}
    if summary != threshold_summary:
        raise ValueError("calibration manifest summary disagrees with thresholds")

    validation = _mapping(manifest["validation"], field="calibration validation")
    metric_reproduction = _mapping(
        validation.get("metric_reproduction"),
        field="calibration validation metric_reproduction",
    )
    runtime_replay = _mapping(
        thresholds["runtime_replay"], field="threshold runtime_replay"
    )
    if metric_reproduction.get("passed") is not True:
        raise ValueError("Gate calibration metric reproduction did not pass")
    if runtime_replay.get("passed") is not True:
        raise ValueError("Gate calibration runtime threshold replay did not pass")
    if validation.get("threshold_runtime_replay") != runtime_replay:
        raise ValueError("manifest and thresholds runtime replay receipts disagree")
    num_examples = _positive_int(
        thresholds["num_examples"], field="thresholds.num_examples"
    )
    if (
        validation.get("num_samples") != num_examples
        or export_config.get("expected_validation_samples") != num_examples
    ):
        raise ValueError("Gate calibration validation sample count mismatch")

    normalized_target = _probability(
        target_with_rate, field="target_with_rate"
    )
    selected_point = _select_point(
        thresholds["calibrations"],
        target_with_rate=normalized_target,
        runtime_replay=runtime_replay,
    )
    gate_checkpoint = _verify_gate_checkpoint_identity(manifest["gate_checkpoint"])
    _verify_manifest_source_bindings(manifest)

    # Recheck every artifact root after parsing to reject concurrent mutation.
    if (
        root_receipt.stat().st_size != complete_size
        or sha256_file(root_receipt) != actual_complete_sha
        or sha256_file(manifest_path) != manifest_file_sha
        or sha256_file(thresholds_path) != threshold_identity["sha256"]
        or sha256_file(records_identity["path"]) != records_identity["sha256"]
    ):
        raise RuntimeError("Gate calibration artifact changed while it was loaded")

    source_files = _mapping(manifest["source_files"], field="source_files")
    receipt = {
        "schema_version": 1,
        "kind": "stage2_gate_calibration_selection_receipt",
        "complete_file": {
            "path": str(root_receipt),
            "sha256": actual_complete_sha,
            "size_bytes": complete_size,
        },
        "manifest_file": {
            "path": str(manifest_path),
            "sha256": manifest_file_sha,
            "semantic_sha256": manifest_semantic_sha,
            "size_bytes": manifest_path.stat().st_size,
        },
        "thresholds_file": threshold_identity,
        "validation_predictions_file": records_identity,
        "source_split": "validation",
        "calibration_algorithm": CALIBRATION_ALGORITHM,
        "decision_rule": DECISION_RULE,
        "configured_video_steps": video_steps,
        "validation_num_examples": num_examples,
        "target_with_rate": normalized_target,
        "selected_point": dict(selected_point),
        "gate_checkpoint_sha256": gate_checkpoint["sha256"],
        "source_identities": {
            "label_manifest_sha256": gate_checkpoint["label_manifest_sha256"],
            "episode_split_assignment_sha256": gate_checkpoint[
                "episode_split_assignment_sha256"
            ],
            "training_config_sha256": gate_checkpoint["training_config_sha256"],
            "adapter_checkpoint_sha256": gate_checkpoint[
                "adapter_checkpoint_sha256"
            ],
            "base_checkpoint_sha256": gate_checkpoint["base_checkpoint_sha256"],
            "data_manifest_sha256": gate_checkpoint["data_manifest_sha256"],
            "normalization_stats_sha256": _mapping(
                source_files["normalization_stats"],
                field="source_files.normalization_stats",
            )["sha256"],
        },
    }
    return LoadedGateCalibration(
        threshold=float(selected_point["threshold"]),
        target_with_rate=normalized_target,
        manifest=manifest,
        receipt=receipt,
    )


__all__ = ["LoadedGateCalibration", "load_gate_calibration_selection"]
