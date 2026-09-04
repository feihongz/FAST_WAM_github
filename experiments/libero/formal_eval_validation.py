"""Fail-closed validation for formal LIBERO Phase-A evaluation artifacts.

The evaluator writes one JSON result per LIBERO task.  This module validates
those files against one immutable condition contract and only then pools the
40 task results.  Raw query records are authoritative: summaries are checked
against them rather than trusted as an alternative source of counts.

Contract schema v1::

    {
      "schema_version": 1,
      "kind": "libero_phase_a_condition_contract",
      "condition_id": "gate_050",
      "expected_tasks": [
        {
          "task_suite_name": "libero_spatial",
          "task_id": 0,
          "source_initial_state_count": 50,
          "environment_assets": {
            "task_bddl": {"path": "/...", "sha256": "...", "size_bytes": 1},
            "initial_states": {"path": "/...", "sha256": "...", "size_bytes": 1}
          }
        }, ...
      ],
      "num_trials_per_task": 50,
      "routing": {
        "routing_mode": "gate",            # static or gate
        "configured_video_steps": 10,
        "inference_mode": null,             # wo/w for static, null for gate
        "gate_threshold": 0.5151455998420715,
        "calibration_complete_sha256": "..."
      },
      "expected_identities": {
        "evaluation_git_identity": {...},
        "base_checkpoint_sha256": "...",
        "alignment_export_sha256": "...",
        "data_manifest_sha256": "...",
        "normalization_stats_sha256": "...",
        "vae_sha256": "...",
        "gate_checkpoint_sha256": "...",   # null for static
        "calibration_manifest_sha256": "..." # null for static
      },
      "simulator_runtime_identity": {...},
      "protocol_shared": {...}
    }

``evaluation_protocol_identity`` has four ordinary task-specific fields
(``task_suite_name``, ``task_id``, ``source_initial_state_count`` and
``max_environment_steps``) plus ``protocol_sha256``.  Every other field must
exactly equal ``protocol_shared``.  The max-step value is derived from the
suite (400 for spatial/object/goal and 700 for libero_10), so it cannot be a
single shared value across the 40-task benchmark.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from fastwam.alignment.libero_simulator_identity import (
    validate_libero_simulator_runtime_identity,
    verify_libero_simulator_runtime_identity,
)


CONDITION_CONTRACT_SCHEMA_VERSION = 1
CONDITION_CONTRACT_KIND = "libero_phase_a_condition_contract"
TASK_RECEIPT_SCHEMA_VERSION = 1
TASK_RECEIPT_KIND = "libero_phase_a_task_validation_receipt"
AGGREGATE_RECEIPT_SCHEMA_VERSION = 1
AGGREGATE_RECEIPT_KIND = "libero_phase_a_condition_validation_receipt"

_CONTRACT_KEYS = {
    "schema_version",
    "kind",
    "condition_id",
    "expected_tasks",
    "num_trials_per_task",
    "routing",
    "expected_identities",
    "simulator_runtime_identity",
    "protocol_shared",
}
_TASK_KEYS = {
    "task_suite_name",
    "task_id",
    "source_initial_state_count",
    "environment_assets",
}
_ROUTING_KEYS = {
    "routing_mode",
    "configured_video_steps",
    "inference_mode",
    "gate_threshold",
    "calibration_complete_sha256",
}
_IDENTITY_KEYS = {
    "evaluation_git_identity",
    "base_checkpoint_sha256",
    "alignment_export_sha256",
    "data_manifest_sha256",
    "normalization_stats_sha256",
    "vae_sha256",
    "gate_checkpoint_sha256",
    "calibration_manifest_sha256",
}
_PROTOCOL_TASK_KEYS = {
    "task_suite_name",
    "task_id",
    "source_initial_state_count",
    "max_environment_steps",
    "environment_assets",
    "protocol_sha256",
}
_SUITE_MAX_STEPS = {
    "libero_spatial": 400,
    "libero_object": 400,
    "libero_goal": 400,
    "libero_10": 700,
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _canonical_json_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("value must be finite JSON data") from error
    return hashlib.sha256(encoded).hexdigest()


def _json_copy(value: Any, *, field: str) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be finite JSON data") from error


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


def _integer(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{field} must be a {qualifier} integer")
    return int(value)


def _real(value: Any, *, field: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a real number")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{field} must be finite and >= {minimum}")
    return result


def _probability(value: Any, *, field: str) -> float:
    result = _real(value, field=field)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be in [0, 1]")
    return result


def _sha256(value: Any, *, field: str, nullable: bool = False) -> str | None:
    if nullable and value is None:
        return None
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must contain 64 lowercase hexadecimal characters")
    return value


def _sequence(value: Any, *, field: str) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{field} must be a sequence")
    return list(value)


def _nested(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ValueError(f"result identity is missing {path}")
        current = current[part]
    return current


def _same_real(actual: Any, expected: Any, *, field: str, atol: float = 1e-12) -> None:
    actual_float = _real(actual, field=field)
    expected_float = _real(expected, field=f"expected {field}")
    if not math.isclose(actual_float, expected_float, rel_tol=0.0, abs_tol=atol):
        raise ValueError(f"{field} mismatch: expected={expected_float}, actual={actual_float}")


def _validate_git_identity(value: Any, *, expected: Any) -> dict[str, Any]:
    identity = _mapping(value, field="evaluation_git_identity")
    _exact_keys(
        identity,
        {"commit", "tracked_dirty", "untracked_source_files"},
        field="evaluation_git_identity",
    )
    if not isinstance(identity["commit"], str) or _GIT_COMMIT_RE.fullmatch(identity["commit"]) is None:
        raise ValueError("evaluation_git_identity.commit must be a lowercase 40-char Git commit")
    if not isinstance(identity["tracked_dirty"], bool):
        raise ValueError("evaluation_git_identity.tracked_dirty must be boolean")
    untracked = _sequence(
        identity["untracked_source_files"],
        field="evaluation_git_identity.untracked_source_files",
    )
    if any(not isinstance(item, str) or not item for item in untracked):
        raise ValueError("evaluation_git_identity.untracked_source_files must contain strings")
    if identity != expected:
        raise ValueError("evaluation_git_identity does not match the condition contract")
    return identity


def _validate_environment_asset(value: Any, *, field: str) -> dict[str, Any]:
    identity = _mapping(value, field=field)
    _exact_keys(identity, {"path", "sha256", "size_bytes"}, field=field)
    raw_path = identity["path"]
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise ValueError(f"{field}.path must be absolute")
    try:
        path = Path(raw_path).expanduser().resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{field}.path does not exist: {raw_path}") from error
    if not path.is_file():
        raise ValueError(f"{field}.path must be a regular file")
    expected_sha = _sha256(identity["sha256"], field=f"{field}.sha256")
    expected_size = _integer(identity["size_bytes"], field=f"{field}.size_bytes", minimum=1)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (
        before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise RuntimeError(f"{field} changed while it was being verified")
    if after.st_size != expected_size or digest.hexdigest() != expected_sha:
        raise ValueError(f"{field} byte identity mismatch")
    return {"path": str(path), "sha256": expected_sha, "size_bytes": expected_size}


def validate_condition_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and detach one Phase-A condition contract.

    Unknown keys and internally inconsistent static/Gate combinations are
    rejected.  The returned mapping is JSON detached and may safely be reused
    by task and aggregate validators.
    """

    normalized = _mapping(contract, field="condition contract")
    _exact_keys(normalized, _CONTRACT_KEYS, field="condition contract")
    if (
        normalized["schema_version"] != CONDITION_CONTRACT_SCHEMA_VERSION
        or normalized["kind"] != CONDITION_CONTRACT_KIND
    ):
        raise ValueError("unsupported LIBERO Phase-A condition contract")
    condition_id = normalized["condition_id"]
    if not isinstance(condition_id, str) or not condition_id.strip():
        raise ValueError("condition_id must be a non-empty string")
    num_trials = _integer(
        normalized["num_trials_per_task"],
        field="num_trials_per_task",
        minimum=1,
    )

    raw_tasks = _sequence(normalized["expected_tasks"], field="expected_tasks")
    if not raw_tasks:
        raise ValueError("expected_tasks must not be empty")
    tasks: list[dict[str, Any]] = []
    task_keys: set[tuple[str, int]] = set()
    for index, raw_task in enumerate(raw_tasks):
        task = _mapping(raw_task, field=f"expected_tasks[{index}]")
        _exact_keys(task, _TASK_KEYS, field=f"expected_tasks[{index}]")
        suite = task["task_suite_name"]
        if suite not in _SUITE_MAX_STEPS:
            raise ValueError(f"expected_tasks[{index}].task_suite_name is unsupported")
        task_id = _integer(task["task_id"], field=f"expected_tasks[{index}].task_id")
        if task_id > 9:
            raise ValueError(f"expected_tasks[{index}].task_id must be in [0, 9]")
        source_initial_state_count = _integer(
            task["source_initial_state_count"],
            field=f"expected_tasks[{index}].source_initial_state_count",
            minimum=1,
        )
        environment_assets = _mapping(
            task["environment_assets"],
            field=f"expected_tasks[{index}].environment_assets",
        )
        _exact_keys(
            environment_assets,
            {"task_bddl", "initial_states"},
            field=f"expected_tasks[{index}].environment_assets",
        )
        verified_environment_assets = {
            name: _validate_environment_asset(
                environment_assets[name],
                field=f"expected_tasks[{index}].environment_assets.{name}",
            )
            for name in ("task_bddl", "initial_states")
        }
        if verified_environment_assets != environment_assets:
            raise ValueError(
                f"expected_tasks[{index}].environment_assets paths must be canonical"
            )
        key = (suite, task_id)
        if key in task_keys:
            raise ValueError(f"expected_tasks contains duplicate task {suite}/{task_id}")
        task_keys.add(key)
        tasks.append(
            {
                "task_suite_name": suite,
                "task_id": task_id,
                "source_initial_state_count": source_initial_state_count,
                "environment_assets": verified_environment_assets,
            }
        )

    routing = _mapping(normalized["routing"], field="routing")
    _exact_keys(routing, _ROUTING_KEYS, field="routing")
    mode = routing["routing_mode"]
    if mode not in {"static", "gate"}:
        raise ValueError("routing.routing_mode must be static or gate")
    steps = _integer(
        routing["configured_video_steps"],
        field="routing.configured_video_steps",
        minimum=1,
    )
    if steps != 10:
        raise ValueError("formal LIBERO Phase-A routing is bound to N=10")
    if mode == "static":
        if routing["inference_mode"] not in {"wo", "w"}:
            raise ValueError("static routing requires inference_mode wo or w")
        if routing["gate_threshold"] is not None:
            raise ValueError("static routing requires gate_threshold=null")
        if routing["calibration_complete_sha256"] is not None:
            raise ValueError("static routing requires calibration_complete_sha256=null")
    else:
        if routing["inference_mode"] is not None:
            raise ValueError("Gate routing requires inference_mode=null")
        _probability(routing["gate_threshold"], field="routing.gate_threshold")
        _sha256(
            routing["calibration_complete_sha256"],
            field="routing.calibration_complete_sha256",
        )

    identities = _mapping(normalized["expected_identities"], field="expected_identities")
    _exact_keys(identities, _IDENTITY_KEYS, field="expected_identities")
    expected_git = _mapping(
        identities["evaluation_git_identity"],
        field="expected_identities.evaluation_git_identity",
    )
    _validate_git_identity(expected_git, expected=expected_git)
    for key in (
        "base_checkpoint_sha256",
        "alignment_export_sha256",
        "data_manifest_sha256",
        "normalization_stats_sha256",
        "vae_sha256",
    ):
        _sha256(identities[key], field=f"expected_identities.{key}")
    gate_sha = _sha256(
        identities["gate_checkpoint_sha256"],
        field="expected_identities.gate_checkpoint_sha256",
        nullable=True,
    )
    calibration_sha = _sha256(
        identities["calibration_manifest_sha256"],
        field="expected_identities.calibration_manifest_sha256",
        nullable=True,
    )
    if mode == "static" and (gate_sha is not None or calibration_sha is not None):
        raise ValueError("static routing requires null Gate/calibration identities")
    if mode == "gate" and (gate_sha is None or calibration_sha is None):
        raise ValueError("Gate routing requires Gate/calibration identities")

    # Contract validation is intentionally non-physical for the simulator
    # runtime. Task validation performs a fresh physical recapture; keeping
    # that operation out of this reusable schema validator avoids repeatedly
    # hashing native libraries and the LIBERO source tree.
    simulator_runtime_identity = validate_libero_simulator_runtime_identity(
        normalized["simulator_runtime_identity"]
    )


    protocol_shared = _mapping(normalized["protocol_shared"], field="protocol_shared")
    collisions = _PROTOCOL_TASK_KEYS.intersection(protocol_shared)
    if collisions:
        raise ValueError(
            "protocol_shared contains task-specific fields: " + ", ".join(sorted(collisions))
        )
    required_protocol = {
        "schema_version",
        "kind",
        "benchmark",
        "num_trials",
        "num_inference_steps",
        "retry_invalid_episodes",
        "env_num",
        "render_resolution",
        "timing",
        "routing_runtime_identity_sha256",
        "simulator_runtime_identity_sha256",
    }
    missing_protocol = required_protocol.difference(protocol_shared)
    if missing_protocol:
        raise ValueError(
            f"protocol_shared is missing required fields: {sorted(missing_protocol)}"
        )
    if (
        protocol_shared["schema_version"] != 1
        or protocol_shared["kind"] != "libero_closed_loop_evaluation_protocol"
        or protocol_shared["benchmark"] != "LIBERO"
    ):
        raise ValueError("protocol_shared has an unsupported protocol identity")
    if protocol_shared["num_trials"] != num_trials:
        raise ValueError("protocol_shared.num_trials disagrees with num_trials_per_task")
    if protocol_shared["num_inference_steps"] != steps:
        raise ValueError("protocol_shared.num_inference_steps disagrees with routing")
    if not isinstance(protocol_shared["retry_invalid_episodes"], bool):
        raise ValueError("protocol_shared.retry_invalid_episodes must be boolean")
    if protocol_shared["env_num"] != 1:
        raise ValueError("formal LIBERO protocol requires env_num=1")
    if protocol_shared["render_resolution"] != 256:
        raise ValueError("formal LIBERO protocol requires render_resolution=256")
    _sha256(
        protocol_shared["routing_runtime_identity_sha256"],
        field="protocol_shared.routing_runtime_identity_sha256",
    )

    simulator_runtime_identity_sha = _sha256(
        protocol_shared["simulator_runtime_identity_sha256"],
        field="protocol_shared.simulator_runtime_identity_sha256",
    )
    if simulator_runtime_identity_sha != simulator_runtime_identity["identity_sha256"]:
        raise ValueError(
            "protocol_shared.simulator_runtime_identity_sha256 disagrees with "
            "simulator_runtime_identity"
        )

    normalized["expected_tasks"] = tasks
    normalized["num_trials_per_task"] = num_trials
    normalized["routing"] = routing
    normalized["expected_identities"] = identities
    normalized["simulator_runtime_identity"] = simulator_runtime_identity
    normalized["protocol_shared"] = protocol_shared
    return _json_copy(normalized, field="condition contract")


def condition_contract_sha256(contract: Mapping[str, Any]) -> str:
    """Return the canonical SHA256 of a validated condition contract."""

    return _canonical_json_sha256(validate_condition_contract(contract))


def _validate_protocol(
    contract: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    suite: str,
    task_id: int,
    expected_task: Mapping[str, Any],
) -> dict[str, Any]:
    protocol = _mapping(
        result.get("evaluation_protocol_identity"),
        field="evaluation_protocol_identity",
    )
    expected_keys = set(contract["protocol_shared"]) | _PROTOCOL_TASK_KEYS
    _exact_keys(protocol, expected_keys, field="evaluation_protocol_identity")
    if protocol["task_suite_name"] != suite or protocol["task_id"] != task_id:
        raise ValueError("evaluation_protocol_identity task does not match result task")
    source_initial_state_count = _integer(
        protocol["source_initial_state_count"],
        field="evaluation_protocol_identity.source_initial_state_count",
        minimum=1,
    )
    if source_initial_state_count != expected_task["source_initial_state_count"]:
        raise ValueError(
            "evaluation_protocol_identity.source_initial_state_count does not "
            "match the condition contract"
        )
    if protocol["max_environment_steps"] != _SUITE_MAX_STEPS[suite]:
        raise ValueError("evaluation_protocol_identity.max_environment_steps is invalid for suite")
    environment_assets = _mapping(
        protocol["environment_assets"],
        field="evaluation_protocol_identity.environment_assets",
    )
    _exact_keys(
        environment_assets,
        {"task_bddl", "initial_states"},
        field="evaluation_protocol_identity.environment_assets",
    )
    verified_environment_assets = {
        name: _validate_environment_asset(
            environment_assets[name],
            field=f"evaluation_protocol_identity.environment_assets.{name}",
        )
        for name in ("task_bddl", "initial_states")
    }
    if verified_environment_assets != environment_assets:
        raise ValueError("environment asset paths are not canonical absolute paths")
    if verified_environment_assets != expected_task["environment_assets"]:
        raise ValueError(
            "evaluation_protocol_identity.environment_assets do not match the "
            "condition contract"
        )
    shared_actual = {
        key: value for key, value in protocol.items() if key not in _PROTOCOL_TASK_KEYS
    }
    if shared_actual != contract["protocol_shared"]:
        raise ValueError("evaluation protocol shared fields do not match condition contract")
    protocol_sha = _sha256(
        protocol["protocol_sha256"],
        field="evaluation_protocol_identity.protocol_sha256",
    )
    unhashed = dict(protocol)
    unhashed.pop("protocol_sha256")
    if _canonical_json_sha256(unhashed) != protocol_sha:
        raise ValueError("evaluation_protocol_identity self-SHA256 mismatch")
    return protocol


def _validate_result_identities(
    contract: Mapping[str, Any], result: Mapping[str, Any]
) -> dict[str, Any]:
    expected = contract["expected_identities"]
    git_identity = _validate_git_identity(
        result.get("evaluation_git_identity"),
        expected=expected["evaluation_git_identity"],
    )
    model = _mapping(result.get("model_artifact_identity"), field="model_artifact_identity")
    if model.get("schema_version") != 1 or model.get("kind") != "stage3_aligned_model_identity":
        raise ValueError("unsupported model_artifact_identity")
    observed = {
        "base_checkpoint_sha256": _nested(model, "base_checkpoint.sha256"),
        "alignment_export_sha256": _nested(model, "alignment_export.sha256"),
        "data_manifest_sha256": _nested(model, "data_manifest_sha256"),
        "normalization_stats_sha256": _nested(
            model, "runtime_assets.normalization_stats.sha256"
        ),
        "vae_sha256": _nested(model, "runtime_assets.vae.sha256"),
    }
    for key, value in observed.items():
        _sha256(value, field=f"model_artifact_identity {key}")
        if value != expected[key]:
            raise ValueError(f"model_artifact_identity {key} mismatch")
    export_metadata = _mapping(
        _nested(model, "alignment_export.export_metadata"),
        field="model alignment export metadata",
    )
    if export_metadata.get("base_checkpoint_sha256") != observed["base_checkpoint_sha256"]:
        raise ValueError("alignment export/base checkpoint identity mismatch")
    if export_metadata.get("data_manifest_sha256") != observed["data_manifest_sha256"]:
        raise ValueError("alignment export/data manifest identity mismatch")

    runtime = _mapping(
        result.get("routing_runtime_identity"), field="routing_runtime_identity"
    )
    if runtime.get("schema_version") != 1 or runtime.get("kind") != "binary_video_routing_runtime":
        raise ValueError("unsupported routing_runtime_identity")
    routing = contract["routing"]
    for key in ("routing_mode", "configured_video_steps", "inference_mode", "gate_threshold"):
        runtime_key = key
        if runtime.get(runtime_key) != routing[key]:
            raise ValueError(f"routing_runtime_identity.{runtime_key} mismatch")
    expected_gate_sha = expected["gate_checkpoint_sha256"]
    expected_calibration_sha = expected["calibration_manifest_sha256"]
    gate = runtime.get("gate_checkpoint")
    calibration = runtime.get("gate_calibration")
    if routing["routing_mode"] == "static":
        if gate is not None or calibration is not None:
            raise ValueError("static routing result unexpectedly contains Gate identity")
        if runtime.get("gate_decision_rule") is not None:
            raise ValueError("static routing result unexpectedly contains Gate decision rule")
    else:
        gate = _mapping(gate, field="routing_runtime_identity.gate_checkpoint")
        calibration = _mapping(
            calibration, field="routing_runtime_identity.gate_calibration"
        )
        if gate.get("sha256") != expected_gate_sha:
            raise ValueError("Gate checkpoint SHA256 mismatch")
        for gate_key, expected_value in (
            ("adapter_checkpoint_sha256", expected["alignment_export_sha256"]),
            ("base_checkpoint_sha256", expected["base_checkpoint_sha256"]),
            ("data_manifest_sha256", expected["data_manifest_sha256"]),
        ):
            if gate.get(gate_key) != expected_value:
                raise ValueError(f"Gate checkpoint {gate_key} disagrees with endpoint")
        if _nested(calibration, "complete_file.sha256") != routing["calibration_complete_sha256"]:
            raise ValueError("Gate calibration COMPLETE SHA256 mismatch")
        if _nested(calibration, "manifest_file.semantic_sha256") != expected_calibration_sha:
            raise ValueError("Gate calibration manifest SHA256 mismatch")
        if calibration.get("gate_checkpoint_sha256") != expected_gate_sha:
            raise ValueError("Gate calibration/Gate checkpoint identity mismatch")
        if calibration.get("configured_video_steps") != routing["configured_video_steps"]:
            raise ValueError("Gate calibration configured_video_steps mismatch")
        if _nested(calibration, "selected_point.threshold") != routing["gate_threshold"]:
            raise ValueError("Gate calibration selected threshold mismatch")
        if runtime.get("gate_decision_rule") != "sigmoid(logit) >= gate_threshold -> w":
            raise ValueError("Gate decision rule mismatch")
        source = _mapping(calibration.get("source_identities"), field="calibration source identities")
        for key, expected_value in (
            ("adapter_checkpoint_sha256", expected["alignment_export_sha256"]),
            ("base_checkpoint_sha256", expected["base_checkpoint_sha256"]),
            ("data_manifest_sha256", expected["data_manifest_sha256"]),
            ("normalization_stats_sha256", expected["normalization_stats_sha256"]),
        ):
            if source.get(key) != expected_value:
                raise ValueError(f"Gate calibration {key} disagrees with endpoint")
    simulator_runtime_identity = validate_libero_simulator_runtime_identity(
        result.get("simulator_runtime_identity")
    )
    if simulator_runtime_identity != contract["simulator_runtime_identity"]:
        raise ValueError(
            "simulator_runtime_identity does not match the condition contract"
        )
    return {
        "evaluation_git_identity": git_identity,
        "model_artifact_identity": model,
        "routing_runtime_identity": runtime,
        "simulator_runtime_identity": simulator_runtime_identity,
    }


def _validate_summary(
    summary_value: Any,
    queries: Sequence[Mapping[str, Any]],
    *,
    field: str,
) -> None:
    summary = _mapping(summary_value, field=field)
    total = len(queries)
    wo_count = sum(query["selected_mode"] == "wo" for query in queries)
    w_count = total - wo_count
    total_steps = sum(int(query["actual_video_steps"]) for query in queries)
    counts = _mapping(summary.get("counts"), field=f"{field}.counts")
    if counts != {"total": total, "wo": wo_count, "w": w_count}:
        raise ValueError(f"{field}.counts disagrees with raw queries")
    _same_real(
        summary.get("with_rate"),
        w_count / total if total else 0.0,
        field=f"{field}.with_rate",
    )
    effective = _mapping(
        summary.get("effective_video_steps"),
        field=f"{field}.effective_video_steps",
    )
    if effective.get("total") != total_steps:
        raise ValueError(f"{field}.effective_video_steps.total disagrees with raw queries")
    _same_real(
        effective.get("mean"),
        total_steps / total if total else 0.0,
        field=f"{field}.effective_video_steps.mean",
    )
    by_route = _mapping(summary.get("by_route"), field=f"{field}.by_route")
    if set(by_route) != {"wo", "w"}:
        raise ValueError(f"{field}.by_route must contain exactly wo and w")
    for route, count, route_steps in (("wo", wo_count, 0), ("w", w_count, 10 * w_count)):
        route_summary = _mapping(by_route[route], field=f"{field}.by_route.{route}")
        if route_summary.get("count") != count:
            raise ValueError(f"{field}.by_route.{route}.count mismatch")
        route_effective = _mapping(
            route_summary.get("effective_video_steps"),
            field=f"{field}.by_route.{route}.effective_video_steps",
        )
        if route_effective.get("total") != route_steps:
            raise ValueError(f"{field}.by_route.{route} video-step total mismatch")
        _same_real(
            route_effective.get("mean"),
            route_steps / count if count else 0.0,
            field=f"{field}.by_route.{route}.effective_video_steps.mean",
        )
    timing = _mapping(summary.get("timing"), field=f"{field}.timing")
    timed = sum(bool(query.get("timing_included", True)) for query in queries)
    if timing.get("query_count") != timed or timing.get("warmup_query_count") != total - timed:
        raise ValueError(f"{field}.timing counts disagree with raw queries")


def _validate_query(
    query_value: Any,
    *,
    contract: Mapping[str, Any],
    suite: str,
    task_id: int,
    episode_index: int,
    replan_index: int,
    global_query_index: int,
) -> dict[str, Any]:
    query = _mapping(query_value, field="routing query")
    for key in (
        "episode_index",
        "replan_index",
        "global_query_index",
        "attempt_index",
        "environment_step",
        "query_id",
        "selected_mode",
        "configured_video_steps",
        "actual_video_steps",
        "logit",
        "probability",
        "gate_latency_s",
        "policy_latency_s",
        "preprocess_plus_gate_latency_s",
        "total_latency_s",
        "timing_included",
        "timing_synchronized",
    ):
        if key not in query:
            raise ValueError(f"routing query is missing {key}")
    if query["episode_index"] != episode_index:
        raise ValueError("routing query episode_index mismatch")
    if query["replan_index"] != replan_index:
        raise ValueError("routing query replan_index must be contiguous from zero")
    if query["global_query_index"] != global_query_index:
        raise ValueError("routing query global_query_index must be globally contiguous")
    expected_query_id = f"{suite}/{task_id}/{episode_index}/{replan_index}"
    if query["query_id"] != expected_query_id:
        raise ValueError("routing query query_id mismatch")
    _integer(query["attempt_index"], field="routing query attempt_index")
    _integer(query["environment_step"], field="routing query environment_step")
    if not isinstance(query["timing_included"], bool) or not isinstance(
        query["timing_synchronized"], bool
    ):
        raise ValueError("routing query timing flags must be boolean")
    runtime_timing = _mapping(
        contract["protocol_shared"].get("timing"), field="protocol_shared.timing"
    )
    warmup = _integer(
        runtime_timing.get("warmup_queries_per_task"),
        field="protocol_shared.timing.warmup_queries_per_task",
    )
    if query["timing_included"] is not (global_query_index >= warmup):
        raise ValueError("routing query timing_included disagrees with warmup protocol")
    if query["timing_synchronized"] is not runtime_timing.get("enabled"):
        raise ValueError("routing query timing_synchronized disagrees with timing protocol")
    for key in ("policy_latency_s", "preprocess_plus_gate_latency_s", "total_latency_s"):
        _real(query[key], field=f"routing query {key}", minimum=0.0)

    routing = contract["routing"]
    route = query["selected_mode"]
    if route not in {"wo", "w"}:
        raise ValueError("routing query selected_mode must be wo or w")
    if query["configured_video_steps"] != routing["configured_video_steps"]:
        raise ValueError("routing query configured_video_steps mismatch")
    required_steps = 0 if route == "wo" else 10
    if query["actual_video_steps"] != required_steps:
        raise ValueError(f"{route} routing query must report {required_steps} video steps")
    if routing["routing_mode"] == "static":
        if route != routing["inference_mode"]:
            raise ValueError("static routing query selected the wrong branch")
        if any(query[key] is not None for key in ("logit", "probability", "gate_latency_s")):
            raise ValueError("static routing query must not contain Gate telemetry")
    else:
        logit = _real(query["logit"], field="routing query logit")
        probability = _probability(query["probability"], field="routing query probability")
        _real(query["gate_latency_s"], field="routing query gate_latency_s", minimum=0.0)
        sigmoid = 1.0 / (1.0 + math.exp(-logit)) if logit >= 0 else math.exp(logit) / (1.0 + math.exp(logit))
        if not math.isclose(probability, sigmoid, rel_tol=0.0, abs_tol=2e-7):
            raise ValueError("routing query probability disagrees with sigmoid(logit)")
        selected = "w" if probability >= routing["gate_threshold"] else "wo"
        if route != selected:
            raise ValueError("Gate routing query violates threshold decision rule")
    return query


def validate_task_result(
    contract: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one loaded task result and return a JSON-safe receipt."""

    contract = validate_condition_contract(contract)
    expected_simulator_identity = contract["simulator_runtime_identity"]
    source_root = Path(expected_simulator_identity["libero_source_tree"]["root"])
    verify_libero_simulator_runtime_identity(
        expected_simulator_identity,
        libero_root=source_root.parents[1],
    )
    result = _mapping(result, field="task result")
    suite = result.get("task_suite")
    if suite not in _SUITE_MAX_STEPS:
        raise ValueError("task result task_suite is unsupported")
    task_id = _integer(result.get("task_id"), field="task result task_id")
    task_key = (suite, task_id)
    expected_tasks = {
        (task["task_suite_name"], task["task_id"]): task
        for task in contract["expected_tasks"]
    }
    if task_key not in expected_tasks:
        raise ValueError(f"task result {suite}/{task_id} is outside contract coverage")
    expected_task = expected_tasks[task_key]

    identities = _validate_result_identities(contract, result)
    protocol = _validate_protocol(
        contract,
        result,
        suite=suite,
        task_id=task_id,
        expected_task=expected_task,
    )
    runtime_sha = _canonical_json_sha256(identities["routing_runtime_identity"])
    if protocol.get("routing_runtime_identity_sha256") != runtime_sha:
        raise ValueError("protocol/routing runtime identity SHA256 mismatch")

    num_trials = contract["num_trials_per_task"]
    if result.get("total_episodes") != num_trials:
        raise ValueError("task result total_episodes mismatch")
    successes = _integer(result.get("successes"), field="task result successes")
    if successes > num_trials:
        raise ValueError("task result successes exceeds total episodes")
    success_episodes = _sequence(result.get("success_episodes"), field="success_episodes")
    failure_episodes = _sequence(result.get("failure_episodes"), field="failure_episodes")
    for field, values in (("success_episodes", success_episodes), ("failure_episodes", failure_episodes)):
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ValueError(f"{field} must contain integer indices")
        if len(set(values)) != len(values):
            raise ValueError(f"{field} contains duplicates")
    if len(success_episodes) != successes:
        raise ValueError("successes disagrees with success_episodes")
    if set(success_episodes).intersection(failure_episodes):
        raise ValueError("success/failure episode sets overlap")
    if set(success_episodes).union(failure_episodes) != set(range(num_trials)):
        raise ValueError("success/failure episodes do not partition all trials")

    invalid_episodes = _sequence(result.get("invalid_episodes"), field="invalid_episodes")
    invalid_count = _integer(
        result.get("invalid_episode_count"), field="invalid_episode_count"
    )
    attempted = _integer(result.get("attempted_episodes"), field="attempted_episodes")
    if invalid_count != len(invalid_episodes) or attempted != num_trials + invalid_count:
        raise ValueError("attempted/invalid episode counts are inconsistent")
    if protocol.get("retry_invalid_episodes") is False and invalid_count != 0:
        raise ValueError("protocol forbids invalid-episode retries")

    routing_container = _mapping(result.get("routing"), field="routing")
    invalid_alias = _sequence(
        routing_container.get("invalid_attempts"), field="routing.invalid_attempts"
    )
    if invalid_alias != invalid_episodes:
        raise ValueError("routing.invalid_attempts disagrees with invalid_episodes")
    invalid_attempt_ids: set[int] = set()
    for index, raw_invalid in enumerate(invalid_episodes):
        invalid = _mapping(raw_invalid, field=f"invalid_episodes[{index}]")
        target_trial = _integer(
            invalid.get("target_trial"), field=f"invalid_episodes[{index}].target_trial"
        )
        if target_trial >= num_trials:
            raise ValueError("invalid episode target_trial is out of range")
        attempt_id = _integer(
            invalid.get("attempt"), field=f"invalid_episodes[{index}].attempt"
        )
        if attempt_id >= attempted or attempt_id in invalid_attempt_ids:
            raise ValueError("invalid episode attempt index is invalid or duplicated")
        invalid_attempt_ids.add(attempt_id)
        if not isinstance(invalid.get("reason"), str) or not invalid["reason"]:
            raise ValueError("invalid episode reason must be non-empty")

    episodes = _sequence(routing_container.get("episodes"), field="routing.episodes")
    if len(episodes) != num_trials:
        raise ValueError("routing.episodes count mismatch")
    all_queries: list[dict[str, Any]] = []
    valid_attempt_ids: set[int] = set()
    previous_attempt = -1
    for episode_index, raw_episode in enumerate(episodes):
        episode = _mapping(raw_episode, field=f"routing.episodes[{episode_index}]")
        if episode.get("episode_index") != episode_index:
            raise ValueError("routing episode indices must be contiguous from zero")
        expected_success = episode_index in set(success_episodes)
        if episode.get("success") is not expected_success:
            raise ValueError("routing episode success disagrees with success partition")
        queries = _sequence(
            episode.get("queries"), field=f"routing.episodes[{episode_index}].queries"
        )
        if not queries:
            raise ValueError("formal routing episodes must contain at least one query")
        validated_queries: list[dict[str, Any]] = []
        for replan_index, query in enumerate(queries):
            validated_queries.append(
                _validate_query(
                    query,
                    contract=contract,
                    suite=suite,
                    task_id=task_id,
                    episode_index=episode_index,
                    replan_index=replan_index,
                    global_query_index=len(all_queries),
                )
            )
            all_queries.append(validated_queries[-1])
        attempt_ids = {query["attempt_index"] for query in validated_queries}
        if len(attempt_ids) != 1:
            raise ValueError("all queries in one valid episode must share attempt_index")
        attempt_id = next(iter(attempt_ids))
        if attempt_id <= previous_attempt or attempt_id in invalid_attempt_ids:
            raise ValueError("valid episode attempt indices must be increasing and non-invalid")
        previous_attempt = attempt_id
        valid_attempt_ids.add(attempt_id)
        environment_steps = [query["environment_step"] for query in validated_queries]
        if environment_steps != sorted(environment_steps) or len(set(environment_steps)) != len(environment_steps):
            raise ValueError("episode routing environment_step values must strictly increase")
        if episode.get("query_count") != len(validated_queries):
            raise ValueError("routing episode query_count mismatch")
        episode_steps = sum(query["actual_video_steps"] for query in validated_queries)
        if episode.get("total_actual_video_steps") != episode_steps:
            raise ValueError("routing episode video-step total mismatch")
        _validate_summary(
            episode.get("summary"),
            validated_queries,
            field=f"routing.episodes[{episode_index}].summary",
        )
    if valid_attempt_ids.union(invalid_attempt_ids) != set(range(attempted)):
        raise ValueError("valid and invalid attempt indices do not cover attempted_episodes")
    _validate_summary(routing_container.get("summary"), all_queries, field="routing.summary")

    wo_count = sum(query["selected_mode"] == "wo" for query in all_queries)
    w_count = len(all_queries) - wo_count
    actual_video_nfe = 10 * w_count
    receipt = {
        "schema_version": TASK_RECEIPT_SCHEMA_VERSION,
        "kind": TASK_RECEIPT_KIND,
        "condition_id": contract["condition_id"],
        "condition_contract_sha256": _canonical_json_sha256(contract),
        "task_suite_name": suite,
        "task_id": task_id,
        "num_episodes": num_trials,
        "successes": successes,
        "success_rate": float(successes / num_trials),
        "attempted_episodes": attempted,
        "invalid_episode_count": invalid_count,
        "query_count": len(all_queries),
        "route_counts": {"wo": wo_count, "w": w_count},
        "actual_total_video_nfe": actual_video_nfe,
        "actual_video_nfe_per_query": float(actual_video_nfe / len(all_queries)),
        "protocol_sha256": protocol["protocol_sha256"],
        "routing_runtime_identity_sha256": runtime_sha,
        "model_artifact_identity_sha256": _canonical_json_sha256(
            identities["model_artifact_identity"]
        ),
        "simulator_runtime_identity_sha256": identities[
            "simulator_runtime_identity"
        ]["identity_sha256"],
        "evaluation_git_identity": identities["evaluation_git_identity"],
    }
    return _json_copy(receipt, field="task validation receipt")


def validate_task_result_file(
    contract: Mapping[str, Any], path: str | Path
) -> dict[str, Any]:
    """Validate one physical result file and attach its byte identity."""

    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError("task result path must be a regular file")
    before = source.stat()
    payload_bytes = source.read_bytes()
    after = source.stat()
    if (
        before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or after.st_size != len(payload_bytes)
    ):
        raise RuntimeError("task result changed while it was being read")
    try:
        payload = json.loads(payload_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot parse task result JSON: {source}") from error
    receipt = validate_task_result(contract, payload)
    receipt["result_file"] = {
        "path": str(source),
        "sha256": hashlib.sha256(payload_bytes).hexdigest(),
        "size_bytes": len(payload_bytes),
    }
    return receipt


def aggregate_condition_result_files(
    contract: Mapping[str, Any],
    result_paths: Sequence[str | Path],
) -> dict[str, Any]:
    """Validate and pool a complete condition, rejecting gaps and duplicates.

    The returned ledger is sorted by ``(task_suite_name, task_id)`` and binds
    every source result by physical file SHA256 and byte size.
    """

    contract = validate_condition_contract(contract)
    paths = _sequence(result_paths, field="result_paths")
    expected = {
        (task["task_suite_name"], task["task_id"])
        for task in contract["expected_tasks"]
    }
    by_task: dict[tuple[str, int], dict[str, Any]] = {}
    seen_paths: set[str] = set()
    for index, raw_path in enumerate(paths):
        source = str(Path(raw_path).expanduser().resolve(strict=True))
        if source in seen_paths:
            raise ValueError(f"result_paths contains duplicate file: {source}")
        seen_paths.add(source)
        receipt = validate_task_result_file(contract, source)
        key = (receipt["task_suite_name"], receipt["task_id"])
        if key in by_task:
            raise ValueError(f"duplicate task result: {key[0]}/{key[1]}")
        by_task[key] = receipt
    observed = set(by_task)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"condition task coverage mismatch: missing={missing}, extra={extra}")

    ordered = [by_task[key] for key in sorted(by_task)]
    episodes = sum(item["num_episodes"] for item in ordered)
    successes = sum(item["successes"] for item in ordered)
    queries = sum(item["query_count"] for item in ordered)
    wo_count = sum(item["route_counts"]["wo"] for item in ordered)
    w_count = sum(item["route_counts"]["w"] for item in ordered)
    actual_nfe = sum(item["actual_total_video_nfe"] for item in ordered)
    ledger = [
        {
            "task_suite_name": item["task_suite_name"],
            "task_id": item["task_id"],
            **item["result_file"],
        }
        for item in ordered
    ]
    aggregate = {
        "schema_version": AGGREGATE_RECEIPT_SCHEMA_VERSION,
        "kind": AGGREGATE_RECEIPT_KIND,
        "condition_id": contract["condition_id"],
        "condition_contract_sha256": _canonical_json_sha256(contract),
        "simulator_runtime_identity_sha256": contract[
            "simulator_runtime_identity"
        ]["identity_sha256"],
        "task_count": len(ordered),
        "episode_count": episodes,
        "successes": successes,
        "success_rate": float(successes / episodes),
        "attempted_episode_count": sum(item["attempted_episodes"] for item in ordered),
        "invalid_episode_count": sum(item["invalid_episode_count"] for item in ordered),
        "query_count": queries,
        "route_counts": {"wo": wo_count, "w": w_count},
        "actual_with_rate": float(w_count / queries),
        "actual_total_video_nfe": actual_nfe,
        "actual_video_nfe_per_query": float(actual_nfe / queries),
        "result_file_ledger": ledger,
        "result_file_ledger_sha256": _canonical_json_sha256(ledger),
    }
    return _json_copy(aggregate, field="condition validation receipt")


__all__ = [
    "AGGREGATE_RECEIPT_KIND",
    "AGGREGATE_RECEIPT_SCHEMA_VERSION",
    "CONDITION_CONTRACT_KIND",
    "CONDITION_CONTRACT_SCHEMA_VERSION",
    "TASK_RECEIPT_KIND",
    "TASK_RECEIPT_SCHEMA_VERSION",
    "aggregate_condition_result_files",
    "condition_contract_sha256",
    "validate_condition_contract",
    "validate_task_result",
    "validate_task_result_file",
]
