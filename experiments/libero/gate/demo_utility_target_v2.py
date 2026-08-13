"""Pure, fail-closed helpers for LIBERO multi-seed Utility Target V2.

The Phase-2.5 stability audit contains a fixed 100-state by 5-seed grid.  This
module turns that *verified* long-form grid into one auditable target per state
and defines the independent seed-47--50 validation record contract.  It does
not construct a dataset or model and it never invokes a GPU.

The target is intentionally conservative.  A state is high confidence only
when all of the following hold:

* the five-seed mean lies outside the +/-1e-4 deadband;
* at least four of five seeds lie outside the deadband in the mean direction;
* the two-sided 95% Student-t interval lies wholly beyond the same deadband.

No training weight is invented here.  Raw uncertainty and a high-confidence
mask are preserved; any weighting policy remains gated on independent
seed-47--50 validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.libero.gate import demo_utility_stability as stability
from experiments.libero.gate.demo_utility import parse_sample_identity, stable_sample_seed


TARGET_BUNDLE_SCHEMA_VERSION = 1
TARGET_RECORD_SCHEMA_VERSION = 1
VALIDATION_RECORD_SCHEMA_VERSION = 1
TARGET_BUNDLE_KIND = "libero_demo_utility_target_v2"
VALIDATION_RECORD_KIND = "libero_demo_utility_target_v2_validation"
SOURCE_BUNDLE_KIND = "libero_demo_utility_multiseed_stability"
TARGET_MANIFEST_FILENAME = "manifest.json"
TARGETS_FILENAME = "targets.jsonl"

TARGET_BASE_SEEDS = (42, 43, 44, 45, 46)
VALIDATION_BASE_SEEDS = (47, 48, 49, 50)
DEFAULT_NUM_STATES = 100
DEFAULT_DEADBAND_EPSILON = 1e-4
DEFAULT_MIN_SIGN_AGREEMENT = 0.8
T95_DF4_CRITICAL = 2.7764451051977987
SHA256_HEX_LENGTH = 64

# Immutable identifiers of the completed Phase-2.5 source used for PR-4.
OFFICIAL_SOURCE_MANIFEST_SHA256 = (
    "c7476d522f47f71df30fb96ebaba5d09f6dd7a0a83400456a79ab1146506d0b9"
)
OFFICIAL_SOURCE_RECORDS_SHA256 = (
    "57abaacb551b4d6094e09812212c2be8098c9d823e58fb1de71d4f40469d4fb8"
)
OFFICIAL_SOURCE_SELECTION_PLAN_SHA256 = (
    "59f13375c815a556073f72cdff5243b73a19fc9dbec7bd58c4d419b4d72e90db"
)


def canonical_json(value: Any) -> str:
    """Return the single canonical JSON representation used by every digest."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _serialize_jsonl(records: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(f"{canonical_json(record)}\n" for record in records).encode("utf-8")


def _sha256_jsonl(records: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(_serialize_jsonl(records)).hexdigest()


def _require_sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256, got {value!r}")
    return value


def _require_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer, not bool")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer, got {value!r}") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{field} must be an integer, got {value!r}")
    if result < minimum:
        raise ValueError(f"{field} must be >= {minimum}, got {result}")
    return result


def _require_finite(value: Any, *, field: str, minimum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite, got {value!r}")
    if minimum is not None and result < minimum:
        raise ValueError(f"{field} must be >= {minimum}, got {result}")
    return result


def _require_bool(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be bool, got {value!r}")
    return value


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _load_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    result: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed JSON in {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{label} row at {path}:{line_number} must be an object")
            result.append(value)
    return result


def _same(left: Any, right: Any) -> bool:
    return canonical_json(left) == canonical_json(right)


def _source_input_projection(
    ordered_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "selection_order": int(row["selection_order"]),
            "source_index": int(row["source_index"]),
            "sample_id": str(row["sample_id"]),
            "input_hashes": dict(row["input_hashes"]),
            "source_pilot_record_sha256": str(row["source_pilot_record_sha256"]),
        }
        for row in ordered_rows
    ]


@dataclass(frozen=True)
class VerifiedSourceBundle:
    """In-memory result of verifying the complete immutable Phase-2.5 grid."""

    manifest_path: Path
    records_path: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    records_sha256: str
    selection_plan_sha256: str
    input_plan_sha256: str
    ordered_states: tuple[dict[str, Any], ...]
    record_index: dict[tuple[int, int], dict[str, Any]]

    @property
    def num_states(self) -> int:
        return len(self.ordered_states)

    @property
    def source_binding(self) -> dict[str, Any]:
        compatibility = self.manifest["compatibility"]
        return {
            "kind": SOURCE_BUNDLE_KIND,
            "manifest_sha256": self.manifest_sha256,
            "records_sha256": self.records_sha256,
            "manifest_compatibility_fingerprint": self.manifest[
                "compatibility_fingerprint"
            ],
            "selection_plan_sha256": self.selection_plan_sha256,
            "input_plan_sha256": self.input_plan_sha256,
            "num_states": self.num_states,
            "base_seeds": list(TARGET_BASE_SEEDS),
            "checkpoint_sha256": compatibility["checkpoint_sha256"],
            "dataset_stats_sha256": compatibility["dataset_stats_sha256"],
            "vae_sha256": compatibility["vae_sha256"],
            "pilot_manifest_fingerprint": compatibility[
                "pilot_manifest_fingerprint"
            ],
        }


def _validate_source_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_selection_plan_sha256: str,
    expected_num_states: int,
) -> tuple[list[dict[str, Any]], str]:
    if manifest.get("kind") != SOURCE_BUNDLE_KIND:
        raise ValueError(f"unexpected source manifest kind={manifest.get('kind')!r}")
    if int(manifest.get("schema_version", -1)) != 1:
        raise ValueError("unsupported Phase-2.5 source manifest schema")
    compatibility = manifest.get("compatibility")
    if not isinstance(compatibility, Mapping):
        raise ValueError("source manifest compatibility must be a mapping")
    fingerprint = _require_sha256(
        manifest.get("compatibility_fingerprint"),
        field="source compatibility_fingerprint",
    )
    if fingerprint != sha256_json(compatibility):
        raise ValueError("source manifest compatibility fingerprint is invalid")
    if compatibility.get("kind") != SOURCE_BUNDLE_KIND:
        raise ValueError("source compatibility kind is invalid")
    if int(compatibility.get("schema_version", -1)) != 1:
        raise ValueError("source compatibility schema is invalid")

    selection = manifest.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("source manifest selection must be a mapping")
    states = selection.get("ordered_states")
    if not isinstance(states, list) or any(not isinstance(row, dict) for row in states):
        raise ValueError("source manifest ordered_states must be a list of objects")
    expected_count = _require_int(
        expected_num_states, field="expected_num_states", minimum=1
    )
    if len(states) != expected_count:
        raise ValueError(
            f"source selection contains {len(states)} states, expected {expected_count}"
        )
    if int(selection.get("num_states", -1)) != len(states):
        raise ValueError("source selection num_states does not match ordered_states")
    state_digest = sha256_json(states)
    if selection.get("ordered_states_sha256") != state_digest:
        raise ValueError("source ordered state plan SHA-256 is invalid")
    expected_digest = _require_sha256(
        expected_selection_plan_sha256,
        field="expected_selection_plan_sha256",
    )
    if state_digest != expected_digest:
        raise ValueError(
            "source selection plan differs from the required immutable plan: "
            f"actual={state_digest}, expected={expected_digest}"
        )
    if compatibility.get("selection_plan_sha256") != state_digest:
        raise ValueError("source compatibility is not bound to its selection plan")
    if int(compatibility.get("num_states", -1)) != len(states):
        raise ValueError("source compatibility num_states is invalid")

    orders = [_require_int(row.get("selection_order"), field="selection_order") for row in states]
    if orders != list(range(len(states))):
        raise ValueError("source selection_order must be exactly 0..num_states-1")
    sources = [_require_int(row.get("source_index"), field="source_index") for row in states]
    samples = [str(row.get("sample_id", "")) for row in states]
    if len(set(sources)) != len(sources) or len(set(samples)) != len(samples):
        raise ValueError("source selection contains duplicate source_index or sample_id")
    for row in states:
        _require_sha256(row.get("pilot_record_sha256"), field="pilot_record_sha256")

    replicates = manifest.get("replicates")
    if not isinstance(replicates, Mapping):
        raise ValueError("source manifest replicates must be a mapping")
    base_seeds = tuple(int(value) for value in replicates.get("base_seeds", []))
    if base_seeds != TARGET_BASE_SEEDS:
        raise ValueError(f"source base seeds must be exactly {list(TARGET_BASE_SEEDS)}")
    if tuple(int(value) for value in compatibility.get("replicate_base_seeds", [])) != base_seeds:
        raise ValueError("source compatibility is not bound to replicate base seeds")
    if int(replicates.get("count", -1)) != len(base_seeds):
        raise ValueError("source replicate count is invalid")
    if int(replicates.get("reuse_base_seed", -1)) != TARGET_BASE_SEEDS[0]:
        raise ValueError("source must reuse base seed 42")
    if int(replicates.get("reuse_replicate_index", -1)) != 0:
        raise ValueError("source seed-42 reuse must be replicate index 0")
    if int(replicates.get("expected_record_count", -1)) != len(states) * len(base_seeds):
        raise ValueError("source expected record count is invalid")

    for field in (
        "pilot_manifest_fingerprint",
        "checkpoint_sha256",
        "dataset_stats_sha256",
        "vae_sha256",
    ):
        _require_sha256(compatibility.get(field), field=f"source compatibility.{field}")
    pilot = manifest.get("pilot")
    if not isinstance(pilot, Mapping):
        raise ValueError("source manifest pilot binding must be a mapping")
    if pilot.get("manifest_fingerprint") != compatibility.get(
        "pilot_manifest_fingerprint"
    ):
        raise ValueError("source pilot fingerprint is not bound to compatibility")
    for field in ("manifest_sha256", "records_sha256"):
        _require_sha256(pilot.get(field), field=f"source pilot.{field}")
        if pilot.get(field) != compatibility.get(f"pilot_{field}"):
            raise ValueError(f"source pilot {field} is not bound to compatibility")
    return [dict(row) for row in states], fingerprint


def load_verified_source_bundle(
    manifest_path: str | Path,
    records_path: str | Path,
    *,
    expected_manifest_sha256: str,
    expected_records_sha256: str,
    expected_selection_plan_sha256: str,
    expected_num_states: int = DEFAULT_NUM_STATES,
) -> VerifiedSourceBundle:
    """Load the exact Phase-2.5 source after exhaustive, fail-closed checks.

    All three expected digests are mandatory external trust anchors.  Internal
    self-consistency is not accepted as a substitute for the known source
    bytes and immutable 100-state plan.
    """

    manifest_file = Path(manifest_path).resolve()
    records_file = Path(records_path).resolve()
    expected_manifest = _require_sha256(
        expected_manifest_sha256, field="expected_manifest_sha256"
    )
    expected_records = _require_sha256(
        expected_records_sha256, field="expected_records_sha256"
    )
    actual_manifest = sha256_file(manifest_file)
    actual_records = sha256_file(records_file)
    if actual_manifest != expected_manifest:
        raise ValueError(
            "Phase-2.5 source manifest SHA-256 mismatch: "
            f"actual={actual_manifest}, expected={expected_manifest}"
        )
    if actual_records != expected_records:
        raise ValueError(
            "Phase-2.5 source records SHA-256 mismatch: "
            f"actual={actual_records}, expected={expected_records}"
        )

    manifest = _load_json(manifest_file, label="Phase-2.5 source manifest")
    ordered_states, fingerprint = _validate_source_manifest(
        manifest,
        expected_selection_plan_sha256=expected_selection_plan_sha256,
        expected_num_states=expected_num_states,
    )
    compatibility = manifest["compatibility"]
    index = stability.load_stability_record_index(
        records_file,
        expected_base_seeds=TARGET_BASE_SEEDS,
        expected_stability_manifest_fingerprint=fingerprint,
        expected_pilot_manifest_fingerprint=compatibility[
            "pilot_manifest_fingerprint"
        ],
        expected_checkpoint_sha256=compatibility["checkpoint_sha256"],
        expected_dataset_stats_sha256=compatibility["dataset_stats_sha256"],
        expected_vae_sha256=compatibility["vae_sha256"],
    )
    expected_keys = {
        (int(state["source_index"]), replicate_index)
        for state in ordered_states
        for replicate_index in range(len(TARGET_BASE_SEEDS))
    }
    actual_keys = set(index)
    outside = sorted(actual_keys - expected_keys)
    missing = sorted(expected_keys - actual_keys)
    if outside:
        raise ValueError(f"source records contain keys outside the 100-state plan: {outside[:10]}")
    if missing:
        raise ValueError(f"source grid is incomplete; missing keys: {missing[:10]}")
    if len(index) != len(ordered_states) * len(TARGET_BASE_SEEDS):
        raise ValueError("source grid record count is invalid")

    cross_seed_fields = (
        "sample_id",
        "dataset_id",
        "dataset_name",
        "suite",
        "episode_index",
        "episode_id",
        "frame_index",
        "task_index",
        "task_id",
        "task_id_source",
        "task",
        "valid_length",
        "target_action_shape",
        "input_hashes",
        "current_proprio",
        "source_metadata",
        "selection_order",
        "selection_bin",
        "pilot_seed",
        "pilot_e0",
        "pilot_efull",
        "pilot_utility",
        "pilot_valid_length",
        "pilot_input_combined_sha256",
        "pilot_manifest_compatibility_fingerprint",
        "source_pilot_record_sha256",
        "stability_manifest_compatibility_fingerprint",
        "checkpoint_sha256",
        "dataset_stats_sha256",
        "vae_sha256",
        "collection_git_sha",
    )
    ordered_reference_rows: list[dict[str, Any]] = []
    for state in ordered_states:
        source_index = int(state["source_index"])
        rows = [index[(source_index, replicate_index)] for replicate_index in range(5)]
        reference = rows[0]
        ordered_reference_rows.append(reference)
        plan_bindings = {
            "selection_order": reference["selection_order"],
            "source_index": reference["source_index"],
            "sample_id": reference["sample_id"],
            "suite": reference["suite"],
            "task_index": reference["task_index"],
            "episode_index": reference["episode_index"],
            "frame_index": reference["frame_index"],
            "selection_bin": reference["selection_bin"],
            "pilot_utility": reference["pilot_utility"],
            "valid_length": reference["valid_length"],
            "pilot_seed": reference["pilot_seed"],
            "pilot_record_sha256": reference["source_pilot_record_sha256"],
        }
        for field, actual in plan_bindings.items():
            if not _same(state.get(field), actual):
                raise ValueError(
                    f"source {source_index} record/selection mismatch for {field}"
                )
        for replicate_index, row in enumerate(rows):
            if int(row["replicate_index"]) != replicate_index:
                raise ValueError("source replicate index is not canonical")
            if int(row["replicate_base_seed"]) != TARGET_BASE_SEEDS[replicate_index]:
                raise ValueError("source replicate base-seed mapping is invalid")
            for field in cross_seed_fields:
                if not _same(row.get(field), reference.get(field)):
                    raise ValueError(
                        f"source {source_index} has cross-seed mismatch for {field}"
                    )
        if reference.get("collection_git_sha") != compatibility.get(
            "collection_git_commit"
        ):
            raise ValueError("source row collection_git_sha differs from manifest")

    input_plan = _source_input_projection(ordered_reference_rows)
    input_plan_sha256 = sha256_json(input_plan)
    return VerifiedSourceBundle(
        manifest_path=manifest_file,
        records_path=records_file,
        manifest=manifest,
        manifest_sha256=actual_manifest,
        records_sha256=actual_records,
        selection_plan_sha256=str(expected_selection_plan_sha256),
        input_plan_sha256=input_plan_sha256,
        ordered_states=tuple(ordered_states),
        record_index=index,
    )


def target_id(sample_id: str) -> str:
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError("sample_id must be a non-empty string")
    return f"{sample_id}/utility_target_v2"


def target_record_sha256(record: Mapping[str, Any]) -> str:
    """Hash a target's full canonical payload, excluding only its own digest."""

    if not isinstance(record, Mapping):
        raise TypeError("target record must be a mapping")
    return sha256_json({key: value for key, value in record.items() if key != "target_sha256"})


def _direction(value: float, deadband: float) -> str:
    if value > deadband:
        return "positive"
    if value < -deadband:
        return "negative"
    return "deadband"


def compute_target_record(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_binding: Mapping[str, Any],
    deadband_epsilon: float = DEFAULT_DEADBAND_EPSILON,
    min_sign_agreement: float = DEFAULT_MIN_SIGN_AGREEMENT,
) -> dict[str, Any]:
    """Aggregate one already-verified five-seed state into Target V2."""

    if len(rows) != len(TARGET_BASE_SEEDS):
        raise ValueError(f"a target requires exactly {len(TARGET_BASE_SEEDS)} source rows")
    deadband = _require_finite(
        deadband_epsilon, field="deadband_epsilon", minimum=0.0
    )
    if deadband != DEFAULT_DEADBAND_EPSILON:
        raise ValueError(f"Target V2 deadband is frozen at {DEFAULT_DEADBAND_EPSILON}")
    agreement_threshold = _require_finite(
        min_sign_agreement, field="min_sign_agreement", minimum=0.0
    )
    if agreement_threshold != DEFAULT_MIN_SIGN_AGREEMENT:
        raise ValueError(
            f"Target V2 min_sign_agreement is frozen at {DEFAULT_MIN_SIGN_AGREEMENT}"
        )
    reference = rows[0]
    expected_source_fields = {
        "manifest_sha256",
        "records_sha256",
        "manifest_compatibility_fingerprint",
        "selection_plan_sha256",
        "input_plan_sha256",
        "num_states",
        "base_seeds",
        "checkpoint_sha256",
        "dataset_stats_sha256",
        "vae_sha256",
        "pilot_manifest_fingerprint",
    }
    if not isinstance(source_binding, Mapping) or not expected_source_fields <= set(source_binding):
        missing = sorted(expected_source_fields - set(source_binding or {}))
        raise ValueError(f"source_binding is missing fields: {missing}")

    utilities: list[float] = []
    utility_rows: list[dict[str, Any]] = []
    for replicate_index, (base_seed, row) in enumerate(zip(TARGET_BASE_SEEDS, rows)):
        if int(row["replicate_index"]) != replicate_index:
            raise ValueError("source rows are not ordered by replicate_index")
        if int(row["replicate_base_seed"]) != base_seed:
            raise ValueError("source rows do not match Target V2 base seeds")
        utility = _require_finite(row["utility"], field="utility")
        e0 = _require_finite(row["e0"], field="e0", minimum=0.0)
        efull = _require_finite(row["efull"], field="efull", minimum=0.0)
        if not math.isclose(utility, e0 - efull, rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError("source utility does not equal E0-Efull")
        utilities.append(utility)
        utility_rows.append(
            {
                "base_seed": base_seed,
                "replicate_index": replicate_index,
                "replicate_seed": int(row["replicate_seed"]),
                "e0": e0,
                "efull": efull,
                "utility": utility,
            }
        )

    mean = statistics.fmean(utilities)
    median = statistics.median(utilities)
    sample_std = statistics.stdev(utilities)
    sem = sample_std / math.sqrt(len(utilities))
    margin = T95_DF4_CRITICAL * sem
    ci_low = mean - margin
    ci_high = mean + margin
    direction = _direction(mean, deadband)
    directions = [_direction(value, deadband) for value in utilities]
    positive_count = directions.count("positive")
    negative_count = directions.count("negative")
    deadband_count = directions.count("deadband")
    direction_count = directions.count(direction)
    sign_agreement = direction_count / len(directions)
    enough_direction = sign_agreement >= agreement_threshold
    ci_clears_deadband = (
        (direction == "positive" and ci_low > deadband)
        or (direction == "negative" and ci_high < -deadband)
    )
    high_confidence = (
        direction in {"positive", "negative"}
        and abs(mean) > deadband
        and enough_direction
        and ci_clears_deadband
    )

    copied_fields = (
        "sample_id",
        "dataset_id",
        "dataset_name",
        "suite",
        "episode_index",
        "episode_id",
        "frame_index",
        "task_index",
        "task_id",
        "task_id_source",
        "task",
        "source_index",
        "selection_order",
        "selection_bin",
        "valid_length",
        "target_action_shape",
        "input_hashes",
        "current_proprio",
        "source_metadata",
        "checkpoint_sha256",
        "dataset_stats_sha256",
        "vae_sha256",
        "source_pilot_record_sha256",
        "pilot_seed",
        "pilot_e0",
        "pilot_efull",
        "pilot_utility",
        "pilot_valid_length",
        "pilot_input_combined_sha256",
    )
    result: dict[str, Any] = {
        "target_record_schema_version": TARGET_RECORD_SCHEMA_VERSION,
        "kind": TARGET_BUNDLE_KIND,
        "target_id": target_id(str(reference["sample_id"])),
        **{field: reference[field] for field in copied_fields},
        "source_binding": dict(source_binding),
        "target_base_seeds": list(TARGET_BASE_SEEDS),
        "utility_by_base_seed": utility_rows,
        "utility_mean": mean,
        "utility_median": median,
        "utility_sample_std": sample_std,
        "utility_sem": sem,
        "t95_degrees_of_freedom": len(utilities) - 1,
        "t95_critical": T95_DF4_CRITICAL,
        "t95_ci_low": ci_low,
        "t95_ci_high": ci_high,
        "deadband_epsilon": deadband,
        "direction": direction,
        "positive_seed_count": positive_count,
        "negative_seed_count": negative_count,
        "deadband_seed_count": deadband_count,
        "direction_seed_count": direction_count,
        "sign_agreement": sign_agreement,
        "min_sign_agreement": agreement_threshold,
        "mean_outside_deadband": abs(mean) > deadband,
        "enough_same_direction_seeds": enough_direction,
        "ci_clears_deadband": ci_clears_deadband,
        "high_confidence": high_confidence,
        "uncertain": not high_confidence,
    }
    result["target_sha256"] = target_record_sha256(result)
    validate_target_record(result)
    return result


def _target_stats_from_record(record: Mapping[str, Any]) -> dict[str, Any]:
    rows = record["utility_by_base_seed"]
    values = [float(row["utility"]) for row in rows]
    mean = statistics.fmean(values)
    sample_std = statistics.stdev(values)
    sem = sample_std / math.sqrt(len(values))
    margin = T95_DF4_CRITICAL * sem
    deadband = float(record["deadband_epsilon"])
    direction = _direction(mean, deadband)
    directions = [_direction(value, deadband) for value in values]
    direction_count = directions.count(direction)
    sign_agreement = direction_count / len(values)
    enough = sign_agreement >= float(record["min_sign_agreement"])
    clears = (
        (direction == "positive" and mean - margin > deadband)
        or (direction == "negative" and mean + margin < -deadband)
    )
    high = direction in {"positive", "negative"} and abs(mean) > deadband and enough and clears
    return {
        "utility_mean": mean,
        "utility_median": statistics.median(values),
        "utility_sample_std": sample_std,
        "utility_sem": sem,
        "t95_degrees_of_freedom": len(values) - 1,
        "t95_critical": T95_DF4_CRITICAL,
        "t95_ci_low": mean - margin,
        "t95_ci_high": mean + margin,
        "direction": direction,
        "positive_seed_count": directions.count("positive"),
        "negative_seed_count": directions.count("negative"),
        "deadband_seed_count": directions.count("deadband"),
        "direction_seed_count": direction_count,
        "sign_agreement": sign_agreement,
        "mean_outside_deadband": abs(mean) > deadband,
        "enough_same_direction_seeds": enough,
        "ci_clears_deadband": clears,
        "high_confidence": high,
        "uncertain": not high,
    }


def validate_target_record(
    record: Mapping[str, Any],
    *,
    expected_source_manifest_sha256: str | None = None,
    expected_source_records_sha256: str | None = None,
    expected_source_manifest_fingerprint: str | None = None,
    expected_source_selection_plan_sha256: str | None = None,
    expected_source_input_plan_sha256: str | None = None,
    expected_deadband_epsilon: float | None = None,
    expected_min_sign_agreement: float | None = None,
) -> None:
    if not isinstance(record, Mapping):
        raise TypeError("target record must be a mapping")
    required = {
        "target_record_schema_version",
        "kind",
        "target_id",
        "target_sha256",
        "sample_id",
        "dataset_id",
        "dataset_name",
        "suite",
        "episode_index",
        "episode_id",
        "frame_index",
        "task_index",
        "task_id",
        "task_id_source",
        "task",
        "source_index",
        "selection_order",
        "selection_bin",
        "valid_length",
        "target_action_shape",
        "input_hashes",
        "current_proprio",
        "source_metadata",
        "checkpoint_sha256",
        "dataset_stats_sha256",
        "vae_sha256",
        "source_pilot_record_sha256",
        "source_binding",
        "target_base_seeds",
        "utility_by_base_seed",
        "utility_mean",
        "utility_median",
        "utility_sample_std",
        "utility_sem",
        "t95_degrees_of_freedom",
        "t95_critical",
        "t95_ci_low",
        "t95_ci_high",
        "deadband_epsilon",
        "direction",
        "positive_seed_count",
        "negative_seed_count",
        "deadband_seed_count",
        "direction_seed_count",
        "sign_agreement",
        "min_sign_agreement",
        "mean_outside_deadband",
        "enough_same_direction_seeds",
        "ci_clears_deadband",
        "high_confidence",
        "uncertain",
    }
    missing = required - set(record)
    if missing:
        raise ValueError(f"target record is missing fields: {sorted(missing)}")
    if int(record["target_record_schema_version"]) != TARGET_RECORD_SCHEMA_VERSION:
        raise ValueError("unsupported target_record_schema_version")
    if record["kind"] != TARGET_BUNDLE_KIND:
        raise ValueError("target record kind is invalid")
    identity = parse_sample_identity(record)
    if record["sample_id"] != identity.sample_id:
        raise ValueError("target sample_id does not match canonical identity")
    if record["target_id"] != target_id(identity.sample_id):
        raise ValueError("target_id does not match sample identity")
    actual_target_sha = _require_sha256(record["target_sha256"], field="target_sha256")
    if actual_target_sha != target_record_sha256(record):
        raise ValueError("target_sha256 does not match target payload")
    if int(record["episode_id"]) != identity.episode_index:
        raise ValueError("target episode_id must equal episode_index")
    if int(record["task_id"]) != identity.task_index:
        raise ValueError("target task_id must equal task_index")
    if record["task_id_source"] != "lerobot_task_index":
        raise ValueError("target task_id_source is invalid")
    source = record["source_metadata"]
    if not isinstance(source, Mapping):
        raise ValueError("target source_metadata must be a mapping")
    source_index = _require_int(record["source_index"], field="source_index")
    if source_index != int(source.get("requested_sample_idx", -1)):
        raise ValueError("target source_index differs from requested_sample_idx")
    if source_index != int(source.get("source_sample_idx", -1)):
        raise ValueError("target source_index differs from source_sample_idx")
    input_hashes = record["input_hashes"]
    if not isinstance(input_hashes, Mapping) or "combined" not in input_hashes:
        raise ValueError("target input_hashes must contain combined")
    for name, value in input_hashes.items():
        _require_sha256(value, field=f"input_hashes.{name}")
    if input_hashes["combined"] != record["pilot_input_combined_sha256"]:
        raise ValueError("target input hashes differ from pilot input hash")
    if int(record["valid_length"]) != int(record["pilot_valid_length"]):
        raise ValueError("target valid_length differs from pilot_valid_length")
    for field in (
        "checkpoint_sha256",
        "dataset_stats_sha256",
        "vae_sha256",
        "source_pilot_record_sha256",
    ):
        _require_sha256(record[field], field=field)

    source_binding = record["source_binding"]
    if not isinstance(source_binding, Mapping):
        raise ValueError("target source_binding must be a mapping")
    if source_binding.get("kind") != SOURCE_BUNDLE_KIND:
        raise ValueError("target source_binding kind is invalid")
    expected_bindings = (
        ("manifest_sha256", expected_source_manifest_sha256),
        ("records_sha256", expected_source_records_sha256),
        ("manifest_compatibility_fingerprint", expected_source_manifest_fingerprint),
        ("selection_plan_sha256", expected_source_selection_plan_sha256),
        ("input_plan_sha256", expected_source_input_plan_sha256),
    )
    for field, expected in expected_bindings:
        value = _require_sha256(source_binding.get(field), field=f"source_binding.{field}")
        if expected is not None and value != expected:
            raise ValueError(f"source_binding.{field} mismatch")
    if list(source_binding.get("base_seeds", [])) != list(TARGET_BASE_SEEDS):
        raise ValueError("target source_binding base_seeds are invalid")
    if list(record["target_base_seeds"]) != list(TARGET_BASE_SEEDS):
        raise ValueError("target_base_seeds are invalid")
    for field in ("checkpoint_sha256", "dataset_stats_sha256", "vae_sha256"):
        if source_binding.get(field) != record[field]:
            raise ValueError(f"target {field} differs from source_binding")

    utility_rows = record["utility_by_base_seed"]
    if not isinstance(utility_rows, list) or len(utility_rows) != len(TARGET_BASE_SEEDS):
        raise ValueError("utility_by_base_seed must contain exactly five rows")
    for index, (base_seed, row) in enumerate(zip(TARGET_BASE_SEEDS, utility_rows)):
        if not isinstance(row, Mapping):
            raise ValueError("utility_by_base_seed members must be mappings")
        if int(row.get("base_seed", -1)) != base_seed or int(
            row.get("replicate_index", -1)
        ) != index:
            raise ValueError("utility_by_base_seed ordering is invalid")
        _require_int(row.get("replicate_seed"), field="replicate_seed")
        e0 = _require_finite(row.get("e0"), field="e0", minimum=0.0)
        efull = _require_finite(row.get("efull"), field="efull", minimum=0.0)
        utility = _require_finite(row.get("utility"), field="utility")
        if not math.isclose(utility, e0 - efull, rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError("target utility_by_base_seed violates U=E0-Efull")

    deadband = _require_finite(record["deadband_epsilon"], field="deadband_epsilon")
    agreement = _require_finite(record["min_sign_agreement"], field="min_sign_agreement")
    if deadband != DEFAULT_DEADBAND_EPSILON:
        raise ValueError("Target V2 deadband policy is not the frozen value")
    if agreement != DEFAULT_MIN_SIGN_AGREEMENT:
        raise ValueError("Target V2 sign-agreement policy is not the frozen value")
    if expected_deadband_epsilon is not None and deadband != expected_deadband_epsilon:
        raise ValueError("target deadband_epsilon mismatch")
    if expected_min_sign_agreement is not None and agreement != expected_min_sign_agreement:
        raise ValueError("target min_sign_agreement mismatch")
    expected_stats = _target_stats_from_record(record)
    for field, expected in expected_stats.items():
        actual = record[field]
        if isinstance(expected, bool):
            if _require_bool(actual, field=field) is not expected:
                raise ValueError(f"target {field} is inconsistent with raw utilities")
        elif isinstance(expected, str):
            if actual != expected:
                raise ValueError(f"target {field} is inconsistent with raw utilities")
        elif not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(f"target {field} is inconsistent with raw utilities")
    if int(record["t95_degrees_of_freedom"]) != 4:
        raise ValueError("Target V2 t interval must use four degrees of freedom")


def _target_selection_projection(targets: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "selection_order": int(record["selection_order"]),
            "source_index": int(record["source_index"]),
            "sample_id": str(record["sample_id"]),
            "target_id": str(record["target_id"]),
            "target_sha256": str(record["target_sha256"]),
            "input_combined_sha256": str(record["input_hashes"]["combined"]),
            "source_pilot_record_sha256": str(record["source_pilot_record_sha256"]),
        }
        for record in targets
    ]


def build_target_bundle(
    source: VerifiedSourceBundle,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build a deterministic target list and self-authenticating manifest."""

    if not isinstance(source, VerifiedSourceBundle):
        raise TypeError("source must be a VerifiedSourceBundle")
    targets: list[dict[str, Any]] = []
    for state in source.ordered_states:
        source_index = int(state["source_index"])
        rows = [source.record_index[(source_index, index)] for index in range(5)]
        targets.append(
            compute_target_record(rows, source_binding=source.source_binding)
        )
    target_ids = [str(record["target_id"]) for record in targets]
    target_hashes = [str(record["target_sha256"]) for record in targets]
    target_records_sha = _sha256_jsonl(targets)
    target_selection = _target_selection_projection(targets)
    target_selection_sha = sha256_json(target_selection)
    compatibility = {
        "schema_version": TARGET_BUNDLE_SCHEMA_VERSION,
        "kind": TARGET_BUNDLE_KIND,
        "target_record_schema_version": TARGET_RECORD_SCHEMA_VERSION,
        "source_manifest_sha256": source.manifest_sha256,
        "source_records_sha256": source.records_sha256,
        "source_manifest_compatibility_fingerprint": source.manifest[
            "compatibility_fingerprint"
        ],
        "source_selection_plan_sha256": source.selection_plan_sha256,
        "source_input_plan_sha256": source.input_plan_sha256,
        "source_num_states": source.num_states,
        "target_base_seeds": list(TARGET_BASE_SEEDS),
        "num_states": len(targets),
        "deadband_epsilon": DEFAULT_DEADBAND_EPSILON,
        "min_sign_agreement": DEFAULT_MIN_SIGN_AGREEMENT,
        "t_interval": {
            "confidence_level": 0.95,
            "two_sided": True,
            "degrees_of_freedom": 4,
            "critical_value": T95_DF4_CRITICAL,
        },
        "target_selection_sha256": target_selection_sha,
        "target_records_sha256": target_records_sha,
    }
    manifest: dict[str, Any] = {
        "schema_version": TARGET_BUNDLE_SCHEMA_VERSION,
        "kind": TARGET_BUNDLE_KIND,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "compatibility_fingerprint": sha256_json(compatibility),
        "compatibility": compatibility,
        "source": {
            **source.source_binding,
            "manifest_filename": source.manifest_path.name,
            "records_filename": source.records_path.name,
        },
        "selection": {
            "num_states": len(targets),
            "ordered_states": target_selection,
            "ordered_states_sha256": target_selection_sha,
            "source_selection_plan_sha256": source.selection_plan_sha256,
            "source_input_plan_sha256": source.input_plan_sha256,
        },
        "targets": {
            "filename": TARGETS_FILENAME,
            "count": len(targets),
            "ordered_target_ids": target_ids,
            "ordered_target_ids_sha256": sha256_json(target_ids),
            "ordered_target_sha256": target_hashes,
            "ordered_target_sha256_sha256": sha256_json(target_hashes),
            "canonical_records_sha256": target_records_sha,
        },
        "policy": {
            "utility_definition": "U = E0 - E10",
            "aggregation": "arithmetic mean over immutable base seeds 42--46",
            "dispersion": "sample standard deviation (ddof=1) and SEM",
            "confidence_interval": "two-sided Student-t 95% interval, df=4",
            "direction": "mean compared with +/-deadband_epsilon",
            "high_confidence": (
                "abs(mean)>epsilon AND >=4/5 epsilon-direction seeds AND "
                "95% t-CI wholly beyond the same +/-epsilon boundary"
            ),
            "uncertain": "logical negation of high_confidence",
            "candidate_training_weight": {
                "status": "not_defined_pending_independent_validation_go",
                "reason": "Target V2 preserves raw uncertainty and does not invent a weight",
            },
        },
        "summary": {
            "num_states": len(targets),
            "high_confidence_count": sum(bool(row["high_confidence"]) for row in targets),
            "uncertain_count": sum(bool(row["uncertain"]) for row in targets),
            "high_confidence_positive_count": sum(
                bool(row["high_confidence"]) and row["direction"] == "positive"
                for row in targets
            ),
            "high_confidence_negative_count": sum(
                bool(row["high_confidence"]) and row["direction"] == "negative"
                for row in targets
            ),
        },
    }
    validate_target_manifest(manifest)
    _validate_targets_against_manifest(manifest, targets)
    return manifest, targets


def validate_target_manifest(manifest: Mapping[str, Any]) -> None:
    if not isinstance(manifest, Mapping):
        raise TypeError("target manifest must be a mapping")
    if manifest.get("kind") != TARGET_BUNDLE_KIND:
        raise ValueError(f"unexpected target manifest kind={manifest.get('kind')!r}")
    if int(manifest.get("schema_version", -1)) != TARGET_BUNDLE_SCHEMA_VERSION:
        raise ValueError("unsupported target bundle schema version")
    compatibility = manifest.get("compatibility")
    if not isinstance(compatibility, Mapping):
        raise ValueError("target compatibility must be a mapping")
    fingerprint = _require_sha256(
        manifest.get("compatibility_fingerprint"), field="compatibility_fingerprint"
    )
    if fingerprint != sha256_json(compatibility):
        raise ValueError("target compatibility fingerprint is invalid")
    if compatibility.get("kind") != TARGET_BUNDLE_KIND:
        raise ValueError("target compatibility kind is invalid")
    if int(compatibility.get("schema_version", -1)) != TARGET_BUNDLE_SCHEMA_VERSION:
        raise ValueError("target compatibility schema version is invalid")
    if int(compatibility.get("target_record_schema_version", -1)) != TARGET_RECORD_SCHEMA_VERSION:
        raise ValueError("target compatibility record schema is invalid")
    for field in (
        "source_manifest_sha256",
        "source_records_sha256",
        "source_manifest_compatibility_fingerprint",
        "source_selection_plan_sha256",
        "source_input_plan_sha256",
        "target_selection_sha256",
        "target_records_sha256",
    ):
        _require_sha256(compatibility.get(field), field=f"compatibility.{field}")
    if list(compatibility.get("target_base_seeds", [])) != list(TARGET_BASE_SEEDS):
        raise ValueError("target compatibility base seeds are invalid")
    if float(compatibility.get("deadband_epsilon", math.nan)) != DEFAULT_DEADBAND_EPSILON:
        raise ValueError("target compatibility deadband is invalid")
    if float(compatibility.get("min_sign_agreement", math.nan)) != DEFAULT_MIN_SIGN_AGREEMENT:
        raise ValueError("target compatibility min sign agreement is invalid")
    interval = compatibility.get("t_interval")
    expected_interval = {
        "confidence_level": 0.95,
        "two_sided": True,
        "degrees_of_freedom": 4,
        "critical_value": T95_DF4_CRITICAL,
    }
    if not isinstance(interval, Mapping) or not _same(interval, expected_interval):
        raise ValueError("target compatibility t interval is invalid")

    selection = manifest.get("selection")
    targets = manifest.get("targets")
    source = manifest.get("source")
    if not isinstance(selection, Mapping) or not isinstance(targets, Mapping):
        raise ValueError("target manifest selection/targets must be mappings")
    if not isinstance(source, Mapping):
        raise ValueError("target manifest source must be a mapping")
    states = selection.get("ordered_states")
    if not isinstance(states, list):
        raise ValueError("target manifest ordered_states must be a list")
    state_sha = sha256_json(states)
    if selection.get("ordered_states_sha256") != state_sha:
        raise ValueError("target ordered_states SHA-256 is invalid")
    if compatibility.get("target_selection_sha256") != state_sha:
        raise ValueError("target compatibility is not bound to target selection")
    count = _require_int(targets.get("count"), field="targets.count", minimum=1)
    if count != len(states) or int(selection.get("num_states", -1)) != count:
        raise ValueError("target manifest state counts are inconsistent")
    if int(compatibility.get("num_states", -1)) != count:
        raise ValueError("target compatibility target count is invalid")
    if int(compatibility.get("source_num_states", -1)) != count:
        raise ValueError("target compatibility source count is invalid")
    if targets.get("filename") != TARGETS_FILENAME:
        raise ValueError("target manifest filename is invalid")
    ids = targets.get("ordered_target_ids")
    hashes = targets.get("ordered_target_sha256")
    if not isinstance(ids, list) or not isinstance(hashes, list):
        raise ValueError("target manifest ordered ids/hashes must be lists")
    if len(ids) != count or len(hashes) != count:
        raise ValueError("target manifest ordered ids/hashes counts are invalid")
    if targets.get("ordered_target_ids_sha256") != sha256_json(ids):
        raise ValueError("target ordered_target_ids digest is invalid")
    if targets.get("ordered_target_sha256_sha256") != sha256_json(hashes):
        raise ValueError("target ordered target-hash digest is invalid")
    target_records_sha = _require_sha256(
        targets.get("canonical_records_sha256"),
        field="targets.canonical_records_sha256",
    )
    if target_records_sha != compatibility.get("target_records_sha256"):
        raise ValueError("target records digest is not bound to compatibility")
    source_map = {
        "manifest_sha256": "source_manifest_sha256",
        "records_sha256": "source_records_sha256",
        "manifest_compatibility_fingerprint": "source_manifest_compatibility_fingerprint",
        "selection_plan_sha256": "source_selection_plan_sha256",
        "input_plan_sha256": "source_input_plan_sha256",
    }
    for source_field, compatibility_field in source_map.items():
        if source.get(source_field) != compatibility.get(compatibility_field):
            raise ValueError(f"target source {source_field} is not bound to compatibility")
    if selection.get("source_selection_plan_sha256") != compatibility.get(
        "source_selection_plan_sha256"
    ):
        raise ValueError("target selection is not bound to source selection")
    if selection.get("source_input_plan_sha256") != compatibility.get(
        "source_input_plan_sha256"
    ):
        raise ValueError("target selection is not bound to source input plan")


def _validate_targets_against_manifest(
    manifest: Mapping[str, Any], targets: Sequence[Mapping[str, Any]]
) -> None:
    validate_target_manifest(manifest)
    compatibility = manifest["compatibility"]
    expected_count = int(manifest["targets"]["count"])
    if len(targets) != expected_count:
        raise ValueError(f"target records contain {len(targets)} rows, expected {expected_count}")
    ids: list[str] = []
    hashes: list[str] = []
    source_indices: list[int] = []
    sample_ids: list[str] = []
    for order, record in enumerate(targets):
        validate_target_record(
            record,
            expected_source_manifest_sha256=compatibility["source_manifest_sha256"],
            expected_source_records_sha256=compatibility["source_records_sha256"],
            expected_source_manifest_fingerprint=compatibility[
                "source_manifest_compatibility_fingerprint"
            ],
            expected_source_selection_plan_sha256=compatibility[
                "source_selection_plan_sha256"
            ],
            expected_source_input_plan_sha256=compatibility[
                "source_input_plan_sha256"
            ],
            expected_deadband_epsilon=compatibility["deadband_epsilon"],
            expected_min_sign_agreement=compatibility["min_sign_agreement"],
        )
        if int(record["selection_order"]) != order:
            raise ValueError("target selection_order must be exactly file order 0..N-1")
        ids.append(str(record["target_id"]))
        hashes.append(str(record["target_sha256"]))
        source_indices.append(int(record["source_index"]))
        sample_ids.append(str(record["sample_id"]))
    if len(set(ids)) != len(ids) or len(set(source_indices)) != len(source_indices):
        raise ValueError("target records contain duplicate target_id or source_index")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("target records contain duplicate sample_id")
    target_section = manifest["targets"]
    if ids != list(target_section["ordered_target_ids"]):
        raise ValueError("target record IDs differ from immutable manifest order")
    if hashes != list(target_section["ordered_target_sha256"]):
        raise ValueError("target record hashes differ from immutable manifest order")
    records_sha = _sha256_jsonl(targets)
    if records_sha != target_section["canonical_records_sha256"]:
        raise ValueError("target records bytes differ from manifest digest")
    projection = _target_selection_projection(targets)
    if not _same(projection, manifest["selection"]["ordered_states"]):
        raise ValueError("target records differ from manifest target selection")
    input_plan_sha = sha256_json(_source_input_projection(targets))
    if input_plan_sha != compatibility["source_input_plan_sha256"]:
        raise ValueError("target records do not reproduce the bound source input plan")
    summary = manifest.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("target manifest summary must be a mapping")
    expected_summary = {
        "num_states": len(targets),
        "high_confidence_count": sum(bool(row["high_confidence"]) for row in targets),
        "uncertain_count": sum(bool(row["uncertain"]) for row in targets),
        "high_confidence_positive_count": sum(
            bool(row["high_confidence"]) and row["direction"] == "positive"
            for row in targets
        ),
        "high_confidence_negative_count": sum(
            bool(row["high_confidence"]) and row["direction"] == "negative"
            for row in targets
        ),
    }
    if not _same(summary, expected_summary):
        raise ValueError("target manifest summary does not match target records")


def write_target_bundle(
    output_dir: str | Path,
    manifest: Mapping[str, Any],
    targets: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path]:
    """Publish both files with one atomic directory rename; never overwrite."""

    _validate_targets_against_manifest(manifest, targets)
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"immutable target output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        manifest_path = staging / TARGET_MANIFEST_FILENAME
        targets_path = staging / TARGETS_FILENAME
        with targets_path.open("xb") as stream:
            stream.write(_serialize_jsonl(targets))
            stream.flush()
            os.fsync(stream.fileno())
        with manifest_path.open("x", encoding="utf-8") as stream:
            json.dump(dict(manifest), stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        directory_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.replace(staging, output)
        parent_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return output / TARGET_MANIFEST_FILENAME, output / TARGETS_FILENAME


def load_target_bundle(
    target_dir: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
    expected_targets_sha256: str | None = None,
    expected_num_states: int = DEFAULT_NUM_STATES,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load and revalidate a Target V2 bundle.

    Production consumers should pass both external file digests.  Supplying
    neither remains useful for immediately re-reading a just-built bundle;
    supplying only one is rejected because that would create partial trust.
    """

    if (expected_manifest_sha256 is None) != (expected_targets_sha256 is None):
        raise ValueError("expected manifest and targets SHA-256 must be supplied together")
    root = Path(target_dir).resolve()
    manifest_path = root / TARGET_MANIFEST_FILENAME
    targets_path = root / TARGETS_FILENAME
    if expected_manifest_sha256 is not None:
        expected_manifest = _require_sha256(
            expected_manifest_sha256, field="expected_manifest_sha256"
        )
        expected_targets = _require_sha256(
            expected_targets_sha256, field="expected_targets_sha256"
        )
        if sha256_file(manifest_path) != expected_manifest:
            raise ValueError("target manifest SHA-256 differs from expected")
        if sha256_file(targets_path) != expected_targets:
            raise ValueError("target records SHA-256 differs from expected")
    manifest = _load_json(manifest_path, label="Target V2 manifest")
    targets = _load_jsonl(targets_path, label="Target V2 records")
    if len(targets) != _require_int(expected_num_states, field="expected_num_states", minimum=1):
        raise ValueError(
            f"Target V2 contains {len(targets)} states, expected {expected_num_states}"
        )
    _validate_targets_against_manifest(manifest, targets)
    if sha256_file(targets_path) != manifest["targets"]["canonical_records_sha256"]:
        raise ValueError("targets.jsonl is not in the canonical bytes bound by manifest")
    return manifest, targets


def validation_record_id(
    target_id_value: str, validation_replicate_index: int, validation_base_seed: int
) -> str:
    if not isinstance(target_id_value, str) or not target_id_value.strip():
        raise ValueError("target_id must be a non-empty string")
    index = _require_int(
        validation_replicate_index, field="validation_replicate_index"
    )
    seed = _require_int(validation_base_seed, field="validation_base_seed")
    return f"{target_id_value}/validation_{index:02d}_base_seed_{seed}"


def validation_record_sha256(record: Mapping[str, Any]) -> str:
    """Digest one completed validation row, excluding the digest itself."""

    if not isinstance(record, Mapping):
        raise TypeError("validation record must be a mapping")
    return sha256_json(
        {
            key: value
            for key, value in record.items()
            if key != "validation_record_sha256"
        }
    )


def _validate_validation_base_seeds(base_seeds: Sequence[int]) -> tuple[int, ...]:
    seeds = tuple(_require_int(value, field="validation base seed") for value in base_seeds)
    if seeds != VALIDATION_BASE_SEEDS:
        raise ValueError(
            f"validation base seeds must be exactly {list(VALIDATION_BASE_SEEDS)}, got {list(seeds)}"
        )
    return seeds


def build_validation_plan(
    target_manifest: Mapping[str, Any],
    targets: Sequence[Mapping[str, Any]],
    *,
    base_seeds: Sequence[int] = VALIDATION_BASE_SEEDS,
) -> list[dict[str, Any]]:
    """Build the immutable state-outer seed-47--50 inference plan."""

    seeds = _validate_validation_base_seeds(base_seeds)
    _validate_targets_against_manifest(target_manifest, targets)
    result: list[dict[str, Any]] = []
    for target in targets:
        identity = parse_sample_identity(target)
        for validation_index, base_seed in enumerate(seeds):
            result.append(
                {
                    "validation_id": validation_record_id(
                        str(target["target_id"]), validation_index, base_seed
                    ),
                    "validation_index": validation_index,
                    "validation_replicate_index": validation_index,
                    "global_seed_index": len(TARGET_BASE_SEEDS) + validation_index,
                    "validation_base_seed": base_seed,
                    "seed": stable_sample_seed(base_seed, identity),
                    "target_id": str(target["target_id"]),
                    "target_sha256": str(target["target_sha256"]),
                    "selection_order": int(target["selection_order"]),
                    "source_index": int(target["source_index"]),
                    "sample_id": str(target["sample_id"]),
                    "input_combined_sha256": str(target["input_hashes"]["combined"]),
                }
            )
    return result


def augment_validation_record(
    core_record: Mapping[str, Any] | Any,
    *,
    target_record: Mapping[str, Any],
    validation_index: int,
    validation_base_seed: int,
    validation_manifest_compatibility_fingerprint: str,
    target_manifest_sha256: str,
    target_targets_sha256: str,
    target_manifest_compatibility_fingerprint: str,
    collection_git_sha: str | None = None,
) -> dict[str, Any]:
    """Bind one paired seed-47--50 inference row to its immutable target."""

    validate_target_record(target_record)
    if hasattr(core_record, "to_dict"):
        core_record = core_record.to_dict()
    if not isinstance(core_record, Mapping):
        raise TypeError("core_record must be a mapping or provide to_dict()")
    # Validate before copying provenance so a mismatched inference row cannot
    # be made to look correct by augmentation.
    stability._validate_core_utility_record(core_record, full_steps=10)
    direct_checks = (
        "sample_id",
        "dataset_id",
        "dataset_name",
        "suite",
        "episode_index",
        "episode_id",
        "frame_index",
        "task_index",
        "task_id",
        "task_id_source",
        "task",
        "valid_length",
        "target_action_shape",
        "input_hashes",
    )
    for field in direct_checks:
        if not _same(core_record.get(field), target_record.get(field)):
            raise ValueError(f"validation core/target mismatch for {field}")
    index = _require_int(validation_index, field="validation_index")
    if index >= len(VALIDATION_BASE_SEEDS):
        raise ValueError("validation_index is outside seed-47--50 grid")
    base_seed = _require_int(validation_base_seed, field="validation_base_seed")
    if base_seed != VALIDATION_BASE_SEEDS[index]:
        raise ValueError("validation_index/base-seed mapping is invalid")
    identity = parse_sample_identity(target_record)
    expected_seed = stable_sample_seed(base_seed, identity)
    if int(core_record["seed"]) != expected_seed:
        raise ValueError("validation core seed is not the stable sample seed")
    if collection_git_sha is None:
        collection_git_sha = str(core_record.get("git_sha", ""))
    if (
        not isinstance(collection_git_sha, str)
        or len(collection_git_sha) != 40
        or any(character not in "0123456789abcdef" for character in collection_git_sha)
    ):
        raise ValueError("collection_git_sha must be a lowercase 40-character hex digest")

    result = dict(core_record)
    for field in (
        "source_metadata",
        "current_proprio",
        "checkpoint_sha256",
        "dataset_stats_sha256",
        "vae_sha256",
    ):
        result[field] = target_record[field]
    source_binding = target_record["source_binding"]
    result.update(
        {
            "validation_record_schema_version": VALIDATION_RECORD_SCHEMA_VERSION,
            "validation_kind": VALIDATION_RECORD_KIND,
            "validation_id": validation_record_id(
                str(target_record["target_id"]), index, base_seed
            ),
            "validation_index": index,
            "validation_replicate_index": index,
            "global_seed_index": len(TARGET_BASE_SEEDS) + index,
            "validation_base_seed": base_seed,
            "validation_seed": expected_seed,
            "inference_origin": "independent_validation",
            "reused_from_target": False,
            "target_id": target_record["target_id"],
            "target_sha256": target_record["target_sha256"],
            "target_manifest_sha256": _require_sha256(
                target_manifest_sha256, field="target_manifest_sha256"
            ),
            "target_targets_sha256": _require_sha256(
                target_targets_sha256, field="target_targets_sha256"
            ),
            "target_manifest_compatibility_fingerprint": _require_sha256(
                target_manifest_compatibility_fingerprint,
                field="target_manifest_compatibility_fingerprint",
            ),
            "validation_manifest_compatibility_fingerprint": _require_sha256(
                validation_manifest_compatibility_fingerprint,
                field="validation_manifest_compatibility_fingerprint",
            ),
            "selection_order": int(target_record["selection_order"]),
            "source_index": int(target_record["source_index"]),
            "target_input_combined_sha256": target_record["input_hashes"]["combined"],
            "source_manifest_sha256": source_binding["manifest_sha256"],
            "source_records_sha256": source_binding["records_sha256"],
            "source_selection_plan_sha256": source_binding["selection_plan_sha256"],
            "source_input_plan_sha256": source_binding["input_plan_sha256"],
            "source_pilot_record_sha256": target_record[
                "source_pilot_record_sha256"
            ],
            "collection_git_sha": collection_git_sha,
        }
    )
    result["validation_record_sha256"] = validation_record_sha256(result)
    validate_validation_record(result)
    return result


def validate_validation_record(
    record: Mapping[str, Any],
    *,
    expected_base_seeds: Sequence[int] = VALIDATION_BASE_SEEDS,
    expected_validation_manifest_fingerprint: str | None = None,
    expected_target_manifest_sha256: str | None = None,
    expected_target_targets_sha256: str | None = None,
    expected_target_manifest_fingerprint: str | None = None,
    expected_checkpoint_sha256: str | None = None,
    expected_dataset_stats_sha256: str | None = None,
    expected_vae_sha256: str | None = None,
) -> None:
    if not isinstance(record, Mapping):
        raise TypeError("validation record must be a mapping")
    stability._validate_core_utility_record(record, full_steps=10)
    required = {
        "validation_record_schema_version",
        "validation_kind",
        "validation_id",
        "validation_index",
        "validation_replicate_index",
        "global_seed_index",
        "validation_base_seed",
        "validation_seed",
        "inference_origin",
        "reused_from_target",
        "target_id",
        "target_sha256",
        "target_manifest_sha256",
        "target_targets_sha256",
        "target_manifest_compatibility_fingerprint",
        "validation_manifest_compatibility_fingerprint",
        "selection_order",
        "source_index",
        "target_input_combined_sha256",
        "source_manifest_sha256",
        "source_records_sha256",
        "source_selection_plan_sha256",
        "source_input_plan_sha256",
        "source_pilot_record_sha256",
        "source_metadata",
        "current_proprio",
        "checkpoint_sha256",
        "dataset_stats_sha256",
        "vae_sha256",
        "collection_git_sha",
        "validation_record_sha256",
    }
    missing = required - set(record)
    if missing:
        raise ValueError(f"validation record is missing fields: {sorted(missing)}")
    if int(record["validation_record_schema_version"]) != VALIDATION_RECORD_SCHEMA_VERSION:
        raise ValueError("unsupported validation_record_schema_version")
    if record["validation_kind"] != VALIDATION_RECORD_KIND:
        raise ValueError("validation record kind is invalid")
    seeds = _validate_validation_base_seeds(expected_base_seeds)
    index = _require_int(record["validation_replicate_index"], field="validation_replicate_index")
    if index >= len(seeds):
        raise ValueError("validation_replicate_index is outside the seed grid")
    if int(record["validation_index"]) != index:
        raise ValueError("validation_index must equal validation_replicate_index")
    if int(record["global_seed_index"]) != len(TARGET_BASE_SEEDS) + index:
        raise ValueError("global_seed_index must be 5..8")
    base_seed = int(record["validation_base_seed"])
    if base_seed != seeds[index]:
        raise ValueError("validation replicate/base-seed mapping is invalid")
    identity = parse_sample_identity(record)
    expected_seed = stable_sample_seed(base_seed, identity)
    if int(record["seed"]) != expected_seed or int(record["validation_seed"]) != expected_seed:
        raise ValueError("validation seed is not the stable sample seed")
    expected_id = validation_record_id(str(record["target_id"]), index, base_seed)
    if record["validation_id"] != expected_id:
        raise ValueError("validation_id does not match target/index/base seed")
    if record["inference_origin"] != "independent_validation":
        raise ValueError("validation inference_origin is invalid")
    if _require_bool(record["reused_from_target"], field="reused_from_target"):
        raise ValueError("validation rows may never be reused from target seeds")
    for field in (
        "target_sha256",
        "target_manifest_sha256",
        "target_targets_sha256",
        "target_manifest_compatibility_fingerprint",
        "validation_manifest_compatibility_fingerprint",
        "target_input_combined_sha256",
        "source_manifest_sha256",
        "source_records_sha256",
        "source_selection_plan_sha256",
        "source_input_plan_sha256",
        "source_pilot_record_sha256",
        "checkpoint_sha256",
        "dataset_stats_sha256",
        "vae_sha256",
    ):
        _require_sha256(record[field], field=field)
    if record["input_hashes"]["combined"] != record["target_input_combined_sha256"]:
        raise ValueError("validation state input hash differs from target")
    source_metadata = record["source_metadata"]
    if not isinstance(source_metadata, Mapping):
        raise ValueError("validation source_metadata must be a mapping")
    if int(record["source_index"]) != int(source_metadata.get("requested_sample_idx", -1)):
        raise ValueError("validation source_index differs from requested_sample_idx")
    if int(record["source_index"]) != int(source_metadata.get("source_sample_idx", -1)):
        raise ValueError("validation source_index differs from source_sample_idx")
    for field, expected in (
        ("validation_manifest_compatibility_fingerprint", expected_validation_manifest_fingerprint),
        ("target_manifest_sha256", expected_target_manifest_sha256),
        ("target_targets_sha256", expected_target_targets_sha256),
        ("target_manifest_compatibility_fingerprint", expected_target_manifest_fingerprint),
        ("checkpoint_sha256", expected_checkpoint_sha256),
        ("dataset_stats_sha256", expected_dataset_stats_sha256),
        ("vae_sha256", expected_vae_sha256),
    ):
        if expected is not None and record[field] != expected:
            raise ValueError(f"validation {field} mismatch")
    collection_git = str(record["collection_git_sha"])
    if len(collection_git) != 40 or any(
        character not in "0123456789abcdef" for character in collection_git
    ):
        raise ValueError("collection_git_sha must be a lowercase 40-character hex digest")
    row_digest = _require_sha256(
        record["validation_record_sha256"], field="validation_record_sha256"
    )
    if row_digest != validation_record_sha256(record):
        raise ValueError("validation_record_sha256 does not match validation payload")


def load_validation_record_index(
    records_path: str | Path,
    *,
    expected_base_seeds: Sequence[int] = VALIDATION_BASE_SEEDS,
    expected_validation_manifest_fingerprint: str | None = None,
    expected_target_manifest_sha256: str | None = None,
    expected_target_targets_sha256: str | None = None,
    expected_target_manifest_fingerprint: str | None = None,
    expected_checkpoint_sha256: str | None = None,
    expected_dataset_stats_sha256: str | None = None,
    expected_vae_sha256: str | None = None,
) -> dict[tuple[int, int], dict[str, Any]]:
    path = Path(records_path)
    if not path.exists():
        return {}
    index: dict[tuple[int, int], dict[str, Any]] = {}
    validation_ids: set[str] = set()
    for line_number, record in enumerate(_load_jsonl(path, label="validation records"), start=1):
        try:
            validate_validation_record(
                record,
                expected_base_seeds=expected_base_seeds,
                expected_validation_manifest_fingerprint=expected_validation_manifest_fingerprint,
                expected_target_manifest_sha256=expected_target_manifest_sha256,
                expected_target_targets_sha256=expected_target_targets_sha256,
                expected_target_manifest_fingerprint=expected_target_manifest_fingerprint,
                expected_checkpoint_sha256=expected_checkpoint_sha256,
                expected_dataset_stats_sha256=expected_dataset_stats_sha256,
                expected_vae_sha256=expected_vae_sha256,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid validation record at {path}:{line_number}: {exc}") from exc
        key = (int(record["source_index"]), int(record["validation_replicate_index"]))
        if key in index:
            raise ValueError(f"duplicate validation resume key {key}")
        validation_id_value = str(record["validation_id"])
        if validation_id_value in validation_ids:
            raise ValueError(f"duplicate validation_id {validation_id_value!r}")
        index[key] = record
        validation_ids.add(validation_id_value)
    return index


def _validate_validation_row_against_target(
    row: Mapping[str, Any], target: Mapping[str, Any]
) -> None:
    direct_fields = (
        "sample_id",
        "dataset_id",
        "dataset_name",
        "suite",
        "episode_index",
        "episode_id",
        "frame_index",
        "task_index",
        "task_id",
        "task_id_source",
        "task",
        "valid_length",
        "target_action_shape",
        "input_hashes",
        "current_proprio",
        "source_metadata",
        "checkpoint_sha256",
        "dataset_stats_sha256",
        "vae_sha256",
        "selection_order",
        "source_index",
        "target_id",
        "target_sha256",
        "source_pilot_record_sha256",
    )
    for field in direct_fields:
        if not _same(row.get(field), target.get(field)):
            raise ValueError(
                f"validation source {target.get('source_index')} target rebind mismatch for {field}"
            )
    expected = {
        "target_input_combined_sha256": target["input_hashes"]["combined"],
        "source_manifest_sha256": target["source_binding"]["manifest_sha256"],
        "source_records_sha256": target["source_binding"]["records_sha256"],
        "source_selection_plan_sha256": target["source_binding"]["selection_plan_sha256"],
        "source_input_plan_sha256": target["source_binding"]["input_plan_sha256"],
    }
    for field, value in expected.items():
        if not _same(row.get(field), value):
            raise ValueError(
                f"validation source {target.get('source_index')} target rebind mismatch for {field}"
            )


def validate_validation_grid(
    record_index: Mapping[tuple[int, int], Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    *,
    base_seeds: Sequence[int] = VALIDATION_BASE_SEEDS,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    """Validate resume/full-grid coverage, rebinding every present cell first."""

    seeds = _validate_validation_base_seeds(base_seeds)
    targets_by_source: dict[int, Mapping[str, Any]] = {}
    orders: set[int] = set()
    for target in targets:
        validate_target_record(target)
        source_index = int(target["source_index"])
        order = int(target["selection_order"])
        if source_index in targets_by_source or order in orders:
            raise ValueError("validation target plan contains duplicate source/order")
        targets_by_source[source_index] = target
        orders.add(order)
    if orders != set(range(len(targets))):
        raise ValueError("validation target order must be exactly 0..N-1")
    expected = {
        (source_index, validation_index)
        for source_index in targets_by_source
        for validation_index in range(len(seeds))
    }
    actual = set(record_index)
    outside = sorted(actual - expected)
    missing = sorted(expected - actual)
    if outside:
        raise ValueError(f"validation records contain keys outside immutable plan: {outside[:10]}")
    rows_by_source: dict[int, list[Mapping[str, Any]]] = {}
    for (source_index, validation_index), row in record_index.items():
        if int(row.get("source_index", -1)) != source_index:
            raise ValueError("validation resume key differs from row source_index")
        if int(row.get("validation_replicate_index", -1)) != validation_index:
            raise ValueError("validation resume key differs from row replicate index")
        _validate_validation_row_against_target(row, targets_by_source[source_index])
        rows_by_source.setdefault(source_index, []).append(row)
    if missing and not allow_incomplete:
        raise ValueError(f"validation grid is incomplete; missing keys: {missing[:10]}")
    cross_fields = (
        "sample_id",
        "input_hashes",
        "current_proprio",
        "source_metadata",
        "target_id",
        "target_sha256",
        "target_manifest_sha256",
        "target_targets_sha256",
        "target_manifest_compatibility_fingerprint",
        "validation_manifest_compatibility_fingerprint",
        "source_manifest_sha256",
        "source_records_sha256",
        "source_selection_plan_sha256",
        "source_input_plan_sha256",
        "checkpoint_sha256",
        "dataset_stats_sha256",
        "vae_sha256",
        "collection_git_sha",
    )
    for source_index, rows in rows_by_source.items():
        reference = rows[0]
        for row in rows[1:]:
            for field in cross_fields:
                if not _same(row.get(field), reference.get(field)):
                    raise ValueError(
                        f"validation source {source_index} has cross-seed mismatch for {field}"
                    )
    return {
        "expected_count": len(expected),
        "completed_count": len(actual),
        "missing_count": len(missing),
        "is_complete": not missing,
        "coverage_fraction": len(actual) / len(expected) if expected else 1.0,
        "missing_key_examples": [list(key) for key in missing[:10]],
    }


def _build_cli(args: argparse.Namespace) -> int:
    source_dir = Path(args.source_dir).resolve()
    source = load_verified_source_bundle(
        source_dir / TARGET_MANIFEST_FILENAME,
        source_dir / "records.jsonl",
        expected_manifest_sha256=args.expected_source_manifest_sha256,
        expected_records_sha256=args.expected_source_records_sha256,
        expected_selection_plan_sha256=args.expected_source_selection_plan_sha256,
        expected_num_states=args.expected_num_states,
    )
    manifest, targets = build_target_bundle(source)
    manifest_path, targets_path = write_target_bundle(args.output_dir, manifest, targets)
    # Read our own durable bytes before claiming success.
    load_target_bundle(
        args.output_dir,
        expected_manifest_sha256=sha256_file(manifest_path),
        expected_targets_sha256=sha256_file(targets_path),
        expected_num_states=args.expected_num_states,
    )
    print(
        canonical_json(
            {
                "status": "built_and_verified",
                "manifest": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "targets": str(targets_path),
                "targets_sha256": sha256_file(targets_path),
                **manifest["summary"],
            }
        )
    )
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="Build immutable Target V2 bundle")
    build.add_argument("--source-dir", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument(
        "--expected-source-manifest-sha256",
        default=OFFICIAL_SOURCE_MANIFEST_SHA256,
    )
    build.add_argument(
        "--expected-source-records-sha256",
        default=OFFICIAL_SOURCE_RECORDS_SHA256,
    )
    build.add_argument(
        "--expected-source-selection-plan-sha256",
        default=OFFICIAL_SOURCE_SELECTION_PLAN_SHA256,
    )
    build.add_argument("--expected-num-states", type=int, default=DEFAULT_NUM_STATES)
    build.set_defaults(handler=_build_cli)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return int(args.handler(args))


__all__ = [
    "DEFAULT_DEADBAND_EPSILON",
    "DEFAULT_MIN_SIGN_AGREEMENT",
    "DEFAULT_NUM_STATES",
    "OFFICIAL_SOURCE_MANIFEST_SHA256",
    "OFFICIAL_SOURCE_RECORDS_SHA256",
    "OFFICIAL_SOURCE_SELECTION_PLAN_SHA256",
    "SOURCE_BUNDLE_KIND",
    "T95_DF4_CRITICAL",
    "TARGET_BASE_SEEDS",
    "TARGET_BUNDLE_KIND",
    "TARGET_BUNDLE_SCHEMA_VERSION",
    "TARGET_MANIFEST_FILENAME",
    "TARGET_RECORD_SCHEMA_VERSION",
    "TARGETS_FILENAME",
    "VALIDATION_BASE_SEEDS",
    "VALIDATION_RECORD_KIND",
    "VALIDATION_RECORD_SCHEMA_VERSION",
    "VerifiedSourceBundle",
    "augment_validation_record",
    "build_arg_parser",
    "build_target_bundle",
    "build_validation_plan",
    "canonical_json",
    "compute_target_record",
    "load_target_bundle",
    "load_validation_record_index",
    "load_verified_source_bundle",
    "main",
    "sha256_file",
    "sha256_json",
    "target_id",
    "target_record_sha256",
    "validate_target_manifest",
    "validate_target_record",
    "validate_validation_grid",
    "validate_validation_record",
    "validation_record_sha256",
    "validation_record_id",
    "write_target_bundle",
]


if __name__ == "__main__":
    raise SystemExit(main())
