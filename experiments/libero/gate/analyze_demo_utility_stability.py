"""Audit and analyze the Phase-2.5 LIBERO multi-seed utility collection.

This module is intentionally fail-closed.  It accepts only the immutable
100-state x 5-replicate stability artifact, verifies that every row represents
the same paired N=0/N=10 experiment for its state, and only then computes label
stability diagnostics.  The resulting GO/CONDITIONAL/NO_GO recommendation is
permission to proceed to an *offline Tiny MLP experiment*, not evidence that a
learned router improves closed-loop success.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy import stats
from experiments.libero.gate import demo_utility_stability as stability_core

from experiments.libero.gate.demo_utility import (
    parse_sample_identity,
    stable_sample_seed,
)
from experiments.libero.gate.demo_utility_stability import (
    replicate_id as stability_replicate_id,
)


LOGGER = logging.getLogger(__name__)
ANALYSIS_SCHEMA_VERSION = 1
STABILITY_KIND = "libero_demo_utility_multiseed_stability"
EXPECTED_STATE_COUNT = 100
EXPECTED_BASE_SEEDS = (42, 43, 44, 45, 46)
EXPECTED_REPLICATE_COUNT = len(EXPECTED_BASE_SEEDS)
PRIMARY_EPSILON = 1e-4
DEADBAND_EPSILONS = (1e-5, PRIMARY_EPSILON, 1e-3)
BOOTSTRAP_SEED = 20260812
BOOTSTRAP_REPLICATES = 2000
STRATUM_PREVALENCE = {
    "SP": 0.152,
    "SN": 0.202,
    "MP": 0.262,
    "MN": 0.184,
    "NZ": 0.200,
}
STRATUM_ORDER = tuple(STRATUM_PREVALENCE)
_HEX64 = set("0123456789abcdef")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_json_constant(token: str) -> None:
    raise ValueError(f"Non-finite JSON constant {token!r} is forbidden")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Malformed or non-finite JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _assert_finite_tree(value: Any, *, path: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite numeric value at {path}")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_finite_tree(child, path=f"{path}.{key}")
        return
    if isinstance(value, Sequence):
        for index, child in enumerate(value):
            _assert_finite_tree(child, path=f"{path}[{index}]")
        return
    raise ValueError(f"Unsupported value type {type(value).__name__} at {path}")


def _required_int(mapping: Mapping[str, Any], key: str, *, minimum: int = 0) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{key} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def _required_string(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string, got {value!r}")
    return value


def _required_float(mapping: Mapping[str, Any], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be finite numeric, got bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be finite numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{key} must be finite, got {result!r}")
    return result


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX64 for character in value)
    )


def classify_pilot_stratum(utility: float) -> str:
    """Classify the original seed-42 label into the five audit strata."""

    utility = float(utility)
    if not math.isfinite(utility):
        raise ValueError("utility must be finite")
    if utility > 1e-3:
        return "SP"
    if utility < -1e-3:
        return "SN"
    if utility > 1e-4:
        return "MP"
    if utility < -1e-4:
        return "MN"
    return "NZ"


def deadband_sign(value: float, epsilon: float = PRIMARY_EPSILON) -> int:
    if epsilon < 0 or not math.isfinite(float(epsilon)):
        raise ValueError("epsilon must be finite and non-negative")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("value must be finite")
    if value > epsilon:
        return 1
    if value < -epsilon:
        return -1
    return 0


def _epsilon_label(epsilon: float) -> str:
    return f"eps_{epsilon:.0e}".replace("-", "m").replace("+", "p")


def _expected_stratum_sign(stratum: str) -> int:
    if stratum in ("SP", "MP"):
        return 1
    if stratum in ("SN", "MN"):
        return -1
    if stratum == "NZ":
        return 0
    raise ValueError(f"Unknown selection stratum {stratum!r}")


def _manifest_base_seeds(manifest: Mapping[str, Any]) -> tuple[int, ...]:
    replicates = manifest.get("replicates")
    if not isinstance(replicates, Mapping):
        raise ValueError("manifest.replicates must be a mapping")
    values = replicates.get("base_seeds")
    if not isinstance(values, list):
        raise ValueError("manifest.replicates.base_seeds must be a list")
    seeds = tuple(
        _required_int({"seed": value}, "seed") for value in values
    )
    if seeds != EXPECTED_BASE_SEEDS:
        raise ValueError(
            f"Expected replicate base seeds {EXPECTED_BASE_SEEDS}, got {seeds}"
        )
    count = replicates.get("count", len(seeds))
    if count != EXPECTED_REPLICATE_COUNT:
        raise ValueError(
            f"manifest.replicates.count must be {EXPECTED_REPLICATE_COUNT}, got {count!r}"
        )
    reuse_index = replicates.get("reuse_replicate_index", replicates.get("reuse_index", 0))
    if reuse_index != 0:
        raise ValueError(f"Seed-42 Pilot reuse must be replicate index 0, got {reuse_index!r}")
    return seeds


def validate_stability_manifest(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate immutable identity and return the ordered 100-state plan."""

    _assert_finite_tree(manifest, path="manifest")
    if manifest.get("kind") != STABILITY_KIND:
        raise ValueError(
            f"manifest.kind must be {STABILITY_KIND!r}, got {manifest.get('kind')!r}"
        )
    if manifest.get("schema_version") != 1:
        raise ValueError("stability manifest schema_version must be 1")
    compatibility = manifest.get("compatibility")
    if not isinstance(compatibility, Mapping):
        raise ValueError("manifest.compatibility must be a mapping")
    fingerprint = _required_string(manifest, "compatibility_fingerprint")
    actual_fingerprint = _sha256_json(compatibility)
    if fingerprint != actual_fingerprint:
        raise ValueError(
            "Stability manifest compatibility fingerprint mismatch: "
            f"stored={fingerprint}, actual={actual_fingerprint}"
        )
    _manifest_base_seeds(manifest)

    selection = manifest.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("manifest.selection must be a mapping")
    ordered_states = selection.get("ordered_states")
    if not isinstance(ordered_states, list):
        raise ValueError("manifest.selection.ordered_states must be a list")
    if len(ordered_states) != EXPECTED_STATE_COUNT:
        raise ValueError(
            f"Expected exactly {EXPECTED_STATE_COUNT} selected states, got {len(ordered_states)}"
        )
    plan_digest = _sha256_json(ordered_states)
    stored_digests = [
        selection.get("ordered_states_sha256"),
        selection.get("selection_plan_sha256"),
        compatibility.get("selection_plan_sha256"),
    ]
    stored_digests = [value for value in stored_digests if value is not None]
    if not stored_digests:
        raise ValueError("manifest is missing an immutable selection-plan SHA-256")
    if any(value != plan_digest for value in stored_digests):
        raise ValueError(
            "Selection-plan SHA-256 mismatch: "
            f"stored={stored_digests}, actual={plan_digest}"
        )

    source_indices: set[int] = set()
    sample_ids: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for position, raw_state in enumerate(ordered_states):
        if not isinstance(raw_state, Mapping):
            raise ValueError(f"selection state {position} must be a mapping")
        state = dict(raw_state)
        order = _required_int(state, "selection_order")
        if order != position:
            raise ValueError(
                f"selection_order mismatch at position {position}: got {order}"
            )
        source_index = _required_int(state, "source_index")
        sample_id = _required_string(state, "sample_id")
        stratum = _required_string(state, "selection_bin")
        pilot_utility = _required_float(state, "pilot_utility")
        if stratum not in STRATUM_PREVALENCE:
            raise ValueError(f"Unknown selection_bin={stratum!r} at state {position}")
        expected_stratum = classify_pilot_stratum(pilot_utility)
        if stratum != expected_stratum:
            raise ValueError(
                f"selection_bin/pilot_utility mismatch for {sample_id}: "
                f"stored={stratum}, expected={expected_stratum}"
            )
        if source_index in source_indices:
            raise ValueError(f"Duplicate source_index={source_index} in selection plan")
        if sample_id in sample_ids:
            raise ValueError(f"Duplicate sample_id={sample_id!r} in selection plan")
        source_indices.add(source_index)
        sample_ids.add(sample_id)
        normalized.append(state)

    missing_strata = sorted(set(STRATUM_PREVALENCE) - {s["selection_bin"] for s in normalized})
    if missing_strata:
        raise ValueError(
            "Population-weighted analysis requires every Pilot stratum; missing "
            f"{missing_strata}"
        )
    # Reuse the collector core preregistered suite/task/episode/bin/partial contract.
    stability_core.validate_stability_selection(normalized)
    return normalized


def load_stability_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line, parse_constant=_reject_json_constant)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"Malformed or non-finite JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            _assert_finite_tree(record, path=f"records[{line_number}]")
            records.append(record)
    if not records:
        raise ValueError(f"No stability records found in {path}")
    return records


def _record_source_index(record: Mapping[str, Any]) -> int:
    direct = record.get("source_index")
    metadata = record.get("source_metadata")
    nested = metadata.get("requested_sample_idx") if isinstance(metadata, Mapping) else None
    if direct is None:
        direct = nested
    source_index = _required_int({"source_index": direct}, "source_index")
    if nested is not None and nested != source_index:
        raise ValueError(
            f"source_index disagrees with source_metadata.requested_sample_idx: "
            f"{source_index} vs {nested!r}"
        )
    if isinstance(metadata, Mapping):
        source_sample = metadata.get("source_sample_idx")
        if source_sample is not None and source_sample != source_index:
            raise ValueError(
                f"source_metadata.source_sample_idx changed identity: "
                f"{source_sample!r} vs {source_index}"
            )
    return source_index


def _aliased_required_string(record: Mapping[str, Any], names: Sequence[str]) -> str:
    present = [(name, record[name]) for name in names if record.get(name) is not None]
    if not present:
        raise ValueError(f"Record is missing required provenance field; expected one of {names}")
    values = {value for _, value in present}
    if len(values) != 1:
        raise ValueError(f"Conflicting aliases {present}")
    value = next(iter(values))
    if not isinstance(value, str) or not value:
        raise ValueError(f"Provenance aliases {names} must contain a non-empty string")
    return value


def _validate_route(record: Mapping[str, Any], *, full_steps: int = 10) -> None:
    if record.get("n0") != 0 or record.get("nfull") != full_steps:
        raise ValueError(
            f"Expected paired N=0/N={full_steps}, got n0={record.get('n0')!r}, "
            f"nfull={record.get('nfull')!r}"
        )
    if record.get("num_inference_steps") != full_steps:
        raise ValueError(
            f"num_inference_steps must be {full_steps}, got "
            f"{record.get('num_inference_steps')!r}"
        )
    for key, prefix in (("n0_route", 0), ("nfull_route", full_steps)):
        route = record.get(key)
        if not isinstance(route, Mapping):
            raise ValueError(f"{key} must be a mapping")
        expected = {
            "inference_mode": "prefix",
            "video_prefix_steps": prefix,
            "num_inference_steps": full_steps,
            "force_custom_prefix": True,
        }
        for field, value in expected.items():
            if route.get(field) != value:
                raise ValueError(
                    f"{key}.{field} must be {value!r}, got {route.get(field)!r}"
                )


def _validate_input_hashes(record: Mapping[str, Any]) -> dict[str, str]:
    hashes = record.get("input_hashes")
    if not isinstance(hashes, Mapping) or not hashes:
        raise ValueError("input_hashes must be a non-empty mapping")
    required = {
        "action_is_pad",
        "combined",
        "context",
        "context_mask",
        "input_image",
        "proprio",
        "valid_target_action",
    }
    if not required.issubset(hashes):
        raise ValueError(f"input_hashes missing fields {sorted(required - set(hashes))}")
    result: dict[str, str] = {}
    for key, value in hashes.items():
        if not _is_sha256(value):
            raise ValueError(f"input_hashes.{key} is not a lowercase SHA-256")
        result[str(key)] = str(value)
    components = {key: result[key] for key in required if key != "combined"}
    if result["combined"] != _sha256_json(components):
        raise ValueError("Input hashes drift: combined digest does not match components")
    return result


def _artifact_digest(manifest: Mapping[str, Any], name: str) -> str | None:
    compatibility = manifest.get("compatibility", {})
    artifacts = manifest.get("artifacts", {})
    candidates: list[Any] = []
    if isinstance(compatibility, Mapping):
        candidates.append(compatibility.get(f"{name}_sha256"))
        nested = compatibility.get("artifacts")
        if isinstance(nested, Mapping) and isinstance(nested.get(name), Mapping):
            candidates.append(nested[name].get("sha256"))
    if isinstance(artifacts, Mapping) and isinstance(artifacts.get(name), Mapping):
        candidates.append(artifacts[name].get("sha256"))
    candidates = [candidate for candidate in candidates if candidate is not None]
    if not candidates:
        return None
    if len(set(candidates)) != 1 or not _is_sha256(candidates[0]):
        raise ValueError(f"Manifest contains inconsistent/invalid {name} SHA-256 values")
    return str(candidates[0])


def _validate_error_log(errors_path: Path | None) -> int:
    if errors_path is None or not errors_path.exists():
        return 0
    count = sum(1 for line in errors_path.read_text(encoding="utf-8").splitlines() if line.strip())
    if count:
        raise ValueError(
            f"Stability collection contains {count} error records in {errors_path}; "
            "analysis requires zero errors"
        )
    return count


def _reconstruct_pilot_selection_row(
    reused_record: Mapping[str, Any], planned: Mapping[str, Any]
) -> dict[str, Any]:
    """Recover the exact Pilot payload and bind it to the manifest digest."""
    stability_only = {
        "stability_record_schema_version", "replicate_id", "replicate_index",
        "replicate_base_seed", "replicate_seed", "pilot_base_seed",
        "stability_manifest_compatibility_fingerprint",
        "source_pilot_record_sha256", "reused_from_pilot",
        "inference_origin", "collection_git_sha",
    }
    pilot = {key: value for key, value in reused_record.items() if key not in stability_only}
    for field in ("selection_order", "source_index", "selection_bin"):
        pilot[field] = planned[field]
    expected = _required_string(planned, "pilot_record_sha256")
    if not _is_sha256(expected):
        raise ValueError("manifest selection pilot_record_sha256 is invalid")
    actual = stability_core.pilot_record_sha256(pilot)
    if actual != expected or reused_record.get("source_pilot_record_sha256") != expected:
        raise ValueError(
            f"Pilot source-record hash mismatch for {planned.get('sample_id')}: "
            f"manifest={expected}, reconstructed={actual}"
        )
    return pilot



def validate_stability_grid(
    records: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    errors_path: Path | None = None,
) -> dict[str, Any]:
    """Fail-closed audit of the 100 x 5 long-record collection."""

    plan = validate_stability_manifest(manifest)
    _validate_error_log(errors_path)
    expected_fingerprint = str(manifest["compatibility_fingerprint"])
    base_seeds = _manifest_base_seeds(manifest)
    if len(records) != EXPECTED_STATE_COUNT * EXPECTED_REPLICATE_COUNT:
        raise ValueError(
            f"Expected exactly 500 long records, got {len(records)}"
        )
    compatibility = manifest["compatibility"]
    pilot_manifest_fingerprint = compatibility.get("pilot_manifest_fingerprint")
    if not isinstance(pilot_manifest_fingerprint, str) or not pilot_manifest_fingerprint:
        raise ValueError("manifest.compatibility.pilot_manifest_fingerprint is required")

    plan_by_source = {int(state["source_index"]): state for state in plan}
    seen_keys: set[tuple[int, int]] = set()
    seen_replicate_ids: set[str] = set()
    rows_by_source: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    artifact_expected = {
        "checkpoint_sha256": _artifact_digest(manifest, "checkpoint"),
        "dataset_stats_sha256": _artifact_digest(manifest, "dataset_stats"),
        "vae_sha256": _artifact_digest(manifest, "vae"),
    }
    reused_count = 0
    new_count = 0

    for row_number, record in enumerate(records, start=1):
        _assert_finite_tree(record, path=f"record[{row_number}]")
        try:
            stability_core.validate_stability_record(
                record,
                expected_full_steps=10,
                expected_base_seeds=base_seeds,
                expected_stability_manifest_fingerprint=expected_fingerprint,
                expected_pilot_manifest_fingerprint=pilot_manifest_fingerprint,
                expected_checkpoint_sha256=artifact_expected["checkpoint_sha256"],
                expected_dataset_stats_sha256=artifact_expected["dataset_stats_sha256"],
                expected_vae_sha256=artifact_expected["vae_sha256"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"record {row_number}: core stability validation failed: {exc}") from exc
        if record.get("stability_record_schema_version") != 1:
            raise ValueError(
                f"record {row_number}: stability_record_schema_version must be 1"
            )
        source_index = _record_source_index(record)
        replicate_index = _required_int(record, "replicate_index")
        if replicate_index >= EXPECTED_REPLICATE_COUNT:
            raise ValueError(f"record {row_number}: invalid replicate_index={replicate_index}")
        replicate_base_seed = _required_int(record, "replicate_base_seed")
        if replicate_base_seed != base_seeds[replicate_index]:
            raise ValueError(
                f"record {row_number}: replicate base-seed mismatch: "
                f"index {replicate_index} requires {base_seeds[replicate_index]}, "
                f"got {replicate_base_seed}"
            )
        key = (source_index, replicate_index)
        if key in seen_keys:
            raise ValueError(f"Duplicate stability cell source/replicate={key}")
        seen_keys.add(key)
        if source_index not in plan_by_source:
            raise ValueError(f"record {row_number}: source_index={source_index} is outside plan")
        planned = plan_by_source[source_index]
        sample_id = _required_string(record, "sample_id")
        if sample_id != planned["sample_id"]:
            raise ValueError(
                f"record {row_number}: sample_id differs from plan for source {source_index}"
            )
        replicate_id = _required_string(record, "replicate_id")
        expected_replicate_id = stability_replicate_id(
            sample_id, replicate_index, replicate_base_seed
        )
        if replicate_id != expected_replicate_id:
            raise ValueError(
                f"record {row_number}: replicate_id={replicate_id!r}, "
                f"expected={expected_replicate_id!r}"
            )
        if replicate_id in seen_replicate_ids:
            raise ValueError(f"Duplicate replicate_id={replicate_id!r}")
        seen_replicate_ids.add(replicate_id)
        if record.get("selection_order") != planned["selection_order"]:
            raise ValueError(f"record {row_number}: selection_order differs from plan")
        if record.get("selection_bin") != planned["selection_bin"]:
            raise ValueError(f"record {row_number}: selection_bin differs from plan")
        pilot_utility = _required_float(record, "pilot_utility")
        if not math.isclose(
            pilot_utility,
            float(planned["pilot_utility"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"record {row_number}: pilot_utility differs from plan")

        e0 = _required_float(record, "e0")
        efull = _required_float(record, "efull")
        utility = _required_float(record, "utility")
        if not math.isclose(utility, e0 - efull, rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError(
                f"record {row_number}: utility != e0-efull ({utility} vs {e0-efull})"
            )
        _validate_route(record)
        _validate_input_hashes(record)

        identity = parse_sample_identity(record)
        if identity.sample_id != sample_id:
            raise ValueError(f"record {row_number}: parsed identity disagrees with sample_id")
        expected_seed = stable_sample_seed(replicate_base_seed, identity)
        seed = _required_int(record, "seed")
        if seed != expected_seed:
            raise ValueError(
                f"record {row_number}: stable inference seed mismatch: "
                f"stored={seed}, expected={expected_seed}"
            )
        pilot_seed = _required_int(record, "pilot_seed")
        expected_pilot_seed = stable_sample_seed(EXPECTED_BASE_SEEDS[0], identity)
        if pilot_seed != expected_pilot_seed:
            raise ValueError(
                f"record {row_number}: pilot_seed mismatch: {pilot_seed} vs {expected_pilot_seed}"
            )
        if record.get("stability_manifest_compatibility_fingerprint") != expected_fingerprint:
            raise ValueError(f"record {row_number}: stability manifest fingerprint mismatch")
        row_pilot_fingerprint = _aliased_required_string(
            record,
            ("source_pilot_manifest_fingerprint", "pilot_manifest_compatibility_fingerprint"),
        )
        if row_pilot_fingerprint != pilot_manifest_fingerprint:
            raise ValueError(f"record {row_number}: Pilot manifest provenance mismatch")
        pilot_record_sha = _required_string(record, "source_pilot_record_sha256")
        if not _is_sha256(pilot_record_sha):
            raise ValueError(f"record {row_number}: invalid source_pilot_record_sha256")
        for field, expected_digest in artifact_expected.items():
            actual_digest = record.get(field)
            if not _is_sha256(actual_digest):
                raise ValueError(f"record {row_number}: invalid/missing {field}")
            if expected_digest is not None and actual_digest != expected_digest:
                raise ValueError(f"record {row_number}: {field} differs from manifest")

        reused = record.get("reused_from_pilot")
        origin = record.get("inference_origin")
        if replicate_index == 0:
            if reused is not True or origin != "pilot_reuse":
                raise ValueError(
                    f"record {row_number}: replicate 0 must be provenance-marked Pilot reuse"
                )
            if seed != pilot_seed or not math.isclose(
                utility, pilot_utility, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    f"record {row_number}: reused replicate does not equal Pilot seed/utility"
                )
            reused_count += 1
        else:
            if reused is not False or origin != "new_inference":
                raise ValueError(
                    f"record {row_number}: replicate {replicate_index} must be new inference"
                )
            _required_string(record, "collection_git_sha")
            new_count += 1
        rows_by_source[source_index].append(record)

    core_record_index = {
        (_record_source_index(row), int(row["replicate_index"])): row for row in records
    }
    pilot_selection = []
    for planned in plan:
        source_index = int(planned["source_index"])
        reused_record = next(
            row for row in rows_by_source[source_index] if int(row["replicate_index"]) == 0
        )
        pilot_selection.append(_reconstruct_pilot_selection_row(reused_record, planned))
    core_grid = stability_core.validate_complete_grid(
        core_record_index, pilot_selection, base_seeds=base_seeds, allow_incomplete=False
    )


    expected_keys = {
        (source_index, replicate_index)
        for source_index in plan_by_source
        for replicate_index in range(EXPECTED_REPLICATE_COUNT)
    }
    missing = sorted(expected_keys - seen_keys)
    extra = sorted(seen_keys - expected_keys)
    if missing or extra:
        raise ValueError(
            f"Stability grid is not complete: missing={missing[:10]}, extra={extra[:10]}"
        )
    if reused_count != EXPECTED_STATE_COUNT or new_count != 400:
        raise ValueError(
            f"Expected 100 Pilot-reused + 400 new rows, got {reused_count} + {new_count}"
        )

    for source_index, state_records in rows_by_source.items():
        by_rep = {int(row["replicate_index"]): row for row in state_records}
        reference = by_rep[0]
        reference_hashes = dict(reference["input_hashes"])
        reference_pilot_sha = reference["source_pilot_record_sha256"]
        reference_pilot_u = float(reference["pilot_utility"])
        identity_fields = (
            "sample_id",
            "dataset_id",
            "dataset_name",
            "suite",
            "episode_index",
            "frame_index",
            "task_index",
            "task",
            "valid_length",
        )
        for replicate_index in range(EXPECTED_REPLICATE_COUNT):
            row = by_rep[replicate_index]
            if dict(row["input_hashes"]) != reference_hashes:
                raise ValueError(
                    f"Input hashes drift across seeds for source_index={source_index}, "
                    f"replicate={replicate_index}"
                )
            if row["source_pilot_record_sha256"] != reference_pilot_sha:
                raise ValueError(
                    f"Pilot source-record provenance drift for source_index={source_index}"
                )
            if not math.isclose(
                float(row["pilot_utility"]), reference_pilot_u, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(f"pilot_utility drift for source_index={source_index}")
            for field in identity_fields:
                if row.get(field) != reference.get(field):
                    raise ValueError(
                        f"State identity field {field!r} drifts across seeds for "
                        f"source_index={source_index}"
                    )

    return {
        "status": "complete_and_verified",
        "core_complete_grid": core_grid,
        "state_count": EXPECTED_STATE_COUNT,
        "replicate_count": EXPECTED_REPLICATE_COUNT,
        "record_count": len(records),
        "expected_record_count": 500,
        "reused_record_count": reused_count,
        "new_inference_record_count": new_count,
        "error_record_count": 0,
        "base_seeds": list(base_seeds),
        "selection_plan_sha256": _sha256_json(plan),
        "stability_manifest_compatibility_fingerprint": expected_fingerprint,
        "pilot_manifest_compatibility_fingerprint": pilot_manifest_fingerprint,
    }


def icc_one_way(values: np.ndarray) -> dict[str, float | None]:
    """Shrout-Fleiss ICC(1,1)/(1,k) for states x stochastic replicates."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 2:
        raise ValueError("ICC requires a finite [states>=2, replicates>=2] matrix")
    if not np.isfinite(array).all():
        raise ValueError("ICC input must be finite")
    n, k = array.shape
    state_means = array.mean(axis=1)
    grand_mean = float(array.mean())
    ms_between = float(k * np.square(state_means - grand_mean).sum() / (n - 1))
    ms_within = float(np.square(array - state_means[:, None]).sum() / (n * (k - 1)))
    denominator = ms_between + (k - 1) * ms_within
    if denominator == 0.0:
        icc_11: float | None = None
    else:
        icc_11 = float((ms_between - ms_within) / denominator)
    if ms_between == 0.0:
        icc_1k: float | None = None
    else:
        icc_1k = float((ms_between - ms_within) / ms_between)
    between_component = float((ms_between - ms_within) / k)
    return {
        "icc_1_1": icc_11,
        "icc_1_k": icc_1k,
        "k": int(k),
        "ms_between": ms_between,
        "ms_within": ms_within,
        "between_state_variance_component": between_component,
        "between_state_variance_component_nonnegative": max(0.0, between_component),
        "within_state_variance": ms_within,
        "grand_mean": grand_mean,
    }


def _rank_pair(x: Sequence[float], y: Sequence[float]) -> dict[str, float | None]:
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if x_array.shape != y_array.shape or x_array.ndim != 1 or x_array.size < 2:
        raise ValueError("Rank correlation needs paired 1D arrays with at least two values")
    spearman = stats.spearmanr(x_array, y_array).statistic
    kendall = stats.kendalltau(x_array, y_array).statistic
    return {
        "spearman_rho": float(spearman) if np.isfinite(spearman) else None,
        "kendall_tau": float(kendall) if np.isfinite(kendall) else None,
    }


def ranking_overlap(
    reference: Sequence[float], comparison: Sequence[float], *, fraction: float = 0.20
) -> dict[str, Any]:
    reference_array = np.asarray(reference, dtype=np.float64)
    comparison_array = np.asarray(comparison, dtype=np.float64)
    if reference_array.shape != comparison_array.shape or reference_array.ndim != 1:
        raise ValueError("ranking_overlap requires equal one-dimensional arrays")
    n = int(reference_array.size)
    if n < 2 or not 0.0 < fraction < 0.5:
        raise ValueError("ranking_overlap needs n>=2 and 0<fraction<0.5")
    k = max(1, int(math.ceil(n * fraction)))

    def indices(values: np.ndarray, *, top: bool) -> set[int]:
        order = np.lexsort((np.arange(n), -values if top else values))
        return set(int(index) for index in order[:k])

    result: dict[str, Any] = {"state_count": n, "fraction": fraction, "k": k}
    for side, top in (("top", True), ("bottom", False)):
        left = indices(reference_array, top=top)
        right = indices(comparison_array, top=top)
        intersection = len(left & right)
        union = len(left | right)
        result[f"{side}_intersection"] = intersection
        result[f"{side}_recall"] = float(intersection / k)
        result[f"{side}_jaccard"] = float(intersection / union)
    result["random_expected_recall"] = float(fraction)
    result["random_expected_jaccard"] = float(fraction / (2.0 - fraction))
    return result


def _bootstrap_ci(
    size: int,
    statistic: Callable[[np.ndarray], float | None],
    *,
    rng: np.random.Generator,
    replicates: int,
) -> dict[str, Any]:
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    samples: list[float] = []
    for _ in range(replicates):
        indices = rng.integers(0, size, size=size)
        value = statistic(indices)
        if value is not None and math.isfinite(float(value)):
            samples.append(float(value))
    if not samples:
        return {
            "lower_95": None,
            "upper_95": None,
            "valid_replicates": 0,
            "requested_replicates": replicates,
        }
    lower, upper = np.quantile(np.asarray(samples), (0.025, 0.975))
    return {
        "lower_95": float(lower),
        "upper_95": float(upper),
        "valid_replicates": len(samples),
        "requested_replicates": replicates,
    }


def population_state_weights(strata: Sequence[str]) -> np.ndarray:
    """Return prevalence-calibrated weights for the tail-oversampled audit set."""

    counts = Counter(strata)
    missing = sorted(set(STRATUM_PREVALENCE) - set(counts))
    unknown = sorted(set(counts) - set(STRATUM_PREVALENCE))
    if missing or unknown:
        raise ValueError(f"Cannot weight strata: missing={missing}, unknown={unknown}")
    weights = np.asarray(
        [STRATUM_PREVALENCE[stratum] / counts[stratum] for stratum in strata],
        dtype=np.float64,
    )
    weights /= weights.sum()
    return weights


def weighted_fraction(mask: Sequence[bool], weights: Sequence[float]) -> float:
    mask_array = np.asarray(mask, dtype=bool)
    weight_array = np.asarray(weights, dtype=np.float64)
    if mask_array.shape != weight_array.shape or mask_array.ndim != 1:
        raise ValueError("mask and weights must be equal 1D arrays")
    if not np.isfinite(weight_array).all() or (weight_array < 0).any():
        raise ValueError("weights must be finite and non-negative")
    denominator = float(weight_array.sum())
    if denominator <= 0:
        raise ValueError("weights must have positive mass")
    return float(weight_array[mask_array].sum() / denominator)


def _metric_at_least(name: str, value: float | None, threshold: float) -> dict[str, Any]:
    passed = value is not None and math.isfinite(value) and value >= threshold
    return {
        "name": name,
        "observed": value,
        "operator": ">=",
        "threshold": threshold,
        "passed": bool(passed),
    }


def decide_tiny_mlp_readiness(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the pre-registered Phase-2.5 readiness thresholds."""

    rank = metrics["rank_stability"]
    strong = metrics["strong_pilot_stability"]
    overlap = metrics["ranking_overlap"]
    reliability = metrics["reliability"]
    weighted = metrics["population_weighted"]
    bootstrap = metrics["bootstrap"]
    loso = metrics["leave_one_seed_out"]

    go_checks = [
        _metric_at_least("strong overall new-seed majority agreement", strong["overall_new4_majority_agreement"], 0.80),
        _metric_at_least("SP new-seed majority agreement", strong["SP_new4_majority_agreement"], 0.75),
        _metric_at_least("SN new-seed majority agreement", strong["SN_new4_majority_agreement"], 0.75),
        _metric_at_least("strong states with >=4/5 expected-direction seeds", strong["overall_four_of_five_expected_direction"], 0.70),
        _metric_at_least("seed42 vs new4-mean Spearman rho", rank["seed42_vs_new4_mean"]["spearman_rho"], 0.50),
        _metric_at_least("Spearman bootstrap 95% lower bound", bootstrap["seed42_vs_new4_mean_spearman"]["lower_95"], 0.30),
        _metric_at_least("LOSO median Spearman", loso["median_spearman_rho"], 0.40),
        _metric_at_least("LOSO positive-rho seeds", float(loso["positive_spearman_count"]), 4.0),
        _metric_at_least("top-20% recall", overlap["top_recall"], 0.40),
        _metric_at_least("top-20% Jaccard", overlap["top_jaccard"], 0.25),
        _metric_at_least("bottom-20% recall", overlap["bottom_recall"], 0.40),
        _metric_at_least("bottom-20% Jaccard", overlap["bottom_jaccard"], 0.25),
        _metric_at_least("ICC(1,5)", reliability["icc_1_k"], 0.75),
        _metric_at_least("ICC(1,1)", reliability["icc_1_1"], 0.35),
        _metric_at_least(
            "population-weighted strong-mean states with >=4/5 same-direction seeds",
            weighted["strong_mean_four_of_five_same_direction"],
            0.60,
        ),
    ]
    if all(check["passed"] for check in go_checks):
        decision = "GO"
    else:
        conditional_checks = [
            _metric_at_least(
                "strong overall new-seed majority agreement",
                strong["overall_new4_majority_agreement"],
                0.65,
            ),
            _metric_at_least(
                "seed42 vs new4-mean Spearman rho",
                rank["seed42_vs_new4_mean"]["spearman_rho"],
                0.30,
            ),
            _metric_at_least("ICC(1,5)", reliability["icc_1_k"], 0.50),
        ]
        decision = (
            "CONDITIONAL" if all(check["passed"] for check in conditional_checks) else "NO_GO"
        )
    return {
        "decision": decision,
        "scope": "readiness_to_start_offline_tiny_mlp_only",
        "does_not_establish": "closed_loop_gate_improvement",
        "go_checks": go_checks,
        "failed_go_checks": [check["name"] for check in go_checks if not check["passed"]],
        "conditional_rule": {
            "strong_majority_min": 0.65,
            "spearman_min": 0.30,
            "icc_1_5_min": 0.50,
        },
    }


def _state_rows_and_matrix(
    records: Sequence[Mapping[str, Any]], plan: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], np.ndarray]:
    by_key = {
        (_record_source_index(record), int(record["replicate_index"])): record
        for record in records
    }
    matrix = np.empty((len(plan), EXPECTED_REPLICATE_COUNT), dtype=np.float64)
    rows: list[dict[str, Any]] = []
    t_critical = float(stats.t.ppf(0.975, df=EXPECTED_REPLICATE_COUNT - 1))
    for state_index, state in enumerate(plan):
        source_index = int(state["source_index"])
        state_records = [by_key[(source_index, rep)] for rep in range(EXPECTED_REPLICATE_COUNT)]
        utilities = np.asarray([float(record["utility"]) for record in state_records])
        matrix[state_index] = utilities
        mean = float(utilities.mean())
        std = float(utilities.std(ddof=1))
        sem = float(std / math.sqrt(EXPECTED_REPLICATE_COUNT))
        reference = state_records[0]
        row: dict[str, Any] = {
            "selection_order": int(state["selection_order"]),
            "source_index": source_index,
            "sample_id": str(state["sample_id"]),
            "suite": str(reference["suite"]),
            "task_index": int(reference["task_index"]),
            "task": str(reference["task"]),
            "episode_index": int(reference["episode_index"]),
            "frame_index": int(reference["frame_index"]),
            "valid_length": int(reference["valid_length"]),
            "selection_bin": str(state["selection_bin"]),
            "pilot_utility": float(state["pilot_utility"]),
            "utility_mean": mean,
            "utility_median": float(np.median(utilities)),
            "utility_std": std,
            "utility_sem": sem,
            "utility_t95_low": float(mean - t_critical * sem),
            "utility_t95_high": float(mean + t_critical * sem),
            "utility_min": float(utilities.min()),
            "utility_max": float(utilities.max()),
        }
        for rep, base_seed in enumerate(EXPECTED_BASE_SEEDS):
            row[f"utility_seed_{base_seed}"] = float(utilities[rep])
        for epsilon in DEADBAND_EPSILONS:
            label = _epsilon_label(epsilon)
            signs = np.asarray([deadband_sign(value, epsilon) for value in utilities])
            counts = Counter(int(sign) for sign in signs)
            max_count = max(counts.values())
            winners = [sign for sign, count in counts.items() if count == max_count]
            majority_sign = winners[0] if len(winners) == 1 else 0
            mean_sign = deadband_sign(mean, epsilon)
            mean_direction_count = int(np.sum(signs == mean_sign)) if mean_sign else 0
            new_signs = signs[1:]
            new_counts = Counter(int(sign) for sign in new_signs)
            new_max = max(new_counts.values())
            new_winners = [sign for sign, count in new_counts.items() if count == new_max]
            new_majority = new_winners[0] if len(new_winners) == 1 and new_max >= 3 else 0
            row.update(
                {
                    f"{label}_positive_count": int(counts.get(1, 0)),
                    f"{label}_negative_count": int(counts.get(-1, 0)),
                    f"{label}_nearzero_count": int(counts.get(0, 0)),
                    f"{label}_majority_sign": int(majority_sign),
                    f"{label}_majority_count": int(max_count),
                    f"{label}_majority_agreement": float(max_count / EXPECTED_REPLICATE_COUNT),
                    f"{label}_mean_sign": int(mean_sign),
                    f"{label}_mean_direction_count": mean_direction_count,
                    f"{label}_four_of_five_same_mean_direction": bool(
                        mean_sign != 0 and mean_direction_count >= 4
                    ),
                    f"{label}_new4_majority_sign": int(new_majority),
                    f"{label}_seed42_matches_new4_majority": bool(
                        new_majority != 0 and int(signs[0]) == new_majority
                    ),
                }
            )
        rows.append(row)
    return rows, matrix


def _strong_metrics(per_state: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    label = _epsilon_label(PRIMARY_EPSILON)
    result: dict[str, Any] = {}
    for name, accepted in (
        ("overall", {"SP", "SN"}),
        ("SP", {"SP"}),
        ("SN", {"SN"}),
    ):
        states = [row for row in per_state if row["selection_bin"] in accepted]
        if not states:
            raise ValueError(f"Strong stratum metric {name} has no states")
        majority_matches: list[bool] = []
        four_of_five: list[bool] = []
        seed42_matches: list[bool] = []
        for row in states:
            expected = _expected_stratum_sign(str(row["selection_bin"]))
            new_majority = int(row[f"{label}_new4_majority_sign"])
            majority_matches.append(new_majority == expected)
            signs = [
                deadband_sign(float(row[f"utility_seed_{seed}"]), PRIMARY_EPSILON)
                for seed in EXPECTED_BASE_SEEDS
            ]
            four_of_five.append(sum(sign == expected for sign in signs) >= 4)
            seed42_matches.append(
                new_majority != 0
                and deadband_sign(float(row["utility_seed_42"]), PRIMARY_EPSILON)
                == new_majority
            )
        result[f"{name}_state_count"] = len(states)
        result[f"{name}_new4_majority_agreement"] = float(np.mean(majority_matches))
        result[f"{name}_four_of_five_expected_direction"] = float(np.mean(four_of_five))
        result[f"{name}_seed42_matches_new4_majority"] = float(np.mean(seed42_matches))
    return result


def _stratum_rows(
    per_state: Sequence[Mapping[str, Any]], weights: np.ndarray
) -> list[dict[str, Any]]:
    label = _epsilon_label(PRIMARY_EPSILON)
    rows: list[dict[str, Any]] = []
    for stratum in STRATUM_ORDER:
        indices = [i for i, row in enumerate(per_state) if row["selection_bin"] == stratum]
        values = np.asarray([float(per_state[i]["utility_mean"]) for i in indices])
        expected = _expected_stratum_sign(stratum)
        general_agreement = [
            float(per_state[i][f"{label}_majority_agreement"]) for i in indices
        ]
        expected_new_majority = [
            int(per_state[i][f"{label}_new4_majority_sign"]) == expected for i in indices
        ]
        expected_four = []
        for i in indices:
            signs = [
                deadband_sign(float(per_state[i][f"utility_seed_{seed}"]), PRIMARY_EPSILON)
                for seed in EXPECTED_BASE_SEEDS
            ]
            expected_four.append(sum(sign == expected for sign in signs) >= 4)
        rows.append(
            {
                "selection_bin": stratum,
                "state_count": len(indices),
                "unweighted_selection_fraction": float(len(indices) / len(per_state)),
                "pilot_population_prevalence": STRATUM_PREVALENCE[stratum],
                "per_state_population_weight": float(weights[indices[0]]),
                "utility_mean_of_state_means": float(values.mean()),
                "utility_median_of_state_means": float(np.median(values)),
                "mean_five_seed_sign_agreement": float(np.mean(general_agreement)),
                "new4_majority_matches_pilot_stratum_direction": float(
                    np.mean(expected_new_majority)
                ),
                "four_of_five_match_pilot_stratum_direction": float(np.mean(expected_four)),
            }
        )
    return rows


def compute_stability_metrics(
    records: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    plan = validate_stability_manifest(manifest)
    per_state, matrix = _state_rows_and_matrix(records, plan)
    pilot = matrix[:, 0]
    new4_mean = matrix[:, 1:].mean(axis=1)
    all5_mean = matrix.mean(axis=1)
    weights = population_state_weights([str(row["selection_bin"]) for row in per_state])
    for row, weight in zip(per_state, weights, strict=True):
        row["population_weight"] = float(weight)

    rank_seed_new = _rank_pair(pilot, new4_mean)
    rank_seed_all = _rank_pair(pilot, all5_mean)
    overlap = ranking_overlap(pilot, new4_mean)
    reliability = icc_one_way(matrix)
    strong = _strong_metrics(per_state)

    seed_rows: list[dict[str, Any]] = []
    loso_rhos: list[float] = []
    for replicate_index, base_seed in enumerate(EXPECTED_BASE_SEEDS):
        other_mean = np.delete(matrix, replicate_index, axis=1).mean(axis=1)
        rank = _rank_pair(matrix[:, replicate_index], other_mean)
        if rank["spearman_rho"] is not None:
            loso_rhos.append(float(rank["spearman_rho"]))
        seed_rows.append(
            {
                "replicate_index": replicate_index,
                "base_seed": base_seed,
                "utility_mean": float(matrix[:, replicate_index].mean()),
                "utility_std_across_states": float(matrix[:, replicate_index].std(ddof=1)),
                "spearman_vs_other4_mean": rank["spearman_rho"],
                "kendall_vs_other4_mean": rank["kendall_tau"],
            }
        )
    loso = {
        "median_spearman_rho": float(np.median(loso_rhos)) if loso_rhos else None,
        "minimum_spearman_rho": float(min(loso_rhos)) if loso_rhos else None,
        "positive_spearman_count": int(sum(value > 0.0 for value in loso_rhos)),
        "finite_spearman_count": len(loso_rhos),
    }

    label = _epsilon_label(PRIMARY_EPSILON)
    strong_mean_mask = np.abs(all5_mean) > 1e-3
    strong_mean_same: list[bool] = []
    for row, mean in zip(per_state, all5_mean, strict=True):
        direction = 1 if mean > 0 else -1
        signs = [
            deadband_sign(float(row[f"utility_seed_{seed}"]), PRIMARY_EPSILON)
            for seed in EXPECTED_BASE_SEEDS
        ]
        strong_mean_same.append(sum(sign == direction for sign in signs) >= 4)
    strong_mean_same_array = np.asarray(strong_mean_same, dtype=bool)
    if strong_mean_mask.any():
        weighted_strong_mean_fraction: float | None = weighted_fraction(
            strong_mean_same_array[strong_mean_mask], weights[strong_mean_mask]
        )
    else:
        weighted_strong_mean_fraction = None

    population_weighted = {
        "method": "post-stratification to Pilot-500 prevalence; audit sample is tail-oversampled",
        "stratum_prevalence": dict(STRATUM_PREVALENCE),
        "mean_utility": float(np.sum(weights * all5_mean)),
        "positive_mean_fraction_at_primary_deadband": weighted_fraction(
            [deadband_sign(value, PRIMARY_EPSILON) == 1 for value in all5_mean], weights
        ),
        "negative_mean_fraction_at_primary_deadband": weighted_fraction(
            [deadband_sign(value, PRIMARY_EPSILON) == -1 for value in all5_mean], weights
        ),
        "nearzero_mean_fraction_at_primary_deadband": weighted_fraction(
            [deadband_sign(value, PRIMARY_EPSILON) == 0 for value in all5_mean], weights
        ),
        "strong_mean_state_count_unweighted": int(strong_mean_mask.sum()),
        "strong_mean_population_mass": float(weights[strong_mean_mask].sum()),
        "strong_mean_four_of_five_same_direction": weighted_strong_mean_fraction,
    }

    deadband_sensitivity: dict[str, Any] = {}
    for epsilon in DEADBAND_EPSILONS:
        eps_label = _epsilon_label(epsilon)
        mean_signs = np.asarray([deadband_sign(value, epsilon) for value in all5_mean])
        majority_agreement = np.asarray(
            [float(row[f"{eps_label}_majority_agreement"]) for row in per_state]
        )
        four_same_mean = np.asarray(
            [bool(row[f"{eps_label}_four_of_five_same_mean_direction"]) for row in per_state]
        )
        deadband_sensitivity[eps_label] = {
            "epsilon": epsilon,
            "positive_mean_count": int(np.sum(mean_signs == 1)),
            "negative_mean_count": int(np.sum(mean_signs == -1)),
            "nearzero_mean_count": int(np.sum(mean_signs == 0)),
            "unweighted_mean_sign_agreement": float(majority_agreement.mean()),
            "unweighted_four_of_five_same_mean_direction": float(four_same_mean.mean()),
            "population_weighted_four_of_five_same_mean_direction": weighted_fraction(
                four_same_mean, weights
            ),
        }

    rng = np.random.default_rng(bootstrap_seed)
    strong_indices = np.asarray(
        [i for i, row in enumerate(per_state) if row["selection_bin"] in ("SP", "SN")],
        dtype=np.int64,
    )
    strong_matches = np.asarray(
        [
            int(per_state[i][f"{label}_new4_majority_sign"])
            == _expected_stratum_sign(str(per_state[i]["selection_bin"]))
            for i in strong_indices
        ],
        dtype=np.float64,
    )
    bootstrap = {
        "seed": int(bootstrap_seed),
        "replicates": int(bootstrap_replicates),
        "seed42_vs_new4_mean_spearman": _bootstrap_ci(
            len(per_state),
            lambda indices: _rank_pair(pilot[indices], new4_mean[indices])["spearman_rho"],
            rng=rng,
            replicates=bootstrap_replicates,
        ),
        "strong_new4_majority_agreement": _bootstrap_ci(
            len(strong_matches),
            lambda indices: float(strong_matches[indices].mean()),
            rng=rng,
            replicates=bootstrap_replicates,
        ),
        "icc_1_5": _bootstrap_ci(
            len(per_state),
            lambda indices: icc_one_way(matrix[indices])["icc_1_k"],
            rng=rng,
            replicates=bootstrap_replicates,
        ),
    }

    stratum_rows = _stratum_rows(per_state, weights)
    metrics: dict[str, Any] = {
        "primary_deadband_epsilon": PRIMARY_EPSILON,
        "rank_stability": {
            "reference": "seed42 Pilot utility",
            "comparison": "mean utility of independent seeds 43-46",
            "seed42_vs_new4_mean": rank_seed_new,
            "seed42_vs_all5_mean": rank_seed_all,
        },
        "leave_one_seed_out": loso,
        "ranking_overlap": overlap,
        "strong_pilot_stability": strong,
        "reliability": reliability,
        "population_weighted": population_weighted,
        "deadband_sensitivity": deadband_sensitivity,
        "bootstrap": bootstrap,
        "unweighted_diagnostic_warning": (
            "The 100-state audit deliberately over-samples utility tails. Unweighted "
            "metrics describe this diagnostic panel, not the Pilot-500 population."
        ),
    }
    metrics["decision"] = decide_tiny_mlp_readiness(metrics)
    return metrics, per_state, seed_rows, stratum_rows


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fieldnames: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)


def _plot_outputs(
    output_dir: Path,
    matrix: np.ndarray,
    per_state: Sequence[Mapping[str, Any]],
    stratum_rows: Sequence[Mapping[str, Any]],
    reliability: Mapping[str, Any],
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    except ImportError:
        LOGGER.warning("matplotlib unavailable; skipping PNG diagnostics")
        return []

    blue = "#3568A8"
    blue_dark = "#183B66"
    gold = "#D69E2E"
    gold_light = "#F6E7BE"
    ink = "#20262E"
    grid = "#D9DEE5"
    written: list[str] = []

    order = np.argsort(matrix[:, 0])[::-1]
    heat_values = matrix[order]
    limit = float(np.quantile(np.abs(heat_values), 0.98))
    limit = max(limit, 1e-8)
    cmap = LinearSegmentedColormap.from_list("gold_white_blue", [gold, "#FFFFFF", blue])
    fig, ax = plt.subplots(figsize=(8.2, 9.0))
    image = ax.imshow(
        heat_values,
        aspect="auto",
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        interpolation="nearest",
    )
    ax.set_xticks(range(EXPECTED_REPLICATE_COUNT), [str(seed) for seed in EXPECTED_BASE_SEEDS])
    ax.set_xlabel("Inference base seed")
    ax.set_ylabel("100 states, sorted by seed-42 utility")
    ax.set_title("Multi-seed utility matrix")
    ax.text(
        0.0,
        1.01,
        "U = E0 − E10 in normalized action space; color clipped at the 98th |U| percentile",
        transform=ax.transAxes,
        fontsize=9,
        color=ink,
    )
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Utility U")
    fig.tight_layout()
    path = output_dir / "utility_seed_heatmap.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    written.append(path.name)

    pilot = matrix[:, 0]
    new_mean = matrix[:, 1:].mean(axis=1)
    bound = max(float(np.quantile(np.abs(np.concatenate([pilot, new_mean])), 0.99)), 1e-8)
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.scatter(pilot, new_mean, s=30, color=blue, edgecolor=blue_dark, linewidth=0.45, alpha=0.82)
    ax.plot([-bound, bound], [-bound, bound], color=ink, linestyle="--", linewidth=1.0, label="identity")
    ax.axhline(0.0, color=grid, linewidth=0.8)
    ax.axvline(0.0, color=grid, linewidth=0.8)
    ax.set_xlim(-bound, bound)
    ax.set_ylim(-bound, bound)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Seed 42 / Pilot utility")
    ax.set_ylabel("Mean utility, new seeds 43–46")
    ax.set_title("Pilot label vs independent-seed mean")
    ax.text(0.0, 1.01, "Each point is one demonstration state (n=100)", transform=ax.transAxes, fontsize=9, color=ink)
    ax.legend(frameon=False)
    ax.grid(color=grid, linewidth=0.6, alpha=0.6)
    fig.tight_layout()
    path = output_dir / "seed42_vs_new4_mean_scatter.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    written.append(path.name)

    labels = [str(row["selection_bin"]) for row in stratum_rows]
    majority = [float(row["new4_majority_matches_pilot_stratum_direction"]) for row in stratum_rows]
    four = [float(row["four_of_five_match_pilot_stratum_direction"]) for row in stratum_rows]
    positions = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.2, 5.1))
    ax.bar(positions - width / 2, majority, width, color=blue, edgecolor=blue_dark, label="new-4 majority retains Pilot direction")
    ax.bar(positions + width / 2, four, width, color=gold_light, edgecolor=gold, label=">=4/5 retain Pilot direction")
    ax.axhline(0.8, color=ink, linestyle="--", linewidth=1.0, label="0.80 reference")
    ax.set_xticks(positions, labels)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Fraction of states")
    ax.set_xlabel("Pilot utility stratum")
    ax.set_title("Sign retention by Pilot utility stratum")
    ax.text(0.0, 1.01, "Primary deadband |U| ≤ 1e−4; panel is deliberately tail-oversampled", transform=ax.transAxes, fontsize=9, color=ink)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower center")
    ax.grid(axis="y", color=grid, linewidth=0.6)
    fig.tight_layout()
    path = output_dir / "stratum_sign_agreement.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    written.append(path.name)

    variance_names = ["Between-state\ncomponent", "Within-state\n(seed noise)"]
    variance_values = [
        float(reliability["between_state_variance_component"]),
        float(reliability["within_state_variance"]),
    ]
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    bars = ax.bar(variance_names, variance_values, color=[blue, gold_light], edgecolor=[blue_dark, gold])
    ax.axhline(0.0, color=ink, linewidth=0.9)
    ax.set_ylabel("Variance in utility U")
    ax.set_title("One-way random-effects variance components")
    ax.text(0.0, 1.01, "100 states × 5 inference seeds; raw method-of-moments estimates", transform=ax.transAxes, fontsize=9, color=ink)
    ax.grid(axis="y", color=grid, linewidth=0.6)
    for bar, value in zip(bars, variance_values, strict=True):
        ax.annotate(f"{value:.3g}", (bar.get_x() + bar.get_width() / 2, value), xytext=(0, 4 if value >= 0 else -12), textcoords="offset points", ha="center", fontsize=9)
    fig.tight_layout()
    path = output_dir / "variance_components.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    written.append(path.name)
    return written


def analyze(
    records_path: Path,
    manifest_path: Path,
    output_dir: Path,
    *,
    errors_path: Path | None = None,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    make_plots: bool = True,
) -> dict[str, Any]:
    records_path = records_path.resolve()
    manifest_path = manifest_path.resolve()
    if not records_path.is_file():
        raise FileNotFoundError(records_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    records = load_stability_records(records_path)
    manifest = _read_json(manifest_path)
    validation = validate_stability_grid(records, manifest, errors_path=errors_path)
    metrics, per_state, seed_rows, stratum_rows = compute_stability_metrics(
        records,
        manifest,
        bootstrap_seed=bootstrap_seed,
        bootstrap_replicates=bootstrap_replicates,
    )
    plan = validate_stability_manifest(manifest)
    _, matrix = _state_rows_and_matrix(records, plan)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "per_state.csv", per_state)
    _write_csv(output_dir / "seed_metrics.csv", seed_rows)
    _write_csv(output_dir / "stratum_metrics.csv", stratum_rows)
    plot_files = (
        _plot_outputs(output_dir, matrix, per_state, stratum_rows, metrics["reliability"])
        if make_plots
        else []
    )
    summary = {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "analysis_scope": "Phase-2.5 stochastic stability of demo-level utility labels",
        "input": {
            "records_path": str(records_path),
            "records_sha256": _sha256_file(records_path),
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256_file(manifest_path),
            "errors_path": str(errors_path.resolve()) if errors_path is not None else None,
        },
        "validation": validation,
        "settings": {
            "primary_deadband_epsilon": PRIMARY_EPSILON,
            "sensitivity_deadband_epsilons": list(DEADBAND_EPSILONS),
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_replicates": bootstrap_replicates,
            "pilot_stratum_prevalence": dict(STRATUM_PREVALENCE),
        },
        "metrics": metrics,
        "outputs": {
            "per_state_csv": "per_state.csv",
            "seed_metrics_csv": "seed_metrics.csv",
            "stratum_metrics_csv": "stratum_metrics.csv",
            "plot_files": plot_files,
        },
        "interpretation_guardrail": (
            "The automatic decision concerns whether labels are stable enough to begin "
            "offline Tiny MLP work. It does not validate Gate calibration, compute savings, "
            "or closed-loop LIBERO success."
        ),
    }
    summary_path = output_dir / "analysis_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--errors", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = _parse_args()
    records_path = args.records.resolve()
    manifest_path = args.manifest or records_path.parent / "manifest.json"
    output_dir = args.output_dir or records_path.parent / "stability_analysis"
    errors_path = args.errors
    if errors_path is None:
        candidate = records_path.parent / "errors.jsonl"
        errors_path = candidate if candidate.exists() else None
    summary = analyze(
        records_path,
        manifest_path,
        output_dir,
        errors_path=errors_path,
        bootstrap_seed=int(args.bootstrap_seed),
        bootstrap_replicates=int(args.bootstrap_replicates),
        make_plots=not bool(args.no_plots),
    )
    decision = summary["metrics"]["decision"]["decision"]
    LOGGER.info(
        "Verified 100x5 stability artifact; Tiny-MLP readiness=%s; outputs=%s",
        decision,
        output_dir,
    )


if __name__ == "__main__":
    main()
