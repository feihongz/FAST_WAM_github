"""Fail-closed contracts for expanding Utility Target V2 to Pilot-500.

The expansion has two deliberately separate artifacts:

* a 400-state x 5-seed stability source for the exact Pilot-500 remainder;
* an immutable 500-state Target V2 bundle combining the already validated
  100 targets with 400 targets derived from that remainder source.

Seed 42 is copied from the immutable Pilot record.  Seeds 43--46 are new
paired N=0/N=10 inference.  Independent validation seeds 47--50 are not used
by this module.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.libero.gate import demo_utility_stability as stability
from experiments.libero.gate import demo_utility_target_v2 as target_v2


PILOT_STATE_COUNT = 500
EXISTING_TARGET_STATE_COUNT = 100
REMAINDER_STATE_COUNT = 400
BASE_SEEDS = target_v2.TARGET_BASE_SEEDS
EXPECTED_REMAINDER_RECORD_COUNT = REMAINDER_STATE_COUNT * len(BASE_SEEDS)
EXPECTED_REUSED_RECORD_COUNT = REMAINDER_STATE_COUNT
EXPECTED_NEW_INFERENCE_COUNT = REMAINDER_STATE_COUNT * (len(BASE_SEEDS) - 1)

EXPANSION_PURPOSE = "libero_pilot500_target_v2_remainder"
EXPANSION_RECORD_SCHEMA_VERSION = 1
EXPANSION_RECORD_HASH_FIELD = "pilot500_expansion_record_sha256"
COMPLETION_SCHEMA_VERSION = 1
COMPLETION_KIND = "libero_pilot500_target_v2_remainder_completion"

COMBINED_SCHEMA_VERSION = 1
COMBINED_KIND = "libero_demo_utility_target_v2_pilot500"
COMBINED_MANIFEST_FILENAME = "manifest.json"
COMBINED_TARGETS_FILENAME = "targets.jsonl"
COMBINED_COMPLETION_FILENAME = "completion.json"


def canonical_json(value: Any) -> str:
    return target_v2.canonical_json(value)


def sha256_json(value: Any) -> str:
    return target_v2.sha256_json(value)


def sha256_file(path: str | Path) -> str:
    return target_v2.sha256_file(path)


def _serialize_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(f"{canonical_json(row)}\n" for row in rows).encode("utf-8")


def _sha256_jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(_serialize_jsonl(rows)).hexdigest()


def _require_sha256(value: Any, *, field: str) -> str:
    return target_v2._require_sha256(value, field=field)


def _source_index(row: Mapping[str, Any]) -> int:
    metadata = row.get("source_metadata")
    value = row.get("source_index")
    if value is None and isinstance(metadata, Mapping):
        value = metadata.get("requested_sample_idx")
    return target_v2._require_int(value, field="source_index")


def pilot_ordered_source_indices(pilot_manifest: Mapping[str, Any]) -> list[int]:
    selection = pilot_manifest.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("Pilot manifest has no selection")
    values = selection.get("ordered_selected_source_indices")
    if not isinstance(values, list):
        raise ValueError("Pilot manifest has no ordered source-index plan")
    result = [target_v2._require_int(value, field="Pilot source index") for value in values]
    if len(result) != PILOT_STATE_COUNT or len(set(result)) != PILOT_STATE_COUNT:
        raise ValueError("Pilot source-index plan must contain 500 unique states")
    if selection.get("ordered_selected_source_indices_sha256") != sha256_json(result):
        raise ValueError("Pilot ordered source-index digest is invalid")
    if int(selection.get("num_samples", -1)) != PILOT_STATE_COUNT:
        raise ValueError("Pilot manifest must contain exactly 500 states")
    return result


def _selection_entry(pilot: Mapping[str, Any], order: int) -> dict[str, Any]:
    result = deepcopy(dict(pilot))
    source_index = _source_index(pilot)
    result.update(
        {
            "selection_order": int(order),
            "source_index": source_index,
            "selection_bin": stability.utility_bin(float(pilot["utility"])),
            "pilot_seed": int(pilot["seed"]),
            "pilot_e0": float(pilot["e0"]),
            "pilot_efull": float(pilot["efull"]),
            "pilot_utility": float(pilot["utility"]),
            "pilot_valid_length": int(pilot["valid_length"]),
            "pilot_input_combined_sha256": str(pilot["input_hashes"]["combined"]),
            "pilot_manifest_compatibility_fingerprint": str(
                pilot["manifest_compatibility_fingerprint"]
            ),
        }
    )
    return result


def build_remainder_selection(
    pilot_records: Sequence[Mapping[str, Any]],
    pilot_manifest: Mapping[str, Any],
    existing_states: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return Pilot order minus the exact immutable 100-state panel."""

    order = pilot_ordered_source_indices(pilot_manifest)
    if len(pilot_records) != PILOT_STATE_COUNT:
        raise ValueError(f"Pilot records must contain {PILOT_STATE_COUNT} rows")
    pilot_by_source: dict[int, Mapping[str, Any]] = {}
    for row in pilot_records:
        source_index = _source_index(row)
        if source_index in pilot_by_source:
            raise ValueError(f"duplicate Pilot source_index={source_index}")
        pilot_by_source[source_index] = row
    if set(order) != set(pilot_by_source):
        raise ValueError("Pilot records do not exactly reproduce the manifest source plan")

    if len(existing_states) != EXISTING_TARGET_STATE_COUNT:
        raise ValueError("existing Target V2 source panel must contain exactly 100 states")
    excluded: set[int] = set()
    for state in existing_states:
        source_index = target_v2._require_int(
            state.get("source_index"), field="existing source_index"
        )
        if source_index in excluded:
            raise ValueError(f"duplicate existing source_index={source_index}")
        pilot = pilot_by_source.get(source_index)
        if pilot is None:
            raise ValueError(f"existing source_index={source_index} is outside Pilot-500")
        if state.get("sample_id") != pilot.get("sample_id"):
            raise ValueError("existing panel sample_id differs from Pilot")
        expected_hash = stability.pilot_record_sha256(pilot)
        if state.get("pilot_record_sha256") != expected_hash:
            raise ValueError("existing panel Pilot row hash differs from Pilot")
        excluded.add(source_index)

    remaining_sources = [value for value in order if value not in excluded]
    if len(remaining_sources) != REMAINDER_STATE_COUNT:
        raise ValueError("Pilot-500 minus existing Target V2 panel is not exactly 400 states")
    result = [
        _selection_entry(pilot_by_source[source_index], remainder_order)
        for remainder_order, source_index in enumerate(remaining_sources)
    ]
    if {_source_index(row) for row in result} & excluded:
        raise AssertionError("remainder selection overlaps existing panel")
    if {_source_index(row) for row in result} | excluded != set(order):
        raise AssertionError("remainder plus existing panel does not cover Pilot-500")
    return result


def expansion_record_sha256(record: Mapping[str, Any]) -> str:
    return sha256_json(
        {key: value for key, value in record.items() if key != EXPANSION_RECORD_HASH_FIELD}
    )


def seal_expansion_record(record: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(record)
    result["pilot500_expansion_record_schema_version"] = EXPANSION_RECORD_SCHEMA_VERSION
    result[EXPANSION_RECORD_HASH_FIELD] = expansion_record_sha256(result)
    validate_expansion_record(result)
    return result


def validate_expansion_record(
    record: Mapping[str, Any],
    *,
    expected_manifest_fingerprint: str | None = None,
    expected_pilot_manifest_fingerprint: str | None = None,
    expected_checkpoint_sha256: str | None = None,
    expected_dataset_stats_sha256: str | None = None,
    expected_vae_sha256: str | None = None,
) -> None:
    stability.validate_stability_record(
        record,
        expected_base_seeds=BASE_SEEDS,
        expected_stability_manifest_fingerprint=expected_manifest_fingerprint,
        expected_pilot_manifest_fingerprint=expected_pilot_manifest_fingerprint,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_dataset_stats_sha256=expected_dataset_stats_sha256,
        expected_vae_sha256=expected_vae_sha256,
    )
    if int(record.get("pilot500_expansion_record_schema_version", -1)) != (
        EXPANSION_RECORD_SCHEMA_VERSION
    ):
        raise ValueError("unsupported Pilot-500 expansion record schema")
    digest = _require_sha256(
        record.get(EXPANSION_RECORD_HASH_FIELD), field=EXPANSION_RECORD_HASH_FIELD
    )
    if digest != expansion_record_sha256(record):
        raise ValueError("Pilot-500 expansion record SHA-256 is invalid")


def load_expansion_record_index(
    records_path: str | Path,
    *,
    expected_manifest_fingerprint: str,
    expected_pilot_manifest_fingerprint: str,
    expected_checkpoint_sha256: str | None = None,
    expected_dataset_stats_sha256: str | None = None,
    expected_vae_sha256: str | None = None,
) -> dict[tuple[int, int], dict[str, Any]]:
    path = Path(records_path)
    if not path.exists():
        return {}
    result: dict[tuple[int, int], dict[str, Any]] = {}
    replicate_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed expansion row at {path}:{line_number}") from exc
            try:
                validate_expansion_record(
                    row,
                    expected_manifest_fingerprint=expected_manifest_fingerprint,
                    expected_pilot_manifest_fingerprint=expected_pilot_manifest_fingerprint,
                    expected_checkpoint_sha256=expected_checkpoint_sha256,
                    expected_dataset_stats_sha256=expected_dataset_stats_sha256,
                    expected_vae_sha256=expected_vae_sha256,
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid expansion row at {path}:{line_number}: {exc}"
                ) from exc
            key = (int(row["source_index"]), int(row["replicate_index"]))
            if key in result:
                raise ValueError(f"duplicate expansion composite key {key}")
            if row["replicate_id"] in replicate_ids:
                raise ValueError(f"duplicate expansion replicate_id={row['replicate_id']!r}")
            result[key] = row
            replicate_ids.add(str(row["replicate_id"]))
    return result


def validate_expansion_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("kind") != target_v2.SOURCE_BUNDLE_KIND:
        raise ValueError("expansion manifest must remain a Target V2 source bundle")
    if manifest.get("purpose") != EXPANSION_PURPOSE:
        raise ValueError("unexpected expansion manifest purpose")
    compatibility = manifest.get("compatibility")
    if not isinstance(compatibility, Mapping):
        raise ValueError("expansion manifest has no compatibility object")
    if manifest.get("compatibility_fingerprint") != sha256_json(compatibility):
        raise ValueError("expansion manifest compatibility fingerprint is invalid")
    if compatibility.get("purpose") != EXPANSION_PURPOSE:
        raise ValueError("expansion compatibility purpose is invalid")
    selection = manifest.get("selection")
    if not isinstance(selection, Mapping) or not isinstance(
        selection.get("ordered_states"), list
    ):
        raise ValueError("expansion manifest has no ordered remainder plan")
    states = selection["ordered_states"]
    if len(states) != REMAINDER_STATE_COUNT:
        raise ValueError("expansion plan must contain exactly 400 states")
    selection_sha = sha256_json(states)
    if selection.get("ordered_states_sha256") != selection_sha:
        raise ValueError("expansion selection digest is invalid")
    if compatibility.get("selection_plan_sha256") != selection_sha:
        raise ValueError("expansion compatibility is not bound to its selection")
    orders = [int(row.get("selection_order", -1)) for row in states]
    sources = [int(row.get("source_index", -1)) for row in states]
    if orders != list(range(REMAINDER_STATE_COUNT)) or len(set(sources)) != len(sources):
        raise ValueError("expansion selection order/source keys are invalid")
    replicates = manifest.get("replicates")
    if not isinstance(replicates, Mapping):
        raise ValueError("expansion manifest has no replicate plan")
    expected_replicates = {
        "base_seeds": list(BASE_SEEDS),
        "count": len(BASE_SEEDS),
        "reuse_base_seed": BASE_SEEDS[0],
        "reuse_replicate_index": 0,
        "expected_record_count": EXPECTED_REMAINDER_RECORD_COUNT,
        "expected_reused_record_count": EXPECTED_REUSED_RECORD_COUNT,
        "expected_new_inference_count": EXPECTED_NEW_INFERENCE_COUNT,
    }
    for field, expected in expected_replicates.items():
        if replicates.get(field) != expected:
            raise ValueError(f"expansion replicate plan mismatch for {field}")
    if list(compatibility.get("replicate_base_seeds", [])) != list(BASE_SEEDS):
        raise ValueError("expansion compatibility seeds are invalid")
    if int(compatibility.get("num_states", -1)) != REMAINDER_STATE_COUNT:
        raise ValueError("expansion compatibility state count is invalid")
    if int(compatibility.get("reuse_base_seed", -1)) != BASE_SEEDS[0]:
        raise ValueError("expansion compatibility reuse seed is invalid")
    if selection.get("pilot_ordered_source_indices_sha256") != compatibility.get(
        "pilot_ordered_source_indices_sha256"
    ):
        raise ValueError("expansion Pilot order binding is inconsistent")
    for field in (
        "pilot_manifest_fingerprint",
        "pilot_manifest_sha256",
        "pilot_records_sha256",
        "phase25_manifest_sha256",
        "phase25_records_sha256",
        "phase25_selection_plan_sha256",
        "existing_target_v2_manifest_sha256",
        "existing_target_v2_targets_sha256",
        "pilot_ordered_source_indices_sha256",
        "checkpoint_sha256",
        "dataset_stats_sha256",
        "vae_sha256",
    ):
        _require_sha256(compatibility.get(field), field=f"compatibility.{field}")
    phase25 = manifest.get("excluded_phase25")
    pilot = manifest.get("pilot")
    existing = manifest.get("existing_target_v2")
    if (not isinstance(phase25, Mapping) or not isinstance(pilot, Mapping)
            or not isinstance(existing, Mapping)):
        raise ValueError("expansion manifest source bindings are missing")
    binding_pairs = (
        (pilot, "manifest_fingerprint", "pilot_manifest_fingerprint"),
        (pilot, "manifest_sha256", "pilot_manifest_sha256"),
        (pilot, "records_sha256", "pilot_records_sha256"),
        (phase25, "manifest_sha256", "phase25_manifest_sha256"),
        (phase25, "records_sha256", "phase25_records_sha256"),
        (phase25, "selection_plan_sha256", "phase25_selection_plan_sha256"),
        (existing, "manifest_sha256", "existing_target_v2_manifest_sha256"),
        (existing, "targets_sha256", "existing_target_v2_targets_sha256"),
    )
    for section, field, compatibility_field in binding_pairs:
        if section.get(field) != compatibility.get(compatibility_field):
            raise ValueError(f"expansion source binding mismatch for {compatibility_field}")
    if int(existing.get("state_count", -1)) != EXISTING_TARGET_STATE_COUNT:
        raise ValueError("existing Target V2 binding must contain exactly 100 states")


def validate_component_hashes(
    manifest: Mapping[str, Any], component_hashes: Mapping[str, Any]
) -> None:
    """Validate a pre-seal, freshly rehashed component snapshot."""

    compatibility = manifest["compatibility"]
    expected_pairs = (
        ("pilot", "manifest_sha256", "pilot_manifest_sha256"),
        ("pilot", "records_sha256", "pilot_records_sha256"),
        ("phase25", "manifest_sha256", "phase25_manifest_sha256"),
        ("phase25", "records_sha256", "phase25_records_sha256"),
        ("existing_target_v2", "manifest_sha256", "existing_target_v2_manifest_sha256"),
        ("existing_target_v2", "targets_sha256", "existing_target_v2_targets_sha256"),
    )
    for section, field, compatibility_field in expected_pairs:
        value = component_hashes.get(section)
        if not isinstance(value, Mapping):
            raise ValueError(f"component hash snapshot is missing {section}")
        if value.get(field) != compatibility.get(compatibility_field):
            raise ValueError(f"component hash snapshot differs for {section}.{field}")
    artifacts = component_hashes.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("component hash snapshot is missing artifacts")
    for field in ("checkpoint_sha256", "dataset_stats_sha256", "vae_sha256"):
        if artifacts.get(field) != compatibility.get(field):
            raise ValueError(f"component hash snapshot differs for artifacts.{field}")
    sources = artifacts.get("dataset_sources")
    if not isinstance(sources, list):
        raise ValueError("component hash snapshot has no dataset source trees")
    normalized_sources = [
        {
            "dataset_name": item.get("dataset_name"),
            "sha256": item.get("sha256"),
            "file_count": item.get("file_count"),
            "total_size_bytes": item.get("total_size_bytes"),
        }
        for item in sources
    ]
    if canonical_json(normalized_sources) != canonical_json(
        compatibility.get("dataset_source_content")
    ):
        raise ValueError("component hash snapshot differs for dataset source trees")
    context = artifacts.get("text_embedding_cache")
    if not isinstance(context, Mapping) or context.get("sha256") != compatibility.get(
        "context_cache_sha256"
    ):
        raise ValueError("component hash snapshot differs for text embedding cache")
    scientific = component_hashes.get("scientific_source_files")
    if not isinstance(scientific, Mapping) or canonical_json(scientific) != canonical_json(
        manifest.get("scientific_source_files")
    ):
        raise ValueError("component hash snapshot differs for scientific source files")


def ordered_expansion_record_hash_digest(records_path: str | Path) -> str:
    hashes: list[str] = []
    with Path(records_path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            hashes.append(
                _require_sha256(
                    row.get(EXPANSION_RECORD_HASH_FIELD),
                    field=f"records[{line_number}].{EXPANSION_RECORD_HASH_FIELD}",
                )
            )
    return sha256_json(hashes)


def _jsonl_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as stream:
        return sum(bool(line.strip()) for line in stream)


def build_completion_payload(
    *,
    manifest_path: Path,
    records_path: Path,
    errors_path: Path,
    manifest: Mapping[str, Any],
    component_hashes: Mapping[str, Any],
) -> dict[str, Any]:
    validate_expansion_manifest(manifest)
    validate_component_hashes(manifest, component_hashes)
    records_count = _jsonl_count(records_path)
    errors_count = _jsonl_count(errors_path)
    if records_count != EXPECTED_REMAINDER_RECORD_COUNT:
        raise ValueError(
            f"cannot seal incomplete expansion: {records_count}/{EXPECTED_REMAINDER_RECORD_COUNT}"
        )
    if errors_count:
        raise ValueError("cannot seal expansion with recorded errors")
    compatibility = manifest["compatibility"]
    index = load_expansion_record_index(
        records_path,
        expected_manifest_fingerprint=manifest["compatibility_fingerprint"],
        expected_pilot_manifest_fingerprint=compatibility["pilot_manifest_fingerprint"],
        expected_checkpoint_sha256=compatibility["checkpoint_sha256"],
        expected_dataset_stats_sha256=compatibility["dataset_stats_sha256"],
        expected_vae_sha256=compatibility["vae_sha256"],
    )
    # Re-run the existing exhaustive Target-V2 source verifier before sealing.
    # This binds every self-consistent row back to the immutable selection,
    # checks all cross-seed fields, and rejects rows outside the planned grid.
    target_v2.load_verified_source_bundle(
        manifest_path,
        records_path,
        expected_manifest_sha256=sha256_file(manifest_path),
        expected_records_sha256=sha256_file(records_path),
        expected_selection_plan_sha256=compatibility["selection_plan_sha256"],
        expected_num_states=REMAINDER_STATE_COUNT,
    )
    expected_keys = {
        (int(state["source_index"]), replicate_index)
        for state in manifest["selection"]["ordered_states"]
        for replicate_index in range(len(BASE_SEEDS))
    }
    if set(index) != expected_keys:
        raise ValueError("cannot seal expansion whose composite grid differs from the plan")
    reused = sum(bool(row["reused_from_pilot"]) for row in index.values())
    inferred = sum(row["inference_origin"] == "new_inference" for row in index.values())
    if reused != EXPECTED_REUSED_RECORD_COUNT or inferred != EXPECTED_NEW_INFERENCE_COUNT:
        raise ValueError("expansion reuse/new-inference counts are invalid")
    payload: dict[str, Any] = {
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "kind": COMPLETION_KIND,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_fingerprint": manifest["compatibility_fingerprint"],
        "manifest_sha256": sha256_file(manifest_path),
        "records_sha256": sha256_file(records_path),
        "records_count": records_count,
        "ordered_record_sha256_sha256": ordered_expansion_record_hash_digest(records_path),
        "errors_sha256": sha256_file(errors_path),
        "errors_count": errors_count,
        "reused_record_count": reused,
        "new_inference_record_count": inferred,
        "pilot_manifest_sha256": compatibility["pilot_manifest_sha256"],
        "pilot_records_sha256": compatibility["pilot_records_sha256"],
        "phase25_manifest_sha256": compatibility["phase25_manifest_sha256"],
        "phase25_records_sha256": compatibility["phase25_records_sha256"],
        "selection_plan_sha256": compatibility["selection_plan_sha256"],
        "component_hashes": deepcopy(dict(component_hashes)),
        "component_hashes_sha256": sha256_json(component_hashes),
    }
    payload["completion_sha256"] = sha256_json(payload)
    return payload


def validate_completion_seal(
    completion_path: str | Path,
    *,
    manifest_path: Path,
    records_path: Path,
    errors_path: Path,
    manifest: Mapping[str, Any],
    component_hashes: Mapping[str, Any],
) -> dict[str, Any]:
    seal = json.loads(Path(completion_path).read_text(encoding="utf-8"))
    if seal.get("kind") != COMPLETION_KIND or int(seal.get("schema_version", -1)) != 1:
        raise ValueError("invalid expansion completion kind/schema")
    stored = _require_sha256(seal.get("completion_sha256"), field="completion_sha256")
    if stored != sha256_json(
        {key: value for key, value in seal.items() if key != "completion_sha256"}
    ):
        raise ValueError("expansion completion digest is invalid")
    current = build_completion_payload(
        manifest_path=manifest_path,
        records_path=records_path,
        errors_path=errors_path,
        manifest=manifest,
        component_hashes=component_hashes,
    )
    for field, value in current.items():
        if field not in {"completed_at_utc", "completion_sha256"} and seal.get(field) != value:
            raise ValueError(f"expansion completion seal mismatch for {field}")
    return seal


def ensure_completion_seal(
    completion_path: str | Path,
    *,
    manifest_path: Path,
    records_path: Path,
    errors_path: Path,
    manifest: Mapping[str, Any],
    component_hashes: Mapping[str, Any],
) -> dict[str, Any]:
    destination = Path(completion_path)
    if destination.exists():
        return validate_completion_seal(
            destination,
            manifest_path=manifest_path,
            records_path=records_path,
            errors_path=errors_path,
            manifest=manifest,
            component_hashes=component_hashes,
        )
    payload = build_completion_payload(
        manifest_path=manifest_path,
        records_path=records_path,
        errors_path=errors_path,
        manifest=manifest,
        component_hashes=component_hashes,
    )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent,
            prefix=".completion.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return validate_completion_seal(
        destination,
        manifest_path=manifest_path,
        records_path=records_path,
        errors_path=errors_path,
        manifest=manifest,
        component_hashes=component_hashes,
    )


def _component_binding(
    manifest: Mapping[str, Any],
    targets_sha256: str,
    manifest_sha256: str,
    count: int,
) -> dict[str, Any]:
    return {
        "manifest_sha256": _require_sha256(manifest_sha256, field="component manifest"),
        "targets_sha256": _require_sha256(targets_sha256, field="component targets"),
        "manifest_fingerprint": str(manifest["compatibility_fingerprint"]),
        "count": int(count),
        "source_manifest_sha256": manifest["compatibility"]["source_manifest_sha256"],
        "source_records_sha256": manifest["compatibility"]["source_records_sha256"],
    }


def build_combined_target_bundle(
    *,
    pilot_manifest: Mapping[str, Any],
    pilot_records: Sequence[Mapping[str, Any]],
    pilot_manifest_sha256: str,
    pilot_records_sha256: str,
    existing_manifest: Mapping[str, Any],
    existing_targets: Sequence[Mapping[str, Any]],
    existing_manifest_sha256: str,
    existing_targets_sha256: str,
    remainder_manifest: Mapping[str, Any],
    remainder_targets: Sequence[Mapping[str, Any]],
    remainder_manifest_sha256: str,
    remainder_targets_sha256: str,
    expansion_completion_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Combine two verified Target V2 components in immutable Pilot order."""

    if len(existing_targets) != EXISTING_TARGET_STATE_COUNT:
        raise ValueError("existing Target V2 component must contain 100 targets")
    if len(remainder_targets) != REMAINDER_STATE_COUNT:
        raise ValueError("remainder Target V2 component must contain 400 targets")
    target_v2.validate_target_manifest(existing_manifest)
    target_v2.validate_target_manifest(remainder_manifest)
    target_v2._validate_targets_against_manifest(existing_manifest, existing_targets)
    target_v2._validate_targets_against_manifest(remainder_manifest, remainder_targets)
    for row in [*existing_targets, *remainder_targets]:
        target_v2.validate_target_record(row)
    order = pilot_ordered_source_indices(pilot_manifest)
    if len(pilot_records) != PILOT_STATE_COUNT:
        raise ValueError("Pilot records must contain 500 rows")
    pilot_by_source = {_source_index(row): row for row in pilot_records}
    if len(pilot_by_source) != PILOT_STATE_COUNT or set(pilot_by_source) != set(order):
        raise ValueError("Pilot records do not reproduce Pilot order")

    component_rows = {
        "existing100": list(existing_targets),
        "remainder400": list(remainder_targets),
    }
    by_source: dict[int, tuple[str, Mapping[str, Any]]] = {}
    for component, rows in component_rows.items():
        for row in rows:
            source_index = int(row["source_index"])
            if source_index in by_source:
                raise ValueError("Target V2 components overlap")
            pilot = pilot_by_source.get(source_index)
            if pilot is None or row["sample_id"] != pilot["sample_id"]:
                raise ValueError("Target V2 component contains a state outside Pilot-500")
            if row["source_pilot_record_sha256"] != stability.pilot_record_sha256(pilot):
                raise ValueError("Target V2 component is not bound to the Pilot row")
            by_source[source_index] = (component, row)
    if set(by_source) != set(order):
        raise ValueError("Target V2 components do not exactly cover Pilot-500")

    combined: list[dict[str, Any]] = []
    selection: list[dict[str, Any]] = []
    for pilot500_order, source_index in enumerate(order):
        component, row = by_source[source_index]
        combined.append(dict(row))
        selection.append(
            {
                "pilot500_selection_order": pilot500_order,
                "source_index": source_index,
                "sample_id": row["sample_id"],
                "target_id": row["target_id"],
                "target_sha256": row["target_sha256"],
                "component": component,
                "component_selection_order": int(row["selection_order"]),
                "source_pilot_record_sha256": row["source_pilot_record_sha256"],
            }
        )
    selection_sha = sha256_json(selection)
    targets_sha = _sha256_jsonl(combined)
    existing_binding = _component_binding(
        existing_manifest, existing_targets_sha256, existing_manifest_sha256,
        EXISTING_TARGET_STATE_COUNT,
    )
    remainder_binding = _component_binding(
        remainder_manifest, remainder_targets_sha256, remainder_manifest_sha256,
        REMAINDER_STATE_COUNT,
    )
    compatibility = {
        "schema_version": COMBINED_SCHEMA_VERSION,
        "kind": COMBINED_KIND,
        "pilot_manifest_sha256": _require_sha256(
            pilot_manifest_sha256, field="pilot_manifest_sha256"
        ),
        "pilot_records_sha256": _require_sha256(
            pilot_records_sha256, field="pilot_records_sha256"
        ),
        "pilot_manifest_fingerprint": pilot_manifest["compatibility_fingerprint"],
        "pilot_ordered_source_indices_sha256": sha256_json(order),
        "existing100": existing_binding,
        "remainder400": remainder_binding,
        "expansion_completion_sha256": _require_sha256(
            expansion_completion_sha256, field="expansion_completion_sha256"
        ),
        "target_base_seeds": list(BASE_SEEDS),
        "deadband_epsilon": target_v2.DEFAULT_DEADBAND_EPSILON,
        "min_sign_agreement": target_v2.DEFAULT_MIN_SIGN_AGREEMENT,
        "num_states": PILOT_STATE_COUNT,
        "combined_selection_sha256": selection_sha,
        "combined_targets_sha256": targets_sha,
    }
    manifest = {
        "schema_version": COMBINED_SCHEMA_VERSION,
        "kind": COMBINED_KIND,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "compatibility": compatibility,
        "compatibility_fingerprint": sha256_json(compatibility),
        "pilot": {
            "manifest_sha256": compatibility["pilot_manifest_sha256"],
            "records_sha256": compatibility["pilot_records_sha256"],
            "manifest_fingerprint": compatibility["pilot_manifest_fingerprint"],
            "ordered_source_indices_sha256": compatibility[
                "pilot_ordered_source_indices_sha256"
            ],
        },
        "components": {
            "existing100": existing_binding,
            "remainder400": remainder_binding,
            "expansion_completion_sha256": compatibility[
                "expansion_completion_sha256"
            ],
        },
        "selection": {
            "algorithm": "immutable-pilot500-order-existing100-plus-exact-remainder400-v1",
            "num_states": PILOT_STATE_COUNT,
            "ordered_states": selection,
            "ordered_states_sha256": selection_sha,
        },
        "targets": {
            "filename": COMBINED_TARGETS_FILENAME,
            "count": PILOT_STATE_COUNT,
            "canonical_records_sha256": targets_sha,
            "ordered_target_sha256_sha256": sha256_json(
                [row["target_sha256"] for row in combined]
            ),
        },
        "policy": {
            "utility_definition": "U = E0 - E10",
            "target": "arithmetic mean over seeds 42--46 with preserved uncertainty",
            "existing100_reused_without_target_payload_mutation": True,
            "independent_validation_seeds_excluded": [47, 48, 49, 50],
        },
        "summary": {
            "num_states": PILOT_STATE_COUNT,
            "existing100_count": EXISTING_TARGET_STATE_COUNT,
            "remainder400_count": REMAINDER_STATE_COUNT,
            "high_confidence_count": sum(bool(row["high_confidence"]) for row in combined),
            "uncertain_count": sum(bool(row["uncertain"]) for row in combined),
        },
    }
    validate_combined_target_bundle(manifest, combined)
    return manifest, combined


def validate_combined_target_bundle(
    manifest: Mapping[str, Any], targets: Sequence[Mapping[str, Any]]
) -> None:
    if manifest.get("kind") != COMBINED_KIND or int(
        manifest.get("schema_version", -1)
    ) != 1:
        raise ValueError("invalid combined Target V2 kind/schema")
    compatibility = manifest.get("compatibility")
    if not isinstance(compatibility, Mapping):
        raise ValueError("combined manifest has no compatibility object")
    if manifest.get("compatibility_fingerprint") != sha256_json(compatibility):
        raise ValueError("combined compatibility fingerprint is invalid")
    if list(compatibility.get("target_base_seeds", [])) != list(BASE_SEEDS):
        raise ValueError("combined target seeds are invalid")
    for field in (
        "pilot_manifest_sha256",
        "pilot_records_sha256",
        "pilot_manifest_fingerprint",
        "pilot_ordered_source_indices_sha256",
        "expansion_completion_sha256",
        "combined_selection_sha256",
        "combined_targets_sha256",
    ):
        _require_sha256(compatibility.get(field), field=f"compatibility.{field}")
    if float(compatibility.get("deadband_epsilon", -1)) != (
        target_v2.DEFAULT_DEADBAND_EPSILON
    ):
        raise ValueError("combined deadband policy is invalid")
    if float(compatibility.get("min_sign_agreement", -1)) != (
        target_v2.DEFAULT_MIN_SIGN_AGREEMENT
    ):
        raise ValueError("combined sign-agreement policy is invalid")
    pilot = manifest.get("pilot")
    expected_pilot = {
        "manifest_sha256": compatibility.get("pilot_manifest_sha256"),
        "records_sha256": compatibility.get("pilot_records_sha256"),
        "manifest_fingerprint": compatibility.get("pilot_manifest_fingerprint"),
        "ordered_source_indices_sha256": compatibility.get(
            "pilot_ordered_source_indices_sha256"
        ),
    }
    if not isinstance(pilot, Mapping) or canonical_json(pilot) != canonical_json(
        expected_pilot
    ):
        raise ValueError("combined Pilot binding differs from compatibility")
    if len(targets) != PILOT_STATE_COUNT:
        raise ValueError("combined targets must contain exactly 500 rows")
    for row in targets:
        target_v2.validate_target_record(row)
    if len({row["target_id"] for row in targets}) != PILOT_STATE_COUNT:
        raise ValueError("combined targets contain duplicate IDs")
    if len({int(row["source_index"]) for row in targets}) != PILOT_STATE_COUNT:
        raise ValueError("combined targets contain duplicate source indices")
    selection = manifest.get("selection")
    if not isinstance(selection, Mapping) or not isinstance(
        selection.get("ordered_states"), list
    ):
        raise ValueError("combined manifest has no selection")
    if selection.get("algorithm") != (
        "immutable-pilot500-order-existing100-plus-exact-remainder400-v1"
    ):
        raise ValueError("combined selection algorithm is invalid")
    states = selection["ordered_states"]
    if int(selection.get("num_states", -1)) != PILOT_STATE_COUNT:
        raise ValueError("combined selection state count is invalid")
    if len(states) != PILOT_STATE_COUNT:
        raise ValueError("combined selection must contain 500 rows")
    if [int(row.get("pilot500_selection_order", -1)) for row in states] != list(
        range(PILOT_STATE_COUNT)
    ):
        raise ValueError("combined Pilot-500 order is invalid")
    expected_projection = [
        {
            "pilot500_selection_order": order,
            "source_index": int(target["source_index"]),
            "sample_id": target["sample_id"],
            "target_id": target["target_id"],
            "target_sha256": target["target_sha256"],
            "component": state["component"],
            "component_selection_order": int(target["selection_order"]),
            "source_pilot_record_sha256": target["source_pilot_record_sha256"],
        }
        for order, (state, target) in enumerate(zip(states, targets))
    ]
    if canonical_json(expected_projection) != canonical_json(states):
        raise ValueError("combined selection does not match target rows")
    selection_sha = sha256_json(states)
    if selection.get("ordered_states_sha256") != selection_sha:
        raise ValueError("combined selection digest is invalid")
    if compatibility.get("combined_selection_sha256") != selection_sha:
        raise ValueError("combined compatibility is not bound to selection")
    records_sha = _sha256_jsonl(targets)
    target_section = manifest.get("targets")
    if not isinstance(target_section, Mapping):
        raise ValueError("combined manifest has no target section")
    if target_section.get("filename") != COMBINED_TARGETS_FILENAME:
        raise ValueError("combined target filename is invalid")
    if int(target_section.get("count", -1)) != PILOT_STATE_COUNT:
        raise ValueError("combined target count is invalid")
    target_hashes = [row["target_sha256"] for row in targets]
    if target_section.get("ordered_target_sha256_sha256") != sha256_json(
        target_hashes
    ):
        raise ValueError("combined ordered target digest is invalid")
    if target_section.get("canonical_records_sha256") != records_sha:
        raise ValueError("combined target records digest is invalid")
    if compatibility.get("combined_targets_sha256") != records_sha:
        raise ValueError("combined compatibility is not bound to target records")
    counts = {"existing100": 0, "remainder400": 0}
    components = manifest.get("components")
    if not isinstance(components, Mapping):
        raise ValueError("combined manifest has no component bindings")
    expected_component_counts = {"existing100": 100, "remainder400": 400}
    for component in ("existing100", "remainder400"):
        binding = components.get(component)
        if not isinstance(binding, Mapping):
            raise ValueError(f"combined {component} binding is missing")
        if int(binding.get("count", -1)) != expected_component_counts[component]:
            raise ValueError(f"combined {component} binding count is invalid")
        for field in (
            "manifest_sha256",
            "targets_sha256",
            "manifest_fingerprint",
            "source_manifest_sha256",
            "source_records_sha256",
        ):
            _require_sha256(binding.get(field), field=f"{component}.{field}")
        if canonical_json(binding) != canonical_json(
            compatibility.get(component)
        ):
            raise ValueError(
                f"combined {component} binding differs from compatibility"
            )
    if components.get("expansion_completion_sha256") != compatibility.get(
        "expansion_completion_sha256"
    ):
        raise ValueError("combined expansion completion binding is inconsistent")
    for state, row in zip(states, targets):
        component = state["component"]
        if component not in counts:
            raise ValueError("combined selection has an unknown component")
        counts[component] += 1
        binding = components[component]
        source = row["source_binding"]
        if source["manifest_sha256"] != binding["source_manifest_sha256"]:
            raise ValueError("combined target source-manifest component mismatch")
        if source["records_sha256"] != binding["source_records_sha256"]:
            raise ValueError("combined target source-record component mismatch")
    if counts != {"existing100": 100, "remainder400": 400}:
        raise ValueError("combined component counts are invalid")
    summary = manifest.get("summary")
    expected_summary = {
        "num_states": PILOT_STATE_COUNT,
        "existing100_count": 100,
        "remainder400_count": 400,
        "high_confidence_count": sum(
            bool(row["high_confidence"]) for row in targets
        ),
        "uncertain_count": sum(bool(row["uncertain"]) for row in targets),
    }
    if canonical_json(summary) != canonical_json(expected_summary):
        raise ValueError("combined summary differs from target records")


def write_combined_target_bundle(
    output_dir: str | Path,
    manifest: Mapping[str, Any],
    targets: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path, Path]:
    validate_combined_target_bundle(manifest, targets)
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"immutable combined target output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        targets_path = staging / COMBINED_TARGETS_FILENAME
        manifest_path = staging / COMBINED_MANIFEST_FILENAME
        targets_path.write_bytes(_serialize_jsonl(targets))
        manifest_path.write_text(
            json.dumps(
                dict(manifest), ensure_ascii=False, indent=2, allow_nan=False
            )
            + "\n",
            encoding="utf-8",
        )
        completion = {
            "schema_version": 1,
            "kind": f"{COMBINED_KIND}_completion",
            "manifest_sha256": sha256_file(manifest_path),
            "targets_sha256": sha256_file(targets_path),
            "target_count": len(targets),
            "manifest_fingerprint": manifest["compatibility_fingerprint"],
        }
        completion["completion_sha256"] = sha256_json(completion)
        completion_path = staging / COMBINED_COMPLETION_FILENAME
        completion_path.write_text(
            json.dumps(
                completion, ensure_ascii=False, indent=2, allow_nan=False
            )
            + "\n",
            encoding="utf-8",
        )
        for path in (targets_path, manifest_path, completion_path):
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        os.replace(staging, output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return (
        output / COMBINED_MANIFEST_FILENAME,
        output / COMBINED_TARGETS_FILENAME,
        output / COMBINED_COMPLETION_FILENAME,
    )


def load_combined_target_bundle(
    output_dir: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
    expected_targets_sha256: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    root = Path(output_dir).resolve()
    manifest_path = root / COMBINED_MANIFEST_FILENAME
    targets_path = root / COMBINED_TARGETS_FILENAME
    completion_path = root / COMBINED_COMPLETION_FILENAME
    if (expected_manifest_sha256 is None) != (expected_targets_sha256 is None):
        raise ValueError(
            "expected combined manifest and targets SHA-256 must be supplied together"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    targets = [
        json.loads(line)
        for line in targets_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("kind") != f"{COMBINED_KIND}_completion" or int(
        completion.get("schema_version", -1)
    ) != 1:
        raise ValueError("invalid combined completion kind/schema")
    if int(completion.get("target_count", -1)) != len(targets):
        raise ValueError("combined completion target count mismatch")
    if completion.get("manifest_fingerprint") != manifest.get(
        "compatibility_fingerprint"
    ):
        raise ValueError("combined completion manifest fingerprint mismatch")
    if expected_manifest_sha256 is not None and sha256_file(manifest_path) != _require_sha256(
        expected_manifest_sha256, field="expected_manifest_sha256"
    ):
        raise ValueError("combined manifest SHA-256 differs from expected")
    if expected_targets_sha256 is not None and sha256_file(targets_path) != _require_sha256(
        expected_targets_sha256, field="expected_targets_sha256"
    ):
        raise ValueError("combined targets SHA-256 differs from expected")
    stored = _require_sha256(completion.get("completion_sha256"), field="completion_sha256")
    if stored != sha256_json(
        {
            key: value
            for key, value in completion.items()
            if key != "completion_sha256"
        }
    ):
        raise ValueError("combined completion digest is invalid")
    if completion.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("combined completion manifest digest mismatch")
    if completion.get("targets_sha256") != sha256_file(targets_path):
        raise ValueError("combined completion targets digest mismatch")
    validate_combined_target_bundle(manifest, targets)
    return manifest, targets, completion


__all__ = [
    "BASE_SEEDS",
    "COMBINED_KIND",
    "EXISTING_TARGET_STATE_COUNT",
    "EXPECTED_NEW_INFERENCE_COUNT",
    "EXPECTED_REMAINDER_RECORD_COUNT",
    "EXPECTED_REUSED_RECORD_COUNT",
    "EXPANSION_PURPOSE",
    "PILOT_STATE_COUNT",
    "REMAINDER_STATE_COUNT",
    "build_combined_target_bundle",
    "build_completion_payload",
    "build_remainder_selection",
    "ensure_completion_seal",
    "expansion_record_sha256",
    "load_combined_target_bundle",
    "load_expansion_record_index",
    "pilot_ordered_source_indices",
    "seal_expansion_record",
    "validate_combined_target_bundle",
    "validate_completion_seal",
    "validate_expansion_manifest",
    "validate_expansion_record",
    "write_combined_target_bundle",
]
