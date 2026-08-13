"""Pure helpers for the LIBERO demo-utility cross-seed stability audit.

The Phase-2 pilot measured each state once with paired ``N=0``/``N=10``
inference.  This module selects an exactly balanced 100-state audit set and
defines the durable long-form record used to repeat each selected state with
base seeds 42--46.  It deliberately constructs neither datasets nor models.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from experiments.libero.gate.demo_utility import (
    INPUT_HASH_COMPONENTS,
    LIBERO_DATASET_TO_SUITE,
    parse_sample_identity,
    stable_sample_seed,
)


STABILITY_RECORD_SCHEMA_VERSION = 1
PILOT_BASE_SEED = 42
DEFAULT_REPLICATE_BASE_SEEDS = (42, 43, 44, 45, 46)
UTILITY_BINS = ("SP", "SN", "MP", "MN", "NZ")
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
DEFAULT_SUITE_BIN_QUOTAS: dict[str, dict[str, int]] = {
    "libero_10": {"SP": 7, "SN": 6, "MP": 3, "MN": 3, "NZ": 6},
    "libero_goal": {"SP": 6, "SN": 7, "MP": 3, "MN": 3, "NZ": 6},
    "libero_object": {"SP": 6, "SN": 6, "MP": 4, "MN": 3, "NZ": 6},
    "libero_spatial": {"SP": 6, "SN": 6, "MP": 3, "MN": 3, "NZ": 7},
}
STRONG_UTILITY_THRESHOLD = 1e-3
NEAR_ZERO_UTILITY_THRESHOLD = 1e-4
FULL_ACTION_VALID_LENGTH = 32
SHA256_HEX_LENGTH = 64


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


def pilot_record_sha256(record: Mapping[str, Any]) -> str:
    """Return the canonical SHA-256 binding a stability row to its pilot row."""

    if not isinstance(record, Mapping):
        raise TypeError("pilot record must be a mapping")
    selector_fields = {
        "selection_order",
        "selection_bin",
        "source_index",
        "pilot_seed",
        "pilot_e0",
        "pilot_efull",
        "pilot_utility",
        "pilot_valid_length",
        "pilot_input_combined_sha256",
        "pilot_manifest_compatibility_fingerprint",
    }
    payload = {key: value for key, value in record.items() if key not in selector_fields}
    return _sha256_json(payload)


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


def _require_sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256, got {value!r}")
    return value


def utility_bin(
    utility: float,
    *,
    strong_threshold: float = STRONG_UTILITY_THRESHOLD,
    near_zero_threshold: float = NEAR_ZERO_UTILITY_THRESHOLD,
) -> str:
    """Classify one pilot label using exhaustive, boundary-stable bins.

    ``SP``: U > 1e-3; ``SN``: U < -1e-3; ``MP``: 1e-4 < U <= 1e-3;
    ``MN``: -1e-3 <= U < -1e-4; ``NZ``: |U| <= 1e-4.
    """

    value = _require_finite(utility, field="utility")
    strong = _require_finite(strong_threshold, field="strong_threshold", minimum=0.0)
    near = _require_finite(
        near_zero_threshold,
        field="near_zero_threshold",
        minimum=0.0,
    )
    if not near < strong:
        raise ValueError(
            "near_zero_threshold must be strictly smaller than strong_threshold"
        )
    if value > strong:
        return "SP"
    if value < -strong:
        return "SN"
    if value > near:
        return "MP"
    if value < -near:
        return "MN"
    return "NZ"


def _validate_core_utility_record(record: Mapping[str, Any], *, full_steps: int) -> None:
    required = {
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
        "seed",
        "num_inference_steps",
        "n0",
        "nfull",
        "e0",
        "efull",
        "utility",
        "valid_length",
        "target_action_shape",
        "pred_n0_shape",
        "pred_nfull_shape",
        "input_hashes",
        "n0_latency_ms",
        "nfull_latency_ms",
        "total_latency_ms",
        "n0_route",
        "nfull_route",
    }
    missing = required - set(record)
    if missing:
        raise ValueError(f"utility record is missing fields: {sorted(missing)}")

    identity = parse_sample_identity(record)
    if record["sample_id"] != identity.sample_id:
        raise ValueError(
            f"sample_id mismatch: {record['sample_id']!r} != {identity.sample_id!r}"
        )
    if _require_int(record["episode_id"], field="episode_id") != identity.episode_index:
        raise ValueError("episode_id must equal episode_index")
    if _require_int(record["task_id"], field="task_id") != identity.task_index:
        raise ValueError("task_id must equal task_index")
    if record["task_id_source"] != "lerobot_task_index":
        raise ValueError("task_id_source must equal 'lerobot_task_index'")

    steps = _require_int(full_steps, field="full_steps", minimum=1)
    if _require_int(record["num_inference_steps"], field="num_inference_steps") != steps:
        raise ValueError("num_inference_steps does not match full_steps")
    if _require_int(record["n0"], field="n0") != 0:
        raise ValueError("n0 must equal 0")
    if _require_int(record["nfull"], field="nfull") != steps:
        raise ValueError("nfull does not match full_steps")

    e0 = _require_finite(record["e0"], field="e0", minimum=0.0)
    efull = _require_finite(record["efull"], field="efull", minimum=0.0)
    utility = _require_finite(record["utility"], field="utility")
    if not math.isclose(utility, e0 - efull, rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError(f"utility mismatch: U={utility}, E0-Efull={e0 - efull}")

    shapes: list[list[int]] = []
    for field in ("target_action_shape", "pred_n0_shape", "pred_nfull_shape"):
        raw_shape = record[field]
        if not isinstance(raw_shape, (list, tuple)) or len(raw_shape) != 2:
            raise ValueError(f"{field} must be a two-element shape")
        shape = [_require_int(value, field=f"{field} member", minimum=1) for value in raw_shape]
        shapes.append(shape)
    if not shapes[0] == shapes[1] == shapes[2]:
        raise ValueError("target/N=0/N=full action shapes must be identical")
    valid_length = _require_int(record["valid_length"], field="valid_length", minimum=1)
    if valid_length > shapes[0][0]:
        raise ValueError("valid_length exceeds the action horizon")

    for field in ("n0_latency_ms", "nfull_latency_ms", "total_latency_ms"):
        _require_finite(record[field], field=field, minimum=0.0)
    for route_field, expected_prefix in (("n0_route", 0), ("nfull_route", steps)):
        route = record[route_field]
        expected_route = {
            "inference_mode": "prefix",
            "video_prefix_steps": expected_prefix,
            "num_inference_steps": steps,
            "force_custom_prefix": True,
        }
        if not isinstance(route, Mapping) or dict(route) != expected_route:
            raise ValueError(
                f"{route_field} mismatch: got {route!r}, expected {expected_route!r}"
            )

    input_hashes = record["input_hashes"]
    expected_hash_keys = set(INPUT_HASH_COMPONENTS) | {"combined"}
    if not isinstance(input_hashes, Mapping) or set(input_hashes) != expected_hash_keys:
        raise ValueError(f"input_hashes must contain exactly {sorted(expected_hash_keys)}")
    for name, value in input_hashes.items():
        _require_sha256(value, field=f"input_hashes.{name}")
    components = {name: input_hashes[name] for name in INPUT_HASH_COMPONENTS}
    if input_hashes["combined"] != _sha256_json(components):
        raise ValueError("input_hashes.combined does not match component hashes")


def _validate_pilot_record(
    record: Mapping[str, Any],
    *,
    expected_base_seed: int,
    expected_full_steps: int,
) -> None:
    _validate_core_utility_record(record, full_steps=expected_full_steps)
    required = {
        "schema_version",
        "collector_record_schema_version",
        "source_metadata",
        "current_proprio",
        "manifest_compatibility_fingerprint",
        "checkpoint_sha256",
        "dataset_stats_sha256",
        "vae_sha256",
        "git_sha",
    }
    missing = required - set(record)
    if missing:
        raise ValueError(f"completed pilot record is missing fields: {sorted(missing)}")
    if _require_int(record["schema_version"], field="schema_version") != 1:
        raise ValueError("unsupported pilot core schema_version")
    if _require_int(
        record["collector_record_schema_version"],
        field="collector_record_schema_version",
    ) != 1:
        raise ValueError("unsupported pilot collector_record_schema_version")

    identity = parse_sample_identity(record)
    expected_seed = stable_sample_seed(expected_base_seed, identity)
    if _require_int(record["seed"], field="seed") != expected_seed:
        raise ValueError(
            f"pilot seed mismatch for {identity.sample_id}: expected {expected_seed}"
        )
    _require_sha256(
        record["manifest_compatibility_fingerprint"],
        field="manifest_compatibility_fingerprint",
    )
    for field in ("checkpoint_sha256", "dataset_stats_sha256", "vae_sha256"):
        _require_sha256(record[field], field=field)
    git_sha = record["git_sha"]
    if (
        not isinstance(git_sha, str)
        or len(git_sha) != 40
        or any(character not in "0123456789abcdef" for character in git_sha)
    ):
        raise ValueError("git_sha must be a lowercase 40-character hex digest")

    source = record["source_metadata"]
    if not isinstance(source, Mapping):
        raise ValueError("source_metadata must be a mapping")
    requested = _require_int(
        source.get("requested_sample_idx"), field="source_metadata.requested_sample_idx"
    )
    actual = _require_int(
        source.get("source_sample_idx"), field="source_metadata.source_sample_idx"
    )
    if requested != actual:
        raise ValueError("requested_sample_idx must equal source_sample_idx")
    for record_field, source_field in (
        ("dataset_name", "dataset_name"),
        ("episode_index", "episode_index"),
        ("frame_index", "frame_index"),
        ("task_index", "task_index"),
        ("task", "task"),
    ):
        if str(record[record_field]) != str(source.get(source_field)):
            raise ValueError(f"record/source metadata mismatch for {record_field}")
    proprio = record["current_proprio"]
    if not isinstance(proprio, list) or not proprio:
        raise ValueError("current_proprio must be a non-empty list")
    for index, value in enumerate(proprio):
        _require_finite(value, field=f"current_proprio[{index}]")


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed JSON in {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"record in {path}:{line_number} must be an object")
            records.append(value)
    return records


def load_pilot_records(
    records_path: str | Path,
    *,
    expected_count: int = 500,
    expected_base_seed: int = PILOT_BASE_SEED,
    expected_full_steps: int = 10,
    manifest_path: str | Path | None = None,
    expected_manifest_fingerprint: str | None = None,
) -> list[dict[str, Any]]:
    """Strictly load a complete pilot, returning immutable-plan order.

    The sibling ``manifest.json`` is mandatory by default.  Its fingerprint,
    selection digest, and exact source-index coverage are rebound to every row.
    """

    path = Path(records_path)
    if not path.is_file():
        raise FileNotFoundError(f"pilot records file does not exist: {path}")
    manifest_file = Path(manifest_path) if manifest_path is not None else path.parent / "manifest.json"
    if not manifest_file.is_file():
        raise FileNotFoundError(f"pilot manifest does not exist: {manifest_file}")
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed pilot manifest: {manifest_file}") from exc
    if not isinstance(manifest, Mapping):
        raise ValueError("pilot manifest must be a JSON object")

    compatibility = manifest.get("compatibility")
    fingerprint = manifest.get("compatibility_fingerprint")
    if not isinstance(compatibility, Mapping):
        raise ValueError("pilot manifest compatibility must be a mapping")
    if fingerprint != _sha256_json(compatibility):
        raise ValueError("pilot manifest compatibility fingerprint is invalid")
    if expected_manifest_fingerprint is not None and fingerprint != expected_manifest_fingerprint:
        raise ValueError("pilot manifest compatibility fingerprint differs from expected")
    selection = manifest.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("pilot manifest selection must be a mapping")
    plan = selection.get("ordered_selected_source_indices")
    if not isinstance(plan, list):
        raise ValueError("pilot manifest is missing its ordered source-index plan")
    plan = [_require_int(value, field="selected source index") for value in plan]
    if len(plan) != len(set(plan)):
        raise ValueError("pilot manifest selection contains duplicate source indices")
    selection_sha = _sha256_json(plan)
    if selection.get("ordered_selected_source_indices_sha256") != selection_sha:
        raise ValueError("pilot manifest selection SHA-256 is invalid")
    if _require_int(selection.get("num_samples"), field="selection.num_samples") != len(plan):
        raise ValueError("pilot manifest selection count does not match its plan")
    if compatibility.get("selection_sha256") != selection_sha:
        raise ValueError("pilot manifest compatibility is not bound to its selection")

    expected = _require_int(expected_count, field="expected_count", minimum=1)
    if len(plan) != expected:
        raise ValueError(f"pilot manifest contains {len(plan)} states, expected {expected}")
    records = _load_jsonl_records(path)
    if len(records) != expected:
        raise ValueError(f"pilot records contain {len(records)} rows, expected {expected}")

    by_source: dict[int, dict[str, Any]] = {}
    sample_ids: set[str] = set()
    provenance_values: dict[str, set[Any]] = {
        field: set()
        for field in (
            "manifest_compatibility_fingerprint",
            "checkpoint_sha256",
            "dataset_stats_sha256",
            "vae_sha256",
            "git_sha",
        )
    }
    for line_number, record in enumerate(records, start=1):
        try:
            _validate_pilot_record(
                record,
                expected_base_seed=expected_base_seed,
                expected_full_steps=expected_full_steps,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid pilot record at {path}:{line_number}: {exc}") from exc
        sample_id = str(record["sample_id"])
        if sample_id in sample_ids:
            raise ValueError(f"duplicate pilot sample_id {sample_id!r}")
        sample_ids.add(sample_id)
        source_index = int(record["source_metadata"]["requested_sample_idx"])
        if source_index in by_source:
            raise ValueError(f"duplicate pilot source index {source_index}")
        by_source[source_index] = record
        for field in provenance_values:
            provenance_values[field].add(record[field])
    for field, values in provenance_values.items():
        if len(values) != 1:
            raise ValueError(f"pilot records contain mixed {field} values")
    only_fingerprint = next(iter(provenance_values["manifest_compatibility_fingerprint"]))
    if only_fingerprint != fingerprint:
        raise ValueError("pilot record/manifest compatibility fingerprint mismatch")
    if set(by_source) != set(plan):
        missing = sorted(set(plan) - set(by_source))[:10]
        outside = sorted(set(by_source) - set(plan))[:10]
        raise ValueError(
            f"pilot records do not exactly cover manifest selection; missing={missing}, outside={outside}"
        )
    return [by_source[source_index] for source_index in plan]


def _normalized_quotas(
    suite_bin_quotas: Mapping[str, Mapping[str, int]],
) -> dict[str, dict[str, int]]:
    if not isinstance(suite_bin_quotas, Mapping) or not suite_bin_quotas:
        raise ValueError("suite_bin_quotas must be a non-empty mapping")
    result: dict[str, dict[str, int]] = {}
    for suite, raw_bins in suite_bin_quotas.items():
        if suite not in SUITES:
            raise ValueError(f"unknown suite in quotas: {suite!r}")
        if not isinstance(raw_bins, Mapping) or set(raw_bins) != set(UTILITY_BINS):
            raise ValueError(f"suite {suite!r} must define exactly bins {list(UTILITY_BINS)}")
        result[suite] = {
            utility_class: _require_int(
                raw_bins[utility_class],
                field=f"quota[{suite}][{utility_class}]",
            )
            for utility_class in UTILITY_BINS
        }
    return result


def _tie_value(namespace: str, selection_seed: int, identity: str) -> float:
    payload = f"fastwam-stability-v1\0{namespace}\0{selection_seed}\0{identity}".encode()
    # 52 bits are exactly representable as a Python/NumPy double.
    return int.from_bytes(hashlib.sha256(payload).digest()[:7], "big") >> 4


def build_stability_selection(
    records: Sequence[Mapping[str, Any]],
    *,
    selection_seed: int = 42,
    suite_bin_quotas: Mapping[str, Mapping[str, int]] = DEFAULT_SUITE_BIN_QUOTAS,
    partial_target: int = 16,
    full_valid_length: int = FULL_ACTION_VALID_LENGTH,
    task_min: int = 2,
    task_max: int = 3,
    tasks_at_max: int = 20,
    expected_task_count: int = 40,
    num_states: int | None = None,
) -> list[dict[str, Any]]:
    """Select the exact, deterministic 100-state stability audit plan.

    This is a binary mixed-integer feasibility problem.  Constraints are exact
    suite/bin quotas, 2--3 states per task with 20 tasks at each count, no two
    frames from one episode, and exactly 16 partial action chunks.  A SHA256
    objective and order namespace make the chosen feasible plan reproducible.
    """

    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_array
    except ImportError as exc:  # pragma: no cover - exercised only in broken envs
        raise RuntimeError("build_stability_selection requires scipy.optimize.milp") from exc

    quotas = _normalized_quotas(suite_bin_quotas)
    selection_seed = _require_int(selection_seed, field="selection_seed")
    partial_target = _require_int(partial_target, field="partial_target")
    full_valid_length = _require_int(
        full_valid_length, field="full_valid_length", minimum=1
    )
    task_min = _require_int(task_min, field="task_min", minimum=1)
    task_max = _require_int(task_max, field="task_max", minimum=task_min + 1)
    if task_max != task_min + 1:
        raise ValueError("the selector currently requires task_max == task_min + 1")
    tasks_at_max = _require_int(tasks_at_max, field="tasks_at_max")
    expected_task_count = _require_int(
        expected_task_count, field="expected_task_count", minimum=1
    )
    target_count = sum(sum(bin_quotas.values()) for bin_quotas in quotas.values())
    if num_states is not None and int(num_states) != target_count:
        raise ValueError(f"num_states must equal exact quota total {target_count}")
    expected_from_tasks = expected_task_count * task_min + tasks_at_max
    if target_count != expected_from_tasks:
        raise ValueError(
            f"quota total {target_count} conflicts with task total {expected_from_tasks}"
        )
    if partial_target > target_count:
        raise ValueError("partial_target cannot exceed selection size")

    candidates: list[dict[str, Any]] = []
    seen_samples: set[str] = set()
    seen_sources: set[int] = set()
    for index, raw_record in enumerate(records):
        if not isinstance(raw_record, Mapping):
            raise ValueError(f"candidate {index} must be a mapping")
        for field in (
            "sample_id",
            "suite",
            "task_index",
            "episode_index",
            "valid_length",
            "utility",
            "seed",
            "manifest_compatibility_fingerprint",
            "input_hashes",
            "source_metadata",
        ):
            if field not in raw_record:
                raise ValueError(f"candidate {index} is missing {field!r}")
        sample_id = str(raw_record["sample_id"])
        source_index = _require_int(
            raw_record["source_metadata"].get("requested_sample_idx"),
            field="source_metadata.requested_sample_idx",
        )
        if sample_id in seen_samples or source_index in seen_sources:
            raise ValueError("selection candidates contain duplicate sample/source identities")
        seen_samples.add(sample_id)
        seen_sources.add(source_index)
        suite = str(raw_record["suite"])
        if suite not in quotas:
            raise ValueError(f"candidate suite {suite!r} is absent from suite_bin_quotas")
        utility = _require_finite(raw_record["utility"], field="utility")
        utility_class = utility_bin(utility)
        input_hashes = raw_record["input_hashes"]
        if not isinstance(input_hashes, Mapping):
            raise ValueError("candidate input_hashes must be a mapping")
        combined_hash = _require_sha256(
            input_hashes.get("combined"), field="input_hashes.combined"
        )
        candidates.append(
            {
                "record": raw_record,
                "sample_id": sample_id,
                "source_index": source_index,
                "suite": suite,
                "task_key": (
                    suite,
                    _require_int(raw_record["task_index"], field="task_index"),
                ),
                "episode_key": (
                    suite,
                    _require_int(raw_record["episode_index"], field="episode_index"),
                ),
                "selection_bin": utility_class,
                "partial": _require_int(
                    raw_record["valid_length"], field="valid_length", minimum=1
                )
                < full_valid_length,
                "combined_hash": combined_hash,
            }
        )

    # Canonical solver-column ordering is the deterministic feasibility
    # tie-break and makes selection independent of incoming JSONL row order.
    candidates.sort(
        key=lambda candidate: (
            _tie_value("solver-column", selection_seed, candidate["sample_id"]),
            candidate["sample_id"],
        )
    )

    task_keys = sorted({candidate["task_key"] for candidate in candidates})
    if len(task_keys) != expected_task_count:
        raise ValueError(
            f"candidates cover {len(task_keys)} tasks, expected {expected_task_count}"
        )
    task_to_index = {key: index for index, key in enumerate(task_keys)}
    n_candidates = len(candidates)
    n_tasks = len(task_keys)
    n_variables = n_candidates + n_tasks
    objective = np.zeros(n_variables, dtype=np.float64)

    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []

    def add_constraint(coefficients: Mapping[int, float], lb: float, ub: float) -> None:
        row = len(lower)
        for column, value in coefficients.items():
            row_indices.append(row)
            column_indices.append(column)
            values.append(float(value))
        lower.append(float(lb))
        upper.append(float(ub))

    for suite, bin_quotas in quotas.items():
        for utility_class, quota in bin_quotas.items():
            coefficients = {
                index: 1.0
                for index, candidate in enumerate(candidates)
                if candidate["suite"] == suite
                and candidate["selection_bin"] == utility_class
            }
            if len(coefficients) < quota:
                raise ValueError(
                    f"insufficient candidates for {suite}/{utility_class}: "
                    f"available={len(coefficients)}, required={quota}"
                )
            add_constraint(coefficients, quota, quota)

    for task_key, task_index in task_to_index.items():
        coefficients = {
            index: 1.0
            for index, candidate in enumerate(candidates)
            if candidate["task_key"] == task_key
        }
        coefficients[n_candidates + task_index] = -1.0
        add_constraint(coefficients, task_min, task_min)
    add_constraint(
        {n_candidates + index: 1.0 for index in range(n_tasks)},
        tasks_at_max,
        tasks_at_max,
    )

    episode_to_candidates: dict[tuple[str, int], list[int]] = {}
    for index, candidate in enumerate(candidates):
        episode_to_candidates.setdefault(candidate["episode_key"], []).append(index)
    for candidate_indices in episode_to_candidates.values():
        if len(candidate_indices) > 1:
            add_constraint({index: 1.0 for index in candidate_indices}, 0.0, 1.0)
    add_constraint(
        {
            index: 1.0
            for index, candidate in enumerate(candidates)
            if candidate["partial"]
        },
        partial_target,
        partial_target,
    )

    matrix = coo_array(
        (values, (row_indices, column_indices)),
        shape=(len(lower), n_variables),
        dtype=np.float64,
    ).tocsc()
    result = milp(
        c=objective,
        integrality=np.ones(n_variables, dtype=np.uint8),
        bounds=Bounds(np.zeros(n_variables), np.ones(n_variables)),
        constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
        options={"presolve": True},
    )
    if not result.success or result.x is None:
        raise ValueError(
            "stability selection constraints are infeasible: "
            f"status={result.status}, message={result.message}"
        )
    selected_candidates = [
        candidate
        for index, candidate in enumerate(candidates)
        if result.x[index] > 0.5
    ]
    selected_candidates.sort(
        key=lambda candidate: (
            _tie_value("plan-order", selection_seed, candidate["sample_id"]),
            candidate["sample_id"],
        )
    )
    selection: list[dict[str, Any]] = []
    for order, candidate in enumerate(selected_candidates):
        record = candidate["record"]
        selection.append(
            {
                **dict(record),
                "selection_order": order,
                "source_index": candidate["source_index"],
                "sample_id": candidate["sample_id"],
                "suite": candidate["suite"],
                "task_index": int(record["task_index"]),
                "task": str(record["task"]),
                "episode_index": int(record["episode_index"]),
                "frame_index": int(record["frame_index"]),
                "selection_bin": candidate["selection_bin"],
                "pilot_utility": float(record["utility"]),
                "pilot_e0": float(record["e0"]),
                "pilot_efull": float(record["efull"]),
                "pilot_seed": int(record["seed"]),
                "pilot_valid_length": int(record["valid_length"]),
                "pilot_input_combined_sha256": candidate["combined_hash"],
                "pilot_manifest_compatibility_fingerprint": str(
                    record["manifest_compatibility_fingerprint"]
                ),
            }
        )
    validate_stability_selection(
        selection,
        suite_bin_quotas=quotas,
        partial_target=partial_target,
        full_valid_length=full_valid_length,
        task_min=task_min,
        task_max=task_max,
        tasks_at_max=tasks_at_max,
        expected_task_count=expected_task_count,
    )
    return selection


select_stability_states = build_stability_selection


def validate_stability_selection(
    selection: Sequence[Mapping[str, Any]],
    *,
    suite_bin_quotas: Mapping[str, Mapping[str, int]] = DEFAULT_SUITE_BIN_QUOTAS,
    partial_target: int = 16,
    full_valid_length: int = FULL_ACTION_VALID_LENGTH,
    task_min: int = 2,
    task_max: int = 3,
    tasks_at_max: int = 20,
    expected_task_count: int = 40,
) -> None:
    quotas = _normalized_quotas(suite_bin_quotas)
    target_count = sum(sum(values.values()) for values in quotas.values())
    if len(selection) != target_count:
        raise ValueError(f"selection has {len(selection)} states, expected {target_count}")
    orders = [_require_int(row.get("selection_order"), field="selection_order") for row in selection]
    if orders != list(range(target_count)):
        raise ValueError("selection_order must be contiguous and match plan order")
    sample_ids = [str(row.get("sample_id")) for row in selection]
    sources = [_require_int(row.get("source_index"), field="source_index") for row in selection]
    if len(set(sample_ids)) != target_count or len(set(sources)) != target_count:
        raise ValueError("selection must have unique sample and source identities")
    episodes = [(str(row.get("suite")), int(row.get("episode_index"))) for row in selection]
    if len(set(episodes)) != target_count:
        raise ValueError("selection must contain at most one state per episode")
    observed_suite_bins = Counter(
        (str(row.get("suite")), str(row.get("selection_bin"))) for row in selection
    )
    for suite, bins in quotas.items():
        for utility_class, quota in bins.items():
            if observed_suite_bins[(suite, utility_class)] != quota:
                raise ValueError(f"selection quota mismatch for {suite}/{utility_class}")
    for row in selection:
        if utility_bin(float(row["pilot_utility"])) != row["selection_bin"]:
            raise ValueError("selection_bin does not match pilot_utility")
    partial_count = sum(
        int(row.get("pilot_valid_length", row["valid_length"])) < int(full_valid_length)
        for row in selection
    )
    if partial_count != int(partial_target):
        raise ValueError(
            f"selection has {partial_count} partial chunks, expected {partial_target}"
        )
    task_counts = Counter(
        (str(row.get("suite")), int(row.get("task_index"))) for row in selection
    )
    if len(task_counts) != int(expected_task_count):
        raise ValueError("selection does not cover the expected number of tasks")
    if any(count not in (int(task_min), int(task_max)) for count in task_counts.values()):
        raise ValueError("every task must have task_min or task_max selected states")
    if sum(count == int(task_max) for count in task_counts.values()) != int(tasks_at_max):
        raise ValueError("selection has the wrong number of tasks at task_max")


def validate_replicate_base_seeds(
    base_seeds: Sequence[int],
    *,
    expected: Sequence[int] = DEFAULT_REPLICATE_BASE_SEEDS,
) -> tuple[int, ...]:
    normalized = tuple(_require_int(value, field="replicate base seed") for value in base_seeds)
    expected_tuple = tuple(_require_int(value, field="expected replicate base seed") for value in expected)
    if normalized != expected_tuple:
        raise ValueError(
            f"replicate base seeds must be exactly {list(expected_tuple)}, got {list(normalized)}"
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError("replicate base seeds must be unique")
    return normalized


def replicate_id(sample_id: str, replicate_index: int, replicate_base_seed: int) -> str:
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError("sample_id must be a non-empty string")
    index = _require_int(replicate_index, field="replicate_index")
    base_seed = _require_int(replicate_base_seed, field="replicate_base_seed")
    return f"{sample_id}/replicate_{index:02d}_base_seed_{base_seed}"


def derive_replicate_seed(
    pilot_record: Mapping[str, Any], replicate_base_seed: int
) -> int:
    identity = parse_sample_identity(pilot_record)
    return stable_sample_seed(
        _require_int(replicate_base_seed, field="replicate_base_seed"), identity
    )


def build_replicate_plan(
    selection: Sequence[Mapping[str, Any]],
    *,
    base_seeds: Sequence[int] = DEFAULT_REPLICATE_BASE_SEEDS,
) -> list[dict[str, Any]]:
    seeds = validate_replicate_base_seeds(base_seeds)
    result: list[dict[str, Any]] = []
    for selected in selection:
        sample_id = str(selected["sample_id"])
        source_index = _require_int(selected["source_index"], field="source_index")
        selection_order = _require_int(
            selected["selection_order"], field="selection_order"
        )
        # The compact selection has sufficient canonical identity fields for
        # stable_sample_seed; parse_sample_identity also checks suite/task.
        identity_record = {
            "dataset_name": sample_id.split("/", 1)[0],
            "suite": selected["suite"],
            "episode_index": selected["episode_index"],
            "frame_index": selected["frame_index"],
            "task_index": selected["task_index"],
            "task": selected["task"],
            "sample_id": sample_id,
        }
        identity = parse_sample_identity(identity_record)
        for replicate_index, base_seed in enumerate(seeds):
            result.append(
                {
                    "replicate_id": replicate_id(sample_id, replicate_index, base_seed),
                    "replicate_index": replicate_index,
                    "replicate_base_seed": base_seed,
                    "seed": stable_sample_seed(base_seed, identity),
                    "selection_order": selection_order,
                    "source_index": source_index,
                    "sample_id": sample_id,
                    "selection_bin": str(selected["selection_bin"]),
                }
            )
    return result


def augment_stability_record(
    core_record: Mapping[str, Any] | Any,
    *,
    pilot_record: Mapping[str, Any],
    selection_entry: Mapping[str, Any],
    replicate_index: int,
    replicate_base_seed: int,
    stability_manifest_compatibility_fingerprint: str,
    source_pilot_record_sha256: str | None = None,
    collection_git_sha: str | None = None,
) -> dict[str, Any]:
    """Augment one paired inference result into the durable stability schema."""

    if hasattr(core_record, "to_dict"):
        core_record = core_record.to_dict()
    if not isinstance(core_record, Mapping):
        raise TypeError("core_record must be a mapping or provide to_dict()")
    result = dict(core_record)
    for field in (
        "source_metadata",
        "current_proprio",
        "checkpoint_sha256",
        "dataset_stats_sha256",
        "vae_sha256",
        "git_sha",
    ):
        if field not in pilot_record:
            raise ValueError(f"pilot_record is missing {field!r}")
        result[field] = pilot_record[field]
    source_index = _require_int(selection_entry["source_index"], field="source_index")
    index = _require_int(replicate_index, field="replicate_index")
    base_seed = _require_int(replicate_base_seed, field="replicate_base_seed")
    expected_source_hash = pilot_record_sha256(pilot_record)
    if source_pilot_record_sha256 is None:
        source_pilot_record_sha256 = expected_source_hash
    source_pilot_record_sha256 = _require_sha256(
        source_pilot_record_sha256, field="source_pilot_record_sha256"
    )
    if source_pilot_record_sha256 != expected_source_hash:
        raise ValueError("source_pilot_record_sha256 does not match pilot_record")
    reused_from_pilot = base_seed == PILOT_BASE_SEED
    if collection_git_sha is None:
        collection_git_sha = str(pilot_record["git_sha"])
    result.update(
        {
            "stability_record_schema_version": STABILITY_RECORD_SCHEMA_VERSION,
            "replicate_id": replicate_id(str(pilot_record["sample_id"]), index, base_seed),
            "replicate_index": index,
            "replicate_base_seed": base_seed,
            "replicate_seed": int(result["seed"]),
            "source_index": source_index,
            "selection_order": int(selection_entry["selection_order"]),
            "selection_bin": str(selection_entry["selection_bin"]),
            "pilot_base_seed": PILOT_BASE_SEED,
            "pilot_seed": int(pilot_record["seed"]),
            "pilot_e0": float(pilot_record["e0"]),
            "pilot_efull": float(pilot_record["efull"]),
            "pilot_utility": float(pilot_record["utility"]),
            "pilot_valid_length": int(pilot_record["valid_length"]),
            "pilot_input_combined_sha256": str(pilot_record["input_hashes"]["combined"]),
            "pilot_manifest_compatibility_fingerprint": str(
                pilot_record["manifest_compatibility_fingerprint"]
            ),
            "stability_manifest_compatibility_fingerprint": str(
                stability_manifest_compatibility_fingerprint
            ),
            "source_pilot_record_sha256": source_pilot_record_sha256,
            "reused_from_pilot": reused_from_pilot,
            "inference_origin": "pilot_reuse" if reused_from_pilot else "new_inference",
            "collection_git_sha": collection_git_sha,
        }
    )
    validate_stability_record(result)
    return result


build_stability_record = augment_stability_record


def validate_stability_record(
    record: Mapping[str, Any],
    *,
    expected_full_steps: int = 10,
    expected_base_seeds: Sequence[int] = DEFAULT_REPLICATE_BASE_SEEDS,
    expected_stability_manifest_fingerprint: str | None = None,
    expected_pilot_manifest_fingerprint: str | None = None,
    expected_checkpoint_sha256: str | None = None,
    expected_dataset_stats_sha256: str | None = None,
    expected_vae_sha256: str | None = None,
    expected_git_sha: str | None = None,
) -> None:
    if not isinstance(record, Mapping):
        raise TypeError("stability record must be a mapping")
    _validate_core_utility_record(record, full_steps=expected_full_steps)
    required = {
        "stability_record_schema_version",
        "replicate_id",
        "replicate_index",
        "replicate_base_seed",
        "replicate_seed",
        "source_index",
        "selection_order",
        "selection_bin",
        "pilot_base_seed",
        "pilot_seed",
        "pilot_e0",
        "pilot_efull",
        "pilot_utility",
        "pilot_valid_length",
        "pilot_input_combined_sha256",
        "pilot_manifest_compatibility_fingerprint",
        "stability_manifest_compatibility_fingerprint",
        "source_pilot_record_sha256",
        "reused_from_pilot",
        "inference_origin",
        "collection_git_sha",
        "source_metadata",
        "current_proprio",
        "checkpoint_sha256",
        "dataset_stats_sha256",
        "vae_sha256",
        "git_sha",
    }
    missing = required - set(record)
    if missing:
        raise ValueError(f"stability record is missing fields: {sorted(missing)}")
    if int(record["stability_record_schema_version"]) != STABILITY_RECORD_SCHEMA_VERSION:
        raise ValueError("unsupported stability_record_schema_version")
    seeds = validate_replicate_base_seeds(expected_base_seeds)
    index = _require_int(record["replicate_index"], field="replicate_index")
    if index >= len(seeds):
        raise ValueError("replicate_index is outside the configured seed grid")
    base_seed = _require_int(record["replicate_base_seed"], field="replicate_base_seed")
    if base_seed != seeds[index]:
        raise ValueError("replicate_index/base-seed mapping is invalid")
    expected_id = replicate_id(str(record["sample_id"]), index, base_seed)
    if record["replicate_id"] != expected_id:
        raise ValueError("replicate_id does not match sample/index/base seed")
    identity = parse_sample_identity(record)
    expected_seed = stable_sample_seed(base_seed, identity)
    if int(record["seed"]) != expected_seed or int(record["replicate_seed"]) != expected_seed:
        raise ValueError("replicate seed is not the stable sample seed")
    if int(record["pilot_base_seed"]) != PILOT_BASE_SEED:
        raise ValueError("pilot_base_seed must equal 42")
    expected_pilot_seed = stable_sample_seed(PILOT_BASE_SEED, identity)
    if int(record["pilot_seed"]) != expected_pilot_seed:
        raise ValueError("pilot_seed is not the base-42 stable sample seed")
    source_pilot_record_hash = _require_sha256(
        record["source_pilot_record_sha256"], field="source_pilot_record_sha256"
    )
    if not isinstance(record["reused_from_pilot"], bool):
        raise ValueError("reused_from_pilot must be bool")
    expected_reuse = base_seed == PILOT_BASE_SEED
    expected_origin = "pilot_reuse" if expected_reuse else "new_inference"
    if record["reused_from_pilot"] is not expected_reuse:
        raise ValueError("reused_from_pilot must be true exactly for base seed 42")
    if record["inference_origin"] != expected_origin:
        raise ValueError(f"inference_origin must equal {expected_origin!r}")
    collection_git_sha = record["collection_git_sha"]
    if (
        not isinstance(collection_git_sha, str)
        or len(collection_git_sha) != 40
        or any(character not in "0123456789abcdef" for character in collection_git_sha)
    ):
        raise ValueError("collection_git_sha must be a lowercase 40-character hex digest")

    pilot_e0 = _require_finite(record["pilot_e0"], field="pilot_e0", minimum=0.0)
    pilot_efull = _require_finite(
        record["pilot_efull"], field="pilot_efull", minimum=0.0
    )
    pilot_utility = _require_finite(record["pilot_utility"], field="pilot_utility")
    if not math.isclose(
        pilot_utility, pilot_e0 - pilot_efull, rel_tol=1e-6, abs_tol=1e-8
    ):
        raise ValueError("pilot utility does not equal pilot E0-Efull")
    if record["selection_bin"] not in UTILITY_BINS:
        raise ValueError("selection_bin is unknown")
    if utility_bin(pilot_utility) != record["selection_bin"]:
        raise ValueError("selection_bin does not match pilot_utility")
    pilot_valid_length = _require_int(
        record["pilot_valid_length"], field="pilot_valid_length", minimum=1
    )
    if pilot_valid_length != int(record["valid_length"]):
        raise ValueError("replicate valid_length differs from the pilot")
    pilot_hash = _require_sha256(
        record["pilot_input_combined_sha256"],
        field="pilot_input_combined_sha256",
    )
    if pilot_hash != record["input_hashes"]["combined"]:
        raise ValueError("replicate state input hash differs from the pilot")
    if base_seed == PILOT_BASE_SEED:
        for current, pilot, field in (
            (record["e0"], pilot_e0, "e0"),
            (record["efull"], pilot_efull, "efull"),
            (record["utility"], pilot_utility, "utility"),
        ):
            if not math.isclose(float(current), float(pilot), rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"base-42 replicate {field} does not exactly reproduce pilot")

    source = record["source_metadata"]
    if not isinstance(source, Mapping):
        raise ValueError("source_metadata must be a mapping")
    source_index = _require_int(record["source_index"], field="source_index")
    if source_index != int(source.get("requested_sample_idx", -1)):
        raise ValueError("source_index differs from requested_sample_idx")
    if source_index != int(source.get("source_sample_idx", -1)):
        raise ValueError("source_index differs from source_sample_idx")

    pilot_fingerprint = _require_sha256(
        record["pilot_manifest_compatibility_fingerprint"],
        field="pilot_manifest_compatibility_fingerprint",
    )
    stability_fingerprint = _require_sha256(
        record["stability_manifest_compatibility_fingerprint"],
        field="stability_manifest_compatibility_fingerprint",
    )
    if (
        expected_pilot_manifest_fingerprint is not None
        and pilot_fingerprint != expected_pilot_manifest_fingerprint
    ):
        raise ValueError("pilot manifest compatibility fingerprint mismatch")
    if (
        expected_stability_manifest_fingerprint is not None
        and stability_fingerprint != expected_stability_manifest_fingerprint
    ):
        raise ValueError("stability manifest compatibility fingerprint mismatch")
    for field, expected in (
        ("checkpoint_sha256", expected_checkpoint_sha256),
        ("dataset_stats_sha256", expected_dataset_stats_sha256),
        ("vae_sha256", expected_vae_sha256),
        ("git_sha", expected_git_sha),
    ):
        if expected is not None and record[field] != expected:
            raise ValueError(f"{field} mismatch")


def load_stability_record_index(
    records_path: str | Path,
    *,
    expected_full_steps: int = 10,
    expected_base_seeds: Sequence[int] = DEFAULT_REPLICATE_BASE_SEEDS,
    expected_stability_manifest_fingerprint: str | None = None,
    expected_pilot_manifest_fingerprint: str | None = None,
    expected_checkpoint_sha256: str | None = None,
    expected_dataset_stats_sha256: str | None = None,
    expected_vae_sha256: str | None = None,
    expected_git_sha: str | None = None,
    expected_manifest_fingerprint: str | None = None,
) -> dict[tuple[int, int], dict[str, Any]]:
    if expected_manifest_fingerprint is not None:
        if (
            expected_stability_manifest_fingerprint is not None
            and expected_stability_manifest_fingerprint != expected_manifest_fingerprint
        ):
            raise ValueError("conflicting expected stability manifest fingerprints")
        expected_stability_manifest_fingerprint = expected_manifest_fingerprint
    path = Path(records_path)
    if not path.exists():
        return {}
    index: dict[tuple[int, int], dict[str, Any]] = {}
    replicate_ids: set[str] = set()
    for line_number, record in enumerate(_load_jsonl_records(path), start=1):
        try:
            validate_stability_record(
                record,
                expected_full_steps=expected_full_steps,
                expected_base_seeds=expected_base_seeds,
                expected_stability_manifest_fingerprint=(
                    expected_stability_manifest_fingerprint
                ),
                expected_pilot_manifest_fingerprint=expected_pilot_manifest_fingerprint,
                expected_checkpoint_sha256=expected_checkpoint_sha256,
                expected_dataset_stats_sha256=expected_dataset_stats_sha256,
                expected_vae_sha256=expected_vae_sha256,
                expected_git_sha=expected_git_sha,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid stability record at {path}:{line_number}: {exc}") from exc
        key = (int(record["source_index"]), int(record["replicate_index"]))
        if key in index:
            raise ValueError(f"duplicate stability resume key {key}")
        if record["replicate_id"] in replicate_ids:
            raise ValueError(f"duplicate replicate_id {record['replicate_id']!r}")
        index[key] = record
        replicate_ids.add(str(record["replicate_id"]))
    return index


load_stability_resume_index = load_stability_record_index


def _validate_grid_row_against_pilot_selection(
    record: Mapping[str, Any],
    selected: Mapping[str, Any],
) -> None:
    """Rebind one durable row to the immutable, real Pilot source row.

    Per-row schema validation only proves that a record is self-consistent. A
    malicious or stale row can still be internally valid while describing a
    different state or Pilot label. Resume therefore also binds every present
    cell to the selected Pilot row, even when the overall grid is incomplete.
    """

    source_index = _require_int(selected.get("source_index"), field="selection.source_index")
    if _require_int(record.get("source_index"), field="record.source_index") != source_index:
        raise ValueError(f"source {source_index} selection rebind mismatch for source_index")

    selected_source_metadata = selected.get("source_metadata")
    if not isinstance(selected_source_metadata, Mapping):
        raise ValueError(f"selection source {source_index} is missing source_metadata")
    selected_input_hashes = selected.get("input_hashes")
    if not isinstance(selected_input_hashes, Mapping):
        raise ValueError(f"selection source {source_index} is missing input_hashes")

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
    )
    expected_fields: dict[str, Any] = {field: selected.get(field) for field in direct_fields}
    expected_fields.update(
        {
            "selection_order": selected.get("selection_order"),
            "selection_bin": selected.get("selection_bin"),
            "pilot_e0": selected.get("e0"),
            "pilot_efull": selected.get("efull"),
            "pilot_utility": selected.get("utility"),
            "pilot_seed": selected.get("seed"),
            "pilot_valid_length": selected.get("valid_length"),
            "pilot_input_combined_sha256": selected_input_hashes.get("combined"),
            "pilot_manifest_compatibility_fingerprint": selected.get(
                "manifest_compatibility_fingerprint"
            ),
            "source_pilot_record_sha256": pilot_record_sha256(selected),
        }
    )
    for field, expected_value in expected_fields.items():
        if expected_value is None:
            raise ValueError(
                f"selection source {source_index} is missing required Pilot field {field}"
            )
        if _canonical_json(record.get(field)) != _canonical_json(expected_value):
            raise ValueError(
                f"source {source_index} selection rebind mismatch for {field}"
            )

    # ``git_sha`` identifies the code that generated a utility measurement and
    # can legitimately differ for new-inference rows. The reused base-42 row,
    # however, must retain the exact Pilot utility provenance.
    if bool(record.get("reused_from_pilot")):
        if _canonical_json(record.get("git_sha")) != _canonical_json(selected.get("git_sha")):
            raise ValueError(
                f"source {source_index} selection rebind mismatch for reused git_sha"
            )


def validate_complete_grid(
    record_index: Mapping[tuple[int, int], Mapping[str, Any]],
    selection: Sequence[Mapping[str, Any]],
    *,
    base_seeds: Sequence[int] = DEFAULT_REPLICATE_BASE_SEEDS,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    seeds = validate_replicate_base_seeds(base_seeds)
    selection_by_source: dict[int, Mapping[str, Any]] = {}
    selection_orders: set[int] = set()
    for selected in selection:
        source_index = _require_int(selected.get("source_index"), field="selection.source_index")
        if source_index in selection_by_source:
            raise ValueError(f"duplicate selection source_index={source_index}")
        selection_order = _require_int(
            selected.get("selection_order"), field="selection.selection_order"
        )
        if selection_order in selection_orders:
            raise ValueError(f"duplicate selection_order={selection_order}")
        selection_by_source[source_index] = selected
        selection_orders.add(selection_order)
    if selection_orders != set(range(len(selection))):
        raise ValueError("selection_order must be exactly 0..len(selection)-1")

    expected = {
        (source_index, replicate_index)
        for source_index in selection_by_source
        for replicate_index in range(len(seeds))
    }
    actual = set(record_index)
    outside = sorted(actual - expected)
    missing = sorted(expected - actual)
    if outside:
        raise ValueError(f"stability records contain keys outside the plan: {outside[:10]}")
    records_by_source: dict[int, list[Mapping[str, Any]]] = {}
    for (source_index, replicate_index), record in record_index.items():
        # Rebind before reporting incompleteness. This ensures a corrupt
        # completed cell can never hide behind a partially collected grid.
        _validate_grid_row_against_pilot_selection(
            record, selection_by_source[int(source_index)]
        )
        if int(record.get("replicate_index", -1)) != int(replicate_index):
            raise ValueError(
                f"source {source_index} selection rebind mismatch for replicate_index"
            )
        records_by_source.setdefault(int(source_index), []).append(record)
    if missing and not allow_incomplete:
        raise ValueError(f"stability grid is incomplete; missing keys: {missing[:10]}")
    cross_replicate_fields = (
        "sample_id", "dataset_id", "dataset_name", "suite",
        "episode_index", "frame_index", "task_index", "task", "valid_length",
        "target_action_shape", "input_hashes", "current_proprio", "source_metadata",
        "selection_order", "selection_bin", "pilot_seed", "pilot_e0", "pilot_efull",
        "pilot_utility", "pilot_valid_length", "pilot_input_combined_sha256",
        "pilot_manifest_compatibility_fingerprint", "source_pilot_record_sha256",
        "stability_manifest_compatibility_fingerprint", "checkpoint_sha256",
        "dataset_stats_sha256", "vae_sha256", "collection_git_sha",
    )
    for source_index, rows in records_by_source.items():
        reference = rows[0]
        for row in rows[1:]:
            for field in cross_replicate_fields:
                if _canonical_json(row.get(field)) != _canonical_json(reference.get(field)):
                    raise ValueError(
                        f"source {source_index} has cross-replicate mismatch for {field}"
                    )
    reused_count = sum(
        bool(record.get("reused_from_pilot")) for record in record_index.values()
    )
    new_inference_count = sum(
        record.get("inference_origin") == "new_inference"
        for record in record_index.values()
    )
    expected_reused = len(selection)
    expected_new = len(selection) * (len(seeds) - 1)
    if not missing and (reused_count != expected_reused or new_inference_count != expected_new):
        raise ValueError(
            "complete grid reuse/new-inference counts are invalid: "
            f"reused={reused_count}/{expected_reused}, new={new_inference_count}/{expected_new}"
        )
    return {
        "expected_count": len(expected),
        "completed_count": len(actual),
        "missing_count": len(missing),
        "is_complete": not missing,
        "coverage_fraction": len(actual) / len(expected) if expected else 1.0,
        "missing_key_examples": [list(key) for key in missing[:10]],
        "reused_from_pilot_count": reused_count,
        "new_inference_count": new_inference_count,
    }


__all__ = [
    "DEFAULT_REPLICATE_BASE_SEEDS",
    "DEFAULT_SUITE_BIN_QUOTAS",
    "FULL_ACTION_VALID_LENGTH",
    "NEAR_ZERO_UTILITY_THRESHOLD",
    "PILOT_BASE_SEED",
    "STABILITY_RECORD_SCHEMA_VERSION",
    "STRONG_UTILITY_THRESHOLD",
    "SUITES",
    "UTILITY_BINS",
    "augment_stability_record",
    "build_replicate_plan",
    "build_stability_record",
    "build_stability_selection",
    "derive_replicate_seed",
    "load_pilot_records",
    "load_stability_record_index",
    "load_stability_resume_index",
    "pilot_record_sha256",
    "replicate_id",
    "select_stability_states",
    "utility_bin",
    "validate_complete_grid",
    "validate_replicate_base_seeds",
    "validate_stability_record",
    "validate_stability_selection",
]
