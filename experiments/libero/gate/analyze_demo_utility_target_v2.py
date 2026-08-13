"""Independent validation of the LIBERO multi-seed Utility Target V2.

This analysis is deliberately fail-closed and its readiness thresholds were
pre-registered before seeds 47--50 were collected or inspected.  Seeds 42--46
form the frozen five-seed target; seeds 47--50 are an independent validation
group.  A ``GO`` only permits the next *offline* Tiny-MLP training stage.  It
does not establish Gate calibration, compute savings, or closed-loop success.

Pre-registered primary definitions
----------------------------------

* Primary comparison: target-five mean versus independent-four mean.
* Primary deadband: ``epsilon = 1e-4``.
* A target is high-confidence iff its five-seed mean lies outside the
  deadband, at least four of five seeds have that same deadband direction, and
  its two-sided 95% t interval lies wholly beyond ``+epsilon`` or
  ``-epsilon`` in that direction.
* State bootstrap: 2,000 resamples, stratified by the frozen Pilot selection
  bin, with seed 20260813.

All GO thresholds and all CONDITIONAL floors below were locked on 2026-08-13
before inspecting any independent-validation value.  They must not be tuned in
response to seeds 47--50.
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
from typing import Any, Callable, Final, Iterable, Mapping, Sequence

import numpy as np
from scipy import stats

from experiments.libero.gate import (
    collect_demo_utility_target_v2_validation as validation_collector,
)
from experiments.libero.gate import demo_utility_target_v2 as target_core


ANALYSIS_SCHEMA_VERSION: Final = 1
TARGET_BASE_SEEDS: Final = (42, 43, 44, 45, 46)
VALIDATION_BASE_SEEDS: Final = (47, 48, 49, 50)
ALL_BASE_SEEDS: Final = TARGET_BASE_SEEDS + VALIDATION_BASE_SEEDS
EXPECTED_STATE_COUNT: Final = 100
PRIMARY_EPSILON: Final = 1e-4
DEADBAND_EPSILONS: Final = (1e-5, PRIMARY_EPSILON, 1e-3)
TAIL_FRACTION: Final = 0.20
BOOTSTRAP_SEED: Final = 20260813
BOOTSTRAP_REPLICATES: Final = 2000
PREREGISTERED_AT_UTC: Final = "2026-08-13T12:46:00Z"

# A mapping is used rather than inlining literals into decision code so the
# emitted JSON carries the exact, inspectable preregistration contract.
GO_THRESHOLDS: Final = {
    "mean_spearman": 0.50,
    "mean_spearman_bootstrap_lower_95": 0.30,
    "mean_kendall": 0.35,
    "mean_kendall_bootstrap_lower_95": 0.20,
    "actionable_sign_retention": 0.75,
    "high_confidence_sign_retention": 0.80,
    "high_confidence_positive_sign_retention": 0.70,
    "high_confidence_negative_sign_retention": 0.70,
    "high_confidence_min_count_per_direction": 10,
    "top_recall": 0.40,
    "top_jaccard": 0.25,
    "bottom_recall": 0.40,
    "bottom_jaccard": 0.25,
    "lin_ccc": 0.50,
    "absolute_agreement_icc_a1": 0.50,
    "all9_icc_1_9": 0.75,
    "high_confidence_min_count": 20,
    "high_confidence_weighted_coverage": 0.20,
    "median_spearman": 0.40,
    "median_actionable_sign_retention": 0.70,
}

CONDITIONAL_THRESHOLDS: Final = {
    "mean_spearman": 0.30,
    "mean_kendall": 0.20,
    "lin_ccc": 0.30,
    "absolute_agreement_icc_a1": 0.35,
    "actionable_sign_retention": 0.65,
    "high_confidence_sign_retention": 0.70,
    "top_recall": 0.35,
    "bottom_recall": 0.35,
    "all9_icc_1_9": 0.65,
    "high_confidence_min_count": 10,
    "high_confidence_weighted_coverage": 0.10,
    "median_spearman": 0.30,
}

STRATUM_PREVALENCE: Final = {
    "SP": 0.152,
    "SN": 0.202,
    "MP": 0.262,
    "MN": 0.184,
    "NZ": 0.200,
}

LOGGER = logging.getLogger(__name__)
_HEX64 = set("0123456789abcdef")
VALIDATION_MANIFEST_KIND = "libero_demo_utility_target_v2_independent_validation"


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
        value = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Malformed or non-finite JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object in {path}")
    _assert_finite_tree(value, path="json")
    return value


def _load_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, parse_constant=_reject_json_constant)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"Malformed {label} JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            _assert_finite_tree(row, path=f"{label}[{line_number}]")
            rows.append(row)
    if not rows:
        raise ValueError(f"No {label} rows found in {path}")
    return rows


def _assert_finite_tree(value: Any, *, path: str) -> None:
    if value is None or isinstance(value, (bool, str, int)):
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


def _required_float(mapping: Mapping[str, Any], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be finite numeric, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be finite numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{key} must be finite, got {result!r}")
    return result


def _required_string(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string, got {value!r}")
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX64 for character in value)
    )


def deadband_sign(value: float, epsilon: float = PRIMARY_EPSILON) -> int:
    value = float(value)
    epsilon = float(epsilon)
    if not math.isfinite(value) or not math.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("value must be finite and epsilon finite/non-negative")
    if value > epsilon:
        return 1
    if value < -epsilon:
        return -1
    return 0


def _epsilon_label(epsilon: float) -> str:
    return f"eps_{epsilon:.0e}".replace("-", "m").replace("+", "p")


def _safe_fraction(mask: Sequence[bool]) -> float | None:
    values = np.asarray(mask, dtype=bool)
    return float(values.mean()) if values.size else None


def _finite_median(values: Iterable[Any]) -> float | None:
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return float(np.median(finite)) if finite else None


def population_state_weights(strata: Sequence[str]) -> np.ndarray:
    counts = Counter(str(stratum) for stratum in strata)
    missing = sorted(set(STRATUM_PREVALENCE) - set(counts))
    unknown = sorted(set(counts) - set(STRATUM_PREVALENCE))
    if missing or unknown:
        raise ValueError(f"Cannot post-stratify: missing={missing}, unknown={unknown}")
    weights = np.asarray(
        [STRATUM_PREVALENCE[str(stratum)] / counts[str(stratum)] for stratum in strata],
        dtype=np.float64,
    )
    weights /= weights.sum()
    return weights


def weighted_fraction(mask: Sequence[bool], weights: Sequence[float]) -> float | None:
    mask_array = np.asarray(mask, dtype=bool)
    weight_array = np.asarray(weights, dtype=np.float64)
    if mask_array.shape != weight_array.shape or mask_array.ndim != 1:
        raise ValueError("mask and weights must be equal one-dimensional arrays")
    if not np.isfinite(weight_array).all() or (weight_array < 0.0).any():
        raise ValueError("weights must be finite and non-negative")
    denominator = float(weight_array.sum())
    if mask_array.size == 0 or denominator <= 0.0:
        return None
    return float(weight_array[mask_array].sum() / denominator)


def _rank_pair(x: Sequence[float], y: Sequence[float]) -> dict[str, float | None]:
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if x_array.shape != y_array.shape or x_array.ndim != 1 or x_array.size < 2:
        raise ValueError("Rank correlation requires equal 1D arrays with n>=2")
    spearman = stats.spearmanr(x_array, y_array).statistic
    kendall = stats.kendalltau(x_array, y_array).statistic
    return {
        "spearman_rho": float(spearman) if np.isfinite(spearman) else None,
        "kendall_tau": float(kendall) if np.isfinite(kendall) else None,
    }


def ranking_overlap(
    reference: Sequence[float],
    comparison: Sequence[float],
    *,
    fraction: float = TAIL_FRACTION,
) -> dict[str, Any]:
    reference_array = np.asarray(reference, dtype=np.float64)
    comparison_array = np.asarray(comparison, dtype=np.float64)
    if reference_array.shape != comparison_array.shape or reference_array.ndim != 1:
        raise ValueError("ranking_overlap requires equal one-dimensional arrays")
    n = int(reference_array.size)
    if n < 2 or not 0.0 < fraction < 0.5:
        raise ValueError("ranking_overlap requires n>=2 and 0<fraction<0.5")
    k = max(1, int(math.ceil(n * fraction)))

    def selected(values: np.ndarray, *, top: bool) -> set[int]:
        order = np.lexsort((np.arange(n), -values if top else values))
        return set(int(index) for index in order[:k])

    result: dict[str, Any] = {"state_count": n, "fraction": fraction, "k": k}
    for side, top in (("top", True), ("bottom", False)):
        left = selected(reference_array, top=top)
        right = selected(comparison_array, top=top)
        intersection = len(left & right)
        union = len(left | right)
        result[f"{side}_intersection"] = intersection
        result[f"{side}_recall"] = float(intersection / k)
        result[f"{side}_jaccard"] = float(intersection / union)
    result["random_expected_recall"] = float(fraction)
    result["random_expected_jaccard"] = float(fraction / (2.0 - fraction))
    return result


def lin_concordance_correlation(
    reference: Sequence[float], comparison: Sequence[float]
) -> float | None:
    """Lin's population-moment concordance correlation coefficient."""

    x = np.asarray(reference, dtype=np.float64)
    y = np.asarray(comparison, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or x.size < 2:
        raise ValueError("CCC requires equal one-dimensional arrays with n>=2")
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    covariance = float(np.mean(x_centered * y_centered))
    denominator = float(
        np.mean(x_centered**2) + np.mean(y_centered**2) + (x.mean() - y.mean()) ** 2
    )
    if denominator <= 0.0 or not math.isfinite(denominator):
        return None
    value = 2.0 * covariance / denominator
    return float(value) if math.isfinite(value) else None


def icc_absolute_agreement(values: np.ndarray) -> dict[str, float | None]:
    """Two-way absolute-agreement ICC(A,1) and ICC(A,k)."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise ValueError("ICC(A,*) requires an n>=2 by k>=2 matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("ICC values must be finite")
    n, k = matrix.shape
    grand = float(matrix.mean())
    row_means = matrix.mean(axis=1)
    column_means = matrix.mean(axis=0)
    ss_rows = float(k * np.sum((row_means - grand) ** 2))
    ss_columns = float(n * np.sum((column_means - grand) ** 2))
    residual = matrix - row_means[:, None] - column_means[None, :] + grand
    ss_error = float(np.sum(residual**2))
    ms_rows = ss_rows / (n - 1)
    ms_columns = ss_columns / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    numerator = ms_rows - ms_error
    denominator_1 = ms_rows + (k - 1) * ms_error + k * (ms_columns - ms_error) / n
    denominator_k = ms_rows + (ms_columns - ms_error) / n

    def ratio(denominator: float) -> float | None:
        if denominator == 0.0 or not math.isfinite(denominator):
            return None
        value = numerator / denominator
        return float(value) if math.isfinite(value) else None

    return {
        "n_states": int(n),
        "n_measurements": int(k),
        "ms_rows": float(ms_rows),
        "ms_columns": float(ms_columns),
        "ms_error": float(ms_error),
        "icc_a_1": ratio(denominator_1),
        "icc_a_k": ratio(denominator_k),
    }


def icc_one_way(values: np.ndarray) -> dict[str, float | None]:
    """One-way random-effects ICC(1,1) and ICC(1,k)."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise ValueError("ICC(1,*) requires an n>=2 by k>=2 matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("ICC values must be finite")
    n, k = matrix.shape
    row_means = matrix.mean(axis=1)
    grand = float(matrix.mean())
    ms_between = float(k * np.sum((row_means - grand) ** 2) / (n - 1))
    ms_within = float(np.sum((matrix - row_means[:, None]) ** 2) / (n * (k - 1)))
    numerator = ms_between - ms_within
    denominator_single = ms_between + (k - 1) * ms_within

    def ratio(numerator_value: float, denominator: float) -> float | None:
        if denominator == 0.0 or not math.isfinite(denominator):
            return None
        value = numerator_value / denominator
        return float(value) if math.isfinite(value) else None

    return {
        "n_states": int(n),
        "n_seeds": int(k),
        "ms_between_states": ms_between,
        "ms_within_states": ms_within,
        "icc_1_1": ratio(numerator, denominator_single),
        "icc_1_k": ratio(numerator, ms_between),
    }


def _stratified_bootstrap_indices(
    strata: Sequence[str], *, seed: int, replicates: int
) -> list[np.ndarray]:
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    labels = np.asarray([str(value) for value in strata], dtype=object)
    if labels.ndim != 1 or labels.size < 2:
        raise ValueError("bootstrap requires at least two state rows")
    groups = [np.flatnonzero(labels == label) for label in sorted(set(labels))]
    if any(group.size == 0 for group in groups):
        raise ValueError("bootstrap strata cannot be empty")
    rng = np.random.default_rng(seed)
    return [
        np.concatenate([rng.choice(group, size=group.size, replace=True) for group in groups])
        for _ in range(replicates)
    ]


def _bootstrap_ci(
    indices: Sequence[np.ndarray], statistic: Callable[[np.ndarray], float | None]
) -> dict[str, Any]:
    samples: list[float] = []
    for sampled in indices:
        value = statistic(sampled)
        if value is not None and math.isfinite(float(value)):
            samples.append(float(value))
    if not samples:
        return {
            "lower_95": None,
            "upper_95": None,
            "valid_replicates": 0,
            "requested_replicates": len(indices),
        }
    lower, upper = np.quantile(np.asarray(samples, dtype=np.float64), (0.025, 0.975))
    return {
        "lower_95": float(lower),
        "upper_95": float(upper),
        "valid_replicates": len(samples),
        "requested_replicates": len(indices),
    }


def _metric_at_least(name: str, value: Any, threshold: float) -> dict[str, Any]:
    numeric = None
    if not isinstance(value, bool):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = None
    passed = numeric is not None and math.isfinite(numeric) and numeric >= threshold
    return {
        "name": name,
        "observed": value,
        "operator": ">=",
        "threshold": threshold,
        "passed": bool(passed),
    }


def decide_readiness(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the frozen GO/CONDITIONAL/NO_GO contract without tuning."""

    rank = metrics["rank_agreement"]
    sign = metrics["sign_retention"]
    overlap = metrics["ranking_overlap"]
    agreement = metrics["absolute_agreement"]
    reliability = metrics["all9_reliability"]
    coverage = metrics["high_confidence_coverage"]
    bootstrap = metrics["bootstrap"]
    median = metrics["median_guardrail"]

    go_values = {
        "mean_spearman": rank["mean_vs_mean"]["spearman_rho"],
        "mean_spearman_bootstrap_lower_95": bootstrap["mean_spearman"]["lower_95"],
        "mean_kendall": rank["mean_vs_mean"]["kendall_tau"],
        "mean_kendall_bootstrap_lower_95": bootstrap["mean_kendall"]["lower_95"],
        "actionable_sign_retention": sign["actionable"]["retention"],
        "high_confidence_sign_retention": sign["high_confidence"]["retention"],
        "high_confidence_positive_sign_retention": sign["high_confidence_positive"]["retention"],
        "high_confidence_negative_sign_retention": sign["high_confidence_negative"]["retention"],
        "high_confidence_min_count_per_direction_positive": sign["high_confidence_positive"]["state_count"],
        "high_confidence_min_count_per_direction_negative": sign["high_confidence_negative"]["state_count"],
        "top_recall": overlap["top_recall"],
        "top_jaccard": overlap["top_jaccard"],
        "bottom_recall": overlap["bottom_recall"],
        "bottom_jaccard": overlap["bottom_jaccard"],
        "lin_ccc": agreement["lin_ccc"],
        "absolute_agreement_icc_a1": agreement["icc_a_1"],
        "all9_icc_1_9": reliability["icc_1_k"],
        "high_confidence_min_count": coverage["state_count"],
        "high_confidence_weighted_coverage": coverage["population_weighted_fraction"],
        "median_spearman": median["rank"]["spearman_rho"],
        "median_actionable_sign_retention": median["actionable_sign_retention"],
    }
    go_checks: list[dict[str, Any]] = []
    for key, threshold in GO_THRESHOLDS.items():
        if key == "high_confidence_min_count_per_direction":
            go_checks.append(
                _metric_at_least(
                    "high_confidence_min_count_per_direction_positive",
                    go_values["high_confidence_min_count_per_direction_positive"],
                    threshold,
                )
            )
            go_checks.append(
                _metric_at_least(
                    "high_confidence_min_count_per_direction_negative",
                    go_values["high_confidence_min_count_per_direction_negative"],
                    threshold,
                )
            )
        else:
            go_checks.append(_metric_at_least(key, go_values[key], threshold))

    conditional_values = {
        "mean_spearman": rank["mean_vs_mean"]["spearman_rho"],
        "mean_kendall": rank["mean_vs_mean"]["kendall_tau"],
        "lin_ccc": agreement["lin_ccc"],
        "absolute_agreement_icc_a1": agreement["icc_a_1"],
        "actionable_sign_retention": sign["actionable"]["retention"],
        "high_confidence_sign_retention": sign["high_confidence"]["retention"],
        "top_recall": overlap["top_recall"],
        "bottom_recall": overlap["bottom_recall"],
        "all9_icc_1_9": reliability["icc_1_k"],
        "high_confidence_min_count": coverage["state_count"],
        "high_confidence_weighted_coverage": coverage["population_weighted_fraction"],
        "median_spearman": median["rank"]["spearman_rho"],
    }
    conditional_checks = [
        _metric_at_least(key, conditional_values[key], threshold)
        for key, threshold in CONDITIONAL_THRESHOLDS.items()
    ]
    if all(check["passed"] for check in go_checks):
        decision = "GO"
    elif all(check["passed"] for check in conditional_checks):
        decision = "CONDITIONAL"
    else:
        decision = "NO_GO"
    return {
        "decision": decision,
        "scope": "readiness_to_start_offline_tiny_mlp_only",
        "conditional_scope": "high_confidence_small_scale_offline_feasibility_only",
        "does_not_establish": [
            "full_scale_gate_training_readiness_when_conditional",
            "gate_calibration",
            "compute_savings",
            "closed_loop_gate_improvement",
        ],
        "preregistered_at_utc": PREREGISTERED_AT_UTC,
        "go_checks": go_checks,
        "failed_go_checks": [check["name"] for check in go_checks if not check["passed"]],
        "conditional_checks": conditional_checks,
        "failed_conditional_checks": [
            check["name"] for check in conditional_checks if not check["passed"]
        ],
    }


def _target_selection_projection(
    targets: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    projection: list[dict[str, Any]] = []
    for target in targets:
        input_hashes = target.get("input_hashes")
        if not isinstance(input_hashes, Mapping):
            raise ValueError("Target V2 row is missing input_hashes")
        combined = input_hashes.get("combined")
        if not _is_sha256(combined):
            raise ValueError("Target V2 input_hashes.combined must be a SHA-256")
        projection.append(
            {
                "selection_order": _required_int(target, "selection_order"),
                "source_index": _required_int(target, "source_index"),
                "sample_id": _required_string(target, "sample_id"),
                "target_id": _required_string(target, "target_id"),
                "target_sha256": _required_string(target, "target_sha256"),
                "input_combined_sha256": combined,
            }
        )
    projection.sort(key=lambda row: int(row["selection_order"]))
    if [row["selection_order"] for row in projection] != list(range(len(projection))):
        raise ValueError("Target V2 selection_order must be exactly 0..N-1")
    return projection


def validate_validation_manifest(
    manifest: Mapping[str, Any],
    target_manifest: Mapping[str, Any],
    targets: Sequence[Mapping[str, Any]],
    *,
    target_manifest_sha256: str,
    target_targets_sha256: str,
) -> dict[str, Any]:
    """Validate the independent-validation manifest and all source bindings."""

    _assert_finite_tree(manifest, path="validation_manifest")
    if manifest.get("kind") != VALIDATION_MANIFEST_KIND:
        raise ValueError(
            f"validation manifest kind must be {VALIDATION_MANIFEST_KIND!r}"
        )
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported validation manifest schema_version")
    compatibility = manifest.get("compatibility")
    if not isinstance(compatibility, Mapping):
        raise ValueError("validation manifest compatibility must be a mapping")
    fingerprint = manifest.get("compatibility_fingerprint")
    if not _is_sha256(fingerprint) or fingerprint != _sha256_json(compatibility):
        raise ValueError("validation manifest compatibility fingerprint is invalid")
    if compatibility.get("kind") != VALIDATION_MANIFEST_KIND:
        raise ValueError("validation compatibility kind is invalid")
    if compatibility.get("schema_version") != 1:
        raise ValueError("validation compatibility schema is invalid")

    target_core.validate_target_manifest(target_manifest)
    target_compatibility = target_manifest["compatibility"]
    expected_bindings = {
        "phase25_manifest_fingerprint": target_compatibility[
            "source_manifest_compatibility_fingerprint"
        ],
        "phase25_manifest_sha256": target_compatibility["source_manifest_sha256"],
        "phase25_records_sha256": target_compatibility["source_records_sha256"],
        "phase25_selection_plan_sha256": target_compatibility[
            "source_selection_plan_sha256"
        ],
        "target_v2_manifest_fingerprint": target_manifest[
            "compatibility_fingerprint"
        ],
        "target_v2_manifest_sha256": target_manifest_sha256,
        "target_v2_targets_sha256": target_targets_sha256,
        "target_v2_selection_plan_sha256": target_compatibility[
            "source_selection_plan_sha256"
        ],
    }
    for field, expected in expected_bindings.items():
        if compatibility.get(field) != expected:
            raise ValueError(
                f"validation compatibility {field} differs from the loaded target/source"
            )

    selection = manifest.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("validation manifest selection must be a mapping")
    ordered_targets = selection.get("ordered_targets")
    expected_projection = _target_selection_projection(targets)
    if not isinstance(ordered_targets, list) or _canonical_json(
        ordered_targets
    ) != _canonical_json(expected_projection):
        raise ValueError("validation selection differs from the loaded Target V2 rows")
    selection_sha = _sha256_json(ordered_targets)
    if selection.get("ordered_targets_sha256") != selection_sha:
        raise ValueError("validation ordered-target selection digest is invalid")
    if compatibility.get("validation_selection_sha256") != selection_sha:
        raise ValueError("validation compatibility is not bound to its selection")
    if selection.get("num_states") != EXPECTED_STATE_COUNT:
        raise ValueError("validation selection must contain exactly 100 states")
    if compatibility.get("num_states") != EXPECTED_STATE_COUNT:
        raise ValueError("validation compatibility must bind exactly 100 states")

    replicates = manifest.get("replicates")
    if not isinstance(replicates, Mapping):
        raise ValueError("validation manifest replicates must be a mapping")
    expected_global_indices = list(range(len(TARGET_BASE_SEEDS), len(ALL_BASE_SEEDS)))
    expected_record_count = EXPECTED_STATE_COUNT * len(VALIDATION_BASE_SEEDS)
    exact_replicates = {
        "base_seeds": list(VALIDATION_BASE_SEEDS),
        "global_seed_indices": expected_global_indices,
        "count": len(VALIDATION_BASE_SEEDS),
        "expected_record_count": expected_record_count,
        "all_new_inference": True,
    }
    for field, expected in exact_replicates.items():
        if replicates.get(field) != expected:
            raise ValueError(f"validation replicates.{field} must be {expected!r}")
    for field, expected in (
        ("validation_base_seeds", list(VALIDATION_BASE_SEEDS)),
        ("global_seed_indices", expected_global_indices),
        ("expected_record_count", expected_record_count),
    ):
        if compatibility.get(field) != expected:
            raise ValueError(f"validation compatibility.{field} is invalid")
    parameters = compatibility.get("collection_parameters")
    if not isinstance(parameters, Mapping) or parameters.get("all_new_inference") is not True:
        raise ValueError("validation collection must be declared all-new inference")

    phase25 = manifest.get("phase25")
    target_binding = manifest.get("target_v2")
    if not isinstance(phase25, Mapping) or not isinstance(target_binding, Mapping):
        raise ValueError("validation manifest must bind Phase-2.5 and Target V2")
    for field, compatibility_field in (
        ("manifest_fingerprint", "phase25_manifest_fingerprint"),
        ("manifest_sha256", "phase25_manifest_sha256"),
        ("records_sha256", "phase25_records_sha256"),
        ("selection_plan_sha256", "phase25_selection_plan_sha256"),
    ):
        if phase25.get(field) != compatibility.get(compatibility_field):
            raise ValueError(f"validation Phase-2.5 {field} binding is invalid")
    for field, compatibility_field in (
        ("manifest_fingerprint", "target_v2_manifest_fingerprint"),
        ("manifest_sha256", "target_v2_manifest_sha256"),
        ("targets_sha256", "target_v2_targets_sha256"),
        ("selection_plan_sha256", "target_v2_selection_plan_sha256"),
    ):
        if target_binding.get(field) != compatibility.get(compatibility_field):
            raise ValueError(f"validation Target V2 {field} binding is invalid")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("validation manifest artifacts must be a mapping")
    for artifact_name, compatibility_field in (
        ("checkpoint", "checkpoint_sha256"),
        ("dataset_stats", "dataset_stats_sha256"),
        ("vae", "vae_sha256"),
    ):
        artifact = artifacts.get(artifact_name)
        expected = compatibility.get(compatibility_field)
        if not isinstance(artifact, Mapping) or not _is_sha256(expected):
            raise ValueError(f"validation {artifact_name} provenance is invalid")
        if artifact.get("sha256") != expected:
            raise ValueError(
                f"validation artifacts.{artifact_name} is not bound to compatibility"
            )
        target_expected = target_manifest["source"].get(compatibility_field)
        if target_expected is not None and expected != target_expected:
            raise ValueError(
                f"validation {artifact_name} differs from Target V2 source artifact"
            )
    return {
        "compatibility_fingerprint": fingerprint,
        "selection_sha256": selection_sha,
        "expected_record_count": expected_record_count,
    }


def _validate_zero_error_log(errors_path: Path | None) -> int:
    if errors_path is None:
        return 0
    if not errors_path.is_file():
        raise FileNotFoundError(errors_path)
    count = sum(
        bool(line.strip())
        for line in errors_path.read_text(encoding="utf-8").splitlines()
    )
    if count:
        raise ValueError(
            f"Independent validation contains {count} error rows; zero are permitted"
        )
    return 0


def validate_analysis_inputs(
    target_dir: Path,
    validation_records_path: Path,
    validation_manifest_path: Path,
    *,
    errors_path: Path | None = None,
    completion_path: Path | None = None,
) -> tuple[
    dict[str, Any], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, Any]
]:
    """Strictly load and rebind the frozen 100x5 target and fresh 100x4 grid."""

    target_dir = target_dir.resolve()
    target_manifest_path = target_dir / target_core.TARGET_MANIFEST_FILENAME
    target_targets_path = target_dir / target_core.TARGETS_FILENAME
    target_manifest, targets = target_core.load_target_bundle(
        target_dir, expected_num_states=EXPECTED_STATE_COUNT
    )
    target_manifest_sha = _sha256_file(target_manifest_path)
    target_targets_sha = _sha256_file(target_targets_path)
    validation_manifest = _read_json(validation_manifest_path.resolve())
    manifest_evidence = validate_validation_manifest(
        validation_manifest,
        target_manifest,
        targets,
        target_manifest_sha256=target_manifest_sha,
        target_targets_sha256=target_targets_sha,
    )
    resolved_errors = errors_path.resolve() if errors_path is not None else None
    _validate_zero_error_log(resolved_errors)
    resolved_completion = (
        completion_path.resolve()
        if completion_path is not None
        else validation_records_path.resolve().parent / "completion.json"
    )
    completion = validation_collector._validate_completion_seal(
        resolved_completion,
        manifest_path=validation_manifest_path.resolve(),
        records_path=validation_records_path.resolve(),
        errors_path=resolved_errors or validation_records_path.resolve().parent / "errors.jsonl",
        manifest=validation_manifest,
    )
    compatibility = validation_manifest["compatibility"]
    record_index = target_core.load_validation_record_index(
        validation_records_path.resolve(),
        expected_base_seeds=VALIDATION_BASE_SEEDS,
        expected_validation_manifest_fingerprint=validation_manifest[
            "compatibility_fingerprint"
        ],
        expected_target_manifest_sha256=target_manifest_sha,
        expected_target_targets_sha256=target_targets_sha,
        expected_target_manifest_fingerprint=target_manifest[
            "compatibility_fingerprint"
        ],
        expected_checkpoint_sha256=compatibility["checkpoint_sha256"],
        expected_dataset_stats_sha256=compatibility["dataset_stats_sha256"],
        expected_vae_sha256=compatibility["vae_sha256"],
    )
    grid = target_core.validate_validation_grid(
        record_index,
        targets,
        base_seeds=VALIDATION_BASE_SEEDS,
        allow_incomplete=False,
    )
    if grid.get("is_complete") is not True or grid.get("completed_count") != 400:
        raise ValueError("Independent validation grid is not exactly 100 states x 4 seeds")
    ordered_records = [
        record_index[(int(target["source_index"]), validation_index)]
        for target in targets
        for validation_index in range(len(VALIDATION_BASE_SEEDS))
    ]
    validation = {
        "status": "complete_and_verified",
        "target_state_count": len(targets),
        "target_seed_count": len(TARGET_BASE_SEEDS),
        "target_measurement_count": len(targets) * len(TARGET_BASE_SEEDS),
        "validation_state_count": len(targets),
        "validation_seed_count": len(VALIDATION_BASE_SEEDS),
        "validation_measurement_count": len(ordered_records),
        "all9_measurement_count": len(targets) * len(ALL_BASE_SEEDS),
        "error_count": 0,
        "grid": grid,
        "manifest": manifest_evidence,
        "completion": {
            "path": str(resolved_completion),
            "completion_sha256": completion["completion_sha256"],
        },
    }
    return target_manifest, targets, validation_manifest, ordered_records, validation


def _target_seed_values(target: Mapping[str, Any]) -> np.ndarray:
    entries = target.get("utility_by_base_seed")
    if not isinstance(entries, list) or len(entries) != len(TARGET_BASE_SEEDS):
        raise ValueError("utility_by_base_seed must contain exactly five entries")
    values = np.empty(len(TARGET_BASE_SEEDS), dtype=np.float64)
    for index, (entry, expected_seed) in enumerate(zip(entries, TARGET_BASE_SEEDS, strict=True)):
        if not isinstance(entry, Mapping):
            raise ValueError(f"utility_by_base_seed[{index}] must be a mapping")
        if entry.get("base_seed") != expected_seed or entry.get("replicate_index") != index:
            raise ValueError(
                f"target seed entry {index} must be replicate/base seed {index}/{expected_seed}"
            )
        utility = _required_float(entry, "utility")
        e0 = _required_float(entry, "e0")
        efull = _required_float(entry, "efull")
        if not math.isclose(utility, e0 - efull, rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError(f"target seed {expected_seed}: utility != e0-efull")
        values[index] = utility
    return values


def _validation_matrix(
    validation_records: Sequence[Mapping[str, Any]],
    ordered_targets: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    by_key: dict[tuple[int, int], Mapping[str, Any]] = {}
    for record in validation_records:
        source_index = _required_int(record, "source_index")
        replicate_index = _required_int(record, "validation_replicate_index")
        key = (source_index, replicate_index)
        if key in by_key:
            raise ValueError(f"Duplicate validation cell source/replicate={key}")
        by_key[key] = record
    matrix = np.empty(
        (len(ordered_targets), len(VALIDATION_BASE_SEEDS)), dtype=np.float64
    )
    for state_index, target in enumerate(ordered_targets):
        source_index = _required_int(target, "source_index")
        for replicate_index in range(len(VALIDATION_BASE_SEEDS)):
            try:
                record = by_key[(source_index, replicate_index)]
            except KeyError as exc:
                raise ValueError(
                    f"Missing validation cell source/replicate={(source_index, replicate_index)}"
                ) from exc
            matrix[state_index, replicate_index] = _required_float(record, "utility")
    return matrix


def _t_interval(values: np.ndarray) -> tuple[float, float, float, float]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size < 2 or not np.isfinite(vector).all():
        raise ValueError("t interval requires at least two finite values")
    mean = float(vector.mean())
    sample_std = float(vector.std(ddof=1))
    sem = float(sample_std / math.sqrt(vector.size))
    critical = float(stats.t.ppf(0.975, df=vector.size - 1))
    return mean, sample_std, mean - critical * sem, mean + critical * sem


def _high_confidence_from_values(values: np.ndarray, epsilon: float) -> bool:
    mean, _, low, high = _t_interval(values)
    direction = deadband_sign(mean, epsilon)
    signs = np.asarray([deadband_sign(value, epsilon) for value in values])
    same_direction = int(np.sum(signs == direction))
    interval_beyond_deadband = (direction == 1 and low > epsilon) or (
        direction == -1 and high < -epsilon
    )
    return bool(direction != 0 and same_direction >= 4 and interval_beyond_deadband)


def _sign_summary(
    mask: np.ndarray,
    target_signs: np.ndarray,
    validation_signs: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    matches = target_signs[mask] == validation_signs[mask]
    return {
        "state_count": int(mask.sum()),
        "retained_count": int(matches.sum()),
        "retention": _safe_fraction(matches),
        "population_weighted_retention": weighted_fraction(matches, weights[mask]),
    }


def _segment_diagnostics(
    per_state: Sequence[Mapping[str, Any]], *, field: str
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in per_state:
        if field == "task_group":
            key = f"{row['suite']}/task_{int(row['task_index']):02d}"
        else:
            key = str(row[field])
        grouped[key].append(row)
    output: list[dict[str, Any]] = []
    for key in sorted(grouped):
        rows = grouped[key]
        target = np.asarray([float(row["target_mean"]) for row in rows])
        validation = np.asarray([float(row["validation_mean"]) for row in rows])
        rank = _rank_pair(target, validation) if len(rows) >= 2 else {
            "spearman_rho": None,
            "kendall_tau": None,
        }
        actionable = [
            bool(row["target_mean_sign"] != 0 and row["mean_sign_retained"])
            for row in rows
            if int(row["target_mean_sign"]) != 0
        ]
        output.append(
            {
                "segment_type": field,
                "segment": key,
                "state_count": len(rows),
                "actionable_state_count": len(actionable),
                "actionable_sign_retention": _safe_fraction(actionable),
                **rank,
            }
        )
    return output


def compute_target_v2_metrics(
    targets: Sequence[Mapping[str, Any]],
    validation_records: Sequence[Mapping[str, Any]],
    *,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    np.ndarray,
]:
    """Compute every pre-registered metric after hard integrity validation."""

    ordered = sorted(targets, key=lambda row: int(row["selection_order"]))
    target_matrix = np.stack([_target_seed_values(target) for target in ordered])
    validation_matrix = _validation_matrix(validation_records, ordered)
    all9_matrix = np.concatenate([target_matrix, validation_matrix], axis=1)
    target_mean = target_matrix.mean(axis=1)
    target_median = np.median(target_matrix, axis=1)
    validation_mean = validation_matrix.mean(axis=1)
    validation_median = np.median(validation_matrix, axis=1)
    strata = [str(target["selection_bin"]) for target in ordered]
    weights = population_state_weights(strata)

    per_state: list[dict[str, Any]] = []
    for index, (target, weight) in enumerate(zip(ordered, weights, strict=True)):
        target_values = target_matrix[index]
        validation_values = validation_matrix[index]
        target_mean_i, target_std, target_low, target_high = _t_interval(target_values)
        validation_mean_i, validation_std, validation_low, validation_high = _t_interval(
            validation_values
        )
        row: dict[str, Any] = {
            "selection_order": int(target["selection_order"]),
            "source_index": int(target["source_index"]),
            "sample_id": str(target["sample_id"]),
            "suite": str(target["suite"]),
            "task_index": int(target["task_index"]),
            "task": str(target["task"]),
            "episode_index": int(target["episode_index"]),
            "frame_index": int(target["frame_index"]),
            "valid_length": int(target["valid_length"]),
            "selection_bin": str(target["selection_bin"]),
            "population_weight": float(weight),
            "target_mean": target_mean_i,
            "target_median": float(np.median(target_values)),
            "target_sample_std": target_std,
            "target_sem": float(target_std / math.sqrt(len(TARGET_BASE_SEEDS))),
            "target_t95_ci_low": target_low,
            "target_t95_ci_high": target_high,
            "target_high_confidence": _high_confidence_from_values(
                target_values, PRIMARY_EPSILON
            ),
            "validation_mean": validation_mean_i,
            "validation_median": float(np.median(validation_values)),
            "validation_sample_std": validation_std,
            "validation_sem": float(
                validation_std / math.sqrt(len(VALIDATION_BASE_SEEDS))
            ),
            "validation_t95_ci_low": validation_low,
            "validation_t95_ci_high": validation_high,
            "mean_difference_validation_minus_target": validation_mean_i - target_mean_i,
            "median_difference_validation_minus_target": float(
                np.median(validation_values) - np.median(target_values)
            ),
        }
        for seed_index, base_seed in enumerate(ALL_BASE_SEEDS):
            row[f"utility_seed_{base_seed}"] = float(all9_matrix[index, seed_index])
        for epsilon in DEADBAND_EPSILONS:
            label = _epsilon_label(epsilon)
            mean_target_sign = deadband_sign(target_mean_i, epsilon)
            mean_validation_sign = deadband_sign(validation_mean_i, epsilon)
            median_target_sign = deadband_sign(float(np.median(target_values)), epsilon)
            median_validation_sign = deadband_sign(
                float(np.median(validation_values)), epsilon
            )
            high_confidence = _high_confidence_from_values(target_values, epsilon)
            row.update(
                {
                    f"{label}_target_mean_sign": mean_target_sign,
                    f"{label}_validation_mean_sign": mean_validation_sign,
                    f"{label}_mean_sign_retained": bool(
                        mean_target_sign != 0 and mean_target_sign == mean_validation_sign
                    ),
                    f"{label}_target_median_sign": median_target_sign,
                    f"{label}_validation_median_sign": median_validation_sign,
                    f"{label}_median_sign_retained": bool(
                        median_target_sign != 0
                        and median_target_sign == median_validation_sign
                    ),
                    f"{label}_target_high_confidence": high_confidence,
                }
            )
        primary = _epsilon_label(PRIMARY_EPSILON)
        row.update(
            {
                "target_mean_sign": int(row[f"{primary}_target_mean_sign"]),
                "validation_mean_sign": int(row[f"{primary}_validation_mean_sign"]),
                "mean_sign_retained": bool(row[f"{primary}_mean_sign_retained"]),
                "target_median_sign": int(row[f"{primary}_target_median_sign"]),
                "validation_median_sign": int(row[f"{primary}_validation_median_sign"]),
                "median_sign_retained": bool(row[f"{primary}_median_sign_retained"]),
            }
        )
        # Ensure the analysis never silently diverges from the frozen label.
        if bool(target.get("high_confidence")) != bool(row["target_high_confidence"]):
            raise ValueError(
                f"Target high_confidence mismatch for {row['sample_id']}: "
                f"stored={target.get('high_confidence')!r}, recomputed={row['target_high_confidence']!r}"
            )
        per_state.append(row)

    target_signs = np.asarray([row["target_mean_sign"] for row in per_state], dtype=int)
    validation_signs = np.asarray(
        [row["validation_mean_sign"] for row in per_state], dtype=int
    )
    high_confidence = np.asarray(
        [row["target_high_confidence"] for row in per_state], dtype=bool
    )
    actionable = target_signs != 0
    positive_hc = high_confidence & (target_signs == 1)
    negative_hc = high_confidence & (target_signs == -1)
    sign_retention = {
        "actionable": _sign_summary(
            actionable, target_signs, validation_signs, weights
        ),
        "high_confidence": _sign_summary(
            high_confidence, target_signs, validation_signs, weights
        ),
        "high_confidence_positive": _sign_summary(
            positive_hc, target_signs, validation_signs, weights
        ),
        "high_confidence_negative": _sign_summary(
            negative_hc, target_signs, validation_signs, weights
        ),
    }

    rank_agreement = {
        "mean_vs_mean": _rank_pair(target_mean, validation_mean),
        "median_vs_median": _rank_pair(target_median, validation_median),
        "mean_vs_median": _rank_pair(target_mean, validation_median),
        "median_vs_mean": _rank_pair(target_median, validation_mean),
    }
    overlap = ranking_overlap(target_mean, validation_mean)
    absolute_icc = icc_absolute_agreement(
        np.column_stack([target_mean, validation_mean])
    )
    absolute_agreement = {
        "lin_ccc": lin_concordance_correlation(target_mean, validation_mean),
        **absolute_icc,
        "mean_bias_validation_minus_target": float(
            np.mean(validation_mean - target_mean)
        ),
        "median_absolute_difference": float(
            np.median(np.abs(validation_mean - target_mean))
        ),
        "rmse": float(np.sqrt(np.mean((validation_mean - target_mean) ** 2))),
        "bland_altman_lower_95": float(
            np.mean(validation_mean - target_mean)
            - 1.96 * np.std(validation_mean - target_mean, ddof=1)
        ),
        "bland_altman_upper_95": float(
            np.mean(validation_mean - target_mean)
            + 1.96 * np.std(validation_mean - target_mean, ddof=1)
        ),
    }
    target_reliability = icc_one_way(target_matrix)
    validation_reliability = icc_one_way(validation_matrix)
    all9_reliability = icc_one_way(all9_matrix)

    median_target_signs = np.asarray(
        [row["target_median_sign"] for row in per_state], dtype=int
    )
    median_validation_signs = np.asarray(
        [row["validation_median_sign"] for row in per_state], dtype=int
    )
    median_actionable = median_target_signs != 0
    median_guardrail = {
        "rank": rank_agreement["median_vs_median"],
        "actionable_state_count": int(median_actionable.sum()),
        "actionable_sign_retention": _safe_fraction(
            median_target_signs[median_actionable]
            == median_validation_signs[median_actionable]
        ),
    }
    high_confidence_coverage = {
        "state_count": int(high_confidence.sum()),
        "unweighted_fraction": float(high_confidence.mean()),
        "population_weighted_fraction": weighted_fraction(high_confidence, weights),
        "positive_state_count": int(positive_hc.sum()),
        "negative_state_count": int(negative_hc.sum()),
    }

    seed_rows: list[dict[str, Any]] = []
    for seed_index, base_seed in enumerate(ALL_BASE_SEEDS):
        group = "target" if seed_index < len(TARGET_BASE_SEEDS) else "validation"
        group_matrix = target_matrix if group == "target" else validation_matrix
        group_index = (
            seed_index
            if group == "target"
            else seed_index - len(TARGET_BASE_SEEDS)
        )
        other_group_mean = validation_mean if group == "target" else target_mean
        other_same_group = np.delete(group_matrix, group_index, axis=1).mean(axis=1)
        other8 = np.delete(all9_matrix, seed_index, axis=1).mean(axis=1)
        value = all9_matrix[:, seed_index]
        row = {
            "base_seed": base_seed,
            "seed_group": group,
            "group_index": group_index,
            "utility_mean": float(value.mean()),
            "utility_std_across_states": float(value.std(ddof=1)),
        }
        for prefix, comparator in (
            ("other_same_group_mean", other_same_group),
            ("other8_mean", other8),
            ("opposite_group_mean", other_group_mean),
        ):
            rank = _rank_pair(value, comparator)
            row[f"spearman_vs_{prefix}"] = rank["spearman_rho"]
            row[f"kendall_vs_{prefix}"] = rank["kendall_tau"]
        seed_rows.append(row)

    seed_pair_rows: list[dict[str, Any]] = []
    for target_index, target_seed in enumerate(TARGET_BASE_SEEDS):
        for validation_index, validation_seed in enumerate(VALIDATION_BASE_SEEDS):
            rank = _rank_pair(
                target_matrix[:, target_index], validation_matrix[:, validation_index]
            )
            seed_pair_rows.append(
                {
                    "target_base_seed": target_seed,
                    "validation_base_seed": validation_seed,
                    **rank,
                }
            )

    group_rows: list[dict[str, Any]] = []
    for index, seed in enumerate(TARGET_BASE_SEEDS):
        rank = _rank_pair(
            np.delete(target_matrix, index, axis=1).mean(axis=1), validation_mean
        )
        group_rows.append(
            {
                "omitted_group": "target",
                "omitted_base_seed": seed,
                "reference": "target4_mean",
                "comparison": "validation4_mean",
                **rank,
            }
        )
    for index, seed in enumerate(VALIDATION_BASE_SEEDS):
        rank = _rank_pair(
            target_mean, np.delete(validation_matrix, index, axis=1).mean(axis=1)
        )
        group_rows.append(
            {
                "omitted_group": "validation",
                "omitted_base_seed": seed,
                "reference": "target5_mean",
                "comparison": "validation3_mean",
                **rank,
            }
        )

    deadband_rows: list[dict[str, Any]] = []
    for epsilon in DEADBAND_EPSILONS:
        label = _epsilon_label(epsilon)
        target_eps_sign = np.asarray(
            [row[f"{label}_target_mean_sign"] for row in per_state], dtype=int
        )
        validation_eps_sign = np.asarray(
            [row[f"{label}_validation_mean_sign"] for row in per_state], dtype=int
        )
        actionable_eps = target_eps_sign != 0
        hc_eps = np.asarray(
            [row[f"{label}_target_high_confidence"] for row in per_state], dtype=bool
        )
        deadband_rows.append(
            {
                "epsilon": epsilon,
                "target_positive_count": int(np.sum(target_eps_sign == 1)),
                "target_negative_count": int(np.sum(target_eps_sign == -1)),
                "target_deadband_count": int(np.sum(target_eps_sign == 0)),
                "validation_positive_count": int(np.sum(validation_eps_sign == 1)),
                "validation_negative_count": int(np.sum(validation_eps_sign == -1)),
                "validation_deadband_count": int(np.sum(validation_eps_sign == 0)),
                "actionable_state_count": int(actionable_eps.sum()),
                "actionable_sign_retention": _safe_fraction(
                    target_eps_sign[actionable_eps]
                    == validation_eps_sign[actionable_eps]
                ),
                "high_confidence_state_count": int(hc_eps.sum()),
                "high_confidence_weighted_coverage": weighted_fraction(hc_eps, weights),
                "high_confidence_sign_retention": _safe_fraction(
                    target_eps_sign[hc_eps] == validation_eps_sign[hc_eps]
                ),
            }
        )

    bootstrap_indices = _stratified_bootstrap_indices(
        strata, seed=bootstrap_seed, replicates=bootstrap_replicates
    )

    def sampled_sign_retention(indices: np.ndarray, base_mask: np.ndarray) -> float | None:
        selected_mask = base_mask[indices]
        if not selected_mask.any():
            return None
        return float(
            np.mean(
                target_signs[indices][selected_mask]
                == validation_signs[indices][selected_mask]
            )
        )

    bootstrap = {
        "method": "state bootstrap stratified by frozen Pilot selection_bin",
        "seed": bootstrap_seed,
        "replicates": bootstrap_replicates,
        "mean_spearman": _bootstrap_ci(
            bootstrap_indices,
            lambda indices: _rank_pair(
                target_mean[indices], validation_mean[indices]
            )["spearman_rho"],
        ),
        "mean_kendall": _bootstrap_ci(
            bootstrap_indices,
            lambda indices: _rank_pair(
                target_mean[indices], validation_mean[indices]
            )["kendall_tau"],
        ),
        "actionable_sign_retention": _bootstrap_ci(
            bootstrap_indices,
            lambda indices: sampled_sign_retention(indices, actionable),
        ),
        "high_confidence_sign_retention": _bootstrap_ci(
            bootstrap_indices,
            lambda indices: sampled_sign_retention(indices, high_confidence),
        ),
        "lin_ccc": _bootstrap_ci(
            bootstrap_indices,
            lambda indices: lin_concordance_correlation(
                target_mean[indices], validation_mean[indices]
            ),
        ),
        "absolute_agreement_icc_a1": _bootstrap_ci(
            bootstrap_indices,
            lambda indices: icc_absolute_agreement(
                np.column_stack([target_mean[indices], validation_mean[indices]])
            )["icc_a_1"],
        ),
        "all9_icc_1_1": _bootstrap_ci(
            bootstrap_indices,
            lambda indices: icc_one_way(all9_matrix[indices])["icc_1_1"],
        ),
        "all9_icc_1_9": _bootstrap_ci(
            bootstrap_indices,
            lambda indices: icc_one_way(all9_matrix[indices])["icc_1_k"],
        ),
    }

    segment_rows = _segment_diagnostics(per_state, field="suite")
    segment_rows.extend(_segment_diagnostics(per_state, field="task_group"))
    metrics: dict[str, Any] = {
        "primary_comparison": "target seeds42-46 mean vs independent seeds47-50 mean",
        "primary_deadband_epsilon": PRIMARY_EPSILON,
        "rank_agreement": rank_agreement,
        "sign_retention": sign_retention,
        "ranking_overlap": overlap,
        "absolute_agreement": absolute_agreement,
        "target5_reliability": target_reliability,
        "validation4_reliability": validation_reliability,
        "all9_reliability": all9_reliability,
        "single_seed_reliability_guardrail": {
            "metric": "ICC(1,1) across all nine seeds",
            "value": all9_reliability["icc_1_1"],
            "interpretation": (
                "Reported alongside ICC(1,9); the nine-seed aggregate must not be used "
                "to imply that a single-seed label is reliable."
            ),
        },
        "median_guardrail": median_guardrail,
        "high_confidence_coverage": high_confidence_coverage,
        "leave_one_seed_out": {
            "target_median_spearman_vs_other4": _finite_median(
                row["spearman_vs_other_same_group_mean"]
                for row in seed_rows
                if row["seed_group"] == "target"
            ),
            "validation_median_spearman_vs_other3": _finite_median(
                row["spearman_vs_other_same_group_mean"]
                for row in seed_rows
                if row["seed_group"] == "validation"
            ),
            "all9_median_spearman_vs_other8": _finite_median(
                row["spearman_vs_other8_mean"] for row in seed_rows
            ),
        },
        "deadband_sensitivity": {
            _epsilon_label(float(row["epsilon"])): dict(row) for row in deadband_rows
        },
        "bootstrap": bootstrap,
        "population_weighting": {
            "method": "post-stratification to Pilot-500 selection-bin prevalence",
            "stratum_prevalence": dict(STRATUM_PREVALENCE),
            "warning": (
                "The 100-state panel over-samples utility tails; unweighted metrics "
                "describe the diagnostic panel, not Pilot-500 prevalence."
            ),
        },
    }
    metrics["decision"] = decide_readiness(metrics)
    return (
        metrics,
        per_state,
        seed_rows,
        seed_pair_rows,
        group_rows,
        segment_rows,
        deadband_rows,
        all9_matrix,
    )


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
    all9_matrix: np.ndarray,
    per_state: Sequence[Mapping[str, Any]],
    seed_rows: Sequence[Mapping[str, Any]],
    deadband_rows: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
) -> list[str]:
    """Write static diagnostics using an explicit two-root palette."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    except ImportError as exc:
        raise RuntimeError("matplotlib is required when PNG output is enabled") from exc

    blue = "#3568A8"
    blue_dark = "#183B66"
    gold = "#D69E2E"
    gold_light = "#F6E7BE"
    ink = "#20262E"
    grid = "#D9DEE5"
    output: list[str] = []

    def finish(
        figure: Any, *, title: str, subtitle: str, bottom: float = 0.10
    ) -> None:
        figure.suptitle(title, y=0.985, fontsize=13, color=ink)
        figure.text(0.5, 0.94, subtitle, ha="center", va="top", fontsize=9, color="#59636E")
        figure.tight_layout(rect=(0.03, bottom, 0.98, 0.90))

    target = np.asarray([float(row["target_mean"]) for row in per_state])
    validation = np.asarray([float(row["validation_mean"]) for row in per_state])
    high_confidence = np.asarray(
        [bool(row["target_high_confidence"]) for row in per_state]
    )
    extent = float(max(np.max(np.abs(target)), np.max(np.abs(validation)), PRIMARY_EPSILON))
    padding = 0.08 * extent
    bounds = (-extent - padding, extent + padding)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.scatter(
        target[~high_confidence],
        validation[~high_confidence],
        s=30,
        facecolors="none",
        edgecolors=gold,
        linewidths=1.1,
        label="Other target",
    )
    ax.scatter(
        target[high_confidence],
        validation[high_confidence],
        s=34,
        c=blue,
        edgecolors=blue_dark,
        linewidths=0.7,
        label="High-confidence target",
    )
    ax.plot(bounds, bounds, color=ink, linestyle="--", linewidth=1.1, label="Identity")
    ax.axhline(0.0, color=grid, linewidth=0.8)
    ax.axvline(0.0, color=grid, linewidth=0.8)
    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Target-five mean utility")
    ax.set_ylabel("Independent-four mean utility")
    ax.legend(frameon=False, loc="best")
    ax.grid(color=grid, linewidth=0.6, alpha=0.55)
    finish(
        fig,
        title="Target-five and independent-four mean utility",
        subtitle=(
            f"100 LIBERO states; epsilon={PRIMARY_EPSILON:.0e}; dashed line is exact agreement"
        ),
    )
    path = output_dir / "target_vs_validation_mean.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    output.append(path.name)

    ordering = np.argsort(target)
    sorted_values = all9_matrix[ordering]
    vmax = float(np.quantile(np.abs(sorted_values), 0.98))
    vmax = max(vmax, PRIMARY_EPSILON)
    cmap = LinearSegmentedColormap.from_list(
        "utility_signed", [gold, gold_light, "#FFFFFF", "#DCE8F5", blue]
    )
    fig, ax = plt.subplots(figsize=(8.0, 8.5))
    image = ax.imshow(
        sorted_values,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
    )
    ax.axvline(len(TARGET_BASE_SEEDS) - 0.5, color=ink, linewidth=1.4)
    ax.set_xticks(range(len(ALL_BASE_SEEDS)), [str(seed) for seed in ALL_BASE_SEEDS])
    ax.set_xlabel("Base seed (42–46 target | 47–50 independent validation)")
    ax.set_ylabel("States sorted by target-five mean")
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Utility U = E0 - E10")
    finish(
        fig,
        title="Utility across all nine seeds",
        subtitle="Signed per-state utility; color scale clipped at the 98th absolute percentile",
        bottom=0.08,
    )
    path = output_dir / "all9_seed_heatmap.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    output.append(path.name)

    averages = (target + validation) / 2.0
    differences = validation - target
    agreement = metrics["absolute_agreement"]
    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    ax.scatter(
        averages[~high_confidence],
        differences[~high_confidence],
        s=28,
        facecolors="none",
        edgecolors=gold,
        linewidths=1.0,
    )
    ax.scatter(
        averages[high_confidence],
        differences[high_confidence],
        s=32,
        c=blue,
        edgecolors=blue_dark,
        linewidths=0.6,
    )
    for value, label, style in (
        (agreement["mean_bias_validation_minus_target"], "Mean bias", "-"),
        (agreement["bland_altman_lower_95"], "95% limits", "--"),
        (agreement["bland_altman_upper_95"], None, "--"),
    ):
        ax.axhline(value, color=ink, linewidth=1.0, linestyle=style, label=label)
    ax.axhline(0.0, color=grid, linewidth=0.8)
    ax.set_xlabel("Average of target-five and independent-four means")
    ax.set_ylabel("Independent-four minus target-five mean")
    ax.legend(frameon=False)
    ax.grid(color=grid, linewidth=0.6, alpha=0.55)
    finish(
        fig,
        title="Absolute agreement of aggregate utility",
        subtitle="Bland–Altman view; limits are mean difference ± 1.96 sample SD",
    )
    path = output_dir / "aggregate_bland_altman.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    output.append(path.name)

    labels = [f"seed {int(row['base_seed'])}" for row in seed_rows]
    values = [
        (
            float(row["spearman_vs_opposite_group_mean"])
            if row["spearman_vs_opposite_group_mean"] is not None
            else math.nan
        )
        for row in seed_rows
    ]
    colors = [blue if row["seed_group"] == "target" else gold for row in seed_rows]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    positions = np.arange(len(seed_rows))
    ax.scatter(values, positions, c=colors, s=55, edgecolors=ink, linewidths=0.5)
    ax.axvline(0.30, color=grid, linestyle="--", linewidth=1.0, label="Conditional rho floor")
    ax.axvline(0.50, color=ink, linestyle="--", linewidth=1.0, label="GO rho threshold")
    ax.set_yticks(positions, labels)
    ax.set_xlabel("Spearman rho versus opposite-group aggregate")
    ax.set_xlim(-1.0, 1.0)
    ax.grid(axis="x", color=grid, linewidth=0.6)
    ax.legend(frameon=False, loc="lower right")
    finish(
        fig,
        title="Per-seed rank agreement with the opposite seed group",
        subtitle="Blue: target seeds 42–46; gold: independent validation seeds 47–50",
    )
    path = output_dir / "seed_group_rank_diagnostics.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    output.append(path.name)

    epsilons = np.asarray([float(row["epsilon"]) for row in deadband_rows])
    retention = np.asarray(
        [
            (
                float(row["actionable_sign_retention"])
                if row["actionable_sign_retention"] is not None
                else math.nan
            )
            for row in deadband_rows
        ]
    )
    coverage = np.asarray(
        [float(row["high_confidence_weighted_coverage"]) for row in deadband_rows]
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(
        epsilons,
        retention,
        marker="o",
        color=blue,
        linewidth=1.6,
        label="Actionable sign retention",
    )
    ax.plot(
        epsilons,
        coverage,
        marker="s",
        markerfacecolor="white",
        color=gold,
        linewidth=1.6,
        label="Weighted HC coverage",
    )
    ax.axvline(PRIMARY_EPSILON, color=ink, linestyle="--", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Deadband epsilon")
    ax.set_ylabel("Fraction")
    ax.grid(color=grid, linewidth=0.6, alpha=0.65)
    ax.legend(frameon=False)
    finish(
        fig,
        title="Deadband sensitivity",
        subtitle="Primary epsilon=1e-4; coverage is post-stratified to Pilot-500 prevalence",
    )
    path = output_dir / "deadband_sensitivity.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    output.append(path.name)
    return output


def analyze(
    target_dir: Path,
    validation_records_path: Path,
    validation_manifest_path: Path,
    output_dir: Path,
    *,
    errors_path: Path | None = None,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    make_plots: bool = True,
) -> dict[str, Any]:
    """Validate first, then analyze and emit the durable independent audit."""

    if int(bootstrap_seed) != BOOTSTRAP_SEED:
        raise ValueError(
            f"Formal readiness analysis requires preregistered bootstrap_seed={BOOTSTRAP_SEED}"
        )
    if int(bootstrap_replicates) != BOOTSTRAP_REPLICATES:
        raise ValueError(
            "Formal readiness analysis requires preregistered "
            f"bootstrap_replicates={BOOTSTRAP_REPLICATES}"
        )
    target_dir = target_dir.resolve()
    validation_records_path = validation_records_path.resolve()
    validation_manifest_path = validation_manifest_path.resolve()
    if not validation_records_path.is_file():
        raise FileNotFoundError(validation_records_path)
    if not validation_manifest_path.is_file():
        raise FileNotFoundError(validation_manifest_path)
    (
        target_manifest,
        targets,
        validation_manifest,
        validation_records,
        validation,
    ) = validate_analysis_inputs(
        target_dir,
        validation_records_path,
        validation_manifest_path,
        errors_path=errors_path,
    )
    (
        metrics,
        per_state,
        seed_rows,
        seed_pair_rows,
        group_rows,
        segment_rows,
        deadband_rows,
        all9_matrix,
    ) = compute_target_v2_metrics(
        targets,
        validation_records,
        bootstrap_seed=bootstrap_seed,
        bootstrap_replicates=bootstrap_replicates,
    )

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_outputs = {
        "per_state_csv": "per_state.csv",
        "seed_metrics_csv": "seed_metrics.csv",
        "seed_pair_metrics_csv": "seed_pair_metrics.csv",
        "seed_group_loso_csv": "seed_group_loso.csv",
        "segment_metrics_csv": "segment_metrics.csv",
        "deadband_sensitivity_csv": "deadband_sensitivity.csv",
    }
    _write_csv(output_dir / csv_outputs["per_state_csv"], per_state)
    _write_csv(output_dir / csv_outputs["seed_metrics_csv"], seed_rows)
    _write_csv(output_dir / csv_outputs["seed_pair_metrics_csv"], seed_pair_rows)
    _write_csv(output_dir / csv_outputs["seed_group_loso_csv"], group_rows)
    _write_csv(output_dir / csv_outputs["segment_metrics_csv"], segment_rows)
    _write_csv(output_dir / csv_outputs["deadband_sensitivity_csv"], deadband_rows)
    plot_files = (
        _plot_outputs(
            output_dir,
            all9_matrix,
            per_state,
            seed_rows,
            deadband_rows,
            metrics,
        )
        if make_plots
        else []
    )
    target_manifest_path = target_dir / target_core.TARGET_MANIFEST_FILENAME
    target_targets_path = target_dir / target_core.TARGETS_FILENAME
    summary: dict[str, Any] = {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "analysis_scope": (
            "Independent seeds47-50 validation of LIBERO multi-seed Utility Target V2"
        ),
        "preregistration": {
            "locked_at_utc": PREREGISTERED_AT_UTC,
            "locked_before_independent_results_inspection": True,
            "go_thresholds": dict(GO_THRESHOLDS),
            "conditional_thresholds": dict(CONDITIONAL_THRESHOLDS),
            "primary_target": "arithmetic mean of seeds42-46",
            "primary_validation": "arithmetic mean of independent seeds47-50",
            "high_confidence_definition": (
                "target mean outside +/-epsilon, >=4/5 seeds in that epsilon direction, "
                "and two-sided 95% t-CI wholly beyond the same +/-epsilon boundary"
            ),
        },
        "inputs": {
            "target_dir": str(target_dir),
            "target_manifest_path": str(target_manifest_path),
            "target_manifest_sha256": _sha256_file(target_manifest_path),
            "target_targets_path": str(target_targets_path),
            "target_targets_sha256": _sha256_file(target_targets_path),
            "target_manifest_compatibility_fingerprint": target_manifest[
                "compatibility_fingerprint"
            ],
            "validation_records_path": str(validation_records_path),
            "validation_records_sha256": _sha256_file(validation_records_path),
            "validation_manifest_path": str(validation_manifest_path),
            "validation_manifest_sha256": _sha256_file(validation_manifest_path),
            "validation_manifest_compatibility_fingerprint": validation_manifest[
                "compatibility_fingerprint"
            ],
            "errors_path": str(errors_path.resolve()) if errors_path is not None else None,
            "errors_sha256": (
                _sha256_file(errors_path.resolve())
                if errors_path is not None and errors_path.is_file()
                else None
            ),
        },
        "validation": validation,
        "settings": {
            "target_base_seeds": list(TARGET_BASE_SEEDS),
            "validation_base_seeds": list(VALIDATION_BASE_SEEDS),
            "primary_deadband_epsilon": PRIMARY_EPSILON,
            "sensitivity_deadband_epsilons": list(DEADBAND_EPSILONS),
            "tail_fraction": TAIL_FRACTION,
            "bootstrap_method": "state bootstrap stratified by frozen selection_bin",
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_replicates": bootstrap_replicates,
            "pilot_stratum_prevalence": dict(STRATUM_PREVALENCE),
        },
        "metrics": metrics,
        "outputs": {**csv_outputs, "plot_files": plot_files},
        "interpretation_guardrail": (
            "GO only permits the next offline Tiny-MLP training stage. CONDITIONAL only "
            "permits a small high-confidence offline feasibility experiment. Neither "
            "establishes calibration, compute savings, or closed-loop LIBERO success."
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
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--validation-records", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, default=None)
    parser.add_argument("--validation-errors", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = _parse_args()
    records = args.validation_records.resolve()
    manifest = (
        args.validation_manifest.resolve()
        if args.validation_manifest is not None
        else records.parent / "manifest.json"
    )
    errors = args.validation_errors
    if errors is None:
        candidate = records.parent / "errors.jsonl"
        errors = candidate if candidate.exists() else None
    output = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else records.parent / "target_v2_analysis"
    )
    summary = analyze(
        args.target_dir,
        records,
        manifest,
        output,
        errors_path=errors,
        make_plots=not bool(args.no_plots),
    )
    LOGGER.info(
        "Verified Target V2 independent grid; offline readiness=%s; outputs=%s",
        summary["metrics"]["decision"]["decision"],
        output,
    )


if __name__ == "__main__":
    main()
