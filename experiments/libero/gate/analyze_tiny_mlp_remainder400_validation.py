"""Locked original-100 re-test for the remainder-400 Tiny-MLP experiment.

The remainder-400 trainer is deliberately unable to import Target5 labels from
the original 100-state panel or its Validation4 measurements.  It first seals
1,200 predictions: the complete original Phase-3 q100 panel (900 rows), plus
q25/q50/q75 primary-model learning-curve predictions (100 rows each).  Only
then may this module open the original Target5/Validation4 evidence.

Primary readiness is *exactly* the previously frozen 20 GO / 6 CONDITIONAL
panel and uses only ``exact_v1_137/q100``.  The learning curve cannot rescue a
failed primary result.  It only tests the preregistered sample-size attribution
``rho(q100)-rho(q25) >= 0.05`` with a paired task-cluster bootstrap lower bound
strictly above zero.  The post-NO_GO 505-dimensional feature is not run here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import numpy as np

from experiments.libero.gate import analyze_demo_utility_target_v2 as target_analysis
from experiments.libero.gate import analyze_tiny_mlp_validation as phase3
from experiments.libero.gate import offline_tiny_mlp as original_core
from experiments.libero.gate import offline_tiny_mlp_remainder400 as followup_core


ANALYSIS_SCHEMA_VERSION: Final = 1
ANALYSIS_KIND: Final = "libero_gate_remainder400_locked_original100_validation"
PREREGISTRATION: Final = "docs/GATE_OFFLINE_REMAINDER400_FOLLOWUP.md"
REPRESENTATION: Final = "exact_v1_137"
CURVES: Final = (("q25", 0.25), ("q50", 0.50), ("q75", 0.75), ("q100", 1.00))
CURVE_BY_LABEL: Final = dict(CURVES)
PRIMARY_CURVE: Final = "q100"
SAMPLE_REFERENCE_CURVE: Final = "q25"
SAMPLE_DELTA_THRESHOLD: Final = 0.05
EXPECTED_PREDICTION_COUNT: Final = 1_200
FORBIDDEN_PREDICTION_FIELDS: Final = frozenset(
    {
        "target5_utility_mean",
        "target5_utility_sem",
        "target5_high_confidence",
        "utility",
        "utility_mean",
        "validation4_mean",
        "validation_utility",
    }
)
REQUIRED_FOLLOWUP_SHA_BINDINGS: Final = tuple(
    sorted(followup_core.FORMAL_COMPATIBILITY_SHA_FIELDS)
)

DECISION_SAMPLE_SIZE_SUPPORTED: Final = "SAMPLE_SIZE_SUPPORTED"
DECISION_V1_LEARNABLE: Final = "V1_LEARNABLE_ATTRIBUTION_NOT_ESTABLISHED"
DECISION_CONDITIONAL: Final = "CONDITIONAL_ONLY"
DECISION_NO_GO: Final = "EXACT_V1_NO_GO"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _original_fold_lookup(
    fold_plan: Mapping[str, Any],
) -> dict[tuple[str, int], tuple[int, str]]:
    """Map every original state to its frozen task/suite held-out fold."""

    lookup: dict[tuple[str, int], tuple[int, str]] = {}
    for plan_key, scheme in (
        ("task_heldout_folds", "task_heldout"),
        ("suite_heldout_folds", "suite_heldout"),
    ):
        folds = fold_plan.get(plan_key)
        if not isinstance(folds, list):
            raise ValueError(f"original fold plan is missing {plan_key}")
        for fold in folds:
            if not isinstance(fold, Mapping):
                raise ValueError("original fold entry must be a mapping")
            fold_id = int(fold["fold_id"])
            orders = fold.get("test_selection_orders")
            if not isinstance(orders, list):
                raise ValueError("original fold test_selection_orders must be a list")
            for order_value in orders:
                order = int(order_value)
                key = (scheme, order)
                if key in lookup:
                    raise ValueError(f"original folds duplicate {scheme} state {order}")
                lookup[key] = (fold_id, "")
    expected = {
        (scheme, order) for scheme in ("task_heldout", "suite_heldout") for order in range(100)
    }
    if set(lookup) != expected:
        raise ValueError("original folds are not exact 100-state task/suite OOF partitions")
    return lookup


def _assert_embedded_original_folds(
    followup_plan: Mapping[str, Any],
    original_fold_plan: Mapping[str, Any],
) -> None:
    pairs = (
        ("original_task_heldout_folds", "task_heldout_folds"),
        ("original_suite_heldout_folds", "suite_heldout_folds"),
    )
    for followup_key, original_key in pairs:
        embedded = followup_plan.get(followup_key)
        sealed = original_fold_plan.get(original_key)
        if _canonical_json(embedded) != _canonical_json(sealed):
            raise ValueError(
                f"follow-up {followup_key} differs from sealed original folds"
            )


def _ordered_targets(targets: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    if len(targets) != 100:
        raise ValueError("locked external re-test requires exactly 100 original targets")
    ordered = sorted(targets, key=lambda row: int(row["selection_order"]))
    if [int(row["selection_order"]) for row in ordered] != list(range(100)):
        raise ValueError("original target selection_order must be exact 0..99")
    if len({str(row["sample_id"]) for row in ordered}) != 100:
        raise ValueError("original targets contain duplicate sample_id")
    if len({int(row["source_index"]) for row in ordered}) != 100:
        raise ValueError("original targets contain duplicate global source_index")
    return ordered


def _curve_contract(row: Mapping[str, Any]) -> tuple[str, float]:
    label = str(row.get("curve_label"))
    if label not in CURVE_BY_LABEL:
        raise ValueError(f"unexpected learning-curve label {label!r}")
    fraction = float(row.get("curve_fraction", math.nan))
    expected = CURVE_BY_LABEL[label]
    if not math.isfinite(fraction) or fraction != expected:
        raise ValueError(f"{label} curve_fraction must be exactly {expected}")
    if row.get("representation") != REPRESENTATION:
        raise ValueError("formal follow-up accepts only exact_v1_137 predictions")
    return label, fraction


def _prediction_identity(
    row: Mapping[str, Any], target: Mapping[str, Any]
) -> None:
    expected = {
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
    for field, value in expected.items():
        if row.get(field) != value:
            raise ValueError(f"prediction {field} differs from original state identity")
    _require_sha(row.get("feature_record_sha256"), field="feature_record_sha256")
    if not isinstance(row.get("feature_id"), str) or not str(row["feature_id"]).strip():
        raise ValueError("feature_id must be non-empty")
    init_predictions = row.get("init_predictions")
    if not isinstance(init_predictions, list) or len(init_predictions) != 5:
        raise ValueError("prediction must contain five init_predictions")
    values = np.asarray(init_predictions, dtype=np.float64)
    prediction = float(row.get("prediction", math.nan))
    if not np.isfinite(values).all() or not math.isfinite(prediction):
        raise ValueError("prediction values must be finite")
    if not math.isclose(prediction, float(values.mean()), rel_tol=1e-7, abs_tol=1e-9):
        raise ValueError("prediction is not the five-init ensemble mean")


def validate_and_group_predictions(
    prediction_rows: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    original_fold_plan: Mapping[str, Any],
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    """Rebind all 1,200 sealed predictions to original identities and folds.

    Returned q100 rows are adapted to the old analyzer's identity schema by
    adding Target5 IDs only *after* training has completed.  No utility label is
    copied into the predictions.
    """

    if len(prediction_rows) != EXPECTED_PREDICTION_COUNT:
        raise ValueError(f"follow-up run must contain exactly {EXPECTED_PREDICTION_COUNT} rows")
    ordered_targets = _ordered_targets(targets)
    fold_lookup = _original_fold_lookup(original_fold_plan)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    seen: set[tuple[str, str, str, int]] = set()
    state_features: dict[int, tuple[str, str]] = {}

    for source_row in prediction_rows:
        row = dict(source_row)
        forbidden = set(row) & FORBIDDEN_PREDICTION_FIELDS
        if forbidden:
            raise ValueError(
                "sealed trainer prediction leaks forbidden original labels: "
                f"{sorted(forbidden)}"
            )
        label, _ = _curve_contract(row)
        order = int(row.get("selection_order", -1))
        if not 0 <= order < 100:
            raise ValueError("prediction selection_order is outside original100")
        target = ordered_targets[order]
        _prediction_identity(row, target)
        feature_identity = (str(row["feature_id"]), str(row["feature_record_sha256"]))
        if order in state_features and state_features[order] != feature_identity:
            raise ValueError("one original state is bound to inconsistent feature rows")
        state_features[order] = feature_identity

        scheme = str(row.get("outer_scheme"))
        model = str(row.get("model_name"))
        expected_fold = fold_lookup.get((scheme, order))
        if expected_fold is None or int(row.get("fold_id", -1)) != expected_fold[0]:
            raise ValueError("prediction does not use the locked original fold")
        expected_test_group = (
            original_core.group_id(target) if scheme == "task_heldout" else str(target["suite"])
        )
        if row.get("test_group") != expected_test_group:
            raise ValueError("prediction test_group differs from locked original identity")
        uniqueness = (label, scheme, model, order)
        if uniqueness in seen:
            raise ValueError(f"duplicate follow-up prediction {uniqueness}")
        seen.add(uniqueness)

        # These fields are deliberately unavailable to the trainer.  Injecting
        # them here permits exact reuse of the independently audited Phase-3
        # metric implementation without weakening the training boundary.
        row["target_id"] = str(target["target_id"])
        row["target_sha256"] = str(target["target_sha256"])
        key = (label, scheme, model)
        grouped.setdefault(key, []).append(row)

    required: dict[tuple[str, str, str], int] = {}
    for model in phase3.TASK_MODELS:
        required[("q100", "task_heldout", model)] = 100
    required[("q100", "suite_heldout", phase3.PRIMARY_MODEL)] = 100
    for label in ("q25", "q50", "q75"):
        required[(label, "task_heldout", phase3.PRIMARY_MODEL)] = 100
    if set(grouped) != set(required):
        missing = sorted(set(required) - set(grouped))
        extra = sorted(set(grouped) - set(required))
        raise ValueError(f"prediction panel differs from frozen 1200-row contract: missing={missing}, extra={extra}")
    for key, expected_count in required.items():
        values = sorted(grouped[key], key=lambda row: int(row["selection_order"]))
        if len(values) != expected_count or [int(row["selection_order"]) for row in values] != list(range(100)):
            raise ValueError(f"prediction group {key} is not exact original100 OOF")
        grouped[key] = values
    if len(state_features) != 100:
        raise ValueError("follow-up predictions do not bind all 100 original feature rows")
    return grouped


def _legacy_q100_panel(
    grouped: Mapping[tuple[str, str, str], Sequence[Mapping[str, Any]]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in phase3.TASK_MODELS:
        rows.extend(dict(row) for row in grouped[("q100", "task_heldout", model)])
    rows.extend(
        dict(row)
        for row in grouped[("q100", "suite_heldout", phase3.PRIMARY_MODEL)]
    )
    if len(rows) != 900:
        raise AssertionError("internal q100 panel count differs from 900")
    return rows


def _learning_curve_metrics(
    grouped: Mapping[tuple[str, str, str], Sequence[Mapping[str, Any]]],
    truth: np.ndarray,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for label, fraction in CURVES:
        rows = grouped[(label, "task_heldout", phase3.PRIMARY_MODEL)]
        prediction = np.asarray([float(row["prediction"]) for row in rows], dtype=np.float64)
        metrics = phase3._model_metrics(rows, truth)
        output.append(
            {
                "curve_label": label,
                "curve_fraction": fraction,
                "state_count": len(rows),
                **metrics,
            }
        )
    return output


def _sample_size_attribution(
    grouped: Mapping[tuple[str, str, str], Sequence[Mapping[str, Any]]],
    truth: np.ndarray,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    q100_rows = grouped[("q100", "task_heldout", phase3.PRIMARY_MODEL)]
    q25_rows = grouped[("q25", "task_heldout", phase3.PRIMARY_MODEL)]
    q100 = np.asarray([float(row["prediction"]) for row in q100_rows], dtype=np.float64)
    q25 = np.asarray([float(row["prediction"]) for row in q25_rows], dtype=np.float64)
    q100_rho = phase3._rank_metrics(q100, truth)["spearman"]
    q25_rho = phase3._rank_metrics(q25, truth)["spearman"]
    observed = q100_rho - q25_rho
    samples = phase3._cluster_bootstrap_indices(
        q100_rows, replicates=bootstrap_replicates, seed=bootstrap_seed
    )
    interval = phase3._percentile_interval(
        (
            phase3._rank_metrics(q100[index], truth[index])["spearman"]
            - phase3._rank_metrics(q25[index], truth[index])["spearman"]
            for index in samples
        ),
        expected_replicates=bootstrap_replicates,
    )
    checks = [
        phase3._check(
            "delta_rho_q100_minus_q25",
            observed,
            ">=",
            SAMPLE_DELTA_THRESHOLD,
        ),
        phase3._check(
            "paired_task_bootstrap_lower",
            interval["lower_95"],
            ">",
            0.0,
        ),
    ]
    return {
        "q100_spearman": q100_rho,
        "q25_spearman": q25_rho,
        "delta_spearman": observed,
        "paired_task_bootstrap": interval,
        "checks": checks,
        "supported": all(check["passed"] for check in checks),
    }


def compute_followup_metrics(
    targets: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    original_fold_plan: Mapping[str, Any],
    *,
    bootstrap_replicates: int = phase3.BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = phase3.BOOTSTRAP_SEED,
    permutation_replicates: int = phase3.PERMUTATION_REPLICATES,
    permutation_seed: int = phase3.PERMUTATION_SEED,
    enforce_preregistered: bool = True,
) -> dict[str, Any]:
    if enforce_preregistered and (
        bootstrap_replicates != phase3.BOOTSTRAP_REPLICATES
        or bootstrap_seed != phase3.BOOTSTRAP_SEED
        or permutation_replicates != phase3.PERMUTATION_REPLICATES
        or permutation_seed != phase3.PERMUTATION_SEED
    ):
        raise ValueError("formal analysis requires exact Phase-3 resampling settings")
    grouped = validate_and_group_predictions(
        prediction_rows, targets, original_fold_plan
    )
    phase3_metrics = phase3.compute_metrics(
        targets,
        validation_rows,
        _legacy_q100_panel(grouped),
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        permutation_replicates=permutation_replicates,
        permutation_seed=permutation_seed,
        enforce_preregistered=enforce_preregistered,
    )
    truth = phase3._validation_means(targets, validation_rows)
    learning_curve = _learning_curve_metrics(grouped, truth)
    attribution = _sample_size_attribution(
        grouped,
        truth,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    primary_decision = str(phase3_metrics["decision"])
    if primary_decision == "GO":
        decision = (
            DECISION_SAMPLE_SIZE_SUPPORTED
            if attribution["supported"]
            else DECISION_V1_LEARNABLE
        )
    elif primary_decision == "CONDITIONAL":
        decision = DECISION_CONDITIONAL
    elif primary_decision == "NO_GO":
        decision = DECISION_NO_GO
    else:
        raise ValueError(f"unexpected Phase-3 decision {primary_decision!r}")
    return {
        "decision": decision,
        "primary_phase3_decision": primary_decision,
        "primary_phase3": {
            key: value for key, value in phase3_metrics.items() if key != "per_state"
        },
        "sample_size_attribution": attribution,
        "learning_curve": learning_curve,
        "exploratory_505": {
            "status": "not_run",
            "reason": "formal scope is exact_v1_137 only",
            "affects_primary_decision": False,
        },
        "per_state": phase3_metrics["per_state"],
    }


def _assert_followup_bindings(
    manifest: Mapping[str, Any],
    original_run_root: Path,
    expected_original_run_completion_sha256: str,
) -> None:
    """Bind the new seal to its declared inputs and frozen original folds."""

    compatibility = _require_mapping(
        manifest.get("compatibility"), field="follow-up compatibility"
    )
    if manifest.get("compatibility_fingerprint") != _sha256_json(compatibility):
        raise ValueError("follow-up compatibility fingerprint is invalid")
    # A self-consistent seal may not omit an upstream scientific binding.
    for field in REQUIRED_FOLLOWUP_SHA_BINDINGS:
        _require_sha(compatibility.get(field), field=f"compatibility.{field}")
    if compatibility.get("extractor_fingerprint") != (
        followup_core.EXPECTED_EXTRACTOR_FINGERPRINT
    ):
        raise ValueError("follow-up extractor fingerprint differs from exact V1")
    bound_completion = compatibility["original_fold_source_completion_sha256"]
    if bound_completion != expected_original_run_completion_sha256:
        raise ValueError("follow-up run is not bound to trusted original run completion")
    actual_fold_sha = _sha256_file(original_run_root / "fold_plan.json")
    if compatibility.get("original_fold_plan_sha256") != actual_fold_sha:
        raise ValueError("follow-up run is not bound to original sealed fold bytes")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(
    *,
    run_dir: str | Path,
    expected_run_completion_sha256: str,
    original_run_dir: str | Path,
    expected_original_run_completion_sha256: str,
    original_target_dir: str | Path,
    validation_dir: str | Path,
    expected_validation_manifest_sha256: str,
    expected_validation_records_sha256: str,
    expected_validation_completion_sha256: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Analyze a sealed run, touching Validation4 only after trainer validation."""

    expected = {
        "run completion": expected_run_completion_sha256,
        "original run completion": expected_original_run_completion_sha256,
        "validation manifest": expected_validation_manifest_sha256,
        "validation records": expected_validation_records_sha256,
        "validation completion": expected_validation_completion_sha256,
    }
    for field, value in expected.items():
        _require_sha(value, field=field)
    run_root = Path(run_dir).resolve()

    # Order is a scientific boundary: do not hash, stat, or load original
    # Target5/Validation4 until the prediction bundle is sealed and revalidated.
    run_manifest, followup_plan, prediction_rows = followup_core.load_sealed_run(
        run_root,
        expected_completion_sha256=expected_run_completion_sha256,
    )
    original_run_root = Path(original_run_dir).resolve()
    target_root = Path(original_target_dir).resolve()
    validation_root = Path(validation_dir).resolve()
    output_root = Path(output_dir).resolve()
    _assert_followup_bindings(
        run_manifest,
        original_run_root,
        expected_original_run_completion_sha256,
    )
    original_manifest, original_fold_plan, _ = original_core.load_sealed_run(
        original_run_root,
        expected_completion_sha256=expected_original_run_completion_sha256,
    )
    _assert_embedded_original_folds(followup_plan, original_fold_plan)

    validation_paths = {
        "validation manifest": validation_root / "manifest.json",
        "validation records": validation_root / "records.jsonl",
        "validation completion": validation_root / "completion.json",
    }
    for field, path in validation_paths.items():
        if _sha256_file(path) != expected[field]:
            raise ValueError(f"{field} file SHA-256 differs from expected")
    (
        target_manifest,
        targets,
        validation_manifest,
        validation_rows,
        validation_evidence,
    ) = target_analysis.validate_analysis_inputs(
        target_root,
        validation_root / "records.jsonl",
        validation_root / "manifest.json",
        errors_path=validation_root / "errors.jsonl",
        completion_path=validation_root / "completion.json",
    )
    original_core.validate_formal_run_contract(
        original_manifest, original_fold_plan, targets
    )
    # The new core authenticates its full remainder train/feature/plan contract;
    # prediction joins below independently prove use of the exact original fold.
    metrics = compute_followup_metrics(
        targets,
        validation_rows,
        prediction_rows,
        original_fold_plan,
    )
    summary = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "kind": ANALYSIS_KIND,
        "decision": metrics["decision"],
        "primary_phase3_decision": metrics["primary_phase3_decision"],
        "preregistration": {
            "document": PREREGISTRATION,
            "representation": REPRESENTATION,
            "primary_curve": PRIMARY_CURVE,
            "sample_reference_curve": SAMPLE_REFERENCE_CURVE,
            "sample_delta_threshold": SAMPLE_DELTA_THRESHOLD,
            "bootstrap_replicates": phase3.BOOTSTRAP_REPLICATES,
            "bootstrap_seed": phase3.BOOTSTRAP_SEED,
            "permutation_replicates": phase3.PERMUTATION_REPLICATES,
            "permutation_seed": phase3.PERMUTATION_SEED,
            "phase3_go_thresholds": phase3.GO_THRESHOLDS,
            "phase3_conditional_thresholds": phase3.CONDITIONAL_THRESHOLDS,
        },
        "integrity": {
            "status": "complete_and_verified",
            "run_completion_sha256": _sha256_file(run_root / "completion.json"),
            "original_run_completion_sha256": _sha256_file(
                original_run_root / "completion.json"
            ),
            "original_fold_plan_sha256": _sha256_file(
                original_run_root / "fold_plan.json"
            ),
            "original_target_manifest_sha256": _sha256_file(
                target_root / "manifest.json"
            ),
            "original_target_targets_sha256": _sha256_file(
                target_root / "targets.jsonl"
            ),
            "validation_manifest_sha256": _sha256_file(
                validation_root / "manifest.json"
            ),
            "validation_records_sha256": _sha256_file(
                validation_root / "records.jsonl"
            ),
            "validation_completion_sha256": _sha256_file(
                validation_root / "completion.json"
            ),
            "followup_compatibility_fingerprint": run_manifest.get(
                "compatibility_fingerprint"
            ),
            "followup_plan_sha256": _sha256_json(followup_plan),
            "original_target_manifest_fingerprint": target_manifest.get(
                "compatibility_fingerprint"
            ),
            "validation_manifest_fingerprint": validation_manifest.get(
                "compatibility_fingerprint"
            ),
            "prediction_count": len(prediction_rows),
            "original_state_count": len(targets),
            "validation_measurement_count": len(validation_rows),
            "validation_evidence": validation_evidence,
        },
        "results": {key: value for key, value in metrics.items() if key != "per_state"},
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output_root / "analysis_summary.json", summary)
    _write_csv(output_root / "per_state.csv", metrics["per_state"])
    _write_csv(output_root / "learning_curve.csv", metrics["learning_curve"])
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-run-completion-sha256", required=True)
    parser.add_argument("--original-run-dir", type=Path, required=True)
    parser.add_argument("--expected-original-run-completion-sha256", required=True)
    parser.add_argument("--original-target-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--expected-validation-manifest-sha256", required=True)
    parser.add_argument("--expected-validation-records-sha256", required=True)
    parser.add_argument("--expected-validation-completion-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = analyze(
        run_dir=args.run_dir,
        expected_run_completion_sha256=args.expected_run_completion_sha256,
        original_run_dir=args.original_run_dir,
        expected_original_run_completion_sha256=args.expected_original_run_completion_sha256,
        original_target_dir=args.original_target_dir,
        validation_dir=args.validation_dir,
        expected_validation_manifest_sha256=args.expected_validation_manifest_sha256,
        expected_validation_records_sha256=args.expected_validation_records_sha256,
        expected_validation_completion_sha256=args.expected_validation_completion_sha256,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "decision": summary["decision"],
                "primary_phase3_decision": summary["primary_phase3_decision"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
