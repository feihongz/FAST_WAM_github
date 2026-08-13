"""Independent Validation4 analysis for the offline LIBERO Tiny-MLP Gate.

This module is intentionally downstream of the trainer.  It is the only Phase-3
component allowed to load seeds 47--50.  The trainer first publishes a sealed
OOF bundle using Target5 (seeds 42--46); this analyzer then revalidates that
bundle and scores it against the independently collected Validation4 grid.

The readiness thresholds were frozen in
``docs/GATE_OFFLINE_TINY_MLP_FEASIBILITY.md`` before feature extraction or
training.  A ``GO`` here authorizes only the Pilot-500 *offline* experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Final, Iterable, Mapping, Sequence

import numpy as np
from scipy import stats

from experiments.libero.gate import analyze_demo_utility_target_v2 as target_analysis
from experiments.libero.gate import offline_tiny_mlp as gate_core


ANALYSIS_SCHEMA_VERSION: Final = 1
ANALYSIS_KIND: Final = "libero_gate_offline_tiny_mlp_validation"
PRIMARY_EPSILON: Final = 1e-4
BOOTSTRAP_REPLICATES: Final = 2_000
BOOTSTRAP_SEED: Final = 20_260_813
PERMUTATION_REPLICATES: Final = 5_000
PERMUTATION_SEED: Final = 20_260_814
RANDOM_NAMESPACE: Final = "libero_gate_random_v1"
RANDOM_SALTS: Final = 1_000
TAIL_FRACTION: Final = 0.20
MIN_EVALUABLE_WITHIN_TASK_PAIRS: Final = 30

TASK_MODELS: Final = (
    "full_hybrid",
    "full_huber",
    "visual_proprio_hybrid",
    "instruction_proprio_hybrid",
    "instruction_only_hybrid",
    "constant_train_mean",
    "suite_mean_fallback",
    "task_lookup_fallback",
)
PRIMARY_MODEL: Final = "full_hybrid"
BEST_NONVISUAL_MODEL: Final = "instruction_proprio_hybrid"

GO_THRESHOLDS: Final = {
    "spearman": 0.30,
    "spearman_bootstrap_lower": 0.0,
    "permutation_p": 0.05,
    "within_task_pair_accuracy": 0.60,
    "within_task_pair_bootstrap_lower": 0.52,
    "hc_balanced_accuracy": 0.70,
    "hc_positive_recall": 0.60,
    "hc_negative_recall": 0.60,
    "top_recall": 0.40,
    "bottom_recall": 0.40,
    "delta_spearman_nonvisual": 0.05,
    "delta_spearman_bootstrap_lower": 0.0,
    "selected_validation_mean": 0.0,
    "random_gain_bootstrap_lower": 0.0,
    "suite_heldout_pooled_spearman": 0.20,
    "suite_positive_fold_count": 3,
    "suite_min_fold_spearman": -0.20,
    "init_prediction_median_spearman": 0.80,
    "constant_mae_relative_improvement": 0.05,
}

CONDITIONAL_THRESHOLDS: Final = {
    "spearman": 0.20,
    "hc_balanced_accuracy": 0.65,
    "top_recall": 0.35,
    "bottom_recall": 0.35,
    "within_task_pair_accuracy": 0.55,
}


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


def deadband_sign(value: float, epsilon: float = PRIMARY_EPSILON) -> int:
    if not math.isfinite(float(value)) or not math.isfinite(float(epsilon)) or epsilon < 0:
        raise ValueError("deadband sign requires finite value and non-negative epsilon")
    if value > epsilon:
        return 1
    if value < -epsilon:
        return -1
    return 0


def _rank_metrics(prediction: Sequence[float], truth: Sequence[float]) -> dict[str, float]:
    pred = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(truth, dtype=np.float64)
    if pred.ndim != 1 or target.shape != pred.shape or pred.size < 2:
        raise ValueError("rank metrics require same-shaped one-dimensional arrays")
    if not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise ValueError("rank metrics require finite values")
    spearman = float(stats.spearmanr(pred, target).statistic)
    kendall = float(stats.kendalltau(pred, target).statistic)
    return {"spearman": spearman, "kendall": kendall}


def _task_key(row: Mapping[str, Any]) -> tuple[str, int, str]:
    return (str(row["suite"]), int(row["task_index"]), str(row["task"]))


def _cluster_bootstrap_indices(
    rows: Sequence[Mapping[str, Any]],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> list[np.ndarray]:
    """Suite-stratified task-cluster bootstrap indices.

    Whole tasks are sampled with replacement independently within each suite.
    Duplicate sampled tasks duplicate every state in that task.
    """

    by_suite: dict[str, dict[tuple[str, int, str], list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, row in enumerate(rows):
        by_suite[str(row["suite"])][_task_key(row)].append(index)
    if len(by_suite) != 4 or any(len(groups) != 10 for groups in by_suite.values()):
        raise ValueError("expected exactly 4 suites x 10 task clusters")
    ordered = {
        suite: [np.asarray(groups[key], dtype=np.int64) for key in sorted(groups)]
        for suite, groups in sorted(by_suite.items())
    }
    rng = np.random.default_rng(seed)
    samples: list[np.ndarray] = []
    for _ in range(replicates):
        chunks: list[np.ndarray] = []
        for clusters in ordered.values():
            choices = rng.integers(0, len(clusters), size=len(clusters))
            chunks.extend(clusters[int(choice)] for choice in choices)
        samples.append(np.concatenate(chunks))
    return samples


def _percentile_interval(
    values: Iterable[float], *, expected_replicates: int
) -> dict[str, float | int]:
    if isinstance(expected_replicates, bool) or int(expected_replicates) < 1:
        raise ValueError("expected_replicates must be a positive integer")
    vector = np.asarray([float(value) for value in values], dtype=np.float64)
    requested = int(expected_replicates)
    effective = int(np.isfinite(vector).sum())
    if vector.size != requested:
        raise ValueError(
            f"bootstrap produced {vector.size} values, expected exactly {requested}"
        )
    if effective != requested:
        raise ValueError(
            f"bootstrap has {effective} finite replicates, expected exactly {requested}"
        )
    return {
        "lower_95": float(np.quantile(vector, 0.025)),
        "median": float(np.quantile(vector, 0.5)),
        "upper_95": float(np.quantile(vector, 0.975)),
        "requested_replicates": requested,
        "effective_replicates": effective,
        "replicates": effective,
    }


def _random_score(sample_id: str, source_index: int, salt: int) -> float:
    payload = f"{RANDOM_NAMESPACE}\0{salt:04d}\0{sample_id}\0{source_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64


def _tail_overlap(
    prediction: np.ndarray, truth: np.ndarray, *, fraction: float = TAIL_FRACTION
) -> dict[str, float | int]:
    if prediction.shape != truth.shape or prediction.ndim != 1:
        raise ValueError("tail overlap requires matching vectors")
    count = max(1, int(round(prediction.size * fraction)))
    pred_order = np.argsort(prediction, kind="stable")
    truth_order = np.argsort(truth, kind="stable")

    def side(pred_idx: np.ndarray, truth_idx: np.ndarray) -> tuple[float, float]:
        pred_set, truth_set = set(map(int, pred_idx)), set(map(int, truth_idx))
        intersection = len(pred_set & truth_set)
        return intersection / count, intersection / len(pred_set | truth_set)

    bottom_recall, bottom_jaccard = side(pred_order[:count], truth_order[:count])
    top_recall, top_jaccard = side(pred_order[-count:], truth_order[-count:])
    return {
        "tail_count": count,
        "top_recall": top_recall,
        "top_jaccard": top_jaccard,
        "bottom_recall": bottom_recall,
        "bottom_jaccard": bottom_jaccard,
    }


def _prediction_rows_by_model(
    rows: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
    target_by_sample = {str(row["sample_id"]): row for row in targets}
    if len(target_by_sample) != 100:
        raise ValueError("expected exactly 100 unique Target5 states")
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        scheme = str(row["outer_scheme"])
        model = str(row["model_name"])
        sample_id = str(row["sample_id"])
        key = (scheme, model, sample_id)
        if key in seen:
            raise ValueError(f"duplicate OOF prediction {key}")
        seen.add(key)
        if sample_id not in target_by_sample:
            raise ValueError(f"OOF prediction is outside Target5: {sample_id}")
        target = target_by_sample[sample_id]
        exact = {
            "source_index": int(target["source_index"]),
            "suite": str(target["suite"]),
            "task_index": int(target["task_index"]),
            "task": str(target["task"]),
            "target_id": str(target["target_id"]),
            "target_sha256": str(target["target_sha256"]),
            "input_combined_sha256": str(target["input_hashes"]["combined"]),
        }
        for field, expected in exact.items():
            if row.get(field) != expected:
                raise ValueError(f"OOF {sample_id} {field} does not match Target5")
        prediction = float(row["prediction"])
        init_predictions = np.asarray(row["init_predictions"], dtype=np.float64)
        if not math.isfinite(prediction) or init_predictions.shape != (5,):
            raise ValueError("OOF prediction/init_predictions are invalid")
        if not np.isfinite(init_predictions).all() or not math.isclose(
            prediction, float(init_predictions.mean()), rel_tol=1e-7, abs_tol=1e-9
        ):
            raise ValueError("OOF ensemble prediction is not the five-init mean")
        grouped[(scheme, model)].append(row)

    for model in TASK_MODELS:
        key = ("task_heldout", model)
        if len(grouped.get(key, [])) != 100:
            raise ValueError(f"expected 100 task-heldout rows for {model}")
    if len(grouped.get(("suite_heldout", PRIMARY_MODEL), [])) != 100:
        raise ValueError("expected 100 suite-heldout rows for full_hybrid")
    allowed = {("task_heldout", model) for model in TASK_MODELS} | {
        ("suite_heldout", PRIMARY_MODEL)
    }
    extra = set(grouped) - allowed
    if extra or len(rows) != 900:
        raise ValueError(f"OOF bundle has unexpected groups/count: {sorted(extra)}")
    order = {str(target["sample_id"]): index for index, target in enumerate(targets)}
    return {
        key: sorted(value, key=lambda row: order[str(row["sample_id"])])
        for key, value in grouped.items()
    }


def _validation_means(
    targets: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    by_source: dict[int, list[float]] = defaultdict(list)
    for row in validation_rows:
        by_source[int(row["source_index"])].append(float(row["utility"]))
    result = []
    for target in targets:
        values = by_source[int(target["source_index"])]
        if len(values) != 4 or not np.isfinite(values).all():
            raise ValueError("Validation4 must contain four finite utilities per state")
        result.append(float(np.mean(values)))
    return np.asarray(result, dtype=np.float64)


def _model_metrics(
    rows: Sequence[Mapping[str, Any]], truth: np.ndarray
) -> dict[str, Any]:
    prediction = np.asarray([float(row["prediction"]) for row in rows])
    rank: dict[str, float | None]
    if float(np.ptp(prediction)) == 0.0 or float(np.ptp(truth)) == 0.0:
        rank = {"spearman": None, "kendall": None}
    else:
        rank = _rank_metrics(prediction, truth)
    error = prediction - truth
    return {
        **rank,
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        **_tail_overlap(prediction, truth),
    }


def _task_permutation_pvalue(
    rows: Sequence[Mapping[str, Any]],
    prediction: np.ndarray,
    truth: np.ndarray,
    *,
    replicates: int = PERMUTATION_REPLICATES,
    seed: int = PERMUTATION_SEED,
) -> dict[str, Any]:
    observed = _rank_metrics(prediction, truth)["spearman"]
    blocks: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
    task_indices: dict[tuple[str, int, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        task_indices[_task_key(row)].append(index)
    for key in sorted(task_indices):
        indices = np.asarray(
            sorted(task_indices[key], key=lambda idx: str(rows[idx]["sample_id"])),
            dtype=np.int64,
        )
        blocks[(key[0], len(indices))].append(indices)
    rng = np.random.default_rng(seed)
    exceed = 0
    finite = 0
    for _ in range(replicates):
        permuted = np.empty_like(prediction)
        for group_blocks in blocks.values():
            permutation = rng.permutation(len(group_blocks))
            for destination, source_position in zip(group_blocks, permutation, strict=True):
                source = group_blocks[int(source_position)]
                permuted[destination] = prediction[source]
        statistic = float(stats.spearmanr(permuted, truth).statistic)
        if math.isfinite(statistic):
            finite += 1
            exceed += int(statistic >= observed)
    if finite != replicates:
        raise ValueError("non-finite task permutation statistic")
    return {
        "observed_spearman": observed,
        "p_value_one_sided": (exceed + 1) / (replicates + 1),
        "replicates": replicates,
        "seed": seed,
    }


def _within_task_pairs(
    rows: Sequence[Mapping[str, Any]],
    prediction: np.ndarray,
    truth: np.ndarray,
    *,
    epsilon: float = PRIMARY_EPSILON,
) -> tuple[dict[str, Any], dict[tuple[str, int, str], list[bool]]]:
    by_task: dict[tuple[str, int, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_task[_task_key(row)].append(index)
    outcomes: dict[tuple[str, int, str], list[bool]] = {}
    total_pairs = 0
    excluded = 0
    for task, indices in sorted(by_task.items()):
        task_outcomes: list[bool] = []
        for left_position, left in enumerate(indices):
            for right in indices[left_position + 1 :]:
                total_pairs += 1
                delta_truth = float(truth[left] - truth[right])
                if abs(delta_truth) <= epsilon:
                    excluded += 1
                    continue
                delta_prediction = float(prediction[left] - prediction[right])
                task_outcomes.append(delta_prediction * delta_truth > 0.0)
        outcomes[task] = task_outcomes
    flattened = [value for values in outcomes.values() for value in values]
    return (
        {
            "total_pairs": total_pairs,
            "deadband_excluded_pairs": excluded,
            "evaluable_pairs": len(flattened),
            "accuracy": float(np.mean(flattened)) if flattened else None,
        },
        outcomes,
    )


def _within_task_bootstrap(
    outcomes: Mapping[tuple[str, int, str], Sequence[bool]],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    by_suite: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    for key in sorted(outcomes):
        by_suite[key[0]].append(key)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(replicates):
        sampled: list[bool] = []
        for tasks in by_suite.values():
            choices = rng.integers(0, len(tasks), size=len(tasks))
            for choice in choices:
                sampled.extend(outcomes[tasks[int(choice)]])
        if sampled:
            values.append(float(np.mean(sampled)))
    return _percentile_interval(values, expected_replicates=replicates)


def _hc_sign_metrics(
    targets: Sequence[Mapping[str, Any]], prediction: np.ndarray, truth: np.ndarray
) -> dict[str, Any]:
    frozen_hc = np.asarray([bool(target["high_confidence"]) for target in targets])
    truth_sign = np.asarray([deadband_sign(value) for value in truth])
    pred_sign = np.asarray([deadband_sign(value) for value in prediction])
    evaluable = frozen_hc & (truth_sign != 0)
    positive = evaluable & (truth_sign == 1)
    negative = evaluable & (truth_sign == -1)
    positive_recall = float(np.mean(pred_sign[positive] == 1)) if positive.any() else None
    negative_recall = float(np.mean(pred_sign[negative] == -1)) if negative.any() else None
    balanced = (
        (positive_recall + negative_recall) / 2
        if positive_recall is not None and negative_recall is not None
        else None
    )
    return {
        "frozen_target5_hc_count": int(frozen_hc.sum()),
        "validation_deadband_in_hc_count": int((frozen_hc & (truth_sign == 0)).sum()),
        "evaluable_count": int(evaluable.sum()),
        "validation_positive_count": int(positive.sum()),
        "validation_negative_count": int(negative.sum()),
        "balanced_accuracy": balanced,
        "positive_recall": positive_recall,
        "negative_recall": negative_recall,
    }


def _init_prediction_agreement(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    matrix = np.asarray([row["init_predictions"] for row in rows], dtype=np.float64)
    if matrix.shape != (100, 5):
        raise ValueError("init prediction matrix must be [100,5]")
    correlations = []
    for left in range(5):
        for right in range(left + 1, 5):
            correlations.append(float(stats.spearmanr(matrix[:, left], matrix[:, right]).statistic))
    return {
        "pair_count": len(correlations),
        "pairwise_spearman": correlations,
        "median_pairwise_spearman": float(np.median(correlations)),
        "minimum_pairwise_spearman": float(np.min(correlations)),
    }


def _check(name: str, value: Any, operator: str, threshold: float) -> dict[str, Any]:
    finite = value is not None and math.isfinite(float(value))
    if operator == ">=":
        passed = finite and float(value) >= threshold
    elif operator == ">":
        passed = finite and float(value) > threshold
    elif operator == "<=":
        passed = finite and float(value) <= threshold
    else:
        raise ValueError(f"unsupported operator {operator}")
    return {
        "name": name,
        "value": value,
        "operator": operator,
        "threshold": threshold,
        "passed": bool(passed),
    }


def compute_metrics(
    targets: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    permutation_replicates: int = PERMUTATION_REPLICATES,
    permutation_seed: int = PERMUTATION_SEED,
    enforce_preregistered: bool = True,
) -> dict[str, Any]:
    if enforce_preregistered and (
        bootstrap_replicates != BOOTSTRAP_REPLICATES
        or bootstrap_seed != BOOTSTRAP_SEED
    ):
        raise ValueError("formal analysis must use preregistered bootstrap settings")
    if enforce_preregistered and (
        permutation_replicates != PERMUTATION_REPLICATES
        or permutation_seed != PERMUTATION_SEED
    ):
        raise ValueError("formal analysis must use preregistered permutation settings")
    grouped = _prediction_rows_by_model(prediction_rows, targets)
    truth = _validation_means(targets, validation_rows)
    primary_rows = grouped[("task_heldout", PRIMARY_MODEL)]
    primary = np.asarray([float(row["prediction"]) for row in primary_rows])
    nonvisual = np.asarray(
        [float(row["prediction"]) for row in grouped[("task_heldout", BEST_NONVISUAL_MODEL)]]
    )
    constant = np.asarray(
        [float(row["prediction"]) for row in grouped[("task_heldout", "constant_train_mean")]]
    )
    bootstrap_indices = _cluster_bootstrap_indices(
        primary_rows, replicates=bootstrap_replicates, seed=bootstrap_seed
    )
    primary_bootstrap = _percentile_interval(
        (
            _rank_metrics(primary[index], truth[index])["spearman"]
            for index in bootstrap_indices
        ),
        expected_replicates=bootstrap_replicates,
    )
    delta_observed = (
        _rank_metrics(primary, truth)["spearman"]
        - _rank_metrics(nonvisual, truth)["spearman"]
    )
    delta_bootstrap = _percentile_interval(
        (
            _rank_metrics(primary[index], truth[index])["spearman"]
            - _rank_metrics(nonvisual[index], truth[index])["spearman"]
            for index in bootstrap_indices
        ),
        expected_replicates=bootstrap_replicates,
    )
    pair_metrics, pair_outcomes = _within_task_pairs(primary_rows, primary, truth)
    pair_metrics["bootstrap"] = _within_task_bootstrap(
        pair_outcomes, replicates=bootstrap_replicates, seed=bootstrap_seed
    )
    tail = _tail_overlap(primary, truth)
    top_count = int(tail["tail_count"])
    selected = np.argsort(primary, kind="stable")[-top_count:]
    selected_mean = float(np.mean(truth[selected]))
    random_matrix = np.asarray(
        [
            [
                _random_score(str(row["sample_id"]), int(row["source_index"]), salt)
                for row in primary_rows
            ]
            for salt in range(RANDOM_SALTS)
        ],
        dtype=np.float64,
    )
    random_selected_means = np.asarray(
        [float(np.mean(truth[np.argsort(scores, kind="stable")[-top_count:]])) for scores in random_matrix]
    )
    random_mean = float(random_selected_means.mean())
    random_gain = selected_mean - random_mean
    gain_bootstrap_values: list[float] = []
    for index in bootstrap_indices:
        k = max(1, int(round(len(index) * TAIL_FRACTION)))
        selected_index = index[np.argsort(primary[index], kind="stable")[-k:]]
        model_value = float(np.mean(truth[selected_index]))
        score_subset = random_matrix[:, index]
        selected_positions = np.argpartition(score_subset, -k, axis=1)[:, -k:]
        truth_subset = truth[index]
        random_values = truth_subset[selected_positions].mean(axis=1)
        gain_bootstrap_values.append(model_value - float(random_values.mean()))
    random_gain_bootstrap = _percentile_interval(
        gain_bootstrap_values, expected_replicates=bootstrap_replicates
    )

    suite_rows = grouped[("suite_heldout", PRIMARY_MODEL)]
    suite_prediction = np.asarray([float(row["prediction"]) for row in suite_rows])
    suite_pooled = _rank_metrics(suite_prediction, truth)["spearman"]
    suite_folds: dict[str, float] = {}
    for suite in sorted({str(row["suite"]) for row in suite_rows}):
        mask = np.asarray([str(row["suite"]) == suite for row in suite_rows])
        suite_folds[suite] = _rank_metrics(suite_prediction[mask], truth[mask])["spearman"]

    models = {
        model: _model_metrics(grouped[("task_heldout", model)], truth)
        for model in TASK_MODELS
    }
    primary_mae = float(models[PRIMARY_MODEL]["mae"])
    constant_mae = float(models["constant_train_mean"]["mae"])
    constant_improvement = (constant_mae - primary_mae) / constant_mae
    permutation = _task_permutation_pvalue(
        primary_rows,
        primary,
        truth,
        replicates=permutation_replicates,
        seed=permutation_seed,
    )
    hc = _hc_sign_metrics(targets, primary, truth)
    init_agreement = _init_prediction_agreement(primary_rows)

    checks = [
        _check("spearman", models[PRIMARY_MODEL]["spearman"], ">=", GO_THRESHOLDS["spearman"]),
        _check("spearman_bootstrap_lower", primary_bootstrap["lower_95"], ">", 0.0),
        _check("permutation_p", permutation["p_value_one_sided"], "<=", GO_THRESHOLDS["permutation_p"]),
        _check("within_task_evaluable_pairs", pair_metrics["evaluable_pairs"], ">=", MIN_EVALUABLE_WITHIN_TASK_PAIRS),
        _check("within_task_pair_accuracy", pair_metrics["accuracy"], ">=", GO_THRESHOLDS["within_task_pair_accuracy"]),
        _check("within_task_pair_bootstrap_lower", pair_metrics["bootstrap"]["lower_95"], ">=", GO_THRESHOLDS["within_task_pair_bootstrap_lower"]),
        _check("hc_balanced_accuracy", hc["balanced_accuracy"], ">=", GO_THRESHOLDS["hc_balanced_accuracy"]),
        _check("hc_positive_recall", hc["positive_recall"], ">=", GO_THRESHOLDS["hc_positive_recall"]),
        _check("hc_negative_recall", hc["negative_recall"], ">=", GO_THRESHOLDS["hc_negative_recall"]),
        _check("top_recall", tail["top_recall"], ">=", GO_THRESHOLDS["top_recall"]),
        _check("bottom_recall", tail["bottom_recall"], ">=", GO_THRESHOLDS["bottom_recall"]),
        _check("delta_spearman_nonvisual", delta_observed, ">=", GO_THRESHOLDS["delta_spearman_nonvisual"]),
        _check("delta_spearman_bootstrap_lower", delta_bootstrap["lower_95"], ">", 0.0),
        _check("selected_validation_mean", selected_mean, ">", 0.0),
        _check("random_gain_bootstrap_lower", random_gain_bootstrap["lower_95"], ">", 0.0),
        _check("suite_heldout_pooled_spearman", suite_pooled, ">=", GO_THRESHOLDS["suite_heldout_pooled_spearman"]),
        _check("suite_positive_fold_count", sum(value > 0 for value in suite_folds.values()), ">=", GO_THRESHOLDS["suite_positive_fold_count"]),
        _check("suite_min_fold_spearman", min(suite_folds.values()), ">=", GO_THRESHOLDS["suite_min_fold_spearman"]),
        _check("init_prediction_median_spearman", init_agreement["median_pairwise_spearman"], ">=", GO_THRESHOLDS["init_prediction_median_spearman"]),
        _check("constant_mae_relative_improvement", constant_improvement, ">=", GO_THRESHOLDS["constant_mae_relative_improvement"]),
    ]
    go = all(check["passed"] for check in checks)
    conditional_core = {
        "spearman": models[PRIMARY_MODEL]["spearman"] >= CONDITIONAL_THRESHOLDS["spearman"],
        "hc_balanced_accuracy": (hc["balanced_accuracy"] or -math.inf) >= CONDITIONAL_THRESHOLDS["hc_balanced_accuracy"],
        "top_recall": tail["top_recall"] >= CONDITIONAL_THRESHOLDS["top_recall"],
        "bottom_recall": tail["bottom_recall"] >= CONDITIONAL_THRESHOLDS["bottom_recall"],
        "within_task_pair_accuracy": (pair_metrics["accuracy"] or -math.inf) >= CONDITIONAL_THRESHOLDS["within_task_pair_accuracy"],
        "beats_random": random_gain_bootstrap["lower_95"] > 0.0,
    }
    decision = "GO" if go else "CONDITIONAL" if all(conditional_core.values()) else "NO_GO"
    per_state = []
    target5 = np.asarray([float(target["utility_mean"]) for target in targets])
    for index, (target, row) in enumerate(zip(targets, primary_rows, strict=True)):
        per_state.append(
            {
                "selection_order": index,
                "sample_id": str(target["sample_id"]),
                "source_index": int(target["source_index"]),
                "suite": str(target["suite"]),
                "task_index": int(target["task_index"]),
                "task": str(target["task"]),
                "target5_mean": float(target5[index]),
                "target5_high_confidence": bool(target["high_confidence"]),
                "validation4_mean": float(truth[index]),
                "prediction": float(primary[index]),
                "prediction_sign": deadband_sign(primary[index]),
                "validation_sign": deadband_sign(truth[index]),
                "absolute_error": abs(float(primary[index] - truth[index])),
                "fold_id": int(row["fold_id"]),
            }
        )
    effective_by_metric = {
        "primary_spearman": int(primary_bootstrap["effective_replicates"]),
        "delta_spearman_nonvisual": int(delta_bootstrap["effective_replicates"]),
        "within_task_pair_accuracy": int(
            pair_metrics["bootstrap"]["effective_replicates"]
        ),
        "matched_random_gain": int(random_gain_bootstrap["effective_replicates"]),
    }
    if set(effective_by_metric.values()) != {int(bootstrap_replicates)}:
        raise ValueError("bootstrap effective replicate counts differ across metrics")
    return {
        "decision": decision,
        "resampling": {
            "bootstrap_requested_replicates": int(bootstrap_replicates),
            "bootstrap_effective_replicates": int(bootstrap_replicates),
            "bootstrap_effective_by_metric": effective_by_metric,
            "permutation_requested_replicates": int(permutation_replicates),
            "permutation_effective_replicates": int(permutation["replicates"]),
        },
        "go_checks": checks,
        "go_pass_count": sum(check["passed"] for check in checks),
        "go_check_count": len(checks),
        "conditional_checks": conditional_core,
        "primary": {
            "model": PRIMARY_MODEL,
            "metrics": models[PRIMARY_MODEL],
            "spearman_bootstrap": primary_bootstrap,
            "task_permutation": permutation,
            "within_task_pairs": pair_metrics,
            "hc_sign": hc,
            "tail": tail,
            "selected_validation_mean": selected_mean,
            "matched_random_selected_mean": random_mean,
            "matched_random_gain": random_gain,
            "matched_random_gain_bootstrap": random_gain_bootstrap,
            "init_agreement": init_agreement,
            "constant_mae_relative_improvement": constant_improvement,
        },
        "nonvisual_comparison": {
            "model": BEST_NONVISUAL_MODEL,
            "spearman": models[BEST_NONVISUAL_MODEL]["spearman"],
            "delta_spearman": delta_observed,
            "delta_spearman_bootstrap": delta_bootstrap,
        },
        "suite_heldout": {
            "pooled_spearman": suite_pooled,
            "fold_spearman": suite_folds,
        },
        "model_metrics": models,
        "target5_secondary": _rank_metrics(primary, target5),
        "per_state": per_state,
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write empty CSV")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(
    *,
    run_dir: str | Path,
    expected_run_completion_sha256: str,
    target_dir: str | Path,
    validation_dir: str | Path,
    expected_validation_manifest_sha256: str,
    expected_validation_records_sha256: str,
    expected_validation_completion_sha256: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    run_root = Path(run_dir).resolve()
    target_root = Path(target_dir).resolve()
    validation_root = Path(validation_dir).resolve()
    output_root = Path(output_dir).resolve()
    expected_hashes = {
        "run completion": expected_run_completion_sha256,
        "validation manifest": expected_validation_manifest_sha256,
        "validation records": expected_validation_records_sha256,
        "validation completion": expected_validation_completion_sha256,
    }
    for label, expected in expected_hashes.items():
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
        ):
            raise ValueError(f"expected {label} SHA-256 must be lowercase hex")
    validation_paths = {
        "validation manifest": validation_root / "manifest.json",
        "validation records": validation_root / "records.jsonl",
        "validation completion": validation_root / "completion.json",
    }
    for label, path in validation_paths.items():
        if _sha256_file(path) != expected_hashes[label]:
            raise ValueError(f"{label} file SHA-256 differs from expected")
    run_manifest, fold_plan, prediction_rows = gate_core.load_sealed_run(
        run_root, expected_completion_sha256=expected_run_completion_sha256
    )
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
    gate_core.validate_formal_run_contract(run_manifest, fold_plan, targets)
    run_compatibility = run_manifest.get("compatibility")
    if not isinstance(run_compatibility, Mapping):
        raise ValueError("sealed run manifest is missing compatibility")
    external_target_bindings = {
        "target_manifest_sha256": _sha256_file(target_root / "manifest.json"),
        "target_targets_sha256": _sha256_file(target_root / "targets.jsonl"),
        "target_compatibility_fingerprint": target_manifest.get(
            "compatibility_fingerprint"
        ),
        "target_records_sha256": target_manifest["targets"][
            "canonical_records_sha256"
        ],
    }
    for field, expected in external_target_bindings.items():
        if run_compatibility.get(field) != expected:
            raise ValueError(f"sealed run {field} differs from loaded Target5")
    # Rows provide a second, state-level binding.  The sealed-run loader has
    # already authenticated their feature and Target5 provenance.
    metrics = compute_metrics(targets, validation_rows, prediction_rows)
    summary = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "kind": ANALYSIS_KIND,
        "decision": metrics["decision"],
        "preregistration": {
            "document": "docs/GATE_OFFLINE_TINY_MLP_FEASIBILITY.md",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_effective_replicates": metrics["resampling"][
                "bootstrap_effective_replicates"
            ],
            "bootstrap_seed": BOOTSTRAP_SEED,
            "permutation_replicates": PERMUTATION_REPLICATES,
            "permutation_seed": PERMUTATION_SEED,
            "epsilon": PRIMARY_EPSILON,
            "random_namespace": RANDOM_NAMESPACE,
            "random_salts": RANDOM_SALTS,
        },
        "integrity": {
            "run_completion_sha256": _sha256_file(run_root / "completion.json"),
            "target_manifest_sha256": _sha256_file(target_root / "manifest.json"),
            "target_targets_sha256": _sha256_file(target_root / "targets.jsonl"),
            "validation_manifest_sha256": _sha256_file(validation_root / "manifest.json"),
            "validation_records_sha256": _sha256_file(validation_root / "records.jsonl"),
            "validation_completion_sha256": _sha256_file(validation_root / "completion.json"),
            "state_count": len(targets),
            "prediction_count": len(prediction_rows),
            "validation_measurement_count": len(validation_rows),
            "status": "complete_and_verified",
            "run_manifest_fingerprint": run_manifest.get("compatibility_fingerprint"),
            "fold_plan_sha256": _sha256_json(fold_plan),
            "target_manifest_fingerprint": target_manifest.get("compatibility_fingerprint"),
            "validation_manifest_fingerprint": validation_manifest.get("compatibility_fingerprint"),
            "validation_evidence": validation_evidence,
        },
        "results": {key: value for key, value in metrics.items() if key != "per_state"},
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output_root / "analysis_summary.json", summary)
    _write_csv(output_root / "per_state.csv", metrics["per_state"])
    model_rows = [
        {"model_name": model, **values}
        for model, values in metrics["model_metrics"].items()
    ]
    _write_csv(output_root / "model_metrics.csv", model_rows)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-run-completion-sha256", required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
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
        target_dir=args.target_dir,
        validation_dir=args.validation_dir,
        expected_validation_manifest_sha256=args.expected_validation_manifest_sha256,
        expected_validation_records_sha256=args.expected_validation_records_sha256,
        expected_validation_completion_sha256=args.expected_validation_completion_sha256,
        output_dir=args.output_dir,
    )
    print(json.dumps({"decision": summary["decision"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
