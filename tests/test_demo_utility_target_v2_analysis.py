from __future__ import annotations

import copy
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from experiments.libero.gate.analyze_demo_utility_target_v2 import (
    ALL_BASE_SEEDS,
    BOOTSTRAP_REPLICATES,
    GO_THRESHOLDS,
    PREREGISTERED_AT_UTC,
    PRIMARY_EPSILON,
    TARGET_BASE_SEEDS,
    VALIDATION_BASE_SEEDS,
    VALIDATION_MANIFEST_KIND,
    _canonical_json,
    _high_confidence_from_values,
    _plot_outputs,
    _sha256_json,
    _stratified_bootstrap_indices,
    _validate_zero_error_log,
    analyze,
    compute_target_v2_metrics,
    deadband_sign,
    decide_readiness,
    icc_absolute_agreement,
    icc_one_way,
    lin_concordance_correlation,
    ranking_overlap,
    validate_validation_manifest,
)
from experiments.libero.gate import demo_utility_target_v2 as target_core


def test_preregistration_timestamp_and_threshold_contract_are_frozen_utc():
    assert PREREGISTERED_AT_UTC == "2026-08-13T12:46:00Z"
    parsed = datetime.fromisoformat(PREREGISTERED_AT_UTC.replace("Z", "+00:00"))
    assert parsed.tzinfo == timezone.utc
    assert TARGET_BASE_SEEDS == (42, 43, 44, 45, 46)
    assert VALIDATION_BASE_SEEDS == (47, 48, 49, 50)
    assert ALL_BASE_SEEDS == tuple(range(42, 51))
    assert PRIMARY_EPSILON == 1e-4
    assert BOOTSTRAP_REPLICATES == 2000
    assert GO_THRESHOLDS["mean_spearman"] == 0.50
    assert GO_THRESHOLDS["all9_icc_1_9"] == 0.75


def test_formal_analyze_rejects_non_preregistered_bootstrap_settings(tmp_path):
    missing = tmp_path / "not-created"
    with pytest.raises(ValueError, match="bootstrap_seed=20260813"):
        analyze(missing, missing, missing, missing, bootstrap_seed=1)
    with pytest.raises(ValueError, match="bootstrap_replicates=2000"):
        analyze(
            missing,
            missing,
            missing,
            missing,
            bootstrap_replicates=10,
        )


def test_deadband_high_confidence_and_boundaries_are_exact():
    assert deadband_sign(PRIMARY_EPSILON) == 0
    assert deadband_sign(-PRIMARY_EPSILON) == 0
    assert deadband_sign(1.01e-4) == 1
    assert deadband_sign(-1.01e-4) == -1
    assert _high_confidence_from_values(
        np.asarray([0.0020, 0.0021, 0.0019, 0.0020, 0.0021]), PRIMARY_EPSILON
    )
    assert not _high_confidence_from_values(
        np.asarray([0.0020, 0.0021, -0.0020, -0.0021, 0.0020]), PRIMARY_EPSILON
    )


def test_ccc_absolute_icc_and_one_way_icc_have_known_behavior():
    signal = np.linspace(-2.0, 2.0, 30)
    assert lin_concordance_correlation(signal, signal) == pytest.approx(1.0)
    shifted = signal + 0.5
    assert 0.0 < lin_concordance_correlation(signal, shifted) < 1.0

    two_column = np.column_stack([signal, signal])
    absolute = icc_absolute_agreement(two_column)
    assert absolute["icc_a_1"] == pytest.approx(1.0)
    assert absolute["icc_a_k"] == pytest.approx(1.0)
    shifted_absolute = icc_absolute_agreement(np.column_stack([signal, shifted]))
    assert shifted_absolute["icc_a_1"] < 1.0

    repeated = np.repeat(signal[:, None], 9, axis=1)
    reliability = icc_one_way(repeated)
    assert reliability["icc_1_1"] == pytest.approx(1.0)
    assert reliability["icc_1_k"] == pytest.approx(1.0)
    rng = np.random.default_rng(8)
    noisy = signal[:, None] + rng.normal(scale=2.0, size=(30, 9))
    degraded = icc_one_way(noisy)
    assert degraded["icc_1_1"] < degraded["icc_1_k"] < 1.0


def test_stratified_bootstrap_preserves_each_stratum_count():
    strata = ["SP"] * 3 + ["SN"] * 4 + ["MP"] * 5 + ["MN"] * 6 + ["NZ"] * 7
    samples = _stratified_bootstrap_indices(strata, seed=17, replicates=10)
    labels = np.asarray(strata)
    for indices in samples:
        sampled = labels[indices]
        for label in sorted(set(strata)):
            assert np.sum(sampled == label) == np.sum(labels == label)


def test_ranking_overlap_is_exact_and_uses_preregistered_random_baseline():
    values = np.arange(100, dtype=np.float64)
    identical = ranking_overlap(values, values)
    assert identical["k"] == 20
    assert identical["top_recall"] == 1.0
    assert identical["bottom_jaccard"] == 1.0
    assert identical["random_expected_recall"] == pytest.approx(0.20)
    assert identical["random_expected_jaccard"] == pytest.approx(1 / 9)
    reversed_order = ranking_overlap(values, -values)
    assert reversed_order["top_recall"] == 0.0
    assert reversed_order["bottom_recall"] == 0.0


def _synthetic_panel() -> tuple[list[dict], list[dict]]:
    targets: list[dict] = []
    validation: list[dict] = []
    selection_bins = ("SP", "SN", "MP", "MN", "NZ")
    suites = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    utilities = np.linspace(-0.003, 0.003, 100)
    target_offsets = np.asarray([-2, -1, 0, 1, 2], dtype=float) * 2e-6
    validation_offsets = np.asarray([-1.5, -0.5, 0.5, 1.5], dtype=float) * 2e-6
    for index, utility in enumerate(utilities):
        values = utility + target_offsets
        entries = []
        for replicate_index, (base_seed, value) in enumerate(
            zip(TARGET_BASE_SEEDS, values, strict=True)
        ):
            entries.append(
                {
                    "base_seed": base_seed,
                    "replicate_index": replicate_index,
                    "replicate_seed": base_seed * 1000 + index,
                    "utility": float(value),
                    "e0": float(0.01 + value),
                    "efull": 0.01,
                }
            )
        suite = suites[index % len(suites)]
        target = {
            "selection_order": index,
            "source_index": 1000 + index,
            "sample_id": f"dataset/episode_{index:06d}/frame_{index + 1:06d}",
            "suite": suite,
            "task_index": index % 10,
            "task": f"task {index % 10}",
            "episode_index": index,
            "frame_index": index + 1,
            "valid_length": 32,
            "selection_bin": selection_bins[index % len(selection_bins)],
            "utility_by_base_seed": entries,
            "high_confidence": _high_confidence_from_values(values, PRIMARY_EPSILON),
        }
        targets.append(target)
        validation_center = 0.98 * utility
        for replicate_index, (base_seed, offset) in enumerate(
            zip(VALIDATION_BASE_SEEDS, validation_offsets, strict=True)
        ):
            validation.append(
                {
                    "source_index": 1000 + index,
                    "validation_replicate_index": replicate_index,
                    "validation_base_seed": base_seed,
                    "utility": float(validation_center + offset),
                }
            )
    return targets, validation


def test_metrics_cover_primary_and_diagnostic_contract_and_reach_go():
    targets, validation = _synthetic_panel()
    (
        metrics,
        per_state,
        seed_rows,
        seed_pair_rows,
        group_rows,
        segment_rows,
        deadband_rows,
        all9,
    ) = compute_target_v2_metrics(
        targets, validation, bootstrap_seed=9, bootstrap_replicates=30
    )
    assert all9.shape == (100, 9)
    assert len(per_state) == 100
    assert len(seed_rows) == 9
    assert len(seed_pair_rows) == 20
    assert len(group_rows) == 9
    assert len(segment_rows) >= 4
    assert [row["epsilon"] for row in deadband_rows] == [1e-5, 1e-4, 1e-3]
    assert metrics["rank_agreement"]["mean_vs_mean"]["spearman_rho"] == pytest.approx(1.0)
    assert metrics["sign_retention"]["high_confidence"]["retention"] == 1.0
    assert metrics["ranking_overlap"]["top_recall"] == 1.0
    assert metrics["absolute_agreement"]["lin_ccc"] > 0.99
    assert metrics["absolute_agreement"]["icc_a_1"] > 0.99
    assert metrics["all9_reliability"]["icc_1_k"] > 0.99
    assert metrics["single_seed_reliability_guardrail"]["value"] > 0.99
    assert metrics["bootstrap"]["mean_spearman"]["valid_replicates"] == 30
    assert metrics["high_confidence_coverage"]["positive_state_count"] >= 10
    assert metrics["high_confidence_coverage"]["negative_state_count"] >= 10
    assert metrics["decision"]["decision"] == "GO"


def test_conditional_is_limited_and_no_go_applies_below_any_floor():
    targets, validation = _synthetic_panel()
    metrics, *_ = compute_target_v2_metrics(
        targets, validation, bootstrap_seed=3, bootstrap_replicates=5
    )
    conditional = copy.deepcopy(metrics)
    conditional["rank_agreement"]["mean_vs_mean"] = {
        "spearman_rho": 0.40,
        "kendall_tau": 0.25,
    }
    conditional["absolute_agreement"]["lin_ccc"] = 0.40
    conditional["absolute_agreement"]["icc_a_1"] = 0.40
    conditional["sign_retention"]["actionable"]["retention"] = 0.70
    conditional["sign_retention"]["high_confidence"]["retention"] = 0.75
    conditional["sign_retention"]["high_confidence_positive"].update(
        {"state_count": 6, "retention": 0.75}
    )
    conditional["sign_retention"]["high_confidence_negative"].update(
        {"state_count": 6, "retention": 0.75}
    )
    conditional["ranking_overlap"].update(
        {"top_recall": 0.36, "top_jaccard": 0.20, "bottom_recall": 0.36, "bottom_jaccard": 0.20}
    )
    conditional["all9_reliability"]["icc_1_k"] = 0.70
    conditional["high_confidence_coverage"].update(
        {"state_count": 12, "population_weighted_fraction": 0.15}
    )
    conditional["median_guardrail"].update(
        {"rank": {"spearman_rho": 0.35, "kendall_tau": 0.24}, "actionable_sign_retention": 0.65}
    )
    decision = decide_readiness(conditional)
    assert decision["decision"] == "CONDITIONAL"
    assert decision["conditional_scope"] == "high_confidence_small_scale_offline_feasibility_only"

    conditional["absolute_agreement"]["lin_ccc"] = 0.29
    assert decide_readiness(conditional)["decision"] == "NO_GO"


def test_png_outputs_are_nonempty_and_declared(tmp_path: Path):
    pytest.importorskip("matplotlib")
    targets, validation = _synthetic_panel()
    (
        metrics,
        per_state,
        seed_rows,
        _,
        _,
        _,
        deadband_rows,
        all9,
    ) = compute_target_v2_metrics(
        targets, validation, bootstrap_seed=5, bootstrap_replicates=5
    )
    names = _plot_outputs(
        tmp_path, all9, per_state, seed_rows, deadband_rows, metrics
    )
    assert names == [
        "target_vs_validation_mean.png",
        "all9_seed_heatmap.png",
        "aggregate_bland_altman.png",
        "seed_group_rank_diagnostics.png",
        "deadband_sensitivity.png",
    ]
    for name in names:
        path = tmp_path / name
        assert path.stat().st_size > 1_000
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def _manifest_pair() -> tuple[dict, list[dict], dict, str, str]:
    sha = {
        letter: hashlib.sha256(f"fixture-{letter}".encode()).hexdigest()
        for letter in "abcdefghijklmno"
    }
    targets = []
    for index in range(100):
        targets.append(
            {
                "selection_order": index,
                "source_index": 1000 + index,
                "sample_id": f"dataset/episode_{index:06d}/frame_{index + 1:06d}",
                "target_id": f"target_v2/{index:03d}",
                "target_sha256": hashlib.sha256(f"target-{index}".encode()).hexdigest(),
                "input_hashes": {
                    "combined": hashlib.sha256(f"input-{index}".encode()).hexdigest()
                },
            }
        )
    target_selection = [
        {
            "selection_order": row["selection_order"],
            "source_index": row["source_index"],
            "sample_id": row["sample_id"],
            "target_id": row["target_id"],
            "target_sha256": row["target_sha256"],
            "input_combined_sha256": row["input_hashes"]["combined"],
        }
        for row in targets
    ]
    target_selection_sha = _sha256_json(target_selection)
    target_ids = [row["target_id"] for row in targets]
    target_hashes = [row["target_sha256"] for row in targets]
    target_compatibility = {
        "schema_version": 1,
        "kind": target_core.TARGET_BUNDLE_KIND,
        "target_record_schema_version": 1,
        "source_manifest_sha256": sha["a"],
        "source_records_sha256": sha["b"],
        "source_manifest_compatibility_fingerprint": sha["c"],
        "source_selection_plan_sha256": sha["d"],
        "source_input_plan_sha256": sha["e"],
        "source_num_states": 100,
        "target_base_seeds": list(TARGET_BASE_SEEDS),
        "num_states": 100,
        "deadband_epsilon": PRIMARY_EPSILON,
        "min_sign_agreement": 0.8,
        "t_interval": {
            "confidence_level": 0.95,
            "two_sided": True,
            "degrees_of_freedom": 4,
            "critical_value": target_core.T95_DF4_CRITICAL,
        },
        "target_selection_sha256": target_selection_sha,
        "target_records_sha256": sha["f"],
    }
    target_manifest = {
        "schema_version": 1,
        "kind": target_core.TARGET_BUNDLE_KIND,
        "compatibility": target_compatibility,
        "compatibility_fingerprint": _sha256_json(target_compatibility),
        "source": {
            "manifest_sha256": sha["a"],
            "records_sha256": sha["b"],
            "manifest_compatibility_fingerprint": sha["c"],
            "selection_plan_sha256": sha["d"],
            "input_plan_sha256": sha["e"],
            "checkpoint_sha256": sha["g"],
            "dataset_stats_sha256": sha["h"],
            "vae_sha256": sha["i"],
        },
        "selection": {
            "num_states": 100,
            "ordered_states": target_selection,
            "ordered_states_sha256": target_selection_sha,
            "source_selection_plan_sha256": sha["d"],
            "source_input_plan_sha256": sha["e"],
        },
        "targets": {
            "filename": "targets.jsonl",
            "count": 100,
            "ordered_target_ids": target_ids,
            "ordered_target_ids_sha256": _sha256_json(target_ids),
            "ordered_target_sha256": target_hashes,
            "ordered_target_sha256_sha256": _sha256_json(target_hashes),
            "canonical_records_sha256": sha["f"],
        },
    }
    validation_selection_sha = _sha256_json(target_selection)
    target_manifest_file_sha = sha["j"]
    target_targets_file_sha = sha["k"]
    validation_compatibility = {
        "schema_version": 1,
        "kind": VALIDATION_MANIFEST_KIND,
        "phase25_manifest_fingerprint": sha["c"],
        "phase25_manifest_sha256": sha["a"],
        "phase25_records_sha256": sha["b"],
        "phase25_selection_plan_sha256": sha["d"],
        "target_v2_manifest_fingerprint": target_manifest["compatibility_fingerprint"],
        "target_v2_manifest_sha256": target_manifest_file_sha,
        "target_v2_targets_sha256": target_targets_file_sha,
        "target_v2_selection_plan_sha256": sha["d"],
        "validation_selection_sha256": validation_selection_sha,
        "num_states": 100,
        "validation_base_seeds": list(VALIDATION_BASE_SEEDS),
        "global_seed_indices": [5, 6, 7, 8],
        "expected_record_count": 400,
        "checkpoint_sha256": sha["g"],
        "dataset_stats_sha256": sha["h"],
        "vae_sha256": sha["i"],
        "collection_parameters": {"all_new_inference": True},
    }
    validation_manifest = {
        "schema_version": 1,
        "kind": VALIDATION_MANIFEST_KIND,
        "compatibility": validation_compatibility,
        "compatibility_fingerprint": _sha256_json(validation_compatibility),
        "phase25": {
            "manifest_fingerprint": sha["c"],
            "manifest_sha256": sha["a"],
            "records_sha256": sha["b"],
            "selection_plan_sha256": sha["d"],
        },
        "target_v2": {
            "manifest_fingerprint": target_manifest["compatibility_fingerprint"],
            "manifest_sha256": target_manifest_file_sha,
            "targets_sha256": target_targets_file_sha,
            "selection_plan_sha256": sha["d"],
        },
        "selection": {
            "num_states": 100,
            "ordered_targets": target_selection,
            "ordered_targets_sha256": validation_selection_sha,
        },
        "replicates": {
            "base_seeds": list(VALIDATION_BASE_SEEDS),
            "global_seed_indices": [5, 6, 7, 8],
            "count": 4,
            "expected_record_count": 400,
            "all_new_inference": True,
        },
        "artifacts": {
            "checkpoint": {"sha256": sha["g"]},
            "dataset_stats": {"sha256": sha["h"]},
            "vae": {"sha256": sha["i"]},
        },
    }
    return (
        target_manifest,
        targets,
        validation_manifest,
        target_manifest_file_sha,
        target_targets_file_sha,
    )


def test_validation_manifest_is_fully_rebound_and_tamper_fails_closed():
    (
        target_manifest,
        targets,
        validation_manifest,
        target_manifest_file_sha,
        target_targets_file_sha,
    ) = _manifest_pair()
    evidence = validate_validation_manifest(
        validation_manifest,
        target_manifest,
        targets,
        target_manifest_sha256=target_manifest_file_sha,
        target_targets_sha256=target_targets_file_sha,
    )
    assert evidence["expected_record_count"] == 400

    tampered = copy.deepcopy(validation_manifest)
    tampered["selection"]["ordered_targets"][0]["source_index"] += 1
    with pytest.raises(ValueError, match="selection differs"):
        validate_validation_manifest(
            tampered,
            target_manifest,
            targets,
            target_manifest_sha256=target_manifest_file_sha,
            target_targets_sha256=target_targets_file_sha,
        )

    rebound = copy.deepcopy(validation_manifest)
    rebound["compatibility"]["target_v2_targets_sha256"] = hashlib.sha256(
        b"rebound-targets"
    ).hexdigest()
    rebound["compatibility_fingerprint"] = _sha256_json(rebound["compatibility"])
    rebound["target_v2"]["targets_sha256"] = "l" * 64
    with pytest.raises(ValueError, match="target_v2_targets_sha256"):
        validate_validation_manifest(
            rebound,
            target_manifest,
            targets,
            target_manifest_sha256=target_manifest_file_sha,
            target_targets_sha256=target_targets_file_sha,
        )


def test_nonempty_error_log_is_a_hard_failure(tmp_path: Path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert _validate_zero_error_log(empty) == 0
    errors = tmp_path / "errors.jsonl"
    errors.write_text(_canonical_json({"error": "inference failed"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="zero are permitted"):
        _validate_zero_error_log(errors)
