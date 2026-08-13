from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.libero.gate import demo_utility_stability as stability_core
from experiments.libero.gate.analyze_demo_utility_stability import (
    EXPECTED_BASE_SEEDS,
    STABILITY_KIND,
    _sha256_json,
    analyze,
    classify_pilot_stratum,
    deadband_sign,
    decide_tiny_mlp_readiness,
    icc_one_way,
    population_state_weights,
    ranking_overlap,
    validate_stability_grid,
    weighted_fraction,
)
from experiments.libero.gate.demo_utility import stable_sample_seed


SHA = "a" * 64
GIT_SHA = "b" * 40


def test_stratum_boundaries_and_deadband_are_explicit():
    assert classify_pilot_stratum(1.00001e-3) == "SP"
    assert classify_pilot_stratum(1e-3) == "MP"
    assert classify_pilot_stratum(1.00001e-4) == "MP"
    assert classify_pilot_stratum(1e-4) == "NZ"
    assert classify_pilot_stratum(-1e-4) == "NZ"
    assert classify_pilot_stratum(-1.00001e-4) == "MN"
    assert classify_pilot_stratum(-1e-3) == "MN"
    assert classify_pilot_stratum(-1.00001e-3) == "SN"
    assert deadband_sign(1e-4, 1e-4) == 0
    assert deadband_sign(-1e-4, 1e-4) == 0
    assert deadband_sign(1.1e-4, 1e-4) == 1
    assert deadband_sign(-1.1e-4, 1e-4) == -1


def test_icc_distinguishes_repeatable_signal_from_seed_noise():
    signal = np.linspace(-2.0, 2.0, 20)
    perfectly_repeatable = np.repeat(signal[:, None], 5, axis=1)
    perfect = icc_one_way(perfectly_repeatable)
    assert perfect["icc_1_1"] == pytest.approx(1.0)
    assert perfect["icc_1_k"] == pytest.approx(1.0)
    rng = np.random.default_rng(7)
    noisy = signal[:, None] + rng.normal(scale=3.0, size=(20, 5))
    degraded = icc_one_way(noisy)
    assert degraded["icc_1_1"] < perfect["icc_1_1"]
    assert degraded["icc_1_k"] < perfect["icc_1_k"]


def test_rank_overlap_has_known_random_benchmarks_and_exact_sets():
    reference = np.arange(100, dtype=np.float64)
    identical = ranking_overlap(reference, reference)
    assert identical["k"] == 20
    assert identical["top_recall"] == 1.0
    assert identical["bottom_jaccard"] == 1.0
    assert identical["random_expected_recall"] == pytest.approx(0.20)
    assert identical["random_expected_jaccard"] == pytest.approx(1 / 9)
    reversed_result = ranking_overlap(reference, -reference)
    assert reversed_result["top_recall"] == 0.0
    assert reversed_result["bottom_recall"] == 0.0


def test_population_weights_recover_fixed_pilot_prevalence():
    strata = ["SP"] * 50 + ["SN"] * 10 + ["MP"] * 20 + ["MN"] * 10 + ["NZ"] * 10
    weights = population_state_weights(strata)
    assert weights.sum() == pytest.approx(1.0)
    for stratum, expected in {
        "SP": 0.152,
        "SN": 0.202,
        "MP": 0.262,
        "MN": 0.184,
        "NZ": 0.200,
    }.items():
        assert weights[np.asarray(strata) == stratum].sum() == pytest.approx(expected)
    assert weighted_fraction(np.asarray(strata) == "SP", weights) == pytest.approx(0.152)


def _decision_metrics(*, good: bool = True) -> dict:
    high = 0.90 if good else 0.10
    return {
        "rank_stability": {
            "seed42_vs_new4_mean": {"spearman_rho": 0.80 if good else 0.10}
        },
        "strong_pilot_stability": {
            "overall_new4_majority_agreement": high,
            "SP_new4_majority_agreement": high,
            "SN_new4_majority_agreement": high,
            "overall_four_of_five_expected_direction": high,
        },
        "ranking_overlap": {
            "top_recall": 0.70 if good else 0.10,
            "top_jaccard": 0.55 if good else 0.05,
            "bottom_recall": 0.70 if good else 0.10,
            "bottom_jaccard": 0.55 if good else 0.05,
        },
        "reliability": {
            "icc_1_k": 0.90 if good else 0.10,
            "icc_1_1": 0.60 if good else 0.05,
        },
        "population_weighted": {
            "strong_mean_four_of_five_same_direction": high,
        },
        "bootstrap": {
            "seed42_vs_new4_mean_spearman": {"lower_95": 0.55 if good else -0.10}
        },
        "leave_one_seed_out": {
            "median_spearman_rho": 0.70 if good else 0.10,
            "positive_spearman_count": 5 if good else 2,
        },
    }


def test_readiness_decision_is_scope_limited_and_thresholded():
    go = decide_tiny_mlp_readiness(_decision_metrics(good=True))
    assert go["decision"] == "GO"
    assert go["scope"] == "readiness_to_start_offline_tiny_mlp_only"
    assert go["does_not_establish"] == "closed_loop_gate_improvement"
    no_go = decide_tiny_mlp_readiness(_decision_metrics(good=False))
    assert no_go["decision"] == "NO_GO"
    assert no_go["failed_go_checks"]


def _pilot_utility_for(stratum: str) -> float:
    return {"SP": 0.002, "SN": -0.002, "MP": 0.0005, "MN": -0.0005, "NZ": 0.0}[stratum]


def _synthetic_bundle() -> tuple[dict, list[dict]]:
    states = []
    records = []
    suite_specs = (
        ("libero_spatial_no_noops_lerobot", "libero_spatial", {"SP": 6, "SN": 6, "MP": 3, "MN": 3, "NZ": 7}),
        ("libero_object_no_noops_lerobot", "libero_object", {"SP": 6, "SN": 6, "MP": 4, "MN": 3, "NZ": 6}),
        ("libero_goal_no_noops_lerobot", "libero_goal", {"SP": 6, "SN": 7, "MP": 3, "MN": 3, "NZ": 6}),
        ("libero_10_no_noops_lerobot", "libero_10", {"SP": 7, "SN": 6, "MP": 3, "MN": 3, "NZ": 6}),
    )
    index = 0
    for dataset, suite, quotas in suite_specs:
        bins = [name for name in ("SP", "SN", "MP", "MN", "NZ") for _ in range(quotas[name])]
        task_indices = list(range(10)) * 2 + list(range(5))
        for local_index, (selection_bin, task_index) in enumerate(zip(bins, task_indices, strict=True)):
            episode = local_index
            frame = index + 1
            task = f"task {task_index}"
            sample_id = f"{dataset}/episode_{episode:06d}/frame_{frame:06d}"
            pilot_u = _pilot_utility_for(selection_bin)
            source_index = 1000 + index
            states.append({
                "selection_order": index, "source_index": source_index,
                "sample_id": sample_id, "suite": suite, "task_index": task_index,
                "task": task,
                "episode_index": episode, "frame_index": frame,
                "selection_bin": selection_bin, "pilot_utility": pilot_u,
                "valid_length": 16 if local_index < 4 else 32,
            })
            index += 1

    compatibility = {
        "schema_version": 1,
        "kind": STABILITY_KIND,
        "pilot_manifest_fingerprint": "c" * 64,
        "pilot_manifest_sha256": "d" * 64,
        "pilot_records_sha256": "e" * 64,
        "selection_plan_sha256": _sha256_json(states),
        "num_states": 100,
        "replicate_base_seeds": list(EXPECTED_BASE_SEEDS),
        "reuse_base_seed": 42,
        "checkpoint_sha256": SHA,
        "dataset_stats_sha256": SHA,
        "vae_sha256": SHA,
    }
    manifest = {
        "schema_version": 1,
        "kind": STABILITY_KIND,
        "compatibility": compatibility,
        "compatibility_fingerprint": _sha256_json(compatibility),
        "pilot": {
            "manifest_fingerprint": "c" * 64,
            "manifest_sha256": "d" * 64,
            "records_sha256": "e" * 64,
        },
        "selection": {
            "num_states": 100,
            "ordered_states": states,
            "ordered_states_sha256": _sha256_json(states),
        },
        "replicates": {
            "base_seeds": list(EXPECTED_BASE_SEEDS),
            "count": 5,
            "reuse_base_seed": 42,
            "reuse_replicate_index": 0,
            "expected_record_count": 500,
        },
        "artifacts": {
            "checkpoint": {"sha256": SHA},
            "dataset_stats": {"sha256": SHA},
            "vae": {"sha256": SHA},
        },
    }

    for state in states:
        dataset = state["sample_id"].split("/", 1)[0]
        identity_args = (dataset, state["episode_index"], state["frame_index"])
        pilot_seed = stable_sample_seed(42, *identity_args)
        input_parts = {
            "action_is_pad": hashlib.sha256(f"pad-{state['source_index']}".encode()).hexdigest(),
            "context": hashlib.sha256(f"ctx-{state['source_index']}".encode()).hexdigest(),
            "context_mask": hashlib.sha256(f"mask-{state['source_index']}".encode()).hexdigest(),
            "input_image": hashlib.sha256(f"image-{state['source_index']}".encode()).hexdigest(),
            "proprio": hashlib.sha256(f"prop-{state['source_index']}".encode()).hexdigest(),
            "valid_target_action": hashlib.sha256(f"target-{state['source_index']}".encode()).hexdigest(),
        }
        input_hashes = {**input_parts, "combined": _sha256_json(input_parts)}
        pilot_record_sha = hashlib.sha256(state["sample_id"].encode()).hexdigest()
        for replicate_index, base_seed in enumerate(EXPECTED_BASE_SEEDS):
            utility = float(state["pilot_utility"])
            # Small state-dependent, sign-preserving perturbations on new seeds.
            if replicate_index:
                utility += (replicate_index - 2.5) * 1e-5
            seed = stable_sample_seed(base_seed, *identity_args)
            e_full = 0.01
            record = {
                "schema_version": 1,
                "collector_record_schema_version": 1,
                "stability_record_schema_version": 1,
                "replicate_id": (
                    f"{state['sample_id']}/replicate_{replicate_index:02d}_base_seed_{base_seed}"
                ),
                "replicate_index": replicate_index,
                "replicate_base_seed": base_seed,
                "replicate_seed": seed,
                "source_index": state["source_index"],
                "selection_order": state["selection_order"],
                "selection_bin": state["selection_bin"],
                "pilot_base_seed": 42,
                "pilot_seed": pilot_seed,
                "pilot_e0": 0.01 + state["pilot_utility"],
                "pilot_efull": 0.01,
                "pilot_utility": state["pilot_utility"],
                "pilot_valid_length": state["valid_length"],
                "pilot_input_combined_sha256": input_hashes["combined"],
                "manifest_compatibility_fingerprint": "c" * 64,
                "pilot_manifest_compatibility_fingerprint": "c" * 64,
                "source_pilot_record_sha256": pilot_record_sha,
                "stability_manifest_compatibility_fingerprint": manifest[
                    "compatibility_fingerprint"
                ],
                "reused_from_pilot": replicate_index == 0,
                "inference_origin": "pilot_reuse" if replicate_index == 0 else "new_inference",
                "collection_git_sha": GIT_SHA,
                "sample_id": state["sample_id"],
                "dataset_id": dataset,
                "dataset_name": dataset,
                "suite": state["suite"],
                "episode_index": state["episode_index"],
                "episode_id": state["episode_index"],
                "frame_index": state["frame_index"],
                "task_index": state["task_index"],
                "task_id": state["task_index"],
                "task_id_source": "lerobot_task_index",
                "task": f"task {state['task_index']}",
                "seed": seed,
                "num_inference_steps": 10,
                "n0": 0,
                "nfull": 10,
                "e0": e_full + utility,
                "efull": e_full,
                "utility": utility,
                "valid_length": state["valid_length"],
                "target_action_shape": [32, 7],
                "pred_n0_shape": [32, 7],
                "pred_nfull_shape": [32, 7],
                "input_hashes": input_hashes,
                "current_proprio": [0.0] * 8,
                "n0_latency_ms": 1.0,
                "nfull_latency_ms": 2.0,
                "total_latency_ms": 3.0,
                "n0_route": {
                    "inference_mode": "prefix",
                    "video_prefix_steps": 0,
                    "num_inference_steps": 10,
                    "force_custom_prefix": True,
                },
                "nfull_route": {
                    "inference_mode": "prefix",
                    "video_prefix_steps": 10,
                    "num_inference_steps": 10,
                    "force_custom_prefix": True,
                },
                "source_metadata": {
                    "dataset_name": dataset,
                    "requested_sample_idx": state["source_index"],
                    "source_sample_idx": state["source_index"],
                    "episode_index": state["episode_index"],
                    "frame_index": state["frame_index"],
                    "task_index": state["task_index"],
                    "task": f"task {state['task_index']}",
                },
                "checkpoint_sha256": SHA,
                "dataset_stats_sha256": SHA,
                "vae_sha256": SHA,
                "git_sha": GIT_SHA,
            }
            records.append(record)
    stability_only = {
        "stability_record_schema_version", "replicate_id", "replicate_index",
        "replicate_base_seed", "replicate_seed", "pilot_base_seed",
        "stability_manifest_compatibility_fingerprint",
        "source_pilot_record_sha256", "reused_from_pilot",
        "inference_origin", "collection_git_sha",
    }
    by_source = {state["source_index"]: state for state in states}
    for record in records:
        if record["replicate_index"] != 0:
            continue
        state = by_source[record["source_index"]]
        pilot = {key: value for key, value in record.items() if key not in stability_only}
        for field in ("selection_order", "source_index", "selection_bin"):
            pilot[field] = state[field]
        digest = stability_core.pilot_record_sha256(pilot)
        state["pilot_record_sha256"] = digest
        for sibling in records:
            if sibling["source_index"] == record["source_index"]:
                sibling["source_pilot_record_sha256"] = digest
    plan_sha = _sha256_json(states)
    compatibility["selection_plan_sha256"] = plan_sha
    manifest["selection"]["ordered_states_sha256"] = plan_sha
    manifest["compatibility_fingerprint"] = _sha256_json(compatibility)
    for record in records:
        record["stability_manifest_compatibility_fingerprint"] = manifest[
            "compatibility_fingerprint"
        ]
    return manifest, records


def test_complete_grid_validation_rejects_input_drift_and_duplicates():
    manifest, records = _synthetic_bundle()
    result = validate_stability_grid(records, manifest)
    assert result["record_count"] == 500
    assert result["reused_record_count"] == 100
    assert result["new_inference_record_count"] == 400

    drifted = copy.deepcopy(records)
    drifted[1]["input_hashes"]["input_image"] = "f" * 64
    with pytest.raises(ValueError, match="input_hashes|Input hashes drift"):
        validate_stability_grid(drifted, manifest)

    duplicated = records[:-1] + [copy.deepcopy(records[0])]
    with pytest.raises(ValueError, match="Duplicate stability cell"):
        validate_stability_grid(duplicated, manifest)


def test_analyze_writes_required_tables_and_summary(tmp_path: Path):
    pytest.importorskip("matplotlib")
    manifest, records = _synthetic_bundle()
    records_path = tmp_path / "records.jsonl"
    manifest_path = tmp_path / "manifest.json"
    records_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    output_dir = tmp_path / "analysis"
    summary = analyze(
        records_path,
        manifest_path,
        output_dir,
        bootstrap_seed=123,
        bootstrap_replicates=30,
        make_plots=True,
    )
    assert summary["validation"]["status"] == "complete_and_verified"
    assert summary["metrics"]["decision"]["decision"] in {"GO", "CONDITIONAL", "NO_GO"}
    for filename in (
        "analysis_summary.json",
        "per_state.csv",
        "seed_metrics.csv",
        "stratum_metrics.csv",
    ):
        assert (output_dir / filename).is_file()
    expected_plots = [
        "utility_seed_heatmap.png",
        "seed42_vs_new4_mean_scatter.png",
        "stratum_sign_agreement.png",
        "variance_components.png",
    ]
    assert summary["outputs"]["plot_files"] == expected_plots
    for filename in expected_plots:
        plot_path = output_dir / filename
        assert plot_path.is_file()
        assert plot_path.stat().st_size > 1_000
        assert plot_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
