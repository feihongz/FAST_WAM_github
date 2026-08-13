from __future__ import annotations

import copy

import numpy as np
import pytest

from experiments.libero.gate import analyze_tiny_mlp_validation as analysis


def _fixture() -> tuple[list[dict], list[dict], list[dict]]:
    targets: list[dict] = []
    validations: list[dict] = []
    truth_values: list[float] = []
    order = 0
    for suite_index, suite in enumerate(("goal", "libero_10", "object", "spatial")):
        for task_index in range(10):
            state_count = 2 if task_index < 5 else 3
            for state_in_task in range(state_count):
                truth = (order - 49.5) * 2e-4
                truth_values.append(truth)
                sample_id = f"{suite}/task{task_index}/state{state_in_task}"
                target_sha = f"{order + 1:064x}"
                input_sha = f"{order + 1001:064x}"
                targets.append(
                    {
                        "selection_order": order,
                        "sample_id": sample_id,
                        "source_index": 10_000 + order,
                        "suite": suite,
                        "task_index": task_index,
                        "task": f"{suite} instruction {task_index}",
                        "target_id": f"target/{sample_id}",
                        "target_sha256": target_sha,
                        "input_hashes": {"combined": input_sha},
                        "utility_mean": truth,
                        "high_confidence": True,
                    }
                )
                for validation_index, offset in enumerate((-3e-5, -1e-5, 1e-5, 3e-5)):
                    validations.append(
                        {
                            "source_index": 10_000 + order,
                            "validation_replicate_index": validation_index,
                            "utility": truth + offset,
                        }
                    )
                order += 1
    assert order == 100
    truth_vector = np.asarray(truth_values)
    nonvisual = -truth_vector
    rows: list[dict] = []
    model_predictions = {
        "full_hybrid": truth_vector,
        "full_huber": truth_vector * 0.99,
        "visual_proprio_hybrid": truth_vector * 0.95,
        "instruction_proprio_hybrid": nonvisual,
        "instruction_only_hybrid": nonvisual * 0.9,
        "constant_train_mean": np.zeros(100),
        "suite_mean_fallback": np.repeat((-0.006, -0.002, 0.002, 0.006), 25),
        "task_lookup_fallback": np.zeros(100),
    }
    for model_name in analysis.TASK_MODELS:
        for index, target in enumerate(targets):
            prediction = float(model_predictions[model_name][index])
            offsets = np.asarray((-2, -1, 0, 1, 2), dtype=float) * 1e-9
            rows.append(
                {
                    "outer_scheme": "task_heldout",
                    "model_name": model_name,
                    "sample_id": target["sample_id"],
                    "source_index": target["source_index"],
                    "suite": target["suite"],
                    "task_index": target["task_index"],
                    "task": target["task"],
                    "target_id": target["target_id"],
                    "target_sha256": target["target_sha256"],
                    "input_combined_sha256": target["input_hashes"]["combined"],
                    "fold_id": index % 5,
                    "prediction": prediction,
                    "init_predictions": [float(prediction + value) for value in offsets],
                }
            )
    for index, target in enumerate(targets):
        prediction = float(truth_vector[index])
        offsets = np.asarray((-2, -1, 0, 1, 2), dtype=float) * 1e-9
        rows.append(
            {
                "outer_scheme": "suite_heldout",
                "model_name": "full_hybrid",
                "sample_id": target["sample_id"],
                "source_index": target["source_index"],
                "suite": target["suite"],
                "task_index": target["task_index"],
                "task": target["task"],
                "target_id": target["target_id"],
                "target_sha256": target["target_sha256"],
                "input_combined_sha256": target["input_hashes"]["combined"],
                "fold_id": index // 25,
                "prediction": prediction,
                "init_predictions": [float(prediction + value) for value in offsets],
            }
        )
    return targets, validations, rows


def test_deadband_and_random_scores_are_deterministic() -> None:
    assert analysis.deadband_sign(1e-4) == 0
    assert analysis.deadband_sign(1.0001e-4) == 1
    assert analysis.deadband_sign(-1.0001e-4) == -1
    first = analysis._random_score("sample", 123, 7)
    assert first == analysis._random_score("sample", 123, 7)
    assert first != analysis._random_score("sample", 123, 8)


def test_task_cluster_bootstrap_preserves_whole_groups() -> None:
    targets, _, rows = _fixture()
    primary = [row for row in rows if row["outer_scheme"] == "task_heldout" and row["model_name"] == "full_hybrid"]
    samples = analysis._cluster_bootstrap_indices(primary, replicates=3, seed=9)
    assert len(samples) == 3
    for sample in samples:
        # Each sampled source task appears with all of its 2 or 3 states.
        counts: dict[tuple[str, int], int] = {}
        for index in sample:
            target = targets[int(index)]
            key = (target["suite"], target["task_index"])
            counts[key] = counts.get(key, 0) + 1
        for (suite, task_index), count in counts.items():
            size = 2 if task_index < 5 else 3
            assert count % size == 0, suite


def test_prediction_identity_tamper_is_rejected() -> None:
    targets, _, rows = _fixture()
    corrupted = copy.deepcopy(rows)
    corrupted[0]["source_index"] += 1
    with pytest.raises(ValueError, match="source_index"):
        analysis._prediction_rows_by_model(corrupted, targets)


def test_duplicate_prediction_is_rejected() -> None:
    targets, _, rows = _fixture()
    corrupted = copy.deepcopy(rows)
    corrupted[-1] = copy.deepcopy(corrupted[0])
    with pytest.raises(ValueError, match="duplicate|expected"):
        analysis._prediction_rows_by_model(corrupted, targets)


def test_synthetic_perfect_signal_reaches_go_with_test_resamples() -> None:
    targets, validations, rows = _fixture()
    metrics = analysis.compute_metrics(
        targets,
        validations,
        rows,
        bootstrap_replicates=80,
        bootstrap_seed=analysis.BOOTSTRAP_SEED,
        permutation_replicates=200,
        permutation_seed=analysis.PERMUTATION_SEED,
        enforce_preregistered=False,
    )
    assert metrics["decision"] == "GO"
    assert metrics["go_pass_count"] == metrics["go_check_count"]
    assert metrics["primary"]["within_task_pairs"]["total_pairs"] == 80
    assert metrics["primary"]["within_task_pairs"]["evaluable_pairs"] == 80
    assert metrics["primary"]["metrics"]["spearman"] == pytest.approx(1.0)
    assert metrics["nonvisual_comparison"]["delta_spearman"] > 1.9
    assert metrics["resampling"]["bootstrap_requested_replicates"] == 80
    assert metrics["resampling"]["bootstrap_effective_replicates"] == 80
    assert set(
        metrics["resampling"]["bootstrap_effective_by_metric"].values()
    ) == {80}


def test_formal_settings_cannot_be_overridden() -> None:
    targets, validations, rows = _fixture()
    with pytest.raises(ValueError, match="preregistered bootstrap"):
        analysis.compute_metrics(
            targets,
            validations,
            rows,
            bootstrap_replicates=10,
        )



def test_percentile_interval_requires_every_requested_replicate_to_be_finite():
    interval = analysis._percentile_interval(
        [1.0, 2.0, 3.0], expected_replicates=3
    )
    assert interval["requested_replicates"] == 3
    assert interval["effective_replicates"] == 3
    with pytest.raises(ValueError, match="finite replicates"):
        analysis._percentile_interval(
            [1.0, float("nan"), 3.0], expected_replicates=3
        )
    with pytest.raises(ValueError, match="produced 2 values"):
        analysis._percentile_interval([1.0, 2.0], expected_replicates=3)


def test_analyzer_revalidates_formal_run_before_any_metric(
    monkeypatch, tmp_path
):
    calls: list[str] = []
    monkeypatch.setattr(analysis, "_sha256_file", lambda path: "a" * 64)
    monkeypatch.setattr(
        analysis.gate_core,
        "load_sealed_run",
        lambda *args, **kwargs: ({}, {}, []),
    )
    monkeypatch.setattr(
        analysis.target_analysis,
        "validate_analysis_inputs",
        lambda *args, **kwargs: ({}, [{"selection_order": 0}], {}, [], {}),
    )

    def stop_at_formal(manifest, fold_plan, targets):
        calls.append("formal")
        raise ValueError("formal contract stop")

    monkeypatch.setattr(
        analysis.gate_core, "validate_formal_run_contract", stop_at_formal
    )
    monkeypatch.setattr(
        analysis,
        "compute_metrics",
        lambda *args, **kwargs: calls.append("metrics"),
    )
    with pytest.raises(ValueError, match="formal contract stop"):
        analysis.analyze(
            run_dir=tmp_path / "run",
            expected_run_completion_sha256="a" * 64,
            target_dir=tmp_path / "target",
            validation_dir=tmp_path / "validation",
            expected_validation_manifest_sha256="a" * 64,
            expected_validation_records_sha256="a" * 64,
            expected_validation_completion_sha256="a" * 64,
            output_dir=tmp_path / "output",
        )
    assert calls == ["formal"]
