from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from experiments.libero.gate import analyze_tiny_mlp_remainder400_validation as analysis


def _fixture() -> tuple[list[dict], list[dict], list[dict], dict]:
    targets: list[dict] = []
    validation: list[dict] = []
    truth_values: list[float] = []
    suites = ("libero_goal", "libero_10", "libero_object", "libero_spatial")
    order = 0
    for suite in suites:
        for task_index in range(10):
            state_count = 2 if task_index < 5 else 3
            for state_index in range(state_count):
                truth = (order - 49.5) * 2e-4
                truth_values.append(truth)
                sample_id = f"{suite}/task{task_index}/state{state_index}"
                targets.append(
                    {
                        "selection_order": order,
                        "sample_id": sample_id,
                        "source_index": 20_000 + order,
                        "suite": suite,
                        "task_index": task_index,
                        "task": f"{suite} instruction {task_index}",
                        "target_id": f"target/{sample_id}",
                        "target_sha256": f"{order + 1:064x}",
                        "input_hashes": {"combined": f"{order + 1001:064x}"},
                        "utility_mean": truth,
                        "high_confidence": True,
                    }
                )
                for replicate, offset in enumerate((-3e-5, -1e-5, 1e-5, 3e-5)):
                    validation.append(
                        {
                            "source_index": 20_000 + order,
                            "validation_replicate_index": replicate,
                            "utility": truth + offset,
                        }
                    )
                order += 1
    assert order == 100

    task_folds = []
    for fold_id in range(5):
        orders = [
            index
            for index, target in enumerate(targets)
            if int(target["task_index"]) % 5 == fold_id
        ]
        assert len(orders) == 20
        task_folds.append({"fold_id": fold_id, "test_selection_orders": orders})
    suite_folds = []
    for fold_id, suite in enumerate(suites):
        orders = [
            index for index, target in enumerate(targets) if target["suite"] == suite
        ]
        assert len(orders) == 25
        suite_folds.append({"fold_id": fold_id, "test_selection_orders": orders})
    fold_plan = {
        "task_heldout_folds": task_folds,
        "suite_heldout_folds": suite_folds,
    }

    truth = np.asarray(truth_values, dtype=np.float64)
    model_predictions = {
        "full_hybrid": truth,
        "full_huber": truth * 0.99,
        "visual_proprio_hybrid": truth * 0.95,
        "instruction_proprio_hybrid": -truth,
        "instruction_only_hybrid": -truth * 0.9,
        "constant_train_mean": np.zeros(100),
        "suite_mean_fallback": np.repeat((-0.006, -0.002, 0.002, 0.006), 25),
        "task_lookup_fallback": np.zeros(100),
    }

    def row(
        *,
        curve_label: str,
        scheme: str,
        model_name: str,
        index: int,
        prediction: float,
        prediction_order: int,
    ) -> dict:
        target = targets[index]
        fold_id = (
            int(target["task_index"]) % 5
            if scheme == "task_heldout"
            else suites.index(str(target["suite"]))
        )
        offsets = np.asarray((-2, -1, 0, 1, 2), dtype=np.float64) * 1e-9
        return {
            "schema_version": 1,
            "kind": "fake_followup_prediction",
            "prediction_id": f"prediction/{prediction_order:04d}",
            "prediction_sha256": f"{prediction_order + 3001:064x}",
            "prediction_order": prediction_order,
            "selection_order": index,
            "sample_id": target["sample_id"],
            "source_index": target["source_index"],
            "suite": target["suite"],
            "task_index": target["task_index"],
            "task": target["task"],
            "target_id": target["target_id"],
            "target_sha256": target["target_sha256"],
            "input_combined_sha256": target["input_hashes"]["combined"],
            "feature_id": f"feature/{target['sample_id']}",
            "feature_record_sha256": f"{index + 2001:064x}",
            "representation": analysis.REPRESENTATION,
            "curve_label": curve_label,
            "curve_fraction": analysis.CURVE_BY_LABEL[curve_label],
            "outer_scheme": scheme,
            "fold_id": fold_id,
            "test_group": (
                analysis.original_core.group_id(target)
                if scheme == "task_heldout"
                else target["suite"]
            ),
            "model_name": model_name,
            "feature_view": "full",
            "loss_name": "hybrid",
            "init_seeds": [101, 202, 303, 404, 505],
            "init_predictions": [float(prediction + value) for value in offsets],
            "prediction": float(prediction),
        }

    rows: list[dict] = []
    for model_name in analysis.phase3.TASK_MODELS:
        for index in range(100):
            rows.append(
                row(
                    curve_label="q100",
                    scheme="task_heldout",
                    model_name=model_name,
                    index=index,
                    prediction=float(model_predictions[model_name][index]),
                    prediction_order=len(rows),
                )
            )
    for index in range(100):
        rows.append(
            row(
                curve_label="q100",
                scheme="suite_heldout",
                model_name=analysis.phase3.PRIMARY_MODEL,
                index=index,
                prediction=float(truth[index]),
                prediction_order=len(rows),
            )
        )
    curve_predictions = {
        "q25": -truth,
        "q50": truth * 0.2,
        "q75": truth * 0.8,
    }
    for curve_label in ("q25", "q50", "q75"):
        for index in range(100):
            rows.append(
                row(
                    curve_label=curve_label,
                    scheme="task_heldout",
                    model_name=analysis.phase3.PRIMARY_MODEL,
                    index=index,
                    prediction=float(curve_predictions[curve_label][index]),
                    prediction_order=len(rows),
                )
            )
    assert len(rows) == 1_200
    return targets, validation, rows, fold_plan


def test_exact_1200_panel_rebinds_to_original_identities_and_folds() -> None:
    targets, _, rows, fold_plan = _fixture()
    grouped = analysis.validate_and_group_predictions(rows, targets, fold_plan)
    assert len(grouped) == 12
    assert len(analysis._legacy_q100_panel(grouped)) == 900
    assert grouped[("q25", "task_heldout", "full_hybrid")][0]["target_id"] == targets[0]["target_id"]


def test_prediction_identity_feature_and_fold_tampering_fail_closed() -> None:
    targets, _, rows, fold_plan = _fixture()

    identity_tamper = copy.deepcopy(rows)
    identity_tamper[0]["source_index"] += 1
    with pytest.raises(ValueError, match="source_index"):
        analysis.validate_and_group_predictions(identity_tamper, targets, fold_plan)

    feature_tamper = copy.deepcopy(rows)
    feature_tamper[-1]["feature_id"] += "/different"
    with pytest.raises(ValueError, match="inconsistent feature"):
        analysis.validate_and_group_predictions(feature_tamper, targets, fold_plan)

    fold_tamper = copy.deepcopy(rows)
    fold_tamper[0]["fold_id"] = 4
    with pytest.raises(ValueError, match="locked original fold"):
        analysis.validate_and_group_predictions(fold_tamper, targets, fold_plan)


def test_missing_or_extra_curve_model_group_is_rejected() -> None:
    targets, _, rows, fold_plan = _fixture()
    corrupted = copy.deepcopy(rows)
    corrupted[-1]["model_name"] = "instruction_only_hybrid"
    with pytest.raises(ValueError, match="1200-row contract"):
        analysis.validate_and_group_predictions(corrupted, targets, fold_plan)


def test_embedded_original_folds_must_match_sealed_source_exactly() -> None:
    _, _, _, original = _fixture()
    followup = {
        "original_task_heldout_folds": copy.deepcopy(original["task_heldout_folds"]),
        "original_suite_heldout_folds": copy.deepcopy(original["suite_heldout_folds"]),
    }
    analysis._assert_embedded_original_folds(followup, original)
    followup["original_task_heldout_folds"][0]["test_selection_orders"][0] = 99
    with pytest.raises(ValueError, match="sealed original folds"):
        analysis._assert_embedded_original_folds(followup, original)


def test_perfect_q100_and_inverted_q25_support_sample_size_attribution() -> None:
    targets, validation, rows, fold_plan = _fixture()
    result = analysis.compute_followup_metrics(
        targets,
        validation,
        rows,
        fold_plan,
        bootstrap_replicates=80,
        bootstrap_seed=analysis.phase3.BOOTSTRAP_SEED,
        permutation_replicates=200,
        permutation_seed=analysis.phase3.PERMUTATION_SEED,
        enforce_preregistered=False,
    )
    assert result["primary_phase3_decision"] == "GO"
    assert result["decision"] == analysis.DECISION_SAMPLE_SIZE_SUPPORTED
    assert result["primary_phase3"]["go_pass_count"] == 20
    assert result["sample_size_attribution"]["delta_spearman"] > 1.9
    assert result["sample_size_attribution"]["paired_task_bootstrap"]["lower_95"] > 0
    assert [row["curve_label"] for row in result["learning_curve"]] == [
        "q25",
        "q50",
        "q75",
        "q100",
    ]
    assert result["exploratory_505"]["status"] == "not_run"
    assert result["exploratory_505"]["affects_primary_decision"] is False


def test_go_without_positive_q100_q25_delta_does_not_claim_sample_size() -> None:
    targets, validation, rows, fold_plan = _fixture()
    q100_by_order = {
        int(row["selection_order"]): float(row["prediction"])
        for row in rows
        if row["curve_label"] == "q100"
        and row["outer_scheme"] == "task_heldout"
        and row["model_name"] == "full_hybrid"
    }
    offsets = np.asarray((-2, -1, 0, 1, 2), dtype=np.float64) * 1e-9
    for row in rows:
        if row["curve_label"] == "q25":
            prediction = q100_by_order[int(row["selection_order"])]
            row["prediction"] = prediction
            row["init_predictions"] = [float(prediction + value) for value in offsets]
    result = analysis.compute_followup_metrics(
        targets,
        validation,
        rows,
        fold_plan,
        bootstrap_replicates=40,
        bootstrap_seed=analysis.phase3.BOOTSTRAP_SEED,
        permutation_replicates=100,
        permutation_seed=analysis.phase3.PERMUTATION_SEED,
        enforce_preregistered=False,
    )
    assert result["primary_phase3_decision"] == "GO"
    assert result["decision"] == analysis.DECISION_V1_LEARNABLE
    assert result["sample_size_attribution"]["supported"] is False


def test_formal_resampling_settings_cannot_be_overridden() -> None:
    targets, validation, rows, fold_plan = _fixture()
    with pytest.raises(ValueError, match="exact Phase-3 resampling"):
        analysis.compute_followup_metrics(
            targets,
            validation,
            rows,
            fold_plan,
            bootstrap_replicates=10,
        )


def test_analyze_validates_new_seal_before_touching_original_or_validation(
    monkeypatch, tmp_path
) -> None:
    calls: list[str] = []

    def reject_unsealed(*args, **kwargs):
        calls.append("followup_seal")
        raise ValueError("follow-up seal rejected")

    monkeypatch.setattr(analysis.followup_core, "load_sealed_run", reject_unsealed)

    def forbidden_file_read(path, **kwargs):
        calls.append(f"file:{path}")
        raise AssertionError("external evidence touched before trainer seal")

    monkeypatch.setattr(analysis, "_sha256_file", forbidden_file_read)
    with pytest.raises(ValueError, match="follow-up seal rejected"):
        analysis.analyze(
            run_dir=tmp_path / "run",
            expected_run_completion_sha256="a" * 64,
            original_run_dir=tmp_path / "original_run",
            expected_original_run_completion_sha256="b" * 64,
            original_target_dir=tmp_path / "target",
            validation_dir=tmp_path / "validation",
            expected_validation_manifest_sha256="c" * 64,
            expected_validation_records_sha256="d" * 64,
            expected_validation_completion_sha256="e" * 64,
            output_dir=tmp_path / "output",
        )
    assert calls == ["followup_seal"]


def test_followup_manifest_must_bind_trusted_original_completion_and_fold_bytes(
    tmp_path,
) -> None:
    original = tmp_path / "original"
    original.mkdir()
    (original / "fold_plan.json").write_text(json.dumps({"folds": 1}) + "\n")
    expected_completion = "a" * 64
    compatibility = {
        field: "b" * 64 for field in analysis.REQUIRED_FOLLOWUP_SHA_BINDINGS
    }
    compatibility.update(
        {
            "extractor_fingerprint": analysis.followup_core.EXPECTED_EXTRACTOR_FINGERPRINT,
            "original_fold_source_completion_sha256": expected_completion,
            "original_fold_plan_sha256": analysis._sha256_file(original / "fold_plan.json"),
        }
    )
    manifest = {
        "compatibility": compatibility,
        "compatibility_fingerprint": analysis._sha256_json(compatibility),
    }
    analysis._assert_followup_bindings(manifest, original, expected_completion)

    omitted = copy.deepcopy(manifest)
    del omitted["compatibility"]["remainder_feature_completion_sha256"]
    omitted["compatibility_fingerprint"] = analysis._sha256_json(omitted["compatibility"])
    with pytest.raises(ValueError, match="remainder_feature_completion_sha256"):
        analysis._assert_followup_bindings(omitted, original, expected_completion)

    corrupted = copy.deepcopy(manifest)
    corrupted["compatibility"]["original_fold_plan_sha256"] = "f" * 64
    corrupted["compatibility_fingerprint"] = analysis._sha256_json(
        corrupted["compatibility"]
    )
    with pytest.raises(ValueError, match="fold bytes"):
        analysis._assert_followup_bindings(corrupted, original, expected_completion)
