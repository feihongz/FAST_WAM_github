from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

from experiments.libero.gate import offline_tiny_mlp as v1
from experiments.libero.gate import offline_tiny_mlp_remainder400 as core


def _original_features_100() -> list[dict]:
    rows: list[dict] = []
    order = 0
    for suite in v1.SUITES:
        for task_index in range(10):
            count = 2 if task_index < 5 else 3
            for state in range(count):
                sample_id = f"{suite}/task_{task_index}/state_{state}"
                rows.append(
                    {
                        "selection_order": order,
                        "sample_id": sample_id,
                        "source_index": 100_000 + order,
                        "suite": suite,
                        "task_index": task_index,
                        "task": f"{suite} task {task_index}",
                        "target_id": f"{sample_id}/utility_target_v2",
                        "target_sha256": f"{order % 10}" * 64,
                        "input_combined_sha256": f"{(order + 1) % 10}" * 64,
                        "feature_id": f"feature/{sample_id}",
                        "feature_record_sha256": f"{(order + 2) % 10}" * 64,
                    }
                )
                order += 1
    assert len(rows) == 100
    return rows


def _remainder_400(original: list[dict], source_plan: dict) -> tuple[list[dict], str]:
    groups: dict[str, dict] = {}
    for row in original:
        groups.setdefault(v1.group_id(row), row)
    inner = {
        group
        for key in ("task_heldout_folds", "suite_heldout_folds")
        for fold in source_plan[key]
        for group in fold["inner_validation_groups"]
    }
    missing = next(group for group in sorted(groups) if group not in inner)
    available = [group for group in sorted(groups) if group != missing]
    rows: list[dict] = []
    order = 0
    for group_index, group in enumerate(available):
        identity = groups[group]
        state_count = 11 if group_index < 10 else 10
        episode_count = 9 if group_index < 34 else 8
        for state in range(state_count):
            episode = group_index * 100 + min(state, episode_count - 1)
            center = (order - 200) * 1e-5
            rows.append(
                {
                    "selection_order": order,
                    "sample_id": f"remainder/{group_index}/state_{state}",
                    "source_index": 200_000 + order,
                    "suite": identity["suite"],
                    "task_index": identity["task_index"],
                    "task": identity["task"],
                    "dataset_name": f"{identity['suite']}_no_noops_lerobot",
                    "episode_index": episode,
                    "utility_mean": center,
                    "utility_sem": 1e-5,
                    "t95_ci_low": center - 2e-5,
                    "t95_ci_high": center + 2e-5,
                    "high_confidence": abs(center) > 1e-4,
                }
            )
            order += 1
    assert len(rows) == 400
    assert len({(row["dataset_name"], row["episode_index"]) for row in rows}) == 346
    return rows, missing


def _formal_fixture() -> tuple[list[dict], list[dict], dict, dict, str]:
    original = _original_features_100()
    source = v1.build_fold_plan(original, strict_v1=True)
    remainder, missing = _remainder_400(original, source)
    plan = core.build_followup_fold_plan(remainder, original, source, formal=True)
    return remainder, original, source, plan, missing


def test_formal_identity_plan_matches_real_profile_and_is_nested():
    remainder, original, source, plan, missing = _formal_fixture()
    assert plan["num_remainder_states"] == 400
    assert plan["num_original_test_states"] == 100
    assert plan["num_remainder_task_groups"] == 39
    assert plan["num_remainder_episode_groups"] == 346
    assert plan["original_task_heldout_folds"] == source["task_heldout_folds"]
    assert plan["original_suite_heldout_folds"] == source["suite_heldout_folds"]

    missing_seen = 0
    for followup, sealed in zip(
        plan["task_heldout_folds"], source["task_heldout_folds"], strict=True
    ):
        assert followup["remainder_inner_validation_groups"] == sealed[
            "inner_validation_groups"
        ]
        assert len(followup["original_test_selection_orders"]) == 20
        missing_seen += missing in followup["missing_heldout_groups"]
        curves = followup["curve_train_selection_orders"]
        assert set(curves["q25"]) < set(curves["q50"])
        assert set(curves["q50"]) < set(curves["q75"])
        assert set(curves["q75"]) < set(curves["q100"])
        assert not set(curves["q100"]) & set(
            followup["remainder_inner_validation_selection_orders"]
        )
        for task in followup["ordered_episode_groups_by_task"]:
            ordered = task["ordered_episodes"]
            assert [row["episode_sha256"] for row in ordered] == sorted(
                row["episode_sha256"] for row in ordered
            )
    assert missing_seen == 1
    core.validate_followup_fold_plan(plan, remainder, original, source, formal=True)


def test_plan_is_label_independent_and_rejects_unknown_task():
    remainder, original, source, plan, _ = _formal_fixture()
    changed = copy.deepcopy(remainder)
    for index, row in enumerate(changed):
        row["utility_mean"] = 1e9 if index % 2 else -1e9
        row["utility_sem"] = 999.0
        row["t95_ci_low"] = -math.inf
        row["t95_ci_high"] = math.inf
    assert core.build_followup_fold_plan(changed, original, source, formal=True) == plan

    changed[0]["task"] = "unknown task"
    with pytest.raises(ValueError, match="outside the locked original panel"):
        core.build_followup_fold_plan(changed, original, source, formal=True)


def test_fold_plan_is_fsync_staged_before_fit(tmp_path):
    remainder, original, source, plan, _ = _formal_fixture()
    output, staging, frozen = v1.prepare_fold_plan_staging(tmp_path / "run", plan)
    assert output == (tmp_path / "run").resolve()
    assert v1.sha256_file(staging / "fold_plan.json") == frozen
    loaded = json.loads((staging / "fold_plan.json").read_text())
    core.validate_followup_fold_plan(loaded, remainder, original, source, formal=True)


def _formal_prediction_rows(plan: dict, original: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for fold in plan["task_heldout_folds"]:
        for label, fraction in core.CURVES:
            models = v1.TASK_MODELS if label == "q100" else (
                ("full_hybrid", "full", "hybrid"),
            )
            for model, view, loss in models:
                for order in fold["original_test_selection_orders"]:
                    rows.append(
                        core._prediction_record(
                            original=original[order],
                            curve_label=label,
                            curve_fraction=fraction,
                            scheme="task_heldout",
                            fold_id=fold["fold_id"],
                            model_name=model,
                            feature_view=view,
                            loss_name=loss,
                            init_seeds=v1.INIT_SEEDS,
                            init_predictions=[0.01] * 5,
                            prediction=0.01,
                        )
                    )
    for fold in plan["suite_heldout_folds"]:
        for order in fold["original_test_selection_orders"]:
            rows.append(
                core._prediction_record(
                    original=original[order],
                    curve_label="q100",
                    curve_fraction=1.0,
                    scheme="suite_heldout",
                    fold_id=fold["fold_id"],
                    model_name="full_hybrid",
                    feature_view="full",
                    loss_name="hybrid",
                    init_seeds=v1.INIT_SEEDS,
                    init_predictions=[0.01] * 5,
                    prediction=0.01,
                )
            )
    curve_rank = {label: index for index, (label, _) in enumerate(core.CURVES)}
    model_rank = {name: index for index, (name, _, _) in enumerate(v1.TASK_MODELS)}
    rows.sort(
        key=lambda row: (
            curve_rank[row["curve_label"]],
            0 if row["outer_scheme"] == "task_heldout" else 1,
            model_rank[row["model_name"]],
            row["selection_order"],
        )
    )
    for order, row in enumerate(rows):
        row["prediction_order"] = order
        row["prediction_sha256"] = core._payload_sha(row, "prediction_sha256")
    assert len(rows) == 1200
    return rows


def _prediction_manifest(rows: list[dict]) -> dict:
    payload = core._jsonl_bytes(rows)
    return {
        "predictions": {
            "count": len(rows),
            "ordered_prediction_ids": [row["prediction_id"] for row in rows],
            "ordered_prediction_sha256": [row["prediction_sha256"] for row in rows],
            "canonical_records_sha256": hashlib.sha256(payload).hexdigest(),
        }
    }


def test_formal_prediction_panel_is_exact_1200_and_label_free():
    _, original, _, plan, _ = _formal_fixture()
    rows = _formal_prediction_rows(plan, original)
    core.validate_prediction_rows(rows, _prediction_manifest(rows), plan, formal=True)
    assert all(set(row) == core.PREDICTION_FIELDS for row in rows)
    assert all(not (set(row) & core.FORBIDDEN_PREDICTION_FIELDS) for row in rows)

    tampered = copy.deepcopy(rows)
    tampered[0]["target5_utility_mean"] = 0.5
    tampered[0]["prediction_sha256"] = core._payload_sha(
        tampered[0], "prediction_sha256"
    )
    with pytest.raises(ValueError, match="label-free frozen schema"):
        core.validate_prediction_rows(
            tampered, _prediction_manifest(tampered), plan, formal=True
        )


def test_prediction_hash_and_completion_seal_detect_tamper(tmp_path):
    _, original, _, plan, _ = _formal_fixture()
    rows = _formal_prediction_rows(plan, original)
    fold_path = tmp_path / "fold_plan.json"
    fold_path.write_text(json.dumps(plan, indent=2) + "\n")
    fold_sha = v1.sha256_file(fold_path)
    compatibility = {
        "schema_version": 1,
        "kind": core.RUN_KIND,
        "formal_protocol": True,
        "representation": core.REPRESENTATION,
        "curve_namespace": core.CURVE_NAMESPACE,
        "curve_fractions": dict(core.CURVES),
        "extractor_fingerprint": core.EXPECTED_EXTRACTOR_FINGERPRINT,
        "num_remainder_states": 400,
        "num_original_test_states": 100,
        "num_predictions": 1200,
        "followup_fold_plan_sha256": fold_sha,
    }
    for field in core.FORMAL_COMPATIBILITY_SHA_FIELDS:
        compatibility.setdefault(field, "a" * 64)
    manifest = {
        "schema_version": 1,
        "kind": core.RUN_KIND,
        "compatibility": compatibility,
        "compatibility_fingerprint": v1.sha256_json(compatibility),
        "training_config": v1.asdict(v1.TrainingConfig()),
        "models": {
            "q25_q50_q75_task_heldout": ["full_hybrid"],
            "q100_task_heldout": [name for name, _, _ in v1.TASK_MODELS],
            "q100_suite_heldout": ["full_hybrid"],
            "primary": "q100/task_heldout/full_hybrid",
        },
        **_prediction_manifest(rows),
        "diagnostics": {},
    }
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    prediction_path = tmp_path / "oof_predictions.jsonl"
    prediction_path.write_bytes(core._jsonl_bytes(rows))
    completion = {
        "schema_version": 1,
        "kind": core.COMPLETION_KIND,
        "complete": True,
        "run_manifest_sha256": v1.sha256_file(manifest_path),
        "fold_plan_sha256": fold_sha,
        "oof_predictions_sha256": v1.sha256_file(prediction_path),
        "num_predictions": 1200,
        "compatibility_fingerprint": manifest["compatibility_fingerprint"],
    }
    completion["completion_sha256"] = core._payload_sha(
        completion, "completion_sha256"
    )
    completion_path = tmp_path / "completion.json"
    completion_path.write_text(json.dumps(completion, indent=2) + "\n")
    seal = v1.sha256_file(completion_path)
    assert len(core.load_sealed_run(tmp_path, expected_completion_sha256=seal)[2]) == 1200

    prediction_path.write_bytes(prediction_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="oof_predictions_sha256 binding mismatch"):
        core.load_sealed_run(tmp_path, expected_completion_sha256=seal)
