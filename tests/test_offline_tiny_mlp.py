from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from experiments.libero.gate import offline_tiny_mlp as core


def _targets_100() -> list[dict]:
    rows: list[dict] = []
    order = 0
    for suite in core.SUITES:
        for task_index in range(10):
            state_count = 2 if task_index < 5 else 3
            for state in range(state_count):
                rows.append(
                    {
                        "selection_order": order,
                        "sample_id": f"{suite}/task_{task_index}/state_{state}",
                        "source_index": 10_000 + order,
                        "suite": suite,
                        "task_index": task_index,
                        "task": f"{suite} task {task_index}",
                    }
                )
                order += 1
    assert len(rows) == 100
    return rows


def test_identity_only_fold_plan_is_exact_deterministic_and_has_no_group_leakage():
    targets = _targets_100()
    plan = core.build_fold_plan(targets)
    assert plan == core.build_fold_plan(targets)
    assert len(plan["task_heldout_folds"]) == 5
    assert len(plan["suite_heldout_folds"]) == 4

    task_oof: list[int] = []
    for fold in plan["task_heldout_folds"]:
        train = set(fold["train_groups"])
        inner = set(fold["inner_validation_groups"])
        test = set(fold["test_groups"])
        assert not train & inner
        assert not train & test
        assert not inner & test
        assert len(test) == 8
        assert len(fold["test_selection_orders"]) == 20
        assert {
            suite: sum(targets[index]["suite"] == suite for index in fold["test_selection_orders"])
            for suite in core.SUITES
        } == {suite: 5 for suite in core.SUITES}
        task_oof.extend(fold["test_selection_orders"])
    assert sorted(task_oof) == list(range(100))

    suite_oof = sorted(
        order
        for fold in plan["suite_heldout_folds"]
        for order in fold["test_selection_orders"]
    )
    assert suite_oof == list(range(100))
    for fold in plan["suite_heldout_folds"]:
        heldout = fold["heldout_suite"]
        assert {targets[index]["suite"] for index in fold["test_selection_orders"]} == {
            heldout
        }
        assert all(
            json.loads(group)["suite"] != heldout
            for group in fold["train_groups"] + fold["inner_validation_groups"]
        )


def test_fold_plan_uses_suite_task_and_text_not_bare_task_index():
    targets = _targets_100()
    groups = core._groups(targets)
    assert len(groups) == 40
    assert len({row["task_index"] for row in targets}) == 10


def test_tensor_digest_matches_frozen_feature_cache_algorithm():
    tensor = torch.tensor([[1.0, -2.0], [3.5, 0.0]], dtype=torch.float32)
    header = core.canonical_json(
        {"schema_version": 1, "dtype": "torch.float32", "shape": [2, 2]}
    ).encode()
    raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
    assert core.tensor_content_sha256(tensor) == hashlib.sha256(
        header + b"\0" + raw
    ).hexdigest()


def _write_feature_cache(root: Path) -> str:
    visual = torch.arange(2 * 64, dtype=torch.float32).reshape(2, 64) / 100.0
    instruction = torch.arange(2 * 65, dtype=torch.float32).reshape(2, 65) / 50.0
    proprio = torch.arange(2 * 8, dtype=torch.float32).reshape(2, 8) / 10.0
    tensors = {
        "visual": visual,
        "instruction": instruction,
        "proprio": proprio,
        "full": torch.cat((visual, instruction, proprio), dim=1),
    }
    feature_path = root / "features.safetensors"
    save_file(tensors, str(feature_path))
    compatibility = {
        "schema_version": 1,
        "kind": core.FEATURE_CACHE_KIND,
        "num_states": 2,
        "feature_dimensions": core.FEATURE_DIMS,
    }
    manifest = {
        "schema_version": 1,
        "kind": core.FEATURE_CACHE_KIND,
        "compatibility": compatibility,
        "compatibility_fingerprint": core.sha256_json(compatibility),
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    for order in range(2):
        row = {
            "feature_record_schema_version": 1,
            "kind": core.FEATURE_CACHE_KIND,
            "selection_order": order,
            "sample_id": f"sample-{order}",
            "source_index": order + 10,
            "suite": "libero_goal",
            "task_index": order,
            "task": f"task-{order}",
            "target_id": f"target-{order}",
            "target_sha256": f"{order + 1}" * 64,
            "input_combined_sha256": f"{order + 3}" * 64,
            "feature_id": f"feature-{order}",
            "feature_hashes": {
                key: core.tensor_content_sha256(tensor[order])
                for key, tensor in tensors.items()
            },
        }
        row["feature_record_sha256"] = core._payload_sha(
            row, "feature_record_sha256"
        )
        rows.append(row)
    index_path = root / "feature_index.jsonl"
    index_path.write_bytes(core._jsonl_bytes(rows))
    completion = {
        "schema_version": 1,
        "kind": core.FEATURE_COMPLETION_KIND,
        "manifest_sha256": core.sha256_file(manifest_path),
        "manifest_compatibility_fingerprint": manifest[
            "compatibility_fingerprint"
        ],
        "feature_index_sha256": core.sha256_file(index_path),
        "features_sha256": core.sha256_file(feature_path),
        "num_states": 2,
        "tensors": {
            key: {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "content_sha256": core.tensor_content_sha256(tensor),
            }
            for key, tensor in tensors.items()
        },
    }
    completion["completion_sha256"] = core._completion_sha(completion)
    (root / "completion.json").write_text(json.dumps(completion, indent=2) + "\n")
    return completion["completion_sha256"]


def test_feature_loader_is_byte_compatible_with_frozen_collector_contract(tmp_path):
    expected = _write_feature_cache(tmp_path)
    bundle = core.load_feature_bundle(
        tmp_path, expected_completion_sha256=expected
    )
    assert bundle.tensors["full"].shape == (2, 137)
    assert torch.equal(
        bundle.tensors["full"],
        torch.cat(
            (
                bundle.tensors["visual"],
                bundle.tensors["instruction"],
                bundle.tensors["proprio"],
            ),
            dim=1,
        ),
    )


def _write_minimal_sealed_run(root: Path) -> str:
    fold_plan = {
        "schema_version": 1,
        "kind": core.FOLD_PLAN_KIND,
        "task_heldout_folds": [
            {"fold_id": 0, "test_selection_orders": [0]}
        ],
        "suite_heldout_folds": [],
    }
    fold_path = root / "fold_plan.json"
    fold_path.write_text(json.dumps(fold_plan, indent=2) + "\n")
    row = {
        "schema_version": 1,
        "kind": core.PREDICTION_KIND,
        "prediction_id": "task_heldout/fold_0/full_hybrid/sample",
        "selection_order": 0,
        "sample_id": "sample",
        "source_index": 1,
        "suite": "libero_goal",
        "task_index": 0,
        "task": "task",
        "target_id": "target",
        "target_sha256": "a" * 64,
        "input_combined_sha256": "b" * 64,
        "feature_id": "feature",
        "feature_record_sha256": "c" * 64,
        "outer_scheme": "task_heldout",
        "fold_id": 0,
        "test_group": core.canonical_json(
            {"suite": "libero_goal", "task_index": 0, "task": "task"}
        ),
        "model_name": "full_hybrid",
        "feature_view": "full",
        "loss_name": "hybrid",
        "init_seeds": list(core.INIT_SEEDS),
        "init_predictions": [0.1] * 5,
        "prediction": 0.1,
        "target5_utility_mean": 0.2,
        "target5_utility_sem": 0.01,
        "target5_high_confidence": True,
    }
    row["prediction_sha256"] = core._payload_sha(row, "prediction_sha256")
    rows = [row]
    prediction_bytes = core._jsonl_bytes(rows)
    (root / "oof_predictions.jsonl").write_bytes(prediction_bytes)
    compatibility = {
        "schema_version": 1,
        "kind": core.RUN_KIND,
        "fold_plan_sha256": core.sha256_file(fold_path),
    }
    manifest = {
        "schema_version": 1,
        "kind": core.RUN_KIND,
        "compatibility": compatibility,
        "compatibility_fingerprint": core.sha256_json(compatibility),
        "predictions": {
            "count": 1,
            "ordered_prediction_ids": [row["prediction_id"]],
            "ordered_prediction_sha256": [row["prediction_sha256"]],
            "canonical_records_sha256": hashlib.sha256(prediction_bytes).hexdigest(),
        },
    }
    manifest_path = root / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    completion = {
        "schema_version": 1,
        "kind": core.COMPLETION_KIND,
        "complete": True,
        "run_manifest_sha256": core.sha256_file(manifest_path),
        "fold_plan_sha256": core.sha256_file(fold_path),
        "oof_predictions_sha256": core.sha256_file(root / "oof_predictions.jsonl"),
        "num_predictions": 1,
        "compatibility_fingerprint": manifest["compatibility_fingerprint"],
    }
    completion["completion_sha256"] = core._completion_sha(completion)
    completion_path = root / "completion.json"
    completion_path.write_text(json.dumps(completion, indent=2) + "\n")
    return core.sha256_file(completion_path)


def test_load_sealed_run_requires_external_seal_and_rejects_prediction_tamper(tmp_path):
    expected = _write_minimal_sealed_run(tmp_path)
    _, _, rows = core.load_sealed_run(
        tmp_path, expected_completion_sha256=expected
    )
    assert rows[0]["prediction"] == pytest.approx(0.1)

    path = tmp_path / "oof_predictions.jsonl"
    path.write_text(path.read_text().replace('"prediction":0.1', '"prediction":0.9'))
    with pytest.raises(ValueError, match="oof_predictions_sha256 binding mismatch"):
        core.load_sealed_run(tmp_path, expected_completion_sha256=expected)


def test_tiny_mlp_architecture_is_preregistered_137_32_16_scalar():
    model = core.TinyUtilityMLP(137)
    linear = [layer for layer in model.network if isinstance(layer, torch.nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in linear] == [
        (137, 32),
        (32, 16),
        (16, 1),
    ]
    assert model(torch.zeros(3, 137)).shape == (3,)


def test_fold_plan_is_written_and_hash_frozen_before_fit_staging(tmp_path):
    plan = core.build_fold_plan(_targets_100())
    output, staging, frozen_sha = core.prepare_fold_plan_staging(
        tmp_path / "run", plan
    )
    assert output == (tmp_path / "run").resolve()
    fold_path = staging / "fold_plan.json"
    assert fold_path.is_file()
    assert core.sha256_file(fold_path) == frozen_sha
    assert json.loads(fold_path.read_text()) == plan

    fold_path.write_text(fold_path.read_text().replace('"fold_id": 0', '"fold_id": 9', 1))
    assert core.sha256_file(fold_path) != frozen_sha


def _formal_run_manifest() -> dict:
    config = core.asdict(core.TrainingConfig())
    transform = {
        "feature_mean_sha256": "1" * 64,
        "feature_std_sha256": "2" * 64,
        "target_median": 0.0,
        "target_scale": 1e-3,
        "uncertainty_weights_sha256": "3" * 64,
        "ranking_pair_count": 12,
    }
    diagnostics = {"task_heldout": {}, "suite_heldout": {}}
    for fold_id in range(5):
        for name, _, loss in core.TASK_MODELS:
            key = f"fold_{fold_id}/{name}"
            diagnostics["task_heldout"][key] = (
                {"baseline": True}
                if loss == "baseline"
                else {
                    "best_epochs": [100] * 5,
                    "train_transform": dict(transform),
                }
            )
    for fold_id in range(4):
        for name, _, _ in core.SUITE_MODELS:
            diagnostics["suite_heldout"][f"fold_{fold_id}/{name}"] = {
                "best_epochs": [100] * 5,
                "train_transform": dict(transform),
            }
    compatibility = {
        "schema_version": 1,
        "kind": core.RUN_KIND,
        "target_manifest_sha256": "4" * 64,
        "target_targets_sha256": "5" * 64,
        "target_compatibility_fingerprint": "6" * 64,
        "target_records_sha256": "7" * 64,
        "feature_manifest_sha256": "8" * 64,
        "feature_index_sha256": "9" * 64,
        "features_sha256": "a" * 64,
        "feature_completion_sha256": "b" * 64,
        "feature_manifest_compatibility_fingerprint": "c" * 64,
        "feature_index_records_sha256": "d" * 64,
        "feature_tensor_content_sha256": {
            key: f"{index:x}" * 64
            for index, key in enumerate(sorted(core.FEATURE_DIMS), start=1)
        },
        "fold_plan_sha256": "e" * 64,
        "trainer_source_sha256": "f" * 64,
        "config_sha256": core.sha256_json(config),
        "random_namespace": core.RANDOM_NAMESPACE,
        "random_salts_count": 1000,
        "num_states": 100,
        "task_oof_rows": 800,
        "suite_oof_rows": 100,
        "total_oof_rows": 900,
        "formal_protocol": True,
    }
    manifest = {
        "schema_version": 1,
        "kind": core.RUN_KIND,
        "compatibility": compatibility,
        "compatibility_fingerprint": core.sha256_json(compatibility),
        "training_config": config,
        "models": {
            "task_heldout": [name for name, _, _ in core.TASK_MODELS],
            "suite_heldout": [name for name, _, _ in core.SUITE_MODELS],
            "best_nonvisual": "instruction_proprio_hybrid",
        },
        "predictions": {
            "filename": "oof_predictions.jsonl",
            "count": 900,
            "ordered_prediction_ids": [f"prediction-{index}" for index in range(900)],
            "ordered_prediction_sha256": ["1" * 64 for _ in range(900)],
            "canonical_records_sha256": "2" * 64,
        },
        "diagnostics": diagnostics,
    }
    return manifest


def _rebind_formal_compatibility(manifest: dict) -> None:
    manifest["compatibility_fingerprint"] = core.sha256_json(
        manifest["compatibility"]
    )


def test_formal_run_contract_accepts_only_exact_preregistered_protocol():
    targets = _targets_100()
    plan = core.build_fold_plan(targets)
    core.validate_formal_run_contract(_formal_run_manifest(), plan, targets)


def test_formal_run_contract_rejects_self_consistent_wrong_deterministic_fold():
    targets = _targets_100()
    plan = core.build_fold_plan(targets)
    plan["task_heldout_folds"][0]["test_selection_orders"][0] = 99
    with pytest.raises(ValueError, match="deterministic identity-only plan"):
        core.validate_formal_run_contract(_formal_run_manifest(), plan, targets)


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("formal_false", "formal_protocol"),
        ("config", "training_config"),
        ("models", "model set"),
        ("compat_count", "total_oof_rows"),
        ("prediction_count", "prediction filename/count"),
        ("random_namespace", "random_namespace"),
        ("random_count", "random_salts_count"),
        ("diagnostics", "diagnostics key contract"),
        ("trainer_source", "trainer_source_sha256"),
    ],
)
def test_formal_run_contract_rejects_self_consistent_protocol_tamper(
    tamper: str, message: str
):
    targets = _targets_100()
    plan = core.build_fold_plan(targets)
    manifest = _formal_run_manifest()
    if tamper == "formal_false":
        manifest["compatibility"]["formal_protocol"] = False
    elif tamper == "config":
        manifest["training_config"]["max_epochs"] = 999
        manifest["compatibility"]["config_sha256"] = core.sha256_json(
            manifest["training_config"]
        )
    elif tamper == "models":
        manifest["models"]["task_heldout"].pop()
    elif tamper == "compat_count":
        manifest["compatibility"]["total_oof_rows"] = 899
    elif tamper == "prediction_count":
        manifest["predictions"]["count"] = 899
    elif tamper == "random_namespace":
        manifest["compatibility"]["random_namespace"] = "tampered"
    elif tamper == "random_count":
        manifest["compatibility"]["random_salts_count"] = 999
    elif tamper == "diagnostics":
        manifest["diagnostics"]["task_heldout"].pop("fold_0/full_hybrid")
    elif tamper == "trainer_source":
        manifest["compatibility"]["trainer_source_sha256"] = "not-a-sha"
    else:
        raise AssertionError(tamper)
    _rebind_formal_compatibility(manifest)
    with pytest.raises(ValueError, match=message):
        core.validate_formal_run_contract(manifest, plan, targets)
