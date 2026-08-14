from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.libero.gate import offline_tiny_mlp as v1
from experiments.libero.gate import offline_tiny_mlp_remainder400 as core
from experiments.libero.gate import train_tiny_mlp_remainder400 as cli


def _training_rows(count: int = 12) -> list[dict]:
    rows: list[dict] = []
    for index in range(count):
        center = (index - count / 2) * 2e-4
        rows.append(
            {
                "selection_order": index,
                "sample_id": f"sample-{index}",
                "source_index": index,
                "suite": "libero_goal" if index < count // 2 else "libero_object",
                "task_index": index // 2,
                "task": f"task-{index // 2}",
                "dataset_name": "dataset-a" if index < count // 2 else "dataset-b",
                "episode_index": index // 2,
                "utility_mean": center,
                "utility_sem": 2e-5 + index * 1e-6,
                "t95_ci_low": center - 2e-5,
                "t95_ci_high": center + 2e-5,
                "high_confidence": abs(center) > 1e-4,
            }
        )
    return rows


def test_external_training_is_deterministic_and_test_requires_no_labels(tmp_path):
    rows = _training_rows()
    generator = torch.Generator().manual_seed(2026)
    train_features = torch.randn(12, 7, generator=generator)
    external_features = torch.randn(3, 7, generator=generator)
    config = v1.TrainingConfig(
        max_epochs=5,
        min_epochs=2,
        early_stop_patience=2,
        init_seeds=(7, 8),
    )
    kwargs = {
        "train_features": train_features,
        "test_features": external_features,
        "targets": rows,
        "train_orders": list(range(7)),
        "validation_orders": [7, 8],
        "loss_name": "hybrid",
        "config": config,
    }
    first = core.train_external_ensemble(**kwargs)
    # A Validation4-like file is outside the API and cannot affect fitting.
    decoy = tmp_path / "validation4.jsonl"
    decoy.write_bytes(b"first independent exam\n")
    decoy.write_bytes(b"different independent exam\n")
    second = core.train_external_ensemble(**kwargs)
    assert first == second
    assert len(first.init_predictions) == 2
    assert all(len(values) == 3 for values in first.init_predictions)
    assert torch.tensor(first.ensemble_prediction) == pytest.approx(
        torch.tensor(first.init_predictions, dtype=torch.float64).mean(dim=0)
    )


def test_original_fold_source_verifies_seal_without_parsing_old_predictions(tmp_path):
    fold = {"identity_only": True}
    fold_path = tmp_path / "fold_plan.json"
    fold_path.write_text(json.dumps(fold) + "\n")
    # These are deliberately not JSON. The trainer hashes but never parses them.
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_bytes(b"not parsed old manifest bytes")
    prediction_path = tmp_path / "oof_predictions.jsonl"
    prediction_path.write_bytes(b"not parsed old Target5 prediction bytes")
    completion = {
        "schema_version": 1,
        "kind": v1.COMPLETION_KIND,
        "complete": True,
        "run_manifest_sha256": v1.sha256_file(manifest_path),
        "fold_plan_sha256": v1.sha256_file(fold_path),
        "oof_predictions_sha256": v1.sha256_file(prediction_path),
        "num_predictions": 900,
        "compatibility_fingerprint": "a" * 64,
    }
    completion["completion_sha256"] = v1._completion_sha(completion)
    completion_path = tmp_path / "completion.json"
    completion_path.write_text(json.dumps(completion) + "\n")
    loaded = core.load_original_fold_source(
        tmp_path,
        expected_completion_sha256=v1.sha256_file(completion_path),
    )
    assert loaded.fold_plan == fold


def _write(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return v1.sha256_file(path)


def test_all_loaded_input_files_including_feature_completions_are_rehashed(tmp_path):
    remainder_target = tmp_path / "remainder-target"
    remainder_features = tmp_path / "remainder-features"
    original_features = tmp_path / "original-features"
    original_run = tmp_path / "original-run"
    protocol = tmp_path / "protocol.md"

    hashes = {
        "remainder_manifest": _write(remainder_target / "manifest.json", b"rm"),
        "remainder_targets": _write(remainder_target / "targets.jsonl", b"rt"),
        "remainder_feature_manifest": _write(remainder_features / "manifest.json", b"rfm"),
        "remainder_feature_index": _write(remainder_features / "feature_index.jsonl", b"rfi"),
        "remainder_features": _write(remainder_features / "features.safetensors", b"rft"),
        "remainder_completion": _write(remainder_features / "completion.json", b"rfc"),
        "original_feature_manifest": _write(original_features / "manifest.json", b"ofm"),
        "original_feature_index": _write(original_features / "feature_index.jsonl", b"ofi"),
        "original_features": _write(original_features / "features.safetensors", b"oft"),
        "original_completion": _write(original_features / "completion.json", b"ofc"),
        "original_run_completion": _write(original_run / "completion.json", b"orc"),
        "original_fold": _write(original_run / "fold_plan.json", b"orf"),
        "original_run_manifest": _write(original_run / "run_manifest.json", b"orm"),
        "original_run_predictions": _write(
            original_run / "oof_predictions.jsonl", b"orp"
        ),
        "protocol": _write(protocol, b"protocol"),
    }
    inputs = SimpleNamespace(
        remainder=SimpleNamespace(
            target_dir=remainder_target,
            target_manifest_sha256=hashes["remainder_manifest"],
            target_targets_sha256=hashes["remainder_targets"],
            features=SimpleNamespace(
                root=remainder_features,
                manifest_sha256=hashes["remainder_feature_manifest"],
                index_sha256=hashes["remainder_feature_index"],
                features_sha256=hashes["remainder_features"],
            ),
        ),
        original_features=SimpleNamespace(
            root=original_features,
            manifest_sha256=hashes["original_feature_manifest"],
            index_sha256=hashes["original_feature_index"],
            features_sha256=hashes["original_features"],
        ),
        original_fold_source=SimpleNamespace(
            root=original_run,
            completion_file_sha256=hashes["original_run_completion"],
            fold_plan_sha256=hashes["original_fold"],
            completion={
                "run_manifest_sha256": hashes["original_run_manifest"],
                "oof_predictions_sha256": hashes["original_run_predictions"],
            },
        ),
        protocol_doc=protocol,
        protocol_doc_sha256=hashes["protocol"],
        remainder_feature_completion_file_sha256=hashes["remainder_completion"],
        original_feature_completion_file_sha256=hashes["original_completion"],
    )
    core._assert_inputs_unchanged(inputs)
    (remainder_features / "completion.json").write_bytes(b"mutated")
    with pytest.raises(ValueError, match="sealed input changed during fit"):
        core._assert_inputs_unchanged(inputs)


def test_feature_extractor_contract_rejects_projection_drift():
    extractor = {
        "extractor_fingerprint": core.EXPECTED_EXTRACTOR_FINGERPRINT,
        "visual": {"matrix_content_sha256": "a" * 64},
    }
    compatibility = {
        "extractor_fingerprint": core.EXPECTED_EXTRACTOR_FINGERPRINT,
        "feature_dimensions": v1.FEATURE_DIMS,
        "extractor": extractor,
    }
    bundle = SimpleNamespace(manifest={"compatibility": compatibility})
    assert core._extractor_contract(bundle) == extractor
    compatibility["extractor_fingerprint"] = "b" * 64
    with pytest.raises(ValueError, match="frozen exact-V1 extractor"):
        core._extractor_contract(bundle)


def test_cli_and_trainer_have_no_validation4_or_original_target_input():
    forbidden = {
        "validation_dir",
        "validation4_dir",
        "validation_manifest",
        "validation_records",
        "original_target_dir",
        "original_target_manifest_sha256",
        "original_target_targets_sha256",
    }
    assert forbidden.isdisjoint(inspect.signature(core.load_followup_inputs).parameters)
    assert forbidden.isdisjoint(inspect.signature(core.run_remainder400_followup).parameters)
    action_names = {action.dest.lower() for action in cli.build_parser()._actions}
    assert not any("validation" in name for name in action_names)
    assert not any(name.startswith("original_target") for name in action_names)


def test_cli_requires_all_external_trust_anchors():
    required = {
        action.dest
        for action in cli.build_parser()._actions
        if getattr(action, "required", False)
    }
    assert {
        "remainder_target_manifest_sha256",
        "remainder_target_targets_sha256",
        "remainder_feature_completion_sha256",
        "original_feature_completion_sha256",
        "original_fold_source_completion_sha256",
        "protocol_doc_sha256",
    }.issubset(required)
