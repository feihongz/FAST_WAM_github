from __future__ import annotations

import json
from pathlib import Path

import pytest

from fastwam.alignment.checkpointing import canonical_json_sha256, sha256_file
from fastwam.gating.calibration import CALIBRATION_ALGORITHM, DECISION_RULE
from fastwam.gating.calibration_artifact import load_gate_calibration_selection


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _file_identity(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "file_sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _artifact(tmp_path: Path) -> tuple[Path, str]:
    gate_path = tmp_path / "gate.pt"
    gate_path.write_bytes(b"frozen gate bytes")
    records_path = tmp_path / "validation_predictions.jsonl"
    records_path.write_text('{"probability":0.75,"label":1}\n', encoding="utf-8")

    point = {
        "target_with_rate": 0.5,
        "target_count": 1,
        "threshold": 0.75,
        "selected_count": 1,
        "actual_with_rate": 0.5,
        "count_error": 0,
        "rate_error": 0.0,
        "exact_target": True,
        "expected_video_steps_per_query": 5.0,
        "true_positive_count": 1,
        "false_positive_count": 0,
        "false_negative_count": 0,
        "true_negative_count": 1,
        "precision": 1.0,
        "recall": 1.0,
        "block_diagnostics": {},
    }
    replay = {
        "comparison": "python_float_probability >= python_float_threshold",
        "passed": True,
        "points": [
            {
                "actual_with_rate": 0.5,
                "selected_count": 1,
                "threshold": 0.75,
            }
        ],
        "probability_dtype": "float32",
        "sigmoid_location": "gate_device_before_cpu_transfer",
    }
    report = {
        "algorithm": CALIBRATION_ALGORITHM,
        "decision_rule": DECISION_RULE,
        "num_examples": 2,
        "configured_video_steps": 10,
        "label_statistics": {
            "positive_count": 1,
            "negative_count": 1,
            "positive_rate": 0.5,
        },
        "probability_block_diagnostics": {
            "num_unique_probabilities": 2,
        },
        "score_blocks": [],
        "calibrations": [point],
    }
    thresholds = {
        "schema_version": 1,
        "kind": "stage2_gate_validation_thresholds",
        "source_split": "validation",
        "validation_predictions_sha256": sha256_file(records_path),
        "probability_semantics": {
            "logit_dtype": "float32",
            "sigmoid_dtype": "float32",
            "sigmoid_location": "gate_device_before_cpu_transfer",
            "threshold_scalar": "python_float_json_number",
        },
        "runtime_replay": replay,
        **report,
    }
    thresholds_path = tmp_path / "thresholds.json"
    _write_json(thresholds_path, thresholds)

    repeated_sha = {
        "label_manifest_sha256": "1" * 64,
        "adapter_checkpoint_sha256": "2" * 64,
        "base_checkpoint_sha256": "3" * 64,
        "data_manifest_sha256": "4" * 64,
        "episode_split_assignment_sha256": "5" * 64,
        "training_config_sha256": "6" * 64,
    }
    gate_identity = {
        "path": str(gate_path.resolve()),
        "sha256": sha256_file(gate_path),
        "size_bytes": gate_path.stat().st_size,
        "schema_version": 2,
        "kind": "stage2_binary_video_gate_export",
        "parameter_count": 7,
        "global_step": 8,
        "epoch": 9,
        "best_metrics": {"bce": 0.1},
        **repeated_sha,
        "git_identity": {
            "commit": "a" * 40,
            "tracked_dirty": False,
            "untracked_source_files": [],
        },
    }
    export_config = {
        "source_split": "validation",
        "target_with_rates": [0.5],
        "configured_video_steps": 10,
        "expected_validation_samples": 2,
    }
    source_files = {
        "data_manifest": {"semantic_sha256": repeated_sha["data_manifest_sha256"]},
        "episode_split": {
            "semantic_sha256": repeated_sha["episode_split_assignment_sha256"]
        },
        "label_contract": {"semantic_sha256": "7" * 64},
        "merged_label_manifest": {
            "semantic_sha256": repeated_sha["label_manifest_sha256"]
        },
        "merged_label_rows": {"semantic_sha256": "8" * 64},
        "normalization_stats": {"sha256": "9" * 64},
    }
    manifest = {
        "schema_version": 1,
        "kind": "stage2_gate_validation_calibration",
        "source_split": "validation",
        "calibration_algorithm": CALIBRATION_ALGORITHM,
        "export_config": export_config,
        "export_config_sha256": canonical_json_sha256(export_config),
        "gate_checkpoint": gate_identity,
        "gate_run_identity": {},
        "source_files": source_files,
        "selection_binding": {},
        "validation": {
            "num_samples": 2,
            "num_batches": 1,
            "metric_reproduction": {"passed": True},
            "threshold_runtime_replay": replay,
            "records_file": _file_identity(records_path),
        },
        "thresholds_file": _file_identity(thresholds_path),
        "calibration_summary": {
            key: value for key, value in report.items() if key != "score_blocks"
        },
        "runtime": {},
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    manifest_path = tmp_path / "calibration_manifest.json"
    _write_json(manifest_path, manifest)

    complete = {
        "schema_version": 1,
        "kind": "stage2_gate_validation_calibration_complete",
        "manifest_file": manifest_path.name,
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_file_sha256": sha256_file(manifest_path),
        "records_file_sha256": sha256_file(records_path),
        "thresholds_file_sha256": sha256_file(thresholds_path),
    }
    complete_path = tmp_path / "COMPLETE"
    _write_json(complete_path, complete)
    return complete_path, sha256_file(complete_path)


def test_load_gate_calibration_selection_verifies_and_resolves_point(tmp_path):
    complete_path, complete_sha = _artifact(tmp_path)

    loaded = load_gate_calibration_selection(
        complete_path,
        expected_complete_sha256=complete_sha,
        target_with_rate=0.5,
        configured_video_steps=10,
    )

    assert loaded.threshold == 0.75
    assert loaded.target_with_rate == 0.5
    assert loaded.receipt["complete_file"]["sha256"] == complete_sha
    assert loaded.receipt["manifest_file"]["semantic_sha256"] == (
        loaded.manifest["manifest_sha256"]
    )
    assert loaded.receipt["selected_point"]["selected_count"] == 1
    assert loaded.receipt["source_identities"]["normalization_stats_sha256"] == (
        "9" * 64
    )


def test_load_gate_calibration_selection_rejects_tampered_thresholds(tmp_path):
    complete_path, complete_sha = _artifact(tmp_path)
    thresholds_path = tmp_path / "thresholds.json"
    thresholds_path.write_text(
        thresholds_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="thresholds_file identity mismatch"):
        load_gate_calibration_selection(
            complete_path,
            expected_complete_sha256=complete_sha,
            target_with_rate=0.5,
        )


def test_load_gate_calibration_selection_rejects_unpublished_target(tmp_path):
    complete_path, complete_sha = _artifact(tmp_path)

    with pytest.raises(ValueError, match="exactly one published point"):
        load_gate_calibration_selection(
            complete_path,
            expected_complete_sha256=complete_sha,
            target_with_rate=0.25,
        )


def test_load_gate_calibration_selection_rejects_wrong_complete_sha(tmp_path):
    complete_path, _ = _artifact(tmp_path)

    with pytest.raises(ValueError, match="COMPLETE SHA256 mismatch"):
        load_gate_calibration_selection(
            complete_path,
            expected_complete_sha256="0" * 64,
            target_with_rate=0.5,
        )
