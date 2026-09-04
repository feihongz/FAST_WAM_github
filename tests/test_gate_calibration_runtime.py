from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

from hydra import compose, initialize_config_dir
import pytest
import torch

from fastwam.alignment.checkpointing import canonical_json_sha256, sha256_file
from fastwam.gating.calibration import calibrate_probability_thresholds
from fastwam.gating.metrics import compute_gate_binary_metrics
from fastwam.gating.routing import route_with_video_gate
from fastwam.models.video_gate import BinaryVideoGate
from scripts.calibrate_video_gate import (
    _gate_output_views,
    _load_gate_run_identity,
    _metric_reproduction,
    _numerical_runtime_comparison,
    _resolved_configs,
    _threshold_runtime_replay,
    _validate_batch,
    _validation_objective_batch,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts/jihe/calibrate_libero_stage2_gate_1xh100.sh"


def _run_identity() -> dict:
    training_config = {"kind": "test", "value": 7}
    training_config_sha256 = canonical_json_sha256(training_config)
    return {
        "schema_version": 1,
        "kind": "stage2_binary_video_gate_run_identity",
        "training_config": training_config,
        "training_config_sha256": training_config_sha256,
        "training_identity": {
            "label_manifest_sha256": "a" * 64,
            "adapter_checkpoint_sha256": "b" * 64,
            "base_checkpoint_sha256": "c" * 64,
            "data_manifest_sha256": "d" * 64,
            "episode_split_assignment_sha256": "e" * 64,
            "training_config_sha256": training_config_sha256,
            "git_identity": {
                "commit": "f" * 40,
                "tracked_dirty": False,
                "untracked_source_files": [],
            },
        },
    }


def test_gate_run_identity_requires_exact_file_sha_and_self_hash(tmp_path):
    path = tmp_path / "run_identity.json"
    path.write_text(json.dumps(_run_identity()), encoding="utf-8")

    resolved, payload, file_sha = _load_gate_run_identity(
        path, expected_file_sha256=sha256_file(path)
    )

    assert resolved == path.resolve()
    assert payload == _run_identity()
    assert file_sha == sha256_file(path)
    with pytest.raises(ValueError, match="file SHA256 mismatch"):
        _load_gate_run_identity(path, expected_file_sha256="0" * 64)

    drifted = _run_identity()
    drifted["training_config"]["value"] = 8
    path.write_text(json.dumps(drifted), encoding="utf-8")
    with pytest.raises(ValueError, match="self-hash mismatch"):
        _load_gate_run_identity(path, expected_file_sha256=sha256_file(path))


def test_metric_reproduction_is_strict_but_tolerates_small_numeric_delta():
    expected = {
        "bce": 0.2,
        "auroc": 0.7,
        "auprc": 0.6,
        "positive_rate": 0.4,
        "predicted_positive_rate": 0.3,
        "expected_calibration_error": 0.1,
        "num_examples": 4,
        "objective_bce": 0.3,
        "num_batches": 1,
    }
    observed = {
        key: value + 5.0e-7 if isinstance(value, float) else value
        for key, value in expected.items()
    }

    receipt = _metric_reproduction(
        observed=observed, expected=expected, tolerance=1.0e-6
    )

    assert receipt["passed"] is True
    with pytest.raises(ValueError, match="failed reproduction"):
        _metric_reproduction(observed=observed, expected=expected, tolerance=1.0e-8)


def test_calibration_batch_validation_keeps_training_input_schema():
    batch = {
        "input_image": torch.zeros(2, 3, 4, 4),
        "context": torch.zeros(2, 3, 4),
        "context_mask": torch.ones(2, 3, dtype=torch.bool),
        "proprio": torch.zeros(2, 5),
        "label": torch.tensor([True, False]),
        "sample_weight": torch.tensor([1.0, 2.0]),
        "sample_id": ["first", "second"],
    }

    inputs, labels, weights, sample_ids = _validate_batch(
        batch, device=torch.device("cpu")
    )

    assert set(inputs) == {"input_image", "context", "context_mask", "proprio"}
    assert labels.tolist() == [1, 0]
    assert weights.dtype == torch.float64
    assert sample_ids == ("first", "second")


def _frozen_bias_gate(logit: float) -> BinaryVideoGate:
    gate = BinaryVideoGate(
        proprio_dim=2,
        context_dim=4,
        cnn_channels=(2, 2, 2),
        context_feature_dim=2,
        proprio_hidden_dim=2,
        proprio_feature_dim=2,
        fusion_hidden_dim=2,
    )
    with torch.no_grad():
        for parameter in gate.parameters():
            parameter.zero_()
        gate.logit_head[-1].bias.fill_(logit)
    gate.requires_grad_(False)
    return gate.eval()


def _router_inputs() -> dict[str, torch.Tensor]:
    return {
        "input_image": torch.zeros(1, 3, 8, 8),
        "context": torch.zeros(1, 3, 4),
        "context_mask": torch.ones(1, 3, dtype=torch.bool),
        "proprio": torch.zeros(1, 2),
    }


def test_gate_output_views_keep_metric_and_router_sigmoid_numerics_separate():
    raw_logits = torch.tensor([0.1, 0.1, 0.0, -0.1], dtype=torch.float32)

    metric_logits, runtime_logits, runtime_probabilities = _gate_output_views(
        raw_logits
    )

    assert metric_logits.dtype == torch.float64
    assert runtime_logits.dtype == torch.float32
    assert runtime_probabilities.dtype == torch.float32
    assert torch.equal(metric_logits, raw_logits.double())
    assert torch.equal(runtime_logits, raw_logits)
    assert torch.equal(runtime_probabilities, torch.sigmoid(raw_logits.float()))
    assert float(runtime_probabilities[0]) != float(torch.sigmoid(metric_logits)[0])

    labels = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float64)
    expected_metrics = compute_gate_binary_metrics(
        logits=raw_logits.double(), labels=labels
    ).to_dict()
    observed_metrics = compute_gate_binary_metrics(
        logits=metric_logits, labels=labels
    ).to_dict()
    assert observed_metrics == expected_metrics


def test_calibrated_threshold_json_round_trip_replays_real_router_float32():
    raw_logits = torch.tensor([0.1, 0.1, 0.0, -0.1], dtype=torch.float32)
    _, _, runtime_probabilities = _gate_output_views(raw_logits)
    report = calibrate_probability_thresholds(
        runtime_probabilities,
        [1, 1, 0, 0],
        [0.5],
        configured_video_steps=10,
    )
    threshold = json.loads(json.dumps(report["calibrations"][0]["threshold"]))

    assert threshold == float(torch.sigmoid(raw_logits.float())[0])
    replay = _threshold_runtime_replay(
        probabilities=[float(value) for value in runtime_probabilities],
        calibration_report=report,
    )
    assert replay["passed"] is True
    assert replay["points"][0]["selected_count"] == 2

    modes = [
        route_with_video_gate(
            _frozen_bias_gate(float(logit)),
            **_router_inputs(),
            threshold=threshold,
            configured_video_steps=10,
            clock=lambda: 1.0,
        )["selected_mode"]
        for logit in raw_logits
    ]
    assert modes == ["w", "w", "wo", "wo"]

    wrong_float64_boundary = float(torch.sigmoid(raw_logits.double())[0])
    assert wrong_float64_boundary > threshold
    wrong_modes = [
        route_with_video_gate(
            _frozen_bias_gate(float(logit)),
            **_router_inputs(),
            threshold=wrong_float64_boundary,
            configured_video_steps=10,
            clock=lambda: 1.0,
        )["selected_mode"]
        for logit in raw_logits
    ]
    assert wrong_modes.count("w") == 0


def test_validation_objective_batch_replays_weighted_pos_weight_formula():
    loss, weight_sum = _validation_objective_batch(
        raw_logits=torch.zeros(2, dtype=torch.float32),
        labels=torch.tensor([1, 0]),
        weights=torch.tensor([0.25, 2.0], dtype=torch.float64),
        pos_weight=2.0,
    )

    expected = (2.0 * torch.log(torch.tensor(2.0)) * 0.25 + torch.log(
        torch.tensor(2.0)
    ) * 2.0) / 2.25
    assert loss == pytest.approx(float(expected))
    assert weight_sum == 2.25


def test_numerical_runtime_comparison_records_diff_and_can_fail_closed():
    training = {"versions": {"torch": "2.7.1"}, "device": {"cudnn": 90700}}
    current = {"versions": {"torch": "2.7.1"}, "device": {"cudnn": 90701}}

    receipt = _numerical_runtime_comparison(
        training_runtime=training,
        current_runtime=current,
        require_exact=False,
    )

    assert receipt["policy"] == "metric_reproduction_guarded"
    assert receipt["exact_match"] is False
    assert receipt["num_differences"] == 1
    assert receipt["differences"][0]["path"] == "device.cudnn"
    with pytest.raises(RuntimeError, match="differs from Gate training runtime"):
        _numerical_runtime_comparison(
            training_runtime=training,
            current_runtime=current,
            require_exact=True,
        )


def test_calibration_config_composes_over_exact_training_contract(monkeypatch):
    monkeypatch.setenv("FASTWAM_LIBERO_STAGE2_GATE_RUN", "/tmp/gate-run")
    with initialize_config_dir(
        config_dir=str(REPO_ROOT / "configs"), version_base="1.3"
    ):
        config = compose(
            config_name="calibrate_video_gate",
            overrides=[
                "task=libero_stage2_gate_2cam224",
                "calibration.gate_checkpoint=/tmp/gate.pt",
                f"calibration.gate_checkpoint_sha256={'a' * 64}",
                "calibration.gate_run_identity=/tmp/run_identity.json",
                f"calibration.gate_run_identity_sha256={'b' * 64}",
                "calibration.output_dir=/tmp/calibration",
            ],
        )

    training, calibration = _resolved_configs(config)

    assert training["output_dir"] == "/tmp/gate-run"
    assert calibration["source_split"] == "validation"
    assert calibration["target_with_rates"] == [0.1, 0.25, 0.5, 0.75, 0.9]
    assert calibration["configured_video_steps"] == 10
    assert calibration["require_exact_training_numerical_runtime"] is False


def test_libero_calibration_launcher_is_locked_one_h100_dry_run():
    completed = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env={**os.environ, "FASTWAM_DRY_RUN": "1", "RUN_ID": "pytest"},
        check=True,
        capture_output=True,
        text=True,
    )

    output = completed.stdout
    assert "topology=1xH100 single_process" in output
    assert "samples=5408" in output
    assert "batches=85" in output
    assert "target_with_rates=[0.10,0.25,0.50,0.75,0.90]" in output
    assert "calibration.source_split=validation" in output
    assert "calibration.configured_video_steps=10" in output
    assert "calibration.gate_checkpoint_sha256=67db6f46" in output
    assert "calibration.gate_run_identity_sha256=c7562b91" in output
    assert "calibration.require_exact_training_numerical_runtime=false" in output
