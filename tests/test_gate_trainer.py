import copy
import inspect
import math
import random

import numpy as np
import pytest
import torch

import fastwam.gating.trainer as trainer_module
from fastwam.gating.trainer import GateTrainer
from fastwam.models.video_gate import BinaryVideoGate


def _gate() -> BinaryVideoGate:
    return BinaryVideoGate(
        proprio_dim=2,
        context_dim=5,
        cnn_channels=(2, 3, 4),
        context_feature_dim=3,
        proprio_hidden_dim=3,
        proprio_feature_dim=2,
        fusion_hidden_dim=4,
    )


def _training_identity(**overrides):
    identity = {
        "label_manifest_sha256": "a" * 64,
        "adapter_checkpoint_sha256": "b" * 64,
        "base_checkpoint_sha256": "c" * 64,
        "data_manifest_sha256": "d" * 64,
        "episode_split_assignment_sha256": "e" * 64,
        "training_config_sha256": "f" * 64,
        "git_identity": {
            "commit": "1" * 40,
            "tracked_dirty": False,
            "untracked_source_files": [],
        },
    }
    identity.update(overrides)
    return identity


def _batch(*, labels=(1.0, 0.0), weights=(1.0, 1.0), prefix="sample"):
    batch_size = len(labels)
    generator = torch.Generator().manual_seed(91 + len(prefix))
    return {
        "input_image": torch.randn(
            batch_size,
            3,
            8,
            8,
            generator=generator,
        ),
        "context": torch.randn(
            batch_size,
            3,
            5,
            generator=generator,
        ),
        "context_mask": torch.tensor(
            [[True, True, False]] * batch_size,
        ),
        "proprio": torch.randn(
            batch_size,
            2,
            generator=generator,
        ),
        "label": torch.tensor(labels),
        "sample_weight": torch.tensor(weights),
        "sample_id": [f"{prefix}-{index}" for index in range(batch_size)],
    }


def _state(gate):
    return {
        name: value.detach().clone()
        for name, value in gate.state_dict().items()
    }


def _trees_equal(left, right):
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return np.array_equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _trees_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _trees_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return left == right


def test_training_identity_is_mandatory_validated_and_deeply_immutable():
    with pytest.raises(TypeError, match="training_identity"):
        GateTrainer(_gate(), train_labels=[0, 1])

    missing = _training_identity()
    del missing["base_checkpoint_sha256"]
    with pytest.raises(ValueError, match="fields do not match schema"):
        GateTrainer(
            _gate(),
            train_labels=[0, 1],
            training_identity=missing,
        )

    malformed = _training_identity(label_manifest_sha256="not-a-sha")
    with pytest.raises(ValueError, match="label_manifest_sha256"):
        GateTrainer(
            _gate(),
            train_labels=[0, 1],
            training_identity=malformed,
        )

    non_json = _training_identity()
    non_json["git_identity"]["untracked_source_files"] = [{"not-json-safe"}]
    with pytest.raises(ValueError, match="canonical-JSON serializable"):
        GateTrainer(
            _gate(),
            train_labels=[0, 1],
            training_identity=non_json,
        )

    source_identity = _training_identity()
    trainer = GateTrainer(
        _gate(),
        train_labels=[0, 1],
        training_identity=source_identity,
    )
    source_identity["label_manifest_sha256"] = "9" * 64
    source_identity["git_identity"]["tracked_dirty"] = True
    assert trainer.training_identity["label_manifest_sha256"] == "a" * 64
    assert trainer.training_identity["git_identity"]["tracked_dirty"] is False
    with pytest.raises(TypeError):
        trainer.training_identity["label_manifest_sha256"] = "9" * 64
    with pytest.raises(TypeError):
        trainer.training_identity["git_identity"]["tracked_dirty"] = True
    with pytest.raises(AttributeError):
        trainer.training_identity = _training_identity()


def test_weighted_pos_weight_loss_matches_manual_value_and_updates_gate():
    gate = _gate()
    for parameter in gate.parameters():
        parameter.data.zero_()
    trainer = GateTrainer(
        gate,
        train_labels=[1, 0, 0],
        training_identity=_training_identity(),
        lr=0.1,
    )
    before = _state(gate)
    batch = _batch(labels=(1.0, 0.0), weights=(0.25, 2.0))

    actual = trainer.train_batch(batch)
    expected = (2.0 * math.log(2.0) * 0.25 + math.log(2.0) * 2.0) / 2.25

    assert trainer.pos_weight == pytest.approx(2.0)
    assert actual == pytest.approx(expected)
    assert trainer.global_step == 1
    assert any(
        not torch.equal(value, before[name])
        for name, value in gate.state_dict().items()
    )


def test_train_and_eval_epochs_report_binary_metrics_without_eval_mutation():
    torch.manual_seed(7)
    gate = _gate()
    trainer = GateTrainer(
        gate,
        train_labels=[0, 1, 0, 1],
        training_identity=_training_identity(),
    )
    batches = [_batch(prefix="a"), _batch(prefix="b")]

    train_result = trainer.train_epoch(batches)
    before_eval = _state(gate)
    eval_result = trainer.evaluate_epoch(batches)

    assert trainer.epoch == 1
    assert trainer.global_step == 2
    assert train_result.num_batches == 2
    assert train_result.metrics.num_examples == 4
    assert train_result.metrics.positive_rate == pytest.approx(0.5)
    assert 0.0 <= train_result.metrics.auroc <= 1.0
    assert 0.0 <= train_result.metrics.auprc <= 1.0
    assert math.isfinite(train_result.metrics.expected_calibration_error)
    assert math.isfinite(eval_result.objective_bce)
    assert all(
        torch.equal(value, before_eval[name])
        for name, value in gate.state_dict().items()
    )


@pytest.mark.parametrize("labels", [[0, 0], [1, 1]])
def test_training_split_requires_both_classes(labels):
    with pytest.raises(ValueError, match="both label classes"):
        GateTrainer(
            _gate(),
            train_labels=labels,
            training_identity=_training_identity(),
        )


@pytest.mark.parametrize("forbidden", ["action", "future", "E0", "E10"])
def test_batch_rejects_every_non_current_feature(forbidden):
    trainer = GateTrainer(
        _gate(),
        train_labels=[0, 1],
        training_identity=_training_identity(),
    )
    batch = _batch()
    batch[forbidden] = torch.zeros(2)

    with pytest.raises(ValueError, match="current-only"):
        trainer.evaluate_epoch([batch])


def test_optimizer_defaults_and_parameter_contract():
    gate = _gate()
    trainer = GateTrainer(
        gate,
        train_labels=[0, 1],
        training_identity=_training_identity(),
    )

    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)
    assert trainer.optimizer.param_groups[0]["weight_decay"] == pytest.approx(
        1e-4
    )

    foreign_gate = _gate()
    foreign_optimizer = torch.optim.AdamW(foreign_gate.parameters())
    with pytest.raises(ValueError, match="exactly Gate parameters"):
        GateTrainer(
            gate,
            train_labels=[0, 1],
            training_identity=_training_identity(),
            optimizer=foreign_optimizer,
        )


def test_strict_resume_matches_uninterrupted_training_and_restores_rng(tmp_path):
    torch.manual_seed(19)
    initial = _state(_gate())
    batches = [_batch(prefix="one"), _batch(prefix="two")]

    full_gate = _gate()
    full_gate.load_state_dict(initial)
    full = GateTrainer(
        full_gate,
        train_labels=[0, 1, 0, 1],
        training_identity=_training_identity(),
    )
    torch.manual_seed(101)
    full.train_epoch(batches)
    full.train_epoch(batches)

    split_gate = _gate()
    split_gate.load_state_dict(initial)
    split = GateTrainer(
        split_gate,
        train_labels=[0, 1, 0, 1],
        training_identity=_training_identity(),
    )
    torch.manual_seed(101)
    split.train_epoch(batches)
    state_path = split.save_training_state(tmp_path / "state.pt")

    expected_python = random.random()
    expected_numpy = float(np.random.rand())
    expected_torch = torch.rand(3)

    resumed = GateTrainer(
        _gate(),
        train_labels=[0, 1, 0, 1],
        training_identity=_training_identity(),
    )
    resumed.load_training_state(state_path)
    assert random.random() == expected_python
    assert float(np.random.rand()) == expected_numpy
    assert torch.equal(torch.rand(3), expected_torch)
    resumed.load_training_state(state_path)
    resumed.train_epoch(batches)

    assert resumed.epoch == full.epoch == 2
    assert resumed.global_step == full.global_step == 4
    assert all(
        torch.equal(value, full_gate.state_dict()[name])
        for name, value in resumed.gate.state_dict().items()
    )


def test_resume_identity_mismatch_has_no_model_optimizer_progress_or_rng_side_effect(
    tmp_path,
):
    source = GateTrainer(
        _gate(),
        train_labels=[0, 1],
        training_identity=_training_identity(),
    )
    source.train_batch(_batch(prefix="source"))
    state_path = source.save_training_state(tmp_path / "wrong-identity.pt")

    target = GateTrainer(
        _gate(),
        train_labels=[0, 1],
        training_identity=_training_identity(label_manifest_sha256="9" * 64),
    )
    target.train_batch(_batch(prefix="target"))
    random.seed(1234)
    np.random.seed(5678)
    torch.manual_seed(9012)

    before_gate = _state(target.gate)
    before_gradients = {
        name: (
            None if parameter.grad is None else parameter.grad.detach().clone()
        )
        for name, parameter in target.gate.named_parameters()
    }
    before_optimizer = copy.deepcopy(target.optimizer.state_dict())
    before_progress = (
        target.epoch,
        target.global_step,
        target.best_epoch,
        target.best_global_step,
        target.best_val_bce,
        copy.deepcopy(target.best_metrics),
        target.epochs_without_improvement,
        copy.deepcopy(target._best_gate_state),
    )
    before_python_rng = random.getstate()
    before_numpy_rng = copy.deepcopy(np.random.get_state())
    before_torch_rng = torch.get_rng_state().clone()
    before_training_mode = target.gate.training

    with pytest.raises(ValueError, match="training_identity mismatch"):
        target.load_training_state(state_path)

    assert all(
        torch.equal(value, before_gate[name])
        for name, value in target.gate.state_dict().items()
    )
    assert all(
        _trees_equal(
            None if parameter.grad is None else parameter.grad,
            before_gradients[name],
        )
        for name, parameter in target.gate.named_parameters()
    )
    assert _trees_equal(target.optimizer.state_dict(), before_optimizer)
    assert _trees_equal(
        (
            target.epoch,
            target.global_step,
            target.best_epoch,
            target.best_global_step,
            target.best_val_bce,
            target.best_metrics,
            target.epochs_without_improvement,
            target._best_gate_state,
        ),
        before_progress,
    )
    assert random.getstate() == before_python_rng
    assert _trees_equal(np.random.get_state(), before_numpy_rng)
    assert torch.equal(torch.get_rng_state(), before_torch_rng)
    assert target.gate.training is before_training_mode


def test_fit_early_stops_on_validation_bce_and_exports_best_and_last(tmp_path):
    torch.manual_seed(23)
    trainer = GateTrainer(
        _gate(),
        train_labels=[0, 1],
        training_identity=_training_identity(),
    )
    batches = [_batch()]
    result = trainer.fit(
        batches,
        batches,
        num_epochs=4,
        early_stop_patience=1,
        min_delta=100.0,
    )

    assert result.stopped_early
    assert len(result.epochs) == 2
    assert result.best_epoch == 1
    identities = {
        "label_manifest_sha256": "a" * 64,
        "adapter_checkpoint_sha256": "b" * 64,
        "data_manifest_sha256": "d" * 64,
        "episode_split_assignment_sha256": "e" * 64,
        "training_config_sha256": "f" * 64,
        "git_identity": {
            "commit": "1" * 40,
            "tracked_dirty": False,
            "untracked_source_files": [],
        },
    }
    best_path = trainer.export_checkpoint(
        tmp_path / "best.pt",
        selection="best",
        **identities,
    )
    last_path = trainer.export_checkpoint(
        tmp_path / "last.pt",
        selection="last",
        **identities,
    )

    best = torch.load(best_path, map_location="cpu", weights_only=False)
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    assert best["epoch"] == 1
    assert last["epoch"] == 2
    assert best["best_metrics"]["bce"] == pytest.approx(result.best_val_bce)

    wrong_identity = copy.deepcopy(identities)
    wrong_identity["adapter_checkpoint_sha256"] = "9" * 64
    wrong_path = tmp_path / "wrong.pt"
    with pytest.raises(ValueError, match="differs from the training identity"):
        trainer.export_checkpoint(
            wrong_path,
            selection="last",
            **wrong_identity,
        )
    assert not wrong_path.exists()


def test_trainer_source_has_no_wam_dependency():
    source = inspect.getsource(trainer_module)
    assert "fastwam.models.wan22" not in source
    assert "FastWAM" not in source
