from __future__ import annotations

import inspect

import pytest
import torch

from experiments.libero.gate import offline_tiny_mlp as core
from experiments.libero.gate import train_tiny_mlp as cli


def _training_rows(count: int = 12) -> list[dict]:
    rows = []
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
                "utility_mean": center,
                "utility_sem": 2e-5 + index * 1e-6,
                "t95_ci_low": center - 2e-5,
                "t95_ci_high": center + 2e-5,
                "high_confidence": abs(center) > 1e-4,
            }
        )
    return rows


def test_robust_target_scale_uses_linear_median_for_even_samples():
    scale = core.fit_robust_target_scale(torch.tensor([0.0, 2.0, 4.0, 10.0]))
    assert scale.median == pytest.approx(3.0)
    # |x-3| = [3,1,1,7], linear median = 2.
    assert scale.scale == pytest.approx(1.4826 * 2.0)


def test_feature_and_target_transforms_fit_only_fold_train_rows():
    train = torch.tensor([[0.0, 2.0], [2.0, 4.0]], dtype=torch.float32)
    heldout_a = torch.tensor([[10.0, 20.0]], dtype=torch.float32)
    heldout_b = torch.tensor([[1e9, -1e9]], dtype=torch.float32)
    first = core.fit_standardizer(train)
    second = core.fit_standardizer(train)
    assert torch.equal(first.mean, second.mean)
    assert torch.equal(first.std, second.std)
    assert not torch.equal(first.transform(heldout_a), first.transform(heldout_b))
    assert torch.equal(first.mean, torch.tensor([1.0, 3.0]))

    target_scale = core.fit_robust_target_scale(torch.tensor([-2.0, 0.0, 2.0, 4.0]))
    assert target_scale.median == pytest.approx(1.0)


def test_uncertainty_weights_are_clipped_then_renormalized():
    weights = core.uncertainty_weights(
        torch.tensor([0.0, 1e-5, 1e-3, 100.0]), robust_scale=1e-3
    )
    assert float(weights.mean()) == pytest.approx(1.0)
    assert torch.isfinite(weights).all()
    # The second normalization can move the literal bounds, but preserves order.
    assert list(weights.argsort(descending=True).tolist()) == [0, 1, 2, 3]


def test_ranking_pairs_only_use_nonoverlapping_target5_intervals_and_double_same_task():
    rows = _training_rows(6)
    # Force state 0/1 to the same task, with disjoint intervals.
    rows[0]["suite"] = rows[1]["suite"] = "libero_goal"
    rows[0]["task_index"] = rows[1]["task_index"] = 9
    rows[0]["task"] = rows[1]["task"] = "same"
    high, low, weights = core.build_ranking_pairs(rows, list(range(6)))
    assert high.numel() > 0
    assert torch.all(weights >= 1.0)
    pair_to_weight = {
        (int(a), int(b)): float(weight)
        for a, b, weight in zip(high, low, weights, strict=True)
    }
    assert pair_to_weight[(1, 0)] == pytest.approx(2.0)


def test_small_cpu_fold_training_is_deterministic_and_ensembles_all_fixed_test_seeds(
    tmp_path,
):
    rows = _training_rows()
    generator = torch.Generator().manual_seed(99)
    features = torch.randn(12, 7, generator=generator)
    config = core.TrainingConfig(
        max_epochs=5,
        min_epochs=2,
        early_stop_patience=2,
        init_seeds=(7, 8),
    )
    kwargs = {
        "features": features,
        "targets": rows,
        "train_orders": list(range(7)),
        "validation_orders": [7, 8],
        "test_orders": [9, 10, 11],
        "loss_name": "hybrid",
        "config": config,
    }
    before = core.train_fold_ensemble(**kwargs)
    # A Validation4-like file is outside the API and cannot affect training.
    validation_decoy = tmp_path / "validation4.jsonl"
    validation_decoy.write_bytes(b"first independent exam bytes\n")
    validation_decoy.write_bytes(b"completely different independent exam bytes\n")
    after = core.train_fold_ensemble(**kwargs)
    assert before == after
    assert len(before.init_predictions) == 2
    assert all(len(values) == 3 for values in before.init_predictions)
    expected = torch.tensor(before.init_predictions, dtype=torch.float64).mean(dim=0)
    assert torch.tensor(before.ensemble_prediction, dtype=torch.float64) == pytest.approx(
        expected
    )


def test_formal_config_is_frozen_and_validation4_is_not_a_trainer_input():
    core.TrainingConfig().validate(formal=True)
    with pytest.raises(ValueError, match="exact preregistered"):
        core.TrainingConfig(max_epochs=999).validate(formal=True)

    forbidden_names = {
        "validation_dir",
        "validation4_dir",
        "validation_manifest",
        "validation_records",
    }
    assert forbidden_names.isdisjoint(inspect.signature(core.load_training_inputs).parameters)
    assert forbidden_names.isdisjoint(inspect.signature(core.run_offline_feasibility).parameters)
    assert not any(
        "validation" in action.dest.lower()
        for action in cli.build_parser()._actions
    )
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            [
                "--target-dir",
                "t",
                "--target-manifest-sha256",
                "a" * 64,
                "--target-targets-sha256",
                "b" * 64,
                "--feature-dir",
                "f",
                "--feature-completion-sha256",
                "c" * 64,
                "--output-dir",
                "o",
                "--validation4-dir",
                "forbidden",
            ]
        )


def test_training_restores_process_cpu_thread_count(monkeypatch):
    previous = torch.get_num_threads()
    observed: list[int] = []

    def fake_impl(*args, **kwargs):
        observed.append(torch.get_num_threads())
        return [], {}

    monkeypatch.setattr(core, "_train_oof_predictions_impl", fake_impl)
    config = core.TrainingConfig(
        max_epochs=1,
        min_epochs=1,
        early_stop_patience=1,
        init_seeds=(7,),
        cpu_threads=1,
    )
    assert core.train_oof_predictions(None, {}, config=config, formal=False) == ([], {})
    assert observed == [1]
    assert torch.get_num_threads() == previous


def test_training_restores_threads_even_when_fit_fails(monkeypatch):
    previous = torch.get_num_threads()

    def fail(*args, **kwargs):
        assert torch.get_num_threads() == 1
        raise RuntimeError("synthetic fit failure")

    monkeypatch.setattr(core, "_train_oof_predictions_impl", fail)
    config = core.TrainingConfig(
        max_epochs=1,
        min_epochs=1,
        early_stop_patience=1,
        init_seeds=(7,),
        cpu_threads=1,
    )
    with pytest.raises(RuntimeError, match="synthetic fit failure"):
        core.train_oof_predictions(None, {}, config=config, formal=False)
    assert torch.get_num_threads() == previous
