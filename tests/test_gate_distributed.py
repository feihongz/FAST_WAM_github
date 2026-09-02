from __future__ import annotations

import os
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel


def _training_identity() -> dict[str, Any]:
    return {
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


def _gate():
    from fastwam.models.video_gate import BinaryVideoGate

    return BinaryVideoGate(
        proprio_dim=2,
        context_dim=5,
        cnn_channels=(2, 3, 4),
        context_feature_dim=3,
        proprio_hidden_dim=3,
        proprio_feature_dim=2,
        fusion_hidden_dim=4,
    )


def _batch(
    *,
    labels: tuple[float, ...],
    weights: tuple[float, ...],
    prefix: str,
) -> dict[str, Any]:
    batch_size = len(labels)
    generator = torch.Generator().manual_seed(
        10_000 + sum(ord(character) for character in prefix)
    )
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
            dtype=torch.bool,
        ),
        "proprio": torch.randn(
            batch_size,
            2,
            generator=generator,
        ),
        "label": torch.tensor(labels, dtype=torch.float32),
        "sample_weight": torch.tensor(weights, dtype=torch.float32),
        "sample_id": tuple(
            f"{prefix}-{index}" for index in range(batch_size)
        ),
    }


def _concatenate_batches(
    batches: list[Mapping[str, Any]],
) -> dict[str, Any]:
    tensor_keys = {
        "input_image",
        "context",
        "context_mask",
        "proprio",
        "label",
        "sample_weight",
    }
    combined = {
        key: torch.cat([batch[key] for batch in batches], dim=0)
        for key in tensor_keys
    }
    combined["sample_id"] = tuple(
        sample_id
        for batch in batches
        for sample_id in batch["sample_id"]
    )
    return combined


def _gloo_context(rank: int, world_size: int, init_file: str):
    from fastwam.gating.distributed import DistributedGateContext

    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    return DistributedGateContext(
        rank=rank,
        local_rank=rank,
        world_size=world_size,
        backend="gloo",
    )


def _spawn(worker, *, world_size: int, tmp_path: Path) -> None:
    init_file = tmp_path / f"{worker.__name__}.init"
    mp.spawn(
        worker,
        args=(world_size, str(init_file), str(tmp_path)),
        nprocs=world_size,
        join=True,
    )


class _IndexDataset(torch.utils.data.Dataset):
    def __init__(self, size: int):
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> int:
        return index


def _loader_order(loader) -> list[int]:
    return [int(index) for batch in loader for index in batch]


def _sampler_worker(
    rank: int,
    world_size: int,
    init_file: str,
    output_dir: str,
) -> None:
    from scripts import train_video_gate as train_cli

    context = _gloo_context(rank, world_size, init_file)
    try:
        assert context.is_main is (rank == 0)
        context.assert_same(
            {"world_size": world_size, "purpose": "gate-test"},
            label="shared contract",
        )
        try:
            context.assert_same(
                {"rank_specific_value": rank},
                label="rank-specific contract",
            )
        except RuntimeError as error:
            assert "rank-specific contract" in str(error)
        else:
            raise AssertionError("rank-specific distributed drift was accepted")
        context.barrier()

        training = {
            "seed": 42,
            # This remains the global batch. Each of four ranks must receive 2.
            "batch_size": 8,
            "num_workers": 0,
            "pin_memory": False,
        }
        dataset = _IndexDataset(24)
        first_train, first_val = train_cli._epoch_loaders(
            train_dataset=dataset,
            val_dataset=dataset,
            training=training,
            epoch_index=2,
            distributed_context=context,
        )
        resumed_train, resumed_val = train_cli._epoch_loaders(
            train_dataset=dataset,
            val_dataset=dataset,
            training=training,
            epoch_index=2,
            distributed_context=context,
        )
        next_train, _ = train_cli._epoch_loaders(
            train_dataset=dataset,
            val_dataset=dataset,
            training=training,
            epoch_index=3,
            distributed_context=context,
        )
        assert first_train.batch_size == 2
        assert first_val.batch_size == 2

        errors: dict[str, str] = {}
        for name, invalid_training, invalid_dataset in (
            (
                "global_batch",
                {**training, "batch_size": 6},
                dataset,
            ),
            (
                "dataset_size",
                training,
                _IndexDataset(26),
            ),
        ):
            try:
                train_cli._epoch_loaders(
                    train_dataset=invalid_dataset,
                    val_dataset=invalid_dataset,
                    training=invalid_training,
                    epoch_index=2,
                    distributed_context=context,
                )
            except ValueError as error:
                errors[name] = str(error)
            else:
                raise AssertionError(f"{name} must fail closed")

        torch.save(
            {
                "rank": rank,
                "first_train": _loader_order(first_train),
                "first_val": _loader_order(first_val),
                "resumed_train": _loader_order(resumed_train),
                "resumed_val": _loader_order(resumed_val),
                "next_train": _loader_order(next_train),
                "errors": errors,
            },
            Path(output_dir) / f"sampler-rank-{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


def test_four_rank_sampler_is_exact_disjoint_and_resume_stable(tmp_path):
    _spawn(_sampler_worker, world_size=4, tmp_path=tmp_path)
    records = [
        torch.load(
            tmp_path / f"sampler-rank-{rank}.pt",
            map_location="cpu",
            weights_only=False,
        )
        for rank in range(4)
    ]

    train_shards = [record["first_train"] for record in records]
    val_shards = [record["first_val"] for record in records]
    assert all(len(shard) == 6 for shard in train_shards)
    assert all(len(shard) == 6 for shard in val_shards)
    assert sorted(index for shard in train_shards for index in shard) == list(
        range(24)
    )
    assert sorted(index for shard in val_shards for index in shard) == list(
        range(24)
    )
    for left_rank, left in enumerate(train_shards):
        for right in train_shards[left_rank + 1 :]:
            assert set(left).isdisjoint(right)
    for left_rank, left in enumerate(val_shards):
        for right in val_shards[left_rank + 1 :]:
            assert set(left).isdisjoint(right)

    assert all(
        record["first_train"] == record["resumed_train"]
        and record["first_val"] == record["resumed_val"]
        for record in records
    )
    assert any(
        record["first_train"] != record["next_train"]
        for record in records
    )
    assert all(
        set(record["errors"]) == {"global_batch", "dataset_size"}
        for record in records
    )


def _assert_epoch_results_close(actual, expected) -> None:
    assert actual.num_batches == expected.num_batches
    assert actual.objective_bce == pytest.approx(
        expected.objective_bce,
        rel=1e-6,
        abs=1e-7,
    )
    actual_metrics = actual.metrics.to_dict()
    expected_metrics = expected.metrics.to_dict()
    assert set(actual_metrics) == set(expected_metrics)
    for name, expected_value in expected_metrics.items():
        actual_value = actual_metrics[name]
        if expected_value is None:
            assert actual_value is None
        elif isinstance(expected_value, int):
            assert actual_value == expected_value
        else:
            assert actual_value == pytest.approx(
                expected_value,
                rel=1e-6,
                abs=1e-7,
            )


def _trainer_worker(
    rank: int,
    world_size: int,
    init_file: str,
    output_dir: str,
) -> None:
    import fastwam.gating.trainer as trainer_module
    from fastwam.gating.trainer import GateTrainer

    context = _gloo_context(rank, world_size, init_file)
    try:
        torch.manual_seed(123)
        gate = _gate()
        initial_state = {
            name: value.detach().clone()
            for name, value in gate.state_dict().items()
        }
        forward_gate = DistributedDataParallel(gate)
        trainer = GateTrainer(
            gate,
            train_labels=[0, 1, 0, 1],
            training_identity=_training_identity(),
            lr=1.0e-2,
            weight_decay=0.0,
            forward_gate=forward_gate,
            distributed_context=context,
        )
        rank_batches = [
            _batch(
                labels=(1.0, 0.0),
                weights=(1.0, 9.0),
                prefix="rank-zero",
            ),
            _batch(
                labels=(0.0, 1.0),
                weights=(7.0, 1.0),
                prefix="rank-one",
            ),
        ]
        distributed_loss = trainer.train_batch(rank_batches[rank])

        reference_gate = _gate()
        reference_gate.load_state_dict(initial_state, strict=True)
        reference = GateTrainer(
            reference_gate,
            train_labels=[0, 1, 0, 1],
            training_identity=_training_identity(),
            lr=1.0e-2,
            weight_decay=0.0,
        )
        reference_loss = reference.train_batch(
            _concatenate_batches(rank_batches)
        )
        assert distributed_loss == pytest.approx(
            reference_loss,
            rel=1e-6,
            abs=1e-7,
        )
        for name, parameter in gate.state_dict().items():
            torch.testing.assert_close(
                parameter,
                reference_gate.state_dict()[name],
                rtol=1e-5,
                atol=1e-6,
            )

        # Validation must aggregate variable local lengths and return one
        # identical global metric record on every rank.
        metric_batches = [
            _batch(
                labels=(1.0,),
                weights=(11.0,),
                prefix="metric-zero",
            ),
            _batch(
                labels=(0.0, 1.0, 0.0),
                weights=(1.0, 2.0, 5.0),
                prefix="metric-one",
            ),
        ]
        distributed_metrics = trainer.evaluate_epoch([metric_batches[rank]])
        reference_metrics = reference.evaluate_epoch(
            [_concatenate_batches(metric_batches)]
        )
        _assert_epoch_results_close(distributed_metrics, reference_metrics)
        assert distributed_metrics.metrics.num_examples == 4
        assert distributed_metrics.num_batches == 1

        # Early stopping is a single global decision, not four independent
        # decisions based on rank-local validation shards.
        fit = trainer.fit(
            [metric_batches[rank]],
            [metric_batches[rank]],
            num_epochs=2,
            early_stop_patience=1,
            min_delta=1.0e6,
        )
        decisions: list[Any] = [None] * world_size
        dist.all_gather_object(
            decisions,
            (
                fit.stopped_early,
                fit.best_epoch,
                fit.best_val_bce,
                trainer.epoch,
                trainer.global_step,
                trainer.epochs_without_improvement,
            ),
        )
        assert all(decision == decisions[0] for decision in decisions)
        assert fit.stopped_early
        assert trainer.epochs_without_improvement == 1

        # A distributed training state has exactly one physical writer, while
        # retaining each rank's RNG stream for strict epoch-boundary resume.
        random.seed(1000 + rank)
        np.random.seed(2000 + rank)
        torch.manual_seed(3000 + rank)
        state_path = Path(output_dir) / "distributed-training-state.pt"
        original_torch_save = trainer_module.torch.save
        save_calls = 0

        temporary_state_path = state_path.with_name(f".{state_path.name}.tmp")

        def counted_save(value, destination, *args, **kwargs):
            nonlocal save_calls
            if isinstance(destination, (str, os.PathLike)) and Path(
                destination
            ) == temporary_state_path:
                save_calls += 1
            return original_torch_save(value, destination, *args, **kwargs)

        trainer_module.torch.save = counted_save
        try:
            trainer.save_training_state(state_path)
        finally:
            trainer_module.torch.save = original_torch_save
        context.barrier()

        save_counts: list[Any] = [None] * world_size
        dist.all_gather_object(save_counts, save_calls)
        assert save_counts == [1, 0]
        assert state_path.is_file()

        expected_python = random.random()
        expected_numpy = float(np.random.rand())
        expected_torch = torch.rand(3)
        random.seed(91)
        np.random.seed(92)
        torch.manual_seed(93)
        trainer.load_training_state(state_path)
        assert random.random() == expected_python
        assert float(np.random.rand()) == expected_numpy
        assert torch.equal(torch.rand(3), expected_torch)

        rng_fingerprints: list[Any] = [None] * world_size
        dist.all_gather_object(
            rng_fingerprints,
            (expected_python, expected_numpy, expected_torch.tolist()),
        )
        assert len({repr(value) for value in rng_fingerprints}) == world_size
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_two_rank_training_matches_global_weighted_objective_and_resumes_rng(
    tmp_path,
):
    _spawn(_trainer_worker, world_size=2, tmp_path=tmp_path)
