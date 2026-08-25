from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from accelerate import Accelerator
from safetensors.torch import load_file
import pytest
import torch
from torch import nn

from fastwam.alignment.checkpointing import (
    BaseCheckpointIdentity,
    GitIdentity,
)
from fastwam.alignment.formal_trainer import Stage3AlignmentTrainer
from fastwam.alignment.losses import Stage3LossOutput
from fastwam.models.wan22.video_action_alignment import (
    VideoActionResidualAdapter,
    load_alignment_checkpoint,
)


DATA_MANIFEST_SHA256 = "d" * 64


class _TinyDataset:
    def __init__(self, length: int = 4):
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"index": torch.tensor(index, dtype=torch.long)}


class _TinyAlignedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(3, 3)
        self.alignment_adapter = VideoActionResidualAdapter(
            action_hidden_dim=2,
            video_hidden_dim=2,
            action_dim=2,
            bottleneck_dim=2,
            num_heads=1,
            ffn_multiplier=1,
        )

    def configure_alignment_training(self) -> set[str]:
        self.eval()
        self.requires_grad_(False)
        self.alignment_adapter.train()
        self.alignment_adapter.requires_grad_(True)
        return {
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }


def _config(
    output_dir: Path,
    *,
    accumulation: int = 1,
    max_steps: int = 2,
) -> dict:
    return {
        "output_dir": str(output_dir),
        "model": {"name": "tiny-aligned"},
        "data": {"name": "tiny-dataset"},
        "runtime": {"log_every": 0},
        "training": {
            "batch_size": 1,
            "num_workers": 0,
            "drop_last": True,
            "gradient_accumulation_steps": accumulation,
            "learning_rate": 1.0e-3,
            "weight_decay": 1.0e-4,
            "max_grad_norm": 1.0,
            "num_epochs": 2,
            "max_steps": max_steps,
            "seed": 17,
            "betas": [0.9, 0.999],
            "warmup_ratio": 0.0,
            "lr_scheduler_type": "constant",
        },
        "stage3": {
            "num_solver_steps": 10,
            "sigma_shift": None,
            "helpful_relative_margin": 0.05,
            "lambda_action": 1.0,
            "lambda_align": 1.0,
            "lambda_safe": 0.5,
        },
        "checkpoint": {
            "save_every": 0,
            "keep_last": 2,
            "save_final": False,
            "resume": None,
            "strict_resume": True,
        },
    }


def _make_trainer(
    tmp_path: Path,
    *,
    accumulation: int = 1,
    max_steps: int = 2,
    dataset_length: int = 4,
    batch_size: int = 1,
    resume: Path | None = None,
    data_manifest_sha256: str | None = DATA_MANIFEST_SHA256,
) -> tuple[Stage3AlignmentTrainer, _TinyAlignedModel, Accelerator]:
    model = _TinyAlignedModel()
    accelerator = Accelerator(
        cpu=True,
        gradient_accumulation_steps=accumulation,
        step_scheduler_with_optimizer=False,
    )
    config = _config(
        tmp_path / f"accum-{accumulation}",
        accumulation=accumulation,
        max_steps=max_steps,
    )
    config["training"]["batch_size"] = batch_size
    config["checkpoint"]["resume"] = None if resume is None else str(resume)
    trainer = Stage3AlignmentTrainer(
        accelerator=accelerator,
        model=model,
        train_dataset=_TinyDataset(length=dataset_length),
        config=config,
        base_identity=BaseCheckpointIdentity(
            path="/frozen/base-checkpoint.pt",
            sha256="a" * 64,
            size_bytes=5_000_000_000,
        ),
        git_identity=GitIdentity(commit="deadbeef", tracked_dirty=False),
        data_identity=(
            {}
            if data_manifest_sha256 is None
            else {"sha256": data_manifest_sha256}
        ),
    )
    return trainer, model, accelerator


def _adapter_state(model: _TinyAlignedModel) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in model.alignment_adapter.state_dict().items()
    }


def _assert_state_equal(
    actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]
) -> None:
    assert actual.keys() == expected.keys()
    for name in expected:
        assert torch.equal(actual[name], expected[name]), name


def _assert_nested_equal(actual, expected) -> None:
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_value, expected_value in zip(actual, expected, strict=True):
            _assert_nested_equal(actual_value, expected_value)
    else:
        assert actual == expected


def _fake_two_rank_trainer() -> Stage3AlignmentTrainer:
    trainer = Stage3AlignmentTrainer.__new__(Stage3AlignmentTrainer)
    trainer.accelerator = SimpleNamespace(num_processes=2)
    return trainer


def _checkpoint_identity() -> dict[str, int | str]:
    return {
        "manifest_sha256": "a" * 64,
        "global_step": 7,
        "epoch": 2,
        "batch_in_epoch": 3,
        "scheduler_last_epoch": 7,
    }


def test_require_all_rank_identical_accepts_matching_fake_two_rank_identity(
    monkeypatch,
):
    trainer = _fake_two_rank_trainer()
    identity = _checkpoint_identity()

    def fake_gather(payload):
        assert payload == [identity]
        return [dict(identity), dict(identity)]

    monkeypatch.setattr(
        "fastwam.alignment.formal_trainer.gather_object",
        fake_gather,
    )

    assert trainer._require_all_rank_identical(
        identity,
        phase="resume checkpoint",
    ) == identity


@pytest.mark.parametrize(
    "remote_override",
    [
        {"manifest_sha256": "b" * 64},
        {"batch_in_epoch": 4},
    ],
    ids=["manifest", "cursor"],
)
def test_require_all_rank_identical_rejects_fake_two_rank_identity_mismatch(
    monkeypatch,
    remote_override,
):
    trainer = _fake_two_rank_trainer()
    local_identity = _checkpoint_identity()
    remote_identity = {**local_identity, **remote_override}
    monkeypatch.setattr(
        "fastwam.alignment.formal_trainer.gather_object",
        lambda _: [local_identity, remote_identity],
    )

    with pytest.raises(
        RuntimeError,
        match="resume checkpoint identity differs across ranks",
    ):
        trainer._require_all_rank_identical(
            local_identity,
            phase="resume checkpoint",
        )


def test_accelerator_and_optimizer_own_only_adapter(tmp_path):
    trainer, model, accelerator = _make_trainer(tmp_path)

    assert len(accelerator._models) == 1
    owned = accelerator.unwrap_model(accelerator._models[0])
    assert owned is accelerator.unwrap_model(trainer.adapter_module)
    assert set(dict(owned.named_parameters())) == {
        f"adapter.{name}"
        for name, _ in model.alignment_adapter.named_parameters()
    }

    adapter_ids = {id(parameter) for parameter in trainer.adapter_module.parameters()}
    optimizer_ids = {
        id(parameter)
        for group in trainer.optimizer.param_groups
        for parameter in group["params"]
    }
    base_ids = {id(parameter) for parameter in model.base.parameters()}
    assert optimizer_ids == adapter_ids
    assert optimizer_ids.isdisjoint(base_ids)
    assert all(not parameter.requires_grad for parameter in model.base.parameters())


def test_formal_trainer_requires_data_manifest_sha256(tmp_path):
    with pytest.raises(ValueError, match="data_identity.sha256"):
        _make_trainer(tmp_path, data_manifest_sha256=None)


def test_checkpoint_and_export_contain_adapter_but_no_base_weights(tmp_path):
    trainer, model, _ = _make_trainer(tmp_path)
    trainer.global_step = 0
    trainer.epoch = 1
    trainer.batch_in_epoch = 2

    state_dir = trainer.save_checkpoint()
    export_path = trainer.exports_dir / "step_000000.pt"
    payload = torch.load(export_path, map_location="cpu", weights_only=False)
    accelerator_state = load_file(state_dir / "accelerator" / "model.safetensors")

    expected_adapter_keys = set(model.alignment_adapter.state_dict())
    assert set(payload["adapter"]) == expected_adapter_keys
    assert set(accelerator_state) == {
        f"adapter.{name}" for name in expected_adapter_keys
    }
    assert not any(name.startswith("base.") for name in accelerator_state)
    assert not any(path.name == "base-checkpoint.pt" for path in state_dir.rglob("*"))

    manifest = json.loads((state_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["base_checkpoint"] == "/frozen/base-checkpoint.pt"
    assert manifest["base_checkpoint_size_bytes"] == 5_000_000_000
    assert manifest["data_manifest_sha256"] == DATA_MANIFEST_SHA256
    assert payload["schema_version"] == 2
    assert payload["data_manifest_sha256"] == DATA_MANIFEST_SHA256
    assert all("base-checkpoint.pt" not in name for name in manifest["files"])

    restored = _TinyAlignedModel().alignment_adapter
    load_alignment_checkpoint(
        export_path,
        restored,
        expected_base_checkpoint_sha256="a" * 64,
        expected_data_manifest_sha256=DATA_MANIFEST_SHA256,
    )
    _assert_state_equal(restored.state_dict(), model.alignment_adapter.state_dict())


def test_strict_resume_restores_adapter_optimizer_position_and_step(tmp_path):
    trainer, model, accelerator = _make_trainer(tmp_path)

    base_velocity = torch.zeros(1, 2, 2)
    action_tokens = torch.ones(1, 2, 2)
    video_tokens = torch.ones(1, 4, 2)
    output = trainer.adapter_module(
        base_velocity,
        action_tokens=action_tokens,
        video_tokens=video_tokens,
        video_meta={"tokens_per_frame": 2},
    )
    accelerator.backward(output.sum())
    trainer.optimizer.step()
    trainer.scheduler.step()
    trainer.optimizer.zero_grad(set_to_none=True)

    trainer.global_step = 1
    trainer.epoch = 2
    trainer.batch_in_epoch = 1
    expected_adapter = _adapter_state(model)
    expected_optimizer_step = max(
        int(state["step"])
        for state in trainer.optimizer.state.values()
        if "step" in state
    )
    state_dir = trainer.save_checkpoint()

    with torch.no_grad():
        for parameter in model.alignment_adapter.parameters():
            parameter.add_(10.0)
    trainer.global_step = 0
    trainer.epoch = 0
    trainer.batch_in_epoch = 0

    trainer.resume_strict(state_dir)

    _assert_state_equal(model.alignment_adapter.state_dict(), expected_adapter)
    assert trainer.global_step == 1
    assert trainer.epoch == 2
    assert trainer.batch_in_epoch == 1
    assert trainer.train_sampler.epoch_offset == 2
    assert trainer.train_sampler.resume_batch_offset == 1
    resumed_optimizer_step = max(
        int(state["step"])
        for state in trainer.optimizer.state.values()
        if "step" in state
    )
    assert resumed_optimizer_step == expected_optimizer_step


def test_train_accumulates_two_microbatches_per_optimizer_step(tmp_path):
    trainer, model, _ = _make_trainer(
        tmp_path,
        accumulation=2,
        max_steps=2,
    )
    before = _adapter_state(model)
    microbatches: list[int] = []

    def fake_build_loss(sample, *, k=None):
        del k
        microbatches.extend(int(value) for value in sample["index"].reshape(-1))
        parameter = next(trainer.adapter_module.parameters())
        loss = parameter.float().sum()
        detached = loss.detach().reshape(())
        per_sample = detached.reshape(1)
        return Stage3LossOutput(
            loss=loss,
            action_loss=detached,
            alignment_loss=detached,
            safe_loss=detached,
            helpful_fraction=detached,
            e0=per_sample,
            egt=per_sample,
            eself=per_sample,
        )

    trainer.build_loss = fake_build_loss
    trainer.train()

    assert trainer.global_step == 2
    assert len(microbatches) == 4
    assert trainer.scheduler.scheduler.last_epoch == 2
    assert any(
        not torch.equal(value, before[name])
        for name, value in model.alignment_adapter.state_dict().items()
    )


def test_mid_epoch_resume_matches_uninterrupted_rng_and_updates(tmp_path):
    def make(root: Path, *, resume: Path | None = None):
        torch.manual_seed(991)
        model = _TinyAlignedModel()
        config = _config(root, max_steps=3)
        config["checkpoint"]["resume"] = None if resume is None else str(resume)
        accelerator = Accelerator(
            cpu=True,
            gradient_accumulation_steps=1,
            step_scheduler_with_optimizer=False,
        )
        trainer = Stage3AlignmentTrainer(
            accelerator=accelerator,
            model=model,
            train_dataset=_TinyDataset(length=4),
            config=config,
            base_identity=BaseCheckpointIdentity(
                path="/frozen/base-checkpoint.pt",
                sha256="a" * 64,
                size_bytes=5_000_000_000,
            ),
            git_identity=GitIdentity(commit="deadbeef", tracked_dirty=False),
            data_identity={"sha256": DATA_MANIFEST_SHA256},
        )
        return trainer, model

    def attach_random_loss(trainer, records):
        def fake_build_loss(sample, *, k=None):
            del k
            sampled_k = int(torch.randint(10, (1,)).item())
            sampled_noise = float(torch.randn(()).item())
            sample_index = int(sample["index"].reshape(-1)[0])
            records.append((sample_index, sampled_k, sampled_noise))
            parameter = next(trainer.adapter_module.parameters())
            scale = 1.0 + 0.01 * sampled_k + 0.001 * sampled_noise
            loss = parameter.float().square().mean() * scale
            detached = loss.detach().reshape(())
            per_sample = detached.reshape(1)
            return Stage3LossOutput(
                loss=loss,
                action_loss=detached,
                alignment_loss=detached,
                safe_loss=detached,
                helpful_fraction=detached,
                e0=per_sample,
                egt=per_sample,
                eself=per_sample,
            )

        trainer.build_loss = fake_build_loss

    full_records = []
    full_trainer, full_model = make(tmp_path / "full")
    attach_random_loss(full_trainer, full_records)
    full_trainer.train()

    split_records = []
    split_root = tmp_path / "split"
    first_trainer, _ = make(split_root)
    attach_random_loss(first_trainer, split_records)
    first_trainer.max_steps = 1
    first_trainer.train()
    first_trainer.max_steps = 3
    state_dir = first_trainer.save_checkpoint()

    resumed_trainer, resumed_model = make(split_root, resume=state_dir)
    attach_random_loss(resumed_trainer, split_records)
    resumed_trainer.train()

    assert split_records == full_records
    assert resumed_trainer.global_step == full_trainer.global_step == 3
    assert resumed_trainer.scheduler.scheduler.last_epoch == 3
    _assert_state_equal(
        resumed_model.alignment_adapter.state_dict(),
        full_model.alignment_adapter.state_dict(),
    )
    _assert_nested_equal(
        resumed_trainer.optimizer.state_dict(),
        full_trainer.optimizer.state_dict(),
    )


def test_epoch_tail_checkpoint_resumes_next_epoch_exactly(tmp_path):
    def attach_random_loss(trainer, records):
        def fake_build_loss(sample, *, k=None):
            del k
            sample_index = int(sample["index"].reshape(-1)[0])
            sampled_k = int(torch.randint(10, (1,)).item())
            sampled_noise = float(torch.randn(()).item())
            records.append((sample_index, sampled_k, sampled_noise))
            parameter = next(trainer.adapter_module.parameters())
            scale = 1.0 + 0.01 * sampled_k + 0.001 * sampled_noise
            loss = parameter.float().square().mean() * scale
            detached = loss.detach().reshape(())
            per_sample = detached.reshape(1)
            return Stage3LossOutput(
                loss=loss,
                action_loss=detached,
                alignment_loss=detached,
                safe_loss=detached,
                helpful_fraction=detached,
                e0=per_sample,
                egt=per_sample,
                eself=per_sample,
            )

        trainer.build_loss = fake_build_loss

    torch.manual_seed(1441)
    full_trainer, full_model, _ = _make_trainer(
        tmp_path / "epoch-tail-full",
        max_steps=6,
        dataset_length=4,
    )
    full_records = []
    attach_random_loss(full_trainer, full_records)
    full_trainer.train()

    torch.manual_seed(1441)
    first_trainer, _, _ = _make_trainer(
        tmp_path / "epoch-tail-split",
        max_steps=6,
        dataset_length=4,
    )
    split_records = []
    attach_random_loss(first_trainer, split_records)
    first_trainer.max_steps = 4
    first_trainer.train()
    assert first_trainer.epoch == 0
    assert first_trainer.batch_in_epoch == 4
    first_trainer.max_steps = 6
    state_dir = first_trainer.save_checkpoint()

    torch.manual_seed(1441)
    resumed_trainer, resumed_model, _ = _make_trainer(
        tmp_path / "epoch-tail-split",
        max_steps=6,
        dataset_length=4,
        resume=state_dir,
    )
    attach_random_loss(resumed_trainer, split_records)
    resumed_trainer.train()

    assert split_records == full_records
    assert resumed_trainer.global_step == full_trainer.global_step == 6
    assert resumed_trainer.epoch == full_trainer.epoch == 1
    assert resumed_trainer.batch_in_epoch == full_trainer.batch_in_epoch == 2
    _assert_state_equal(
        resumed_model.alignment_adapter.state_dict(),
        full_model.alignment_adapter.state_dict(),
    )
    _assert_nested_equal(
        resumed_trainer.optimizer.state_dict(),
        full_trainer.optimizer.state_dict(),
    )


def test_explicit_max_steps_rejects_dataset_smaller_than_global_batch(tmp_path):
    with pytest.raises(ValueError, match="drop_last|no batches|global batch"):
        _make_trainer(
            tmp_path,
            max_steps=1,
            dataset_length=1,
            batch_size=2,
        )


def test_formal_v1_rejects_partial_accumulation_group_per_epoch(tmp_path):
    with pytest.raises(
        ValueError,
        match="accumulation|divisible|complete.*group",
    ):
        _make_trainer(
            tmp_path,
            accumulation=2,
            max_steps=2,
            dataset_length=3,
            batch_size=1,
        )
