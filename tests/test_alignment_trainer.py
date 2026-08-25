import pytest
import torch
from torch import nn

from fastwam.alignment import AlignmentTrainer


class TinyAlignedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.5))
        self.alignment_adapter = nn.Linear(2, 2, bias=False)

    def configure_alignment_training(self):
        self.eval()
        self.requires_grad_(False)
        self.alignment_adapter.train()
        self.alignment_adapter.requires_grad_(True)
        return {"alignment_adapter.weight"}


def test_trainer_keeps_base_frozen_and_updates_adapter():
    model = TinyAlignedModel()
    base_before = {
        name: value.detach().clone() for name, value in model.base.state_dict().items()
    }
    adapter_before = model.alignment_adapter.weight.detach().clone()
    trainer = AlignmentTrainer(model, lr=0.1)
    x = torch.ones(1, 2, 2)
    v_self = model.alignment_adapter(x)
    zeros = torch.zeros_like(v_self)
    trainer.step_velocities(zeros, zeros, v_self, zeros)
    assert trainer.global_step == 1
    assert model.base.training is False
    assert all(not parameter.requires_grad for parameter in model.base.parameters())
    assert all(
        torch.equal(value, base_before[name])
        for name, value in model.base.state_dict().items()
    )
    assert not torch.equal(model.alignment_adapter.weight, adapter_before)


def test_trainer_rejects_detached_self_velocity():
    model = TinyAlignedModel()
    trainer = AlignmentTrainer(model)
    zeros = torch.zeros(1, 2, 2)
    with pytest.raises(RuntimeError, match="computation graph"):
        trainer.step_velocities(zeros, zeros, zeros, zeros)


def test_trainer_rejects_optimizer_with_base_parameters():
    model = TinyAlignedModel()
    optimizer = torch.optim.AdamW(model.parameters())
    with pytest.raises(ValueError, match="optimizer parameters"):
        AlignmentTrainer(model, optimizer=optimizer)


def test_trainer_uses_stage3_optimizer_defaults():
    model = TinyAlignedModel()
    trainer = AlignmentTrainer(model)

    assert trainer.optimizer.param_groups[0]["weight_decay"] == pytest.approx(1e-4)
    assert trainer.max_grad_norm == pytest.approx(1.0)
