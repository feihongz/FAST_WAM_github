from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from fastwam.alignment import AlignmentTrainer
from fastwam.alignment.rollout import (
    PreparedStage3Batch,
    SolverPanel,
    Stage3VelocityPanel,
    build_solver_panel,
    compute_stage3_velocity_panel,
    prepare_stage3_batch,
    rollout_self_video,
    validate_video_only_joint_equivalence,
)
from fastwam.models.wan22.schedulers.scheduler_continuous import (
    WanContinuousFlowMatchScheduler,
)


class _RecordingVideoExpert(nn.Module):
    def __init__(self, *, action_conditioned: bool = False):
        super().__init__()
        self.action_conditioned = action_conditioned
        self.seperated_timestep = True
        self.fuse_vae_embedding_in_latents = True
        self.video_attention_mask_mode = "first_frame_causal"
        self.calls: list[dict[str, object]] = []

    def forward(self, **kwargs):
        self.calls.append(
            {
                "x": kwargs["x"].detach().clone(),
                "timestep": kwargs["timestep"].detach().clone(),
                "action": kwargs.get("action"),
                "grad_enabled": torch.is_grad_enabled(),
                "training": self.training,
            }
        )
        # A nonzero velocity deliberately moves the first frame too. The
        # rollout helper must restore that frame after every solver update.
        return torch.full_like(kwargs["x"], 2.0)


class _TinyAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.25))

    def forward(self, action_tokens: torch.Tensor) -> torch.Tensor:
        return self.scale * action_tokens


class _MockAlignedModel(nn.Module):
    def __init__(self, *, action_conditioned: bool = False):
        super().__init__()
        self.device = torch.device("cpu")
        self.torch_dtype = torch.float32
        self.video_expert = _RecordingVideoExpert(
            action_conditioned=action_conditioned
        )
        self.alignment_adapter = _TinyAdapter()
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=100,
            shift=1.0,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=200,
            shift=1.0,
        )
        self.train_action_scheduler = self.infer_action_scheduler
        self._unified_inference_mode = "wo"
        self.action_calls: list[dict[str, object]] = []
        self.joint_calls: list[dict[str, object]] = []
        self.hook_calls: list[dict[str, object]] = []
        self.raise_in_hook = False

        self.input_latents = torch.tensor(
            [
                [[[[3.0]], [[4.0]], [[5.0]]]],
                [[[[6.0]], [[7.0]], [[8.0]]]],
            ],
            requires_grad=True,
        )
        self.first_frame_latents = self.input_latents[:, :, :1].detach().clone()
        self.action = torch.tensor(
            [[[1.0], [2.0]], [[3.0], [4.0]]],
            requires_grad=True,
        )
        self.context = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
        self.context_mask = torch.ones(2, 2, dtype=torch.bool)
        self.action_is_pad = torch.tensor(
            [[False, False], [False, True]], dtype=torch.bool
        )

    def build_inputs(self, sample, tiled: bool = False):
        del sample, tiled
        return {
            "context": self.context,
            "context_mask": self.context_mask,
            "input_latents": self.input_latents,
            "first_frame_latents": self.first_frame_latents,
            "fuse_vae_embedding_in_latents": True,
            "action": self.action,
            "action_is_pad": self.action_is_pad,
            "image_is_pad": None,
        }

    def configure_alignment_training(self):
        self.eval()
        self.requires_grad_(False)
        self.alignment_adapter.train()
        self.alignment_adapter.requires_grad_(True)
        return {
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }

    def _predict_wo_action_noise(self, **kwargs):
        self.action_calls.append(
            {
                **kwargs,
                "mode": self._unified_inference_mode,
                "grad_enabled": torch.is_grad_enabled(),
                "training": self.training,
            }
        )
        return kwargs["latents_action"] + 1.0

    def _predict_joint_base(self, **kwargs):
        call_number = len(self.joint_calls)
        self.joint_calls.append(
            {
                **kwargs,
                "mode": self._unified_inference_mode,
                "grad_enabled": torch.is_grad_enabled(),
                "training": self.training,
            }
        )
        action_tokens = kwargs["latents_action"] + float(call_number + 2)
        video_tokens = kwargs["latents_video"].flatten(2).transpose(1, 2)
        return SimpleNamespace(
            video_velocity=torch.full_like(kwargs["latents_video"], 2.0),
            action_velocity=kwargs["latents_action"] + float(call_number + 3),
            action_tokens=action_tokens,
            video_tokens=video_tokens,
            action_pre={"branch": call_number},
            video_pre={"meta": {"tokens_per_frame": 1}},
        )

    def _apply_action_velocity_hook(
        self,
        base_action_velocity: torch.Tensor,
        *,
        action_tokens: torch.Tensor,
        video_tokens: torch.Tensor,
        action_pre,
        video_pre,
    ) -> torch.Tensor:
        self.hook_calls.append(
            {
                "base_action_velocity": base_action_velocity,
                "action_tokens": action_tokens,
                "video_tokens": video_tokens,
                "action_pre": action_pre,
                "video_pre": video_pre,
                "mode": self._unified_inference_mode,
                "grad_enabled": torch.is_grad_enabled(),
            }
        )
        if self.raise_in_hook:
            raise RuntimeError("adapter failure")
        pooled_video = video_tokens.detach().mean(dim=1, keepdim=True)
        return base_action_velocity.detach() + self.alignment_adapter(
            action_tokens.detach() + pooled_video
        )


def _video_noise() -> torch.Tensor:
    return torch.tensor(
        [
            [[[[10.0]], [[11.0]], [[12.0]]]],
            [[[[13.0]], [[14.0]], [[15.0]]]],
        ]
    )


def _action_noise() -> torch.Tensor:
    return torch.tensor([[[9.0], [8.0]], [[7.0], [6.0]]])


def _solver_panel(model: _MockAlignedModel) -> SolverPanel:
    return build_solver_panel(
        model,
        num_solver_steps=10,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_solver_panel_has_paired_ten_step_schedules():
    model = _MockAlignedModel()
    panel = _solver_panel(model)

    assert isinstance(panel, SolverPanel)
    assert panel.video_timesteps.shape == (10,)
    assert panel.video_deltas.shape == (10,)
    assert panel.action_timesteps.shape == (10,)
    assert panel.action_deltas.shape == (10,)
    torch.testing.assert_close(
        panel.video_sigmas,
        panel.video_timesteps / model.infer_video_scheduler.num_train_timesteps,
    )
    torch.testing.assert_close(
        panel.action_sigmas,
        panel.action_timesteps / model.infer_action_scheduler.num_train_timesteps,
    )


@pytest.mark.parametrize(("k", "expected_calls"), [(0, 0), (9, 9)])
def test_self_rollout_k_is_number_of_completed_updates_and_locks_first_frame(
    k: int,
    expected_calls: int,
):
    model = _MockAlignedModel()
    result = rollout_self_video(
        model,
        video_noise=_video_noise(),
        first_frame_latents=model.first_frame_latents,
        context=model.context,
        context_mask=model.context_mask,
        fuse_vae_embedding_in_latents=True,
        panel=_solver_panel(model),
        k=k,
    )

    assert len(model.video_expert.calls) == expected_calls
    for index, call in enumerate(model.video_expert.calls):
        torch.testing.assert_close(
            call["timestep"],
            _solver_panel(model).video_timesteps[index].expand(2),
        )
        torch.testing.assert_close(
            call["x"][:, :, :1], model.first_frame_latents
        )
        assert call["action"] is None
        assert call["grad_enabled"] is False
    torch.testing.assert_close(result[:, :, :1], model.first_frame_latents)
    assert result.requires_grad is False
    assert result.is_inference() is False


def test_prepare_batch_uses_flow_interpolation_and_one_paired_solver_index():
    model = _MockAlignedModel()
    video_noise = _video_noise()
    action_noise = _action_noise()
    k = 4
    prepared = prepare_stage3_batch(
        model,
        {"sample_id": "mock"},
        k=k,
        video_noise=video_noise,
        action_noise=action_noise,
        num_solver_steps=10,
    )
    panel = _solver_panel(model)

    assert isinstance(prepared, PreparedStage3Batch)
    assert prepared.k == k
    torch.testing.assert_close(
        prepared.video_timestep, panel.video_timesteps[k].expand(2)
    )
    torch.testing.assert_close(
        prepared.action_timestep, panel.action_timesteps[k].expand(2)
    )
    torch.testing.assert_close(prepared.video_sigma, panel.video_sigmas[k])
    torch.testing.assert_close(prepared.action_sigma, panel.action_sigmas[k])

    expected_gt = (
        (1.0 - panel.video_sigmas[k]) * model.input_latents.detach()
        + panel.video_sigmas[k] * video_noise
    )
    expected_gt[:, :, :1] = model.first_frame_latents
    torch.testing.assert_close(prepared.z_gt_k, expected_gt)

    expected_noisy_action = (
        (1.0 - panel.action_sigmas[k]) * model.action.detach()
        + panel.action_sigmas[k] * action_noise
    )
    torch.testing.assert_close(prepared.noisy_action, expected_noisy_action)
    torch.testing.assert_close(
        prepared.action_target, action_noise - model.action.detach()
    )
    assert prepared.z_gt_k.requires_grad is False
    assert prepared.z_self_k.requires_grad is False
    assert prepared.noisy_action.requires_grad is False
    assert prepared.action_target.requires_grad is False
    assert prepared.z_self_k.is_inference() is False


def test_velocity_panel_shares_action_state_and_keeps_only_adapter_graph():
    model = _MockAlignedModel()
    prepared = prepare_stage3_batch(
        model,
        {"sample_id": "mock"},
        k=7,
        video_noise=_video_noise(),
        action_noise=_action_noise(),
    )
    model.video_expert.calls.clear()

    velocity = compute_stage3_velocity_panel(model, prepared)

    assert isinstance(velocity, Stage3VelocityPanel)
    assert len(model.action_calls) == 1
    assert len(model.joint_calls) == 2
    assert len(model.hook_calls) == 1
    assert model.action_calls[0]["mode"] == "wo"
    assert [call["mode"] for call in model.joint_calls] == ["w", "w"]
    assert model.hook_calls[0]["mode"] == "w"
    assert model._unified_inference_mode == "wo"

    action_objects = [
        model.action_calls[0]["latents_action"],
        model.joint_calls[0]["latents_action"],
        model.joint_calls[1]["latents_action"],
    ]
    assert all(value is prepared.noisy_action for value in action_objects)
    assert all(
        call["timestep_action"] is prepared.action_timestep
        for call in model.joint_calls
    )
    assert model.action_calls[0]["timestep_action"] is prepared.action_timestep
    assert all(
        call["timestep_video"] is prepared.video_timestep
        for call in model.joint_calls
    )
    assert model.joint_calls[0]["latents_video"] is prepared.z_gt_k
    assert model.joint_calls[1]["latents_video"] is prepared.z_self_k
    torch.testing.assert_close(
        model.hook_calls[0]["video_tokens"],
        prepared.z_self_k.flatten(2).transpose(1, 2),
    )

    assert velocity.v0.requires_grad is False
    assert velocity.v_gt.requires_grad is False
    assert velocity.v_target.requires_grad is False
    assert velocity.v_self.requires_grad is True
    assert velocity.v_target is prepared.action_target
    assert velocity.action_is_pad is prepared.action_is_pad
    assert velocity.k == prepared.k
    torch.testing.assert_close(velocity.video_sigma, prepared.video_sigma)
    torch.testing.assert_close(velocity.action_sigma, prepared.action_sigma)

    velocity.v_self.sum().backward()
    assert model.alignment_adapter.scale.grad is not None


def test_alignment_trainer_updates_adapter_from_raw_sample():
    model = _MockAlignedModel()
    trainer = AlignmentTrainer(model, lr=0.1)
    before = model.alignment_adapter.scale.detach().clone()
    model.train()

    losses = trainer.step(
        {"sample_id": "mock"},
        k=4,
        video_noise=_video_noise(),
        action_noise=_action_noise(),
    )

    assert torch.isfinite(losses.loss)
    assert trainer.global_step == 1
    assert trainer.last_k == 4
    assert not torch.equal(model.alignment_adapter.scale.detach(), before)
    assert model.input_latents.grad is None
    assert model.action.grad is None
    assert losses.loss.requires_grad is False
    assert model.training is False
    assert model.alignment_adapter.training is True
    assert all(call["training"] is False for call in model.video_expert.calls)
    assert all(call["training"] is False for call in model.action_calls)
    assert all(call["training"] is False for call in model.joint_calls)


def test_video_only_velocity_matches_joint_video_row_and_restores_mode():
    model = _MockAlignedModel()
    prepared = prepare_stage3_batch(
        model,
        {"sample_id": "mock"},
        k=4,
        video_noise=_video_noise(),
        action_noise=_action_noise(),
    )
    model.train()
    model.alignment_adapter.eval()

    max_abs = validate_video_only_joint_equivalence(
        model,
        prepared,
        rtol=0.0,
        atol=0.0,
    )

    assert max_abs == 0.0
    assert model.training is True
    assert model.alignment_adapter.training is False
    assert model.video_expert.calls[-1]["training"] is False


def test_velocity_panel_restores_mode_when_adapter_raises():
    model = _MockAlignedModel()
    prepared = prepare_stage3_batch(
        model,
        {"sample_id": "mock"},
        k=0,
        video_noise=_video_noise(),
        action_noise=_action_noise(),
    )
    model.raise_in_hook = True

    with pytest.raises(RuntimeError, match="adapter failure"):
        compute_stage3_velocity_panel(model, prepared)

    assert model._unified_inference_mode == "wo"


def test_video_only_rollout_rejects_action_conditioned_video_expert():
    model = _MockAlignedModel(action_conditioned=True)

    with pytest.raises(ValueError, match="action_conditioned"):
        rollout_self_video(
            model,
            video_noise=_video_noise(),
            first_frame_latents=model.first_frame_latents,
            context=model.context,
            context_mask=model.context_mask,
            fuse_vae_embedding_in_latents=True,
            panel=_solver_panel(model),
            k=0,
        )
