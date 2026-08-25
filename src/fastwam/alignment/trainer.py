"""Dedicated Adapter-only optimization loop."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from fastwam.models.wan22.video_action_alignment import apply_alignment_velocity

from .losses import Stage3LossOutput, stage3_alignment_loss
from .rollout import (
    STAGE3_SOLVER_STEPS,
    Stage3VelocityPanel,
    compute_stage3_velocity_panel,
    prepare_stage3_batch,
)


class AlignmentVelocityModule(nn.Module):
    """The only module handed to Accelerate/DeepSpeed for Stage 3."""

    def __init__(self, adapter: nn.Module):
        super().__init__()
        self.adapter = adapter

    def forward(
        self,
        base_action_velocity: torch.Tensor,
        *,
        action_tokens: torch.Tensor,
        video_tokens: torch.Tensor,
        video_meta: Mapping[str, Any] | None,
    ) -> torch.Tensor:
        return apply_alignment_velocity(
            self.adapter,
            base_action_velocity,
            action_tokens=action_tokens,
            video_tokens=video_tokens,
            video_meta=video_meta,
        )


class AlignmentTrainer:
    """Keep the frozen base in eval mode and update only the Adapter."""

    def __init__(
        self,
        model: nn.Module,
        *,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        max_grad_norm: float = 1.0,
        optimizer: torch.optim.Optimizer | None = None,
        num_solver_steps: int = STAGE3_SOLVER_STEPS,
        sigma_shift: float | None = None,
    ):
        configure = getattr(model, "configure_alignment_training", None)
        if not callable(configure):
            raise TypeError("model must expose configure_alignment_training()")
        trainable = configure()
        params = [
            parameter
            for name, parameter in model.named_parameters()
            if name in trainable and parameter.requires_grad
        ]
        if not params or any(
            not name.startswith("alignment_adapter.") for name in trainable
        ):
            raise ValueError(
                "AlignmentTrainer requires alignment_adapter.* parameters only"
            )
        if lr <= 0:
            raise ValueError("lr must be positive")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        self.model = model
        self._adapter_params = tuple(params)
        self.optimizer = optimizer or torch.optim.AdamW(
            self._adapter_params,
            lr=lr,
            weight_decay=weight_decay,
        )
        expected_param_ids = [id(parameter) for parameter in params]
        optimizer_param_ids = [
            id(parameter)
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        ]
        if (
            len(optimizer_param_ids) != len(set(optimizer_param_ids))
            or set(optimizer_param_ids) != set(expected_param_ids)
        ):
            raise ValueError(
                "optimizer parameters must be exactly the trainable "
                "alignment_adapter parameters"
            )
        if int(num_solver_steps) != STAGE3_SOLVER_STEPS:
            raise ValueError(
                f"Stage 3 requires exactly {STAGE3_SOLVER_STEPS} solver steps"
            )
        self.num_solver_steps = int(num_solver_steps)
        self.sigma_shift = sigma_shift
        self.max_grad_norm = float(max_grad_norm)
        self.global_step = 0
        self.last_k: int | None = None
        self.model.zero_grad(set_to_none=True)
        self._set_alignment_modes()

    def _set_alignment_modes(self) -> None:
        """Keep every frozen module deterministic while training Adapter."""

        self.model.eval()
        self.model.alignment_adapter.train()

    def build_velocity_panel(
        self,
        sample: Mapping[str, torch.Tensor],
        *,
        k: int,
        video_noise: torch.Tensor | None = None,
        action_noise: torch.Tensor | None = None,
        tiled: bool = False,
    ) -> Stage3VelocityPanel:
        self._set_alignment_modes()
        prepared = prepare_stage3_batch(
            self.model,
            sample,
            k=k,
            video_noise=video_noise,
            action_noise=action_noise,
            num_solver_steps=self.num_solver_steps,
            sigma_shift=self.sigma_shift,
            tiled=tiled,
        )
        return compute_stage3_velocity_panel(self.model, prepared)

    def step(
        self,
        sample: Mapping[str, torch.Tensor],
        *,
        k: int | None = None,
        video_noise: torch.Tensor | None = None,
        action_noise: torch.Tensor | None = None,
        tiled: bool = False,
        **loss_kwargs: Any,
    ) -> Stage3LossOutput:
        """Build the three real branches from a raw sample and update Adapter."""

        if k is None:
            k = int(torch.randint(self.num_solver_steps, (1,)).item())
        panel = self.build_velocity_panel(
            sample,
            k=k,
            video_noise=video_noise,
            action_noise=action_noise,
            tiled=tiled,
        )
        self.last_k = panel.k
        return self.step_velocities(
            panel.v0,
            panel.v_gt,
            panel.v_self,
            panel.v_target,
            panel.action_is_pad,
            action_weight=panel.action_weight,
            **loss_kwargs,
        )

    def step_velocities(
        self,
        v0: torch.Tensor,
        v_gt: torch.Tensor,
        v_self: torch.Tensor,
        v_target: torch.Tensor,
        action_is_pad: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Stage3LossOutput:
        self._set_alignment_modes()
        self.model.zero_grad(set_to_none=True)
        out = stage3_alignment_loss(
            v0, v_gt, v_self, v_target, action_is_pad, **kwargs
        )
        if not out.loss.requires_grad:
            raise RuntimeError("v_self must retain the Adapter computation graph")
        out.loss.backward()
        leaked_gradients = [
            name
            for name, parameter in self.model.named_parameters()
            if not name.startswith("alignment_adapter.") and parameter.grad is not None
        ]
        if leaked_gradients:
            raise RuntimeError(
                "non-Adapter gradients detected: "
                + ", ".join(sorted(leaked_gradients))
            )
        torch.nn.utils.clip_grad_norm_(self._adapter_params, self.max_grad_norm)
        self.optimizer.step()
        self.global_step += 1
        return out.detached()
