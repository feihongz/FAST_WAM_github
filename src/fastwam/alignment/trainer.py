"""Dedicated Adapter-only optimization loop."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .losses import Stage3LossOutput, stage3_alignment_loss


class AlignmentTrainer:
    """Keep the frozen base in eval mode and update only the Adapter."""

    def __init__(
        self,
        model: nn.Module,
        *,
        lr: float = 1e-4,
        optimizer: torch.optim.Optimizer | None = None,
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
        self.model = model
        self.optimizer = optimizer or torch.optim.AdamW(params, lr=lr)
        self.global_step = 0

    def step(
        self,
        v0: torch.Tensor,
        v_gt: torch.Tensor,
        v_self: torch.Tensor,
        v_target: torch.Tensor,
        action_is_pad: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Stage3LossOutput:
        self.model.eval()
        self.model.alignment_adapter.train()
        self.optimizer.zero_grad(set_to_none=True)
        out = stage3_alignment_loss(
            v0, v_gt, v_self, v_target, action_is_pad, **kwargs
        )
        if not out.loss.requires_grad:
            raise RuntimeError("v_self must retain the Adapter computation graph")
        out.loss.backward()
        self.optimizer.step()
        self.global_step += 1
        return out
