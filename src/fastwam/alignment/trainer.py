"""Dedicated Adapter-only optimization loop."""
from __future__ import annotations
from typing import Any
import torch
from torch import nn
from .losses import Stage3LossOutput, stage3_alignment_loss

class AlignmentTrainer:
    def __init__(self, model: nn.Module, *, lr: float = 1e-4):
        configure = getattr(model, "configure_alignment_training", None)
        if not callable(configure):
            raise TypeError("model must expose configure_alignment_training()")
        trainable = configure()
        params = [p for n, p in model.named_parameters() if n in trainable and p.requires_grad]
        if not params or any(not n.startswith("alignment_adapter.") for n in trainable):
            raise ValueError("AlignmentTrainer requires alignment_adapter.* parameters only")
        self.model = model
        self.optimizer = torch.optim.AdamW(params, lr=lr)
        self.global_step = 0

    def step(self, v0: torch.Tensor, v_gt: torch.Tensor, v_self: torch.Tensor, v_target: torch.Tensor, action_is_pad: torch.Tensor | None = None, **kwargs: Any) -> Stage3LossOutput:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        out = stage3_alignment_loss(v0, v_gt, v_self, v_target, action_is_pad, **kwargs)
        out.loss.backward(); self.optimizer.step(); self.global_step += 1
        return out
