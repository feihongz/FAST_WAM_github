"""Padding-aware, detached-target losses for the Stage 3 Adapter."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Stage3LossOutput:
    """Scalar loss and detached diagnostics returned by :func:`stage3_alignment_loss`."""

    loss: torch.Tensor
    action_loss: torch.Tensor
    alignment_loss: torch.Tensor
    safe_loss: torch.Tensor
    helpful_fraction: torch.Tensor
    e0: torch.Tensor
    egt: torch.Tensor
    eself: torch.Tensor


def _valid_mask(action_is_pad: torch.Tensor | None, shape: tuple[int, int], device: torch.device) -> torch.Tensor:
    if action_is_pad is None:
        return torch.ones(shape, dtype=torch.bool, device=device)
    if action_is_pad.ndim != 2 or tuple(action_is_pad.shape) != shape:
        raise ValueError(f"action_is_pad must have shape {shape}, got {tuple(action_is_pad.shape)}")
    return ~action_is_pad.to(device=device, dtype=torch.bool)


def _per_sample_mse(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if pred.shape != target.shape or pred.ndim != 3:
        raise ValueError("velocity tensors must have identical shape [B,T,D]")
    if valid.shape != pred.shape[:2]:
        raise ValueError("padding mask must match the first two velocity dimensions")
    err = (pred - target).square().mean(dim=-1)
    weights = valid.to(dtype=err.dtype)
    return (err * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def stage3_alignment_loss(
    v0: torch.Tensor,
    v_gt: torch.Tensor,
    v_self: torch.Tensor,
    v_target: torch.Tensor,
    action_is_pad: torch.Tensor | None = None,
    *,
    action_weight: torch.Tensor | None = None,
    helpful_relative_margin: float = 0.05,
    lambda_action: float = 1.0,
    lambda_align: float = 1.0,
    lambda_safe: float = 0.5,
) -> Stage3LossOutput:
    """Compute the Stage 3 objective from the three shared-noise branches.

    ``v_gt`` and ``v0`` are detached when constructing the alignment target;
    gradients flow only through ``v_self``. Padded action tokens are excluded
    from every per-sample statistic.
    """
    if not 0 <= helpful_relative_margin < 1:
        raise ValueError("helpful_relative_margin must be in [0, 1)")
    if any(weight < 0 for weight in (lambda_action, lambda_align, lambda_safe)):
        raise ValueError("loss weights must be non-negative")
    valid = _valid_mask(action_is_pad, tuple(v_self.shape[:2]), v_self.device)
    e0 = _per_sample_mse(v0, v_target, valid)
    egt = _per_sample_mse(v_gt, v_target, valid)
    eself = _per_sample_mse(v_self, v_target, valid)
    if action_weight is None:
        action_weight = torch.ones_like(eself)
    else:
        action_weight = action_weight.to(device=eself.device, dtype=eself.dtype)
        if action_weight.ndim == 0:
            action_weight = action_weight.expand_as(eself)
        if action_weight.ndim != 1 or action_weight.shape != eself.shape:
            raise ValueError(
                f"action_weight must be scalar or have shape {tuple(eself.shape)}, "
                f"got {tuple(action_weight.shape)}"
            )
        if not torch.isfinite(action_weight).all() or (action_weight < 0).any():
            raise ValueError("action_weight must be finite and non-negative")
    helpful = egt < ((1.0 - helpful_relative_margin) * e0).detach()
    target_delta = helpful.to(dtype=v_self.dtype).view(-1, 1, 1) * (v_gt.detach() - v0.detach())
    delta = v_self - v0.detach()
    alignment_loss = _per_sample_mse(delta, target_delta, valid).mean()
    safe_loss = torch.relu(eself - e0.detach()).mean()
    action_loss = (eself * action_weight).mean()
    loss = lambda_action * action_loss + lambda_align * alignment_loss + lambda_safe * safe_loss
    return Stage3LossOutput(loss, action_loss, alignment_loss, safe_loss, helpful.float().mean(), e0, egt, eself)
