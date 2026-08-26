"""Fail-closed Stage 2 label math in normalized action space."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GateLabelStatistics:
    """Per-sample errors and hard binary route labels."""

    e0: torch.Tensor
    e10: torch.Tensor
    relative_gain: torch.Tensor
    label: torch.Tensor
    sample_weight: torch.Tensor


def _validate_inputs(
    action_wo: torch.Tensor,
    action_w: torch.Tensor,
    target_action: torch.Tensor,
    action_is_pad: torch.Tensor,
    action_dim_is_pad: torch.Tensor | None,
) -> None:
    if action_wo.ndim != 4:
        raise ValueError(
            "action_wo must be [num_seeds,B,T,D], got "
            f"{tuple(action_wo.shape)}"
        )
    if action_w.shape != action_wo.shape:
        raise ValueError(
            "action_w must match action_wo exactly so every seed is paired"
        )
    if target_action.ndim != 3:
        raise ValueError(
            "target_action must be [B,T,D], got "
            f"{tuple(target_action.shape)}"
        )
    if action_wo.shape[0] < 1:
        raise ValueError("at least one paired action seed is required")
    if tuple(action_wo.shape[1:]) != tuple(target_action.shape):
        raise ValueError(
            "prediction/target shape mismatch: "
            f"{tuple(action_wo.shape[1:])} vs {tuple(target_action.shape)}"
        )
    if action_is_pad.shape != target_action.shape[:2]:
        raise ValueError(
            "action_is_pad must be [B,T], got "
            f"{tuple(action_is_pad.shape)}"
        )
    if action_is_pad.dtype is not torch.bool:
        raise TypeError("action_is_pad must have bool dtype")
    if action_dim_is_pad is not None:
        valid_shapes = {
            (target_action.shape[-1],),
            (target_action.shape[0], target_action.shape[-1]),
        }
        if tuple(action_dim_is_pad.shape) not in valid_shapes:
            raise ValueError(
                "action_dim_is_pad must be [D] or [B,D], got "
                f"{tuple(action_dim_is_pad.shape)}"
            )
        if action_dim_is_pad.dtype is not torch.bool:
            raise TypeError("action_dim_is_pad must have bool dtype")
    if not action_wo.is_floating_point() or not target_action.is_floating_point():
        raise TypeError("predicted and target actions must be floating point")
    if action_w.dtype != action_wo.dtype or action_w.device != action_wo.device:
        raise ValueError("paired wo/w actions must share dtype and device")
    tensors = {
        "action_wo": action_wo,
        "action_w": action_w,
        "target_action": target_action,
    }
    for name, value in tensors.items():
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains a non-finite value")


def _valid_action_mask(
    target_action: torch.Tensor,
    action_is_pad: torch.Tensor,
    action_dim_is_pad: torch.Tensor | None,
) -> torch.Tensor:
    batch_size, action_horizon, action_dim = target_action.shape
    valid_time = (~action_is_pad).reshape(batch_size, action_horizon, 1)
    if action_dim_is_pad is None:
        valid_dim = torch.ones(
            batch_size,
            action_dim,
            dtype=torch.bool,
            device=action_is_pad.device,
        )
    else:
        valid_dim = ~action_dim_is_pad.to(device=action_is_pad.device)
        if valid_dim.ndim == 1:
            valid_dim = valid_dim.unsqueeze(0).expand(batch_size, -1)
    valid = valid_time & valid_dim.reshape(batch_size, 1, action_dim)
    if torch.any(valid.sum(dim=(1, 2)) <= 0):
        raise ValueError("every sample must contain at least one valid action value")
    return valid


def _paired_masked_mse(
    prediction: torch.Tensor,
    target_action: torch.Tensor,
    valid_action: torch.Tensor,
) -> torch.Tensor:
    # Label decisions sit on a strict relative margin.  Always accumulate the
    # squared error in FP32: model rollouts may be BF16/FP16, where direct
    # squaring can quantize a boundary case or overflow before the reduction.
    prediction_fp32 = prediction.to(dtype=torch.float32)
    target = target_action.to(
        device=prediction.device,
        dtype=torch.float32,
    )
    valid = valid_action.to(
        device=prediction.device,
        dtype=torch.float32,
    ).unsqueeze(0)
    squared_error = (prediction_fp32 - target.unsqueeze(0)).square()
    numerator = (squared_error * valid).sum(dim=(2, 3))
    denominator = valid.sum(dim=(2, 3))
    return (numerator / denominator).mean(dim=0)


def paired_gate_label_statistics(
    *,
    action_wo: torch.Tensor,
    action_w: torch.Tensor,
    target_action: torch.Tensor,
    action_is_pad: torch.Tensor,
    action_dim_is_pad: torch.Tensor | None = None,
    relative_margin: float = 0.05,
    relative_gain_epsilon: float = 1.0e-12,
) -> GateLabelStatistics:
    """Compute E0/E10 from complete, same-seed-paired action rollouts.

    Inputs must already be in the model's normalized action space. Predictions
    use shape ``[num_seeds, B, T, D]`` so the seed pairing is explicit and the
    two route errors are averaged over the same number of full rollouts.
    """

    margin = float(relative_margin)
    epsilon = float(relative_gain_epsilon)
    if not 0.0 <= margin < 1.0:
        raise ValueError("relative_margin must be in [0, 1)")
    if epsilon <= 0.0:
        raise ValueError("relative_gain_epsilon must be positive")
    _validate_inputs(
        action_wo,
        action_w,
        target_action,
        action_is_pad,
        action_dim_is_pad,
    )
    valid_action = _valid_action_mask(
        target_action,
        action_is_pad,
        action_dim_is_pad,
    )

    e0 = _paired_masked_mse(action_wo, target_action, valid_action)
    e10 = _paired_masked_mse(action_w, target_action, valid_action)
    if not torch.isfinite(e0).all() or not torch.isfinite(e10).all():
        raise ValueError("Gate label errors are non-finite after FP32 accumulation")
    denominator = e0.clamp_min(epsilon)
    relative_gain = torch.where(
        e0 > epsilon,
        (e0 - e10) / denominator,
        torch.zeros_like(e0),
    )
    label = e10 < (1.0 - margin) * e0
    return GateLabelStatistics(
        e0=e0,
        e10=e10,
        relative_gain=relative_gain,
        label=label,
        sample_weight=torch.ones_like(e0),
    )
