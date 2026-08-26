"""Strict same-noise paired inference for Stage 2 Gate labels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import torch


STAGE2_NUM_INFERENCE_STEPS = 10
MIN_SEED_PAIRS = 2
MAX_SEED_PAIRS = 4


@dataclass(frozen=True)
class PairedActionRollouts:
    """Complete normalized-action rollouts for matched wo/w noise seeds."""

    action_wo: torch.Tensor
    action_w: torch.Tensor
    seeds: tuple[int, ...]
    action_horizon: int
    num_video_frames: int
    num_inference_steps: int


def _tensor(sample: Mapping[str, Any], key: str) -> torch.Tensor:
    value = sample.get(key)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Stage 2 label sample {key!r} must be a torch.Tensor")
    return value


def _finite_float_tensor(value: torch.Tensor, *, name: str) -> None:
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains a non-finite value")


def _validated_seed_pairs(seeds: Sequence[int]) -> tuple[int, ...]:
    if isinstance(seeds, (str, bytes)) or not isinstance(seeds, Sequence):
        raise TypeError("seeds must be a sequence of integers")
    normalized: list[int] = []
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("every paired action seed must be an integer")
        if not 0 <= seed < 2**63:
            raise ValueError("every paired action seed must be in [0, 2**63)")
        normalized.append(int(seed))
    if not MIN_SEED_PAIRS <= len(normalized) <= MAX_SEED_PAIRS:
        raise ValueError(
            f"formal Stage 2 labels require {MIN_SEED_PAIRS}--"
            f"{MAX_SEED_PAIRS} seed pairs"
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError("paired action seeds must be unique")
    return tuple(normalized)


def _validate_sample(sample: Mapping[str, Any]) -> dict[str, torch.Tensor | int]:
    video = _tensor(sample, "video")
    action = _tensor(sample, "action")
    proprio = _tensor(sample, "proprio")
    context = _tensor(sample, "context")
    context_mask = _tensor(sample, "context_mask")

    if video.ndim != 4 or video.shape[0] != 3:
        raise ValueError("sample video must have shape [3,T,H,W]")
    if video.shape[1] < 2 or video.shape[1] % 4 != 1:
        raise ValueError("sample video frame count must be at least two and satisfy T % 4 == 1")
    if min(video.shape[2:]) <= 0 or any(size % 16 for size in video.shape[2:]):
        raise ValueError("sample video spatial dimensions must be positive multiples of 16")
    if action.ndim != 2 or min(action.shape) <= 0:
        raise ValueError("sample action must have shape [T,D] with positive dimensions")
    if proprio.ndim != 2 or min(proprio.shape) <= 0:
        raise ValueError("sample proprio must have shape [T,P] with positive dimensions")
    if proprio.shape[0] != action.shape[0]:
        raise ValueError("sample proprio/action horizons must match")
    if action.shape[0] % (video.shape[1] - 1) != 0:
        raise ValueError(
            "sample action horizon must be divisible by video transitions"
        )
    if context.ndim != 2 or min(context.shape) <= 0:
        raise ValueError("sample context must have shape [L,C] with positive dimensions")
    if context_mask.ndim != 1 or context_mask.shape[0] != context.shape[0]:
        raise ValueError("sample context_mask must have shape [L] matching context")
    if context_mask.dtype != torch.bool:
        raise TypeError("sample context_mask must have bool dtype")
    if not bool(context_mask.any().item()):
        raise ValueError("sample context_mask must select at least one token")
    for name, value in {
        "video": video,
        "action": action,
        "proprio": proprio,
        "context": context,
    }.items():
        _finite_float_tensor(value, name=f"sample {name}")
    return {
        "input_image": video[:, 0],
        "target_action": action,
        "proprio": proprio[0],
        "context": context,
        "context_mask": context_mask,
        "action_horizon": int(action.shape[0]),
        "action_dim": int(action.shape[1]),
        "num_video_frames": int(video.shape[1]),
    }


def _action_from_result(
    result: Any,
    *,
    mode: str,
    action_horizon: int,
    action_dim: int,
) -> torch.Tensor:
    if not isinstance(result, Mapping) or "action" not in result:
        raise TypeError(f"{mode} inference must return a mapping containing action")
    action = result["action"]
    if not isinstance(action, torch.Tensor):
        raise TypeError(f"{mode} inference action must be a torch.Tensor")
    if tuple(action.shape) != (action_horizon, action_dim):
        raise ValueError(
            f"{mode} inference action shape mismatch: got {tuple(action.shape)}, "
            f"expected {(action_horizon, action_dim)}"
        )
    _finite_float_tensor(action, name=f"{mode} inference action")
    return action.detach().to(device="cpu", dtype=torch.float32)


def run_paired_action_rollouts(
    model: Any,
    sample: Mapping[str, Any],
    *,
    seeds: Sequence[int],
    num_inference_steps: int = STAGE2_NUM_INFERENCE_STEPS,
    sigma_shift: float | None = None,
    rand_device: str = "cpu",
    tiled: bool = False,
) -> PairedActionRollouts:
    """Run exactly one wo and one w rollout for each matched action seed.

    Only the current image and current proprio are exposed to either branch.
    The target action is inspected for output shape but is never passed to the
    model.  The w branch generates its own future; the wo branch never receives
    ``num_video_frames`` at all.
    """

    if isinstance(num_inference_steps, bool) or not isinstance(
        num_inference_steps, int
    ):
        raise TypeError("num_inference_steps must be an integer")
    if num_inference_steps != STAGE2_NUM_INFERENCE_STEPS:
        raise ValueError(
            f"formal Stage 2 labels require exactly {STAGE2_NUM_INFERENCE_STEPS} "
            "inference steps"
        )
    if sigma_shift is not None:
        sigma_shift = float(sigma_shift)
        if not math.isfinite(sigma_shift) or sigma_shift <= 0.0:
            raise ValueError("sigma_shift must be finite and positive")
    if not isinstance(rand_device, str) or not rand_device:
        raise ValueError("rand_device must be a non-empty string")
    if not isinstance(tiled, bool):
        raise TypeError("tiled must be bool")
    infer = getattr(model, "infer_action_mode", None)
    evaluate = getattr(model, "eval", None)
    if not callable(infer) or not callable(evaluate):
        raise TypeError("model must provide eval() and infer_action_mode()")

    paired_seeds = _validated_seed_pairs(seeds)
    inputs = _validate_sample(sample)
    action_horizon = int(inputs["action_horizon"])
    action_dim = int(inputs["action_dim"])
    num_video_frames = int(inputs["num_video_frames"])
    common = {
        "prompt": None,
        "input_image": inputs["input_image"],
        "action_horizon": action_horizon,
        "proprio": inputs["proprio"],
        "context": inputs["context"],
        "context_mask": inputs["context_mask"],
        "num_inference_steps": num_inference_steps,
        "sigma_shift": sigma_shift,
        "rand_device": rand_device,
        "tiled": tiled,
    }

    model.eval()
    actions_wo: list[torch.Tensor] = []
    actions_w: list[torch.Tensor] = []
    with torch.inference_mode():
        for seed in paired_seeds:
            result_wo = infer(
                **common,
                seed=seed,
                inference_mode="wo",
            )
            result_w = infer(
                **common,
                seed=seed,
                inference_mode="w",
                num_video_frames=num_video_frames,
            )
            actions_wo.append(
                _action_from_result(
                    result_wo,
                    mode="wo",
                    action_horizon=action_horizon,
                    action_dim=action_dim,
                )
            )
            actions_w.append(
                _action_from_result(
                    result_w,
                    mode="w",
                    action_horizon=action_horizon,
                    action_dim=action_dim,
                )
            )

    return PairedActionRollouts(
        action_wo=torch.stack(actions_wo, dim=0).unsqueeze(1),
        action_w=torch.stack(actions_w, dim=0).unsqueeze(1),
        seeds=paired_seeds,
        action_horizon=action_horizon,
        num_video_frames=num_video_frames,
        num_inference_steps=num_inference_steps,
    )


__all__ = [
    "MAX_SEED_PAIRS",
    "MIN_SEED_PAIRS",
    "PairedActionRollouts",
    "STAGE2_NUM_INFERENCE_STEPS",
    "run_paired_action_rollouts",
]
