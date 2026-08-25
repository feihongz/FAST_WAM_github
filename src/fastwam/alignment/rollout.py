"""Frozen self-video rollout and shared-noise Stage 3 branch construction."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Mapping

import torch

STAGE3_SOLVER_STEPS = 10

@dataclass(frozen=True)
class RolloutStep:
    sample_id: str
    k: int
    sigma: float
    seed: int
    base_model_sha256: str

    def __post_init__(self) -> None:
        if not 0 <= self.k < STAGE3_SOLVER_STEPS:
            raise ValueError(f"k must be in [0,{STAGE3_SOLVER_STEPS}), got {self.k}")
        if not self.sample_id or not self.base_model_sha256:
            raise ValueError("sample_id and base_model_sha256 are required")


@dataclass(frozen=True)
class SolverPanel:
    """Paired video/action inference schedules for the fixed 10-step solver."""

    video_timesteps: torch.Tensor
    video_deltas: torch.Tensor
    action_timesteps: torch.Tensor
    action_deltas: torch.Tensor
    video_sigmas: torch.Tensor
    action_sigmas: torch.Tensor

    def __post_init__(self) -> None:
        tensors = {
            "video_timesteps": self.video_timesteps,
            "video_deltas": self.video_deltas,
            "action_timesteps": self.action_timesteps,
            "action_deltas": self.action_deltas,
        }
        for name, value in tensors.items():
            if value.ndim != 1 or value.numel() != STAGE3_SOLVER_STEPS:
                raise ValueError(
                    f"{name} must have shape [{STAGE3_SOLVER_STEPS}], "
                    f"got {tuple(value.shape)}"
                )
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite")
        validate_solver_panel(self.video_sigmas)
        validate_solver_panel(self.action_sigmas)


@dataclass(frozen=True)
class PreparedStage3Batch:
    """Shared inputs at solver step k before the three action forwards."""

    k: int
    first_frame_latents: torch.Tensor
    z_self_k: torch.Tensor
    z_gt_k: torch.Tensor
    noisy_action: torch.Tensor
    video_timestep: torch.Tensor
    action_timestep: torch.Tensor
    video_sigma: torch.Tensor
    action_sigma: torch.Tensor
    action_target: torch.Tensor
    action_weight: torch.Tensor
    context: torch.Tensor
    context_mask: torch.Tensor
    action_is_pad: torch.Tensor | None
    fuse_vae_embedding_in_latents: bool


@dataclass(frozen=True)
class Stage3VelocityPanel:
    """The three shared-action-noise velocity branches consumed by the loss."""

    v0: torch.Tensor
    v_gt: torch.Tensor
    v_self: torch.Tensor
    v_target: torch.Tensor
    action_weight: torch.Tensor
    action_is_pad: torch.Tensor | None
    k: int
    video_sigma: torch.Tensor
    action_sigma: torch.Tensor


def validate_solver_panel(sigmas: torch.Tensor, *, steps: int = STAGE3_SOLVER_STEPS) -> None:
    if sigmas.ndim != 1 or sigmas.numel() != steps:
        raise ValueError(f"sigma panel must be [{steps}], got {tuple(sigmas.shape)}")
    if (
        not torch.isfinite(sigmas).all()
        or (sigmas < 0).any()
        or (sigmas > 1).any()
    ):
        raise ValueError("sigma panel must be finite and in [0,1]")


def _validate_k(k: int) -> int:
    k = int(k)
    if not 0 <= k < STAGE3_SOLVER_STEPS:
        raise ValueError(f"k must be in [0,{STAGE3_SOLVER_STEPS}), got {k}")
    return k


def _batch_timestep(
    timestep: torch.Tensor,
    *,
    batch_size: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    if timestep.ndim != 0:
        raise ValueError(f"solver timestep must be scalar, got {tuple(timestep.shape)}")
    return timestep.to(device=reference.device, dtype=reference.dtype).reshape(1).expand(
        batch_size
    )


def _lock_first_frame(
    latents: torch.Tensor,
    first_frame_latents: torch.Tensor,
) -> torch.Tensor:
    if latents.ndim != 5:
        raise ValueError(f"video latents must be 5D, got {tuple(latents.shape)}")
    expected = (
        latents.shape[0],
        latents.shape[1],
        1,
        latents.shape[3],
        latents.shape[4],
    )
    if tuple(first_frame_latents.shape) != expected:
        raise ValueError(
            "first_frame_latents must match video latents except for temporal "
            f"length 1, got {tuple(first_frame_latents.shape)} vs {expected}"
        )
    locked = latents.clone()
    locked[:, :, 0:1] = first_frame_latents.to(
        device=latents.device,
        dtype=latents.dtype,
    )
    return locked


def _validate_video_rollout_contract(model: Any) -> None:
    video_expert = model.video_expert
    if getattr(video_expert, "action_conditioned", None) is not False:
        raise ValueError(
            "video-only Stage 3 rollout requires "
            "video_expert.action_conditioned=false"
        )
    if getattr(video_expert, "seperated_timestep", None) is not True:
        raise ValueError(
            "Stage 3 rollout requires video_expert.seperated_timestep=true"
        )
    if (
        str(getattr(video_expert, "video_attention_mask_mode", ""))
        != "first_frame_causal"
    ):
        raise ValueError(
            "Stage 3 rollout requires "
            "video_attention_mask_mode='first_frame_causal'"
        )


def build_solver_panel(
    model: Any,
    *,
    num_solver_steps: int = STAGE3_SOLVER_STEPS,
    sigma_shift: float | None = None,
    dtype: torch.dtype,
    device: torch.device | str,
) -> SolverPanel:
    """Build paired deployment schedules used to select a shared step."""

    if int(num_solver_steps) != STAGE3_SOLVER_STEPS:
        raise ValueError(
            f"Stage 3 requires exactly {STAGE3_SOLVER_STEPS} solver steps, "
            f"got {num_solver_steps}"
        )
    video_timesteps, video_deltas = (
        model.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=STAGE3_SOLVER_STEPS,
            device=torch.device(device),
            dtype=dtype,
            shift_override=sigma_shift,
        )
    )
    action_timesteps, action_deltas = (
        model.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=STAGE3_SOLVER_STEPS,
            device=torch.device(device),
            dtype=dtype,
            shift_override=sigma_shift,
        )
    )
    video_steps = float(model.infer_video_scheduler.num_train_timesteps)
    action_steps = float(model.infer_action_scheduler.num_train_timesteps)
    if video_steps <= 0 or action_steps <= 0:
        raise ValueError("scheduler num_train_timesteps must be positive")
    return SolverPanel(
        video_timesteps=video_timesteps.detach(),
        video_deltas=video_deltas.detach(),
        action_timesteps=action_timesteps.detach(),
        action_deltas=action_deltas.detach(),
        video_sigmas=(video_timesteps / video_steps).detach(),
        action_sigmas=(action_timesteps / action_steps).detach(),
    )


@torch.no_grad()
def rollout_self_video(
    model: Any,
    *,
    video_noise: torch.Tensor,
    first_frame_latents: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    fuse_vae_embedding_in_latents: bool,
    panel: SolverPanel,
    k: int,
) -> torch.Tensor:
    """Run exactly the first k frozen video-only solver updates."""

    k = _validate_k(k)
    _validate_video_rollout_contract(model)
    if not fuse_vae_embedding_in_latents:
        raise ValueError("Stage 3 rollout requires first-frame latent fusion")
    z_self = _lock_first_frame(video_noise.detach(), first_frame_latents.detach())
    batch_size = z_self.shape[0]
    for step_index in range(k):
        timestep = _batch_timestep(
            panel.video_timesteps[step_index],
            batch_size=batch_size,
            reference=z_self,
        )
        pred_video = model.video_expert(
            x=z_self,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=True,
        )
        z_self = model.infer_video_scheduler.step(
            pred_video,
            panel.video_deltas[step_index],
            z_self,
        )
        z_self = _lock_first_frame(z_self, first_frame_latents)
    return z_self.detach()


def _shared_noise_or_random(
    reference: torch.Tensor,
    noise: torch.Tensor | None,
    *,
    name: str,
) -> torch.Tensor:
    if noise is None:
        return torch.randn_like(reference)
    if noise.shape != reference.shape:
        raise ValueError(
            f"{name} must have shape {tuple(reference.shape)}, got {tuple(noise.shape)}"
        )
    return noise.to(device=reference.device, dtype=reference.dtype).detach()


@torch.no_grad()
def prepare_stage3_batch(
    model: Any,
    sample: Mapping[str, torch.Tensor],
    *,
    k: int,
    video_noise: torch.Tensor | None = None,
    action_noise: torch.Tensor | None = None,
    num_solver_steps: int = STAGE3_SOLVER_STEPS,
    sigma_shift: float | None = None,
    tiled: bool = False,
) -> PreparedStage3Batch:
    """Encode one raw batch and construct shared states at solver step k."""

    k = _validate_k(k)
    inputs = model.build_inputs(sample, tiled=tiled)
    input_latents = inputs["input_latents"].detach()
    action = inputs["action"].detach()
    first_frame_latents = inputs["first_frame_latents"]
    fuse_flag = bool(inputs["fuse_vae_embedding_in_latents"])
    if first_frame_latents is None or not fuse_flag:
        raise ValueError(
            "Stage 3 requires first_frame_latents and "
            "fuse_vae_embedding_in_latents=true"
        )
    first_frame_latents = first_frame_latents.detach()
    context = inputs["context"].detach()
    context_mask = inputs["context_mask"].detach()
    panel = build_solver_panel(
        model,
        num_solver_steps=num_solver_steps,
        sigma_shift=sigma_shift,
        dtype=input_latents.dtype,
        device=input_latents.device,
    )
    if (
        model.train_action_scheduler.num_train_timesteps
        != model.infer_action_scheduler.num_train_timesteps
    ):
        raise ValueError(
            "train/infer action schedulers must share num_train_timesteps"
        )

    video_noise = _shared_noise_or_random(
        input_latents,
        video_noise,
        name="video_noise",
    )
    action_noise = _shared_noise_or_random(action, action_noise, name="action_noise")
    z_self_k = rollout_self_video(
        model,
        video_noise=video_noise,
        first_frame_latents=first_frame_latents,
        context=context,
        context_mask=context_mask,
        fuse_vae_embedding_in_latents=fuse_flag,
        panel=panel,
        k=k,
    )

    batch_size = input_latents.shape[0]
    video_timestep = _batch_timestep(
        panel.video_timesteps[k],
        batch_size=batch_size,
        reference=input_latents,
    )
    z_gt_k = model.infer_video_scheduler.add_noise(
        input_latents,
        video_noise,
        video_timestep,
    )
    z_gt_k = _lock_first_frame(z_gt_k, first_frame_latents).detach()

    action_timestep = _batch_timestep(
        panel.action_timesteps[k],
        batch_size=batch_size,
        reference=action,
    )
    noisy_action = model.train_action_scheduler.add_noise(
        action,
        action_noise,
        action_timestep,
    ).detach()
    action_target = model.train_action_scheduler.training_target(
        action,
        action_noise,
        action_timestep,
    ).detach()
    action_weight = model.train_action_scheduler.training_weight(
        action_timestep
    ).detach()
    action_is_pad = inputs["action_is_pad"]
    if action_is_pad is not None:
        action_is_pad = action_is_pad.detach()

    return PreparedStage3Batch(
        k=k,
        first_frame_latents=first_frame_latents,
        z_self_k=z_self_k,
        z_gt_k=z_gt_k,
        noisy_action=noisy_action,
        video_timestep=video_timestep.detach(),
        action_timestep=action_timestep.detach(),
        video_sigma=panel.video_sigmas[k].detach(),
        action_sigma=panel.action_sigmas[k].detach(),
        action_target=action_target,
        action_weight=action_weight,
        context=context,
        context_mask=context_mask,
        action_is_pad=action_is_pad,
        fuse_vae_embedding_in_latents=fuse_flag,
    )


@contextmanager
def _unified_mode(model: Any, mode: str) -> Iterator[None]:
    missing = object()
    previous = getattr(model, "_unified_inference_mode", missing)
    model._unified_inference_mode = mode
    try:
        yield
    finally:
        if previous is missing:
            delattr(model, "_unified_inference_mode")
        else:
            model._unified_inference_mode = previous


@torch.no_grad()
def validate_video_only_joint_equivalence(
    model: Any,
    prepared: PreparedStage3Batch,
    *,
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> float:
    """Fail fast unless the self-rollout primitive matches joint video MoT.

    Stage 3 relies on the video row being independent of action tokens. This
    check exercises both real execution paths on the same latent and returns
    the maximum absolute difference for experiment metadata.
    """

    if rtol < 0 or atol < 0:
        raise ValueError("rtol and atol must be non-negative")
    _validate_video_rollout_contract(model)

    was_training = bool(model.training)
    adapter = getattr(model, "alignment_adapter", None)
    adapter_was_training = (
        bool(adapter.training) if isinstance(adapter, torch.nn.Module) else None
    )
    model.eval()
    try:
        direct_velocity = model.video_expert(
            x=prepared.z_self_k,
            timestep=prepared.video_timestep,
            context=prepared.context,
            context_mask=prepared.context_mask,
            action=None,
            fuse_vae_embedding_in_latents=(
                prepared.fuse_vae_embedding_in_latents
            ),
        )
        with _unified_mode(model, "w"):
            joint_prediction = model._predict_joint_base(
                latents_video=prepared.z_self_k,
                latents_action=prepared.noisy_action,
                timestep_video=prepared.video_timestep,
                timestep_action=prepared.action_timestep,
                context=prepared.context,
                context_mask=prepared.context_mask,
                fuse_vae_embedding_in_latents=(
                    prepared.fuse_vae_embedding_in_latents
                ),
                gt_action=None,
            )
        joint_velocity = joint_prediction.video_velocity
        if direct_velocity.shape != joint_velocity.shape:
            raise RuntimeError(
                "video-only/joint velocity shape mismatch: "
                f"{tuple(direct_velocity.shape)} vs {tuple(joint_velocity.shape)}"
            )
        max_abs = float(
            (direct_velocity.float() - joint_velocity.float()).abs().max().item()
        )
        if not torch.allclose(
            direct_velocity.float(),
            joint_velocity.float(),
            rtol=rtol,
            atol=atol,
        ):
            raise RuntimeError(
                "video-only rollout does not match joint MoT video velocity; "
                f"max_abs={max_abs:.6g}, rtol={rtol}, atol={atol}"
            )
        return max_abs
    finally:
        model.train(was_training)
        if adapter_was_training is not None:
            adapter.train(adapter_was_training)


def compute_stage3_velocity_panel(
    model: Any,
    prepared: PreparedStage3Batch,
) -> Stage3VelocityPanel:
    """Compute detached wo/GT anchors and one differentiable self branch."""

    with torch.no_grad(), _unified_mode(model, "wo"):
        v0 = model._predict_wo_action_noise(
            first_frame_latents=prepared.first_frame_latents,
            latents_action=prepared.noisy_action,
            timestep_action=prepared.action_timestep,
            context=prepared.context,
            context_mask=prepared.context_mask,
            fuse_vae_embedding_in_latents=(
                prepared.fuse_vae_embedding_in_latents
            ),
        ).detach()

    with torch.no_grad(), _unified_mode(model, "w"):
        gt_prediction = model._predict_joint_base(
            latents_video=prepared.z_gt_k,
            latents_action=prepared.noisy_action,
            timestep_video=prepared.video_timestep,
            timestep_action=prepared.action_timestep,
            context=prepared.context,
            context_mask=prepared.context_mask,
            fuse_vae_embedding_in_latents=(
                prepared.fuse_vae_embedding_in_latents
            ),
            gt_action=None,
        )
        v_gt = gt_prediction.action_velocity.detach()
        del gt_prediction

        self_prediction = model._predict_joint_base(
            latents_video=prepared.z_self_k,
            latents_action=prepared.noisy_action,
            timestep_video=prepared.video_timestep,
            timestep_action=prepared.action_timestep,
            context=prepared.context,
            context_mask=prepared.context_mask,
            fuse_vae_embedding_in_latents=(
                prepared.fuse_vae_embedding_in_latents
            ),
            gt_action=None,
        )
        self_action_velocity = self_prediction.action_velocity.detach()
        self_action_tokens = self_prediction.action_tokens.detach()
        self_video_tokens = self_prediction.video_tokens.detach()
        self_video_meta = self_prediction.video_pre.get("meta")
        del self_prediction

    with _unified_mode(model, "w"):
        v_self = model._apply_action_velocity_hook(
            self_action_velocity,
            action_tokens=self_action_tokens,
            video_tokens=self_video_tokens,
            action_pre={},
            video_pre={"meta": self_video_meta},
        )

    expected_shape = prepared.action_target.shape
    for name, value in {"v0": v0, "v_gt": v_gt, "v_self": v_self}.items():
        if value.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {tuple(expected_shape)}, "
                f"got {tuple(value.shape)}"
            )
    if v0.requires_grad or v_gt.requires_grad or prepared.action_target.requires_grad:
        raise RuntimeError("v0, v_gt, and v_target must be detached")
    if not v_self.requires_grad:
        raise RuntimeError("v_self must retain the alignment Adapter graph")
    return Stage3VelocityPanel(
        v0=v0,
        v_gt=v_gt,
        v_self=v_self,
        v_target=prepared.action_target,
        action_weight=prepared.action_weight,
        action_is_pad=prepared.action_is_pad,
        k=prepared.k,
        video_sigma=prepared.video_sigma,
        action_sigma=prepared.action_sigma,
    )

def shared_action_noise(*, shape: tuple[int, ...], seed: int,
                        device: torch.device | str = "cpu",
                        dtype: torch.dtype = torch.float32) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(int(seed))
    return torch.randn(shape, generator=generator, device=device, dtype=dtype)

def perturb_with_shared_noise(latent: torch.Tensor, noise: torch.Tensor, sigma: float) -> torch.Tensor:
    if latent.shape != noise.shape:
        raise ValueError("latent and shared noise must have identical shapes")
    sigma_value = float(sigma)
    if not 0.0 <= sigma_value <= 1.0:
        raise ValueError(f"sigma must be in [0,1], got {sigma_value}")
    return (1.0 - sigma_value) * latent + sigma_value * noise
