"""Contracts for Stage 3 self-video rollout and shared action noise."""
from __future__ import annotations
from dataclasses import dataclass
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

def validate_solver_panel(sigmas: torch.Tensor, *, steps: int = STAGE3_SOLVER_STEPS) -> None:
    if sigmas.ndim != 1 or sigmas.numel() != steps:
        raise ValueError(f"sigma panel must be [{steps}], got {tuple(sigmas.shape)}")
    if not torch.isfinite(sigmas).all() or (sigmas < 0).any():
        raise ValueError("sigma panel must be finite and non-negative")

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
