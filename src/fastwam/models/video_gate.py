"""Stage 2 binary video Gate using query-time inputs only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn


VIDEO_GATE_CONFIG_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class VideoGateConfig:
    """Serializable architecture contract for :class:`BinaryVideoGate`."""

    proprio_dim: int
    context_dim: int = 4096
    cnn_channels: tuple[int, int, int] = (32, 64, 128)
    context_feature_dim: int = 128
    proprio_hidden_dim: int = 64
    proprio_feature_dim: int = 32
    fusion_hidden_dim: int = 128

    def __post_init__(self) -> None:
        dimensions = {
            "proprio_dim": self.proprio_dim,
            "context_dim": self.context_dim,
            "context_feature_dim": self.context_feature_dim,
            "proprio_hidden_dim": self.proprio_hidden_dim,
            "proprio_feature_dim": self.proprio_feature_dim,
            "fusion_hidden_dim": self.fusion_hidden_dim,
        }
        if any(int(value) <= 0 for value in dimensions.values()):
            raise ValueError(f"Gate dimensions must be positive: {dimensions}")
        channels = tuple(int(value) for value in self.cnn_channels)
        if len(channels) != 3 or any(value <= 0 for value in channels):
            raise ValueError("cnn_channels must contain three positive dimensions")
        object.__setattr__(self, "cnn_channels", channels)
        for name, value in dimensions.items():
            object.__setattr__(self, name, int(value))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe, versioned architecture payload."""

        return {
            "schema_version": VIDEO_GATE_CONFIG_SCHEMA_VERSION,
            "proprio_dim": self.proprio_dim,
            "context_dim": self.context_dim,
            "cnn_channels": list(self.cnn_channels),
            "context_feature_dim": self.context_feature_dim,
            "proprio_hidden_dim": self.proprio_hidden_dim,
            "proprio_feature_dim": self.proprio_feature_dim,
            "fusion_hidden_dim": self.fusion_hidden_dim,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> VideoGateConfig:
        """Restore a config while rejecting unknown schema versions."""

        values = dict(payload)
        version = values.pop("schema_version", None)
        if version != VIDEO_GATE_CONFIG_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported video Gate config schema_version: "
                f"{version!r}"
            )
        channels = values.get("cnn_channels")
        if isinstance(channels, Sequence) and not isinstance(channels, str):
            values["cnn_channels"] = tuple(channels)
        return cls(**values)


class BinaryVideoGate(nn.Module):
    """Choose whether a query should use the complete N=0 or N=10 branch.

    The caller must provide a normalized proprio vector. The API deliberately
    accepts no predicted video, ground-truth future, action error, or other
    post-generation feature.
    """

    def __init__(
        self,
        *,
        proprio_dim: int,
        context_dim: int = 4096,
        cnn_channels: Sequence[int] = (32, 64, 128),
        context_feature_dim: int = 128,
        proprio_hidden_dim: int = 64,
        proprio_feature_dim: int = 32,
        fusion_hidden_dim: int = 128,
    ):
        super().__init__()
        self.gate_config = VideoGateConfig(
            proprio_dim=proprio_dim,
            context_dim=context_dim,
            cnn_channels=tuple(cnn_channels),
            context_feature_dim=context_feature_dim,
            proprio_hidden_dim=proprio_hidden_dim,
            proprio_feature_dim=proprio_feature_dim,
            fusion_hidden_dim=fusion_hidden_dim,
        )
        first, second, image_feature_dim = self.gate_config.cnn_channels
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, first, kernel_size=5, stride=2, padding=2),
            nn.GELU(approximate="tanh"),
            nn.Conv2d(first, second, kernel_size=3, stride=2, padding=1),
            nn.GELU(approximate="tanh"),
            nn.Conv2d(
                second,
                image_feature_dim,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.GELU(approximate="tanh"),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
        )
        self.context_encoder = nn.Linear(
            self.gate_config.context_dim,
            self.gate_config.context_feature_dim,
        )
        self.proprio_encoder = nn.Sequential(
            nn.Linear(
                self.gate_config.proprio_dim,
                self.gate_config.proprio_hidden_dim,
            ),
            nn.GELU(approximate="tanh"),
            nn.Linear(
                self.gate_config.proprio_hidden_dim,
                self.gate_config.proprio_feature_dim,
            ),
            nn.GELU(approximate="tanh"),
        )
        fusion_dim = (
            image_feature_dim
            + self.gate_config.context_feature_dim
            + self.gate_config.proprio_feature_dim
        )
        self.logit_head = nn.Sequential(
            nn.Linear(fusion_dim, self.gate_config.fusion_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.gate_config.fusion_hidden_dim, 1),
        )

    def config(self) -> dict[str, Any]:
        """Return the JSON-safe constructor contract."""

        return self.gate_config.to_dict()

    @classmethod
    def from_config(cls, payload: Mapping[str, Any]) -> BinaryVideoGate:
        """Construct a Gate from :meth:`config` output."""

        config = VideoGateConfig.from_dict(payload)
        values = config.to_dict()
        values.pop("schema_version")
        return cls(**values)

    def parameter_count(self, *, trainable_only: bool = False) -> int:
        """Count total parameters, or only parameters requiring gradients."""

        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if not trainable_only or parameter.requires_grad
        )

    def _validate_inputs(
        self,
        input_image: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: torch.Tensor,
    ) -> None:
        tensors = {
            "input_image": input_image,
            "context": context,
            "context_mask": context_mask,
            "proprio": proprio,
        }
        for name, value in tensors.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
        if input_image.ndim != 4 or input_image.shape[1] != 3:
            raise ValueError("input_image must have shape [B,3,H,W]")
        if input_image.shape[0] <= 0 or min(input_image.shape[2:]) <= 0:
            raise ValueError("input_image batch and spatial dimensions must be positive")
        if context.ndim != 3:
            raise ValueError("context must have shape [B,L,4096]")
        if context.shape[-1] != self.gate_config.context_dim:
            raise ValueError(
                "context hidden dimension does not match Gate config: "
                f"expected {self.gate_config.context_dim}, got {context.shape[-1]}"
            )
        if context.shape[1] <= 0:
            raise ValueError("context sequence length must be positive")
        if context_mask.ndim != 2 or tuple(context_mask.shape) != tuple(
            context.shape[:2]
        ):
            raise ValueError("context_mask must have shape [B,L] matching context")
        if context_mask.dtype != torch.bool:
            raise ValueError("context_mask must have bool dtype")
        if not bool(context_mask.any(dim=1).all().item()):
            raise ValueError("every context_mask row must select at least one token")
        if proprio.ndim != 2:
            raise ValueError("proprio must have shape [B,D]")
        if proprio.shape[-1] != self.gate_config.proprio_dim:
            raise ValueError(
                "proprio dimension does not match Gate config: "
                f"expected {self.gate_config.proprio_dim}, got {proprio.shape[-1]}"
            )
        batch_size = input_image.shape[0]
        if context.shape[0] != batch_size or proprio.shape[0] != batch_size:
            raise ValueError("input_image/context/proprio batch sizes must match")
        devices = {value.device for value in tensors.values()}
        if len(devices) != 1:
            raise ValueError("all Gate inputs must be on the same device")
        parameter_device = self.context_encoder.weight.device
        if input_image.device != parameter_device:
            raise ValueError("Gate inputs and parameters must be on the same device")
        for name in ("input_image", "context", "proprio"):
            if not tensors[name].is_floating_point():
                raise ValueError(f"{name} must have a floating-point dtype")

    def forward(
        self,
        *,
        input_image: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: torch.Tensor,
    ) -> torch.Tensor:
        """Return one N=10 routing logit per query, with shape ``[B]``."""

        self._validate_inputs(input_image, context, context_mask, proprio)
        parameter_dtype = self.context_encoder.weight.dtype
        input_image = input_image.to(dtype=parameter_dtype)
        context = context.to(dtype=parameter_dtype)
        proprio = proprio.to(dtype=parameter_dtype)
        image_features = self.image_encoder(input_image)
        masked_context = context.masked_fill(~context_mask.unsqueeze(-1), 0.0)
        context_denominator = context_mask.sum(dim=1, keepdim=True).to(
            dtype=context.dtype
        )
        context_features = self.context_encoder(
            masked_context.sum(dim=1) / context_denominator
        )
        proprio_features = self.proprio_encoder(proprio)
        fused = torch.cat(
            [image_features, context_features, proprio_features],
            dim=-1,
        )
        return self.logit_head(fused).squeeze(-1)


__all__ = [
    "BinaryVideoGate",
    "VIDEO_GATE_CONFIG_SCHEMA_VERSION",
    "VideoGateConfig",
]
