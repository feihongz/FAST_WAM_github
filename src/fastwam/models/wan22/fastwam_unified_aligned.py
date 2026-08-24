"""UnifiedShared model with an optional Stage 3 action-alignment Adapter."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .fastwam_unified_shared import FastWAMUnifiedShared
from .video_action_alignment import VideoActionResidualAdapter


class FastWAMUnifiedAligned(FastWAMUnifiedShared):
    """Frozen UnifiedShared base plus a w-video-only action residual."""

    def __init__(
        self,
        *args,
        alignment_adapter: nn.Module | None = None,
        alignment_config: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        config = dict(alignment_config or {})
        if alignment_adapter is None:
            alignment_adapter = VideoActionResidualAdapter(
                action_hidden_dim=int(config.pop("action_hidden_dim", self.action_expert.hidden_dim)),
                video_hidden_dim=int(config.pop("video_hidden_dim", self.video_expert.hidden_dim)),
                action_dim=int(config.pop("action_dim", self.action_expert.action_dim)),
                **config,
            )
        self.alignment_adapter = alignment_adapter.to(
            device=self.device,
            dtype=self.torch_dtype,
        )
        self.alignment_config = self.alignment_adapter.config()

    @classmethod
    def from_wan22_pretrained(
        cls, *, alignment_config: dict[str, Any] | None = None, **kwargs
    ):
        model = super().from_wan22_pretrained(**kwargs)
        if alignment_config is not None:
            model._replace_alignment_adapter(alignment_config)
        return model

    def _replace_alignment_adapter(self, alignment_config: dict[str, Any]) -> None:
        config = dict(alignment_config)
        self.alignment_adapter = VideoActionResidualAdapter(
            action_hidden_dim=int(config.pop("action_hidden_dim", self.action_expert.hidden_dim)),
            video_hidden_dim=int(config.pop("video_hidden_dim", self.video_expert.hidden_dim)),
            action_dim=int(config.pop("action_dim", self.action_expert.action_dim)),
            **config,
        ).to(device=self.device, dtype=self.torch_dtype)
        self.alignment_config = self.alignment_adapter.config()

    def _apply_action_velocity_hook(
        self,
        base_action_velocity: torch.Tensor,
        *,
        action_tokens: torch.Tensor,
        video_tokens: torch.Tensor,
        action_pre: dict[str, Any],
        video_pre: dict[str, Any],
    ) -> torch.Tensor:
        if str(getattr(self, "_unified_inference_mode", "wo")) != "w":
            return base_action_velocity
        correction = self.alignment_adapter(
            action_tokens=action_tokens,
            video_tokens=video_tokens,
            video_meta=video_pre.get("meta"),
        )
        return base_action_velocity + correction.to(
            device=base_action_velocity.device,
            dtype=base_action_velocity.dtype,
        )

    def configure_alignment_training(self) -> set[str]:
        """Freeze the base and enable gradients only on alignment_adapter."""
        self.eval()
        self.requires_grad_(False)
        self.alignment_adapter.train()
        self.alignment_adapter.requires_grad_(True)
        trainable = {
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        }
        if not trainable or not all(name.startswith("alignment_adapter.") for name in trainable):
            raise RuntimeError(f"invalid alignment trainable set: {sorted(trainable)}")
        return trainable
