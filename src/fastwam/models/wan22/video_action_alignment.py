"""Stage 3 video-to-action residual Adapter and companion checkpoint helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn


ALIGNMENT_CHECKPOINT_SCHEMA_VERSION = 1


class VideoActionResidualAdapter(nn.Module):
    """Small cross-attention residual; zero-init output preserves the base."""

    def __init__(
        self,
        *,
        action_hidden_dim: int,
        video_hidden_dim: int,
        action_dim: int,
        bottleneck_dim: int = 256,
        num_heads: int = 8,
        ffn_multiplier: int = 2,
        drop_first_video_frame: bool = True,
        zero_init_output: bool = True,
    ):
        super().__init__()
        dims = {
            "action_hidden_dim": action_hidden_dim,
            "video_hidden_dim": video_hidden_dim,
            "action_dim": action_dim,
            "bottleneck_dim": bottleneck_dim,
            "num_heads": num_heads,
            "ffn_multiplier": ffn_multiplier,
        }
        if any(int(value) <= 0 for value in dims.values()):
            raise ValueError(f"Adapter dimensions must be positive: {dims}")
        if int(bottleneck_dim) % int(num_heads) != 0:
            raise ValueError("bottleneck_dim must be divisible by num_heads")
        self.action_hidden_dim = int(action_hidden_dim)
        self.video_hidden_dim = int(video_hidden_dim)
        self.action_dim = int(action_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.num_heads = int(num_heads)
        self.ffn_multiplier = int(ffn_multiplier)
        self.drop_first_video_frame = bool(drop_first_video_frame)
        self.zero_init_output = bool(zero_init_output)
        self.action_proj = nn.Linear(self.action_hidden_dim, self.bottleneck_dim)
        self.video_proj = nn.Linear(self.video_hidden_dim, self.bottleneck_dim)
        self.cross_attention = nn.MultiheadAttention(
            self.bottleneck_dim, self.num_heads, batch_first=True
        )
        ffn_dim = self.bottleneck_dim * self.ffn_multiplier
        self.ffn = nn.Sequential(
            nn.LayerNorm(self.bottleneck_dim),
            nn.Linear(self.bottleneck_dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, self.bottleneck_dim),
        )
        self.output_proj = nn.Linear(self.bottleneck_dim, self.action_dim)
        if self.zero_init_output:
            nn.init.zeros_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)

    def config(self) -> dict[str, Any]:
        return {
            "action_hidden_dim": self.action_hidden_dim,
            "video_hidden_dim": self.video_hidden_dim,
            "action_dim": self.action_dim,
            "bottleneck_dim": self.bottleneck_dim,
            "num_heads": self.num_heads,
            "ffn_multiplier": self.ffn_multiplier,
            "drop_first_video_frame": self.drop_first_video_frame,
            "zero_init_output": self.zero_init_output,
        }

    def _pool_video_tokens(
        self, video_tokens: torch.Tensor, video_meta: Mapping[str, Any] | None
    ) -> torch.Tensor:
        if video_tokens.ndim != 3:
            raise ValueError("video_tokens must have shape [B,T,D]")
        tokens_per_frame = None if video_meta is None else video_meta.get("tokens_per_frame")
        if tokens_per_frame is None:
            raise ValueError("video_meta['tokens_per_frame'] is required")
        tokens_per_frame = int(tokens_per_frame)
        if tokens_per_frame <= 0 or video_tokens.shape[1] % tokens_per_frame:
            raise ValueError("video token count must be divisible by tokens_per_frame")
        frames = video_tokens.reshape(
            video_tokens.shape[0],
            video_tokens.shape[1] // tokens_per_frame,
            tokens_per_frame,
            video_tokens.shape[2],
        ).mean(dim=2)
        if self.drop_first_video_frame:
            if frames.shape[1] <= 1:
                raise ValueError("at least two frames are required")
            frames = frames[:, 1:]
        return frames

    def forward(
        self,
        *,
        action_tokens: torch.Tensor,
        video_tokens: torch.Tensor,
        video_meta: Mapping[str, Any] | None = None,
    ) -> torch.Tensor:
        if action_tokens.ndim != 3 or video_tokens.ndim != 3:
            raise ValueError("action_tokens and video_tokens must be [B,T,D]")
        if action_tokens.shape[0] != video_tokens.shape[0]:
            raise ValueError("action/video batch sizes must match")
        if action_tokens.shape[-1] != self.action_hidden_dim:
            raise ValueError("action hidden dimension does not match Adapter")
        if video_tokens.shape[-1] != self.video_hidden_dim:
            raise ValueError("video hidden dimension does not match Adapter")
        query = self.action_proj(action_tokens)
        key_value = self.video_proj(self._pool_video_tokens(video_tokens, video_meta))
        attended, _ = self.cross_attention(query, key_value, key_value, need_weights=False)
        hidden = query + attended
        return self.output_proj(hidden + self.ffn(hidden))


def apply_alignment_velocity(
    adapter: nn.Module,
    base_action_velocity: torch.Tensor,
    *,
    action_tokens: torch.Tensor,
    video_tokens: torch.Tensor,
    video_meta: Mapping[str, Any] | None,
) -> torch.Tensor:
    """Apply the shared Stage 3 residual used by training and w inference."""

    correction = adapter(
        action_tokens=action_tokens.detach(),
        video_tokens=video_tokens.detach(),
        video_meta=video_meta,
    )
    return base_action_velocity.detach() + correction.to(
        device=base_action_velocity.device,
        dtype=base_action_velocity.dtype,
    )


def save_alignment_checkpoint(
    path: str | Path,
    adapter: VideoActionResidualAdapter,
    *,
    base_checkpoint: str,
    base_checkpoint_sha256: str | None = None,
    global_step: int | None = None,
    adapter_state_dict: Mapping[str, torch.Tensor] | None = None,
    git_commit: str | None = None,
    training_contract_sha256: str | None = None,
    asset_identities: Mapping[str, Mapping[str, Any]] | None = None,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    state = adapter.state_dict() if adapter_state_dict is None else adapter_state_dict
    state = {
        str(name): value.detach().to(device="cpu")
        for name, value in state.items()
    }
    torch.save(
        {
            "schema_version": ALIGNMENT_CHECKPOINT_SCHEMA_VERSION,
            "kind": "stage3_alignment_export",
            "adapter": state,
            "base_checkpoint": str(base_checkpoint),
            "base_checkpoint_sha256": base_checkpoint_sha256,
            "alignment_config": adapter.config(),
            "global_step": None if global_step is None else int(global_step),
            "git_commit": git_commit,
            "training_contract_sha256": training_contract_sha256,
            "asset_identities": dict(asset_identities or {}),
        },
        output,
    )
    return output


def load_alignment_checkpoint(
    path: str | Path,
    adapter: VideoActionResidualAdapter,
    *,
    expected_base_checkpoint_sha256: str | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location)
    if (
        not isinstance(payload, dict)
        or int(payload.get("schema_version", -1))
        != ALIGNMENT_CHECKPOINT_SCHEMA_VERSION
        or payload.get("kind") != "stage3_alignment_export"
    ):
        raise ValueError("unsupported alignment checkpoint")
    if payload.get("alignment_config") != adapter.config():
        raise ValueError("alignment checkpoint config does not match Adapter")
    if expected_base_checkpoint_sha256 is not None and payload.get("base_checkpoint_sha256") != expected_base_checkpoint_sha256:
        raise ValueError("alignment checkpoint base hash mismatch")
    state = payload.get("adapter")
    if not isinstance(state, dict):
        raise ValueError("alignment checkpoint is missing adapter state")
    adapter.load_state_dict(state, strict=True)
    return payload
