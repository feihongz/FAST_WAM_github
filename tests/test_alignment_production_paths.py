from __future__ import annotations

import torch

from fastwam.alignment import (
    PreparedStage3Batch,
    validate_video_only_joint_equivalence,
)
from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.fastwam import FastWAM
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.wan_video_dit import WanVideoDiT


def _tiny_real_model() -> FastWAM:
    torch.manual_seed(7)
    common = {
        "hidden_dim": 12,
        "ffn_dim": 24,
        "text_dim": 8,
        "freq_dim": 8,
        "eps": 1e-6,
        "num_heads": 2,
        "attn_head_dim": 6,
        "num_layers": 2,
    }
    video_expert = WanVideoDiT(
        **common,
        in_dim=2,
        out_dim=2,
        patch_size=(1, 1, 1),
        has_image_input=False,
        seperated_timestep=True,
        fuse_vae_embedding_in_latents=True,
        action_conditioned=False,
        video_attention_mask_mode="first_frame_causal",
    )
    action_expert = ActionDiT(**common, action_dim=3)
    mot = MoT(
        mixtures={"video": video_expert, "action": action_expert},
        mot_checkpoint_mixed_attn=False,
    )
    return FastWAM(
        video_expert=video_expert,
        action_expert=action_expert,
        mot=mot,
        vae=torch.nn.Identity(),
        text_dim=8,
        device="cpu",
        torch_dtype=torch.float32,
    ).eval()


def _prepared(model: FastWAM, *, batch_size: int = 2) -> PreparedStage3Batch:
    z_self = torch.randn(batch_size, 2, 3, 2, 2)
    noisy_action = torch.randn(batch_size, 4, 3)
    context = torch.randn(batch_size, 5, 8)
    context_mask = torch.ones(batch_size, 5, dtype=torch.bool)
    return PreparedStage3Batch(
        k=4,
        first_frame_latents=z_self[:, :, :1].clone(),
        z_self_k=z_self,
        z_gt_k=torch.randn_like(z_self),
        noisy_action=noisy_action,
        video_timestep=torch.full((batch_size,), 500.0),
        action_timestep=torch.full((batch_size,), 500.0),
        video_sigma=torch.tensor(0.5),
        action_sigma=torch.tensor(0.5),
        action_target=torch.randn_like(noisy_action),
        action_weight=torch.ones(batch_size),
        context=context,
        context_mask=context_mask,
        action_is_pad=None,
        fuse_vae_embedding_in_latents=True,
    )


def test_tiny_real_video_only_matches_joint_mot_video_row():
    model = _tiny_real_model()
    prepared = _prepared(model)

    max_abs = validate_video_only_joint_equivalence(
        model,
        prepared,
        rtol=1e-5,
        atol=1e-6,
    )

    assert max_abs < 1e-6


def test_tiny_real_deployment_cache_matches_full_wo_path():
    model = _tiny_real_model()
    prepared = _prepared(model)

    cached = model._predict_wo_action_noise(
        first_frame_latents=prepared.first_frame_latents,
        latents_action=prepared.noisy_action,
        timestep_action=prepared.action_timestep,
        context=prepared.context,
        context_mask=prepared.context_mask,
        fuse_vae_embedding_in_latents=True,
    )
    full = model._predict_action_noise(
        first_frame_latents=prepared.first_frame_latents,
        latents_action=prepared.noisy_action,
        timestep_action=prepared.action_timestep,
        context=prepared.context,
        context_mask=prepared.context_mask,
        fuse_vae_embedding_in_latents=True,
    )

    torch.testing.assert_close(cached, full, rtol=1e-5, atol=1e-6)
