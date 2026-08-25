import torch
from torch import nn

from fastwam.models.wan22.fastwam import FastWAM, FrozenJointPrediction
from fastwam.models.wan22.fastwam_unified_aligned import FastWAMUnifiedAligned
from fastwam.models.wan22.video_action_alignment import (
    VideoActionResidualAdapter,
    load_alignment_checkpoint,
    save_alignment_checkpoint,
)


def make_adapter():
    return VideoActionResidualAdapter(
        action_hidden_dim=8,
        video_hidden_dim=12,
        action_dim=3,
        bottleneck_dim=8,
        num_heads=2,
    )


def test_zero_init_and_shapes():
    torch.manual_seed(0)
    adapter = make_adapter()
    output = adapter(
        action_tokens=torch.randn(2, 5, 8),
        video_tokens=torch.randn(2, 9, 12),
        video_meta={"tokens_per_frame": 3},
    )
    assert output.shape == (2, 5, 3)
    assert torch.equal(output, torch.zeros_like(output))


def test_base_hook_is_identity():
    model = object.__new__(FastWAM)
    base = torch.randn(1, 4, 3)
    output = FastWAM._apply_action_velocity_hook(
        model,
        base,
        action_tokens=torch.empty(1),
        video_tokens=torch.empty(1),
        action_pre={},
        video_pre={},
    )
    assert output is base


def test_joint_noise_wrapper_applies_hook_to_frozen_base():
    class TinyFastWAM(FastWAM):
        def __init__(self):
            nn.Module.__init__(self)
            self.hook_calls = 0

        @torch.no_grad()
        def _predict_joint_base(self, **kwargs):
            action = kwargs["latents_action"]
            video = kwargs["latents_video"]
            return FrozenJointPrediction(
                video_velocity=video + 1.0,
                action_velocity=action + 2.0,
                video_tokens=video.detach(),
                action_tokens=action.detach(),
                video_pre={"meta": {}},
                action_pre={},
            )

        def _apply_action_velocity_hook(
            self,
            base_action_velocity,
            *,
            action_tokens,
            video_tokens,
            action_pre,
            video_pre,
        ):
            del action_pre, video_pre
            self.hook_calls += 1
            assert not action_tokens.requires_grad
            assert not video_tokens.requires_grad
            return base_action_velocity + 3.0

    model = TinyFastWAM()
    video = torch.zeros(1, 1, 1, 1, 1, requires_grad=True)
    action = torch.zeros(1, 2, 3, requires_grad=True)
    pred_video, pred_action = model._predict_joint_noise(
        latents_video=video,
        latents_action=action,
        timestep_video=torch.zeros(1),
        timestep_action=torch.zeros(1),
        context=torch.zeros(1, 1, 1),
        context_mask=torch.ones(1, 1, dtype=torch.bool),
        fuse_vae_embedding_in_latents=True,
    )
    assert model.hook_calls == 1
    assert torch.equal(pred_video, torch.ones_like(video))
    assert torch.equal(pred_action, torch.full_like(action, 5.0))
    assert not pred_video.requires_grad
    assert not pred_action.requires_grad


def test_aligned_hook_bypasses_adapter_in_wo():
    model = nn.Module.__new__(FastWAMUnifiedAligned)
    nn.Module.__init__(model)
    model.alignment_adapter = make_adapter()
    model._unified_inference_mode = "wo"
    base = torch.randn(1, 5, 3)
    output = model._apply_action_velocity_hook(
        base,
        action_tokens=torch.randn(1, 5, 8),
        video_tokens=torch.randn(1, 9, 12),
        action_pre={},
        video_pre={"meta": {"tokens_per_frame": 3}},
    )
    assert output is base


def test_aligned_hook_detaches_base_and_adapter_inputs():
    class RecordingAdapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0))
            self.seen_requires_grad = None

        def forward(self, *, action_tokens, video_tokens, video_meta):
            del video_meta
            self.seen_requires_grad = (
                action_tokens.requires_grad,
                video_tokens.requires_grad,
            )
            return self.scale * action_tokens[..., :3]

    model = nn.Module.__new__(FastWAMUnifiedAligned)
    nn.Module.__init__(model)
    model.alignment_adapter = RecordingAdapter()
    model._unified_inference_mode = "w"
    base = torch.randn(1, 5, 3, requires_grad=True)
    action_tokens = torch.randn(1, 5, 8, requires_grad=True)
    video_tokens = torch.randn(1, 9, 12, requires_grad=True)
    output = model._apply_action_velocity_hook(
        base,
        action_tokens=action_tokens,
        video_tokens=video_tokens,
        action_pre={},
        video_pre={"meta": {"tokens_per_frame": 3}},
    )
    output.sum().backward()
    assert model.alignment_adapter.seen_requires_grad == (False, False)
    assert model.alignment_adapter.scale.grad is not None
    assert base.grad is None
    assert action_tokens.grad is None
    assert video_tokens.grad is None


def test_alignment_freeze_contract():
    model = nn.Module.__new__(FastWAMUnifiedAligned)
    nn.Module.__init__(model)
    model.backbone = nn.Linear(4, 4)
    model.alignment_adapter = make_adapter()
    trainable = model.configure_alignment_training()
    assert trainable
    assert all(name.startswith("alignment_adapter.") for name in trainable)
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("alignment_adapter.")
    )


def test_alignment_checkpoint_roundtrip(tmp_path):
    adapter = make_adapter()
    with torch.no_grad():
        adapter.output_proj.weight.normal_()
    path = save_alignment_checkpoint(
        tmp_path / "alignment.pt",
        adapter,
        base_checkpoint="base.pt",
        base_checkpoint_sha256="a" * 64,
        global_step=7,
    )
    restored = make_adapter()
    payload = load_alignment_checkpoint(
        path,
        restored,
        expected_base_checkpoint_sha256="a" * 64,
    )
    assert payload["global_step"] == 7
    for key, value in adapter.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
