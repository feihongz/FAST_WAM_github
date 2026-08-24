import torch
from torch import nn

from fastwam.models.wan22.fastwam import FastWAM
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
