import torch

from fastwam.alignment.losses import stage3_alignment_loss


def test_stage3_loss_masks_padding_and_detaches_teacher_target():
    v0 = torch.zeros(2, 3, 1, requires_grad=True)
    vgt = torch.tensor([[[1.0], [1.0], [9.0]], [[0.0], [0.0], [0.0]]], requires_grad=True)
    vself = torch.zeros(2, 3, 1, requires_grad=True)
    target = torch.ones(2, 3, 1)
    pad = torch.tensor([[False, False, True], [False, True, True]])
    out = stage3_alignment_loss(v0, vgt, vself, target, pad, lambda_safe=0.0)
    assert torch.isclose(out.e0[0], torch.tensor(1.0))
    assert torch.isclose(out.e0[1], torch.tensor(1.0))
    assert out.helpful_fraction.item() == 0.5
    out.loss.backward()
    assert vself.grad is not None
    assert vgt.grad is None
    assert v0.grad is None


def test_stage3_safe_term_only_penalizes_regression():
    z = torch.zeros(1, 2, 1)
    out = stage3_alignment_loss(z, z, torch.ones_like(z), z, lambda_action=0.0, lambda_align=0.0, lambda_safe=1.0)
    assert torch.isclose(out.safe_loss, torch.tensor(1.0))


def test_stage3_rejects_bad_mask_shape():
    z = torch.zeros(1, 2, 1)
    try:
        stage3_alignment_loss(z, z, z, z, torch.zeros(1, 1, dtype=torch.bool))
    except ValueError as exc:
        assert "action_is_pad" in str(exc)
    else:
        raise AssertionError("expected mask shape validation")
