import pytest
import torch

from fastwam.gating.labels import paired_gate_label_statistics


def test_paired_gate_labels_average_seeds_and_ignore_padding():
    target = torch.zeros(2, 3, 1)
    action_is_pad = torch.tensor(
        [[False, False, True], [False, True, True]]
    )
    action_wo = torch.tensor(
        [
            [[[2.0], [2.0], [999.0]], [[1.0], [999.0], [999.0]]],
            [[[0.0], [0.0], [999.0]], [[3.0], [999.0], [999.0]]],
        ]
    )
    action_w = torch.tensor(
        [
            [[[1.0], [1.0], [-999.0]], [[2.0], [-999.0], [-999.0]]],
            [[[1.0], [1.0], [-999.0]], [[2.0], [-999.0], [-999.0]]],
        ]
    )

    result = paired_gate_label_statistics(
        action_wo=action_wo,
        action_w=action_w,
        target_action=target,
        action_is_pad=action_is_pad,
    )

    assert torch.allclose(result.e0, torch.tensor([2.0, 5.0]))
    assert torch.allclose(result.e10, torch.tensor([1.0, 4.0]))
    assert torch.allclose(result.relative_gain, torch.tensor([0.5, 0.2]))
    assert result.label.tolist() == [True, True]
    assert torch.equal(result.sample_weight, torch.ones(2))


def test_gate_label_margin_is_strict():
    target = torch.zeros(1, 2, 1)
    action_wo = torch.tensor([[[[2.0], [0.0]]]])
    action_w = torch.tensor([[[[1.0], [1.0]]]])

    result = paired_gate_label_statistics(
        action_wo=action_wo,
        action_w=action_w,
        target_action=target,
        action_is_pad=torch.zeros(1, 2, dtype=torch.bool),
        relative_margin=0.5,
    )

    assert result.e0.item() == 2.0
    assert result.e10.item() == 1.0
    assert result.label.item() is False


def test_zero_baseline_error_has_finite_zero_gain_and_negative_label():
    zeros = torch.zeros(2, 1, 2, 3)
    result = paired_gate_label_statistics(
        action_wo=zeros,
        action_w=zeros,
        target_action=torch.zeros(1, 2, 3),
        action_is_pad=torch.zeros(1, 2, dtype=torch.bool),
    )

    assert result.e0.item() == 0.0
    assert result.relative_gain.item() == 0.0
    assert result.label.item() is False


def test_gate_label_errors_accumulate_in_fp32_for_low_precision_rollouts():
    target = torch.zeros(1, 2, 1, dtype=torch.float16)
    action_wo = torch.full((2, 1, 2, 1), 400.0, dtype=torch.float16)
    action_w = torch.full((2, 1, 2, 1), 200.0, dtype=torch.float16)

    result = paired_gate_label_statistics(
        action_wo=action_wo,
        action_w=action_w,
        target_action=target,
        action_is_pad=torch.zeros(1, 2, dtype=torch.bool),
    )

    assert result.e0.dtype == torch.float32
    assert result.e10.dtype == torch.float32
    assert result.e0.item() == 160_000.0
    assert result.e10.item() == 40_000.0
    assert result.label.item() is True


def test_gate_label_math_ignores_padded_action_dimensions():
    target = torch.zeros(1, 2, 2)
    action_wo = torch.tensor([[[[2.0, 999.0], [0.0, 999.0]]]])
    action_w = torch.tensor([[[[1.0, -999.0], [1.0, -999.0]]]])

    result = paired_gate_label_statistics(
        action_wo=action_wo,
        action_w=action_w,
        target_action=target,
        action_is_pad=torch.zeros(1, 2, dtype=torch.bool),
        action_dim_is_pad=torch.tensor([False, True]),
        relative_margin=0.5,
    )

    assert result.e0.item() == 2.0
    assert result.e10.item() == 1.0
    assert result.label.item() is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("unpaired", "paired"),
        ("target_shape", "shape mismatch"),
        ("mask_shape", r"must be \[B,T\]"),
        ("no_valid", "at least one valid"),
        ("non_finite", "non-finite"),
    ],
)
def test_gate_label_math_fails_closed(mutation, message):
    action_wo = torch.zeros(2, 1, 2, 3)
    action_w = action_wo.clone()
    target = torch.zeros(1, 2, 3)
    action_is_pad = torch.zeros(1, 2, dtype=torch.bool)
    if mutation == "unpaired":
        action_w = action_w[:1]
    elif mutation == "target_shape":
        target = target[:, :1]
    elif mutation == "mask_shape":
        action_is_pad = action_is_pad[:, :1]
    elif mutation == "no_valid":
        action_is_pad[:] = True
    elif mutation == "non_finite":
        action_w[0, 0, 0, 0] = torch.nan

    with pytest.raises((TypeError, ValueError), match=message):
        paired_gate_label_statistics(
            action_wo=action_wo,
            action_w=action_w,
            target_action=target,
            action_is_pad=action_is_pad,
        )


@pytest.mark.parametrize("margin", [-0.1, 1.0])
def test_gate_label_margin_must_be_a_fraction(margin):
    zeros = torch.zeros(1, 1, 1, 1)
    with pytest.raises(ValueError, match="relative_margin"):
        paired_gate_label_statistics(
            action_wo=zeros,
            action_w=zeros,
            target_action=torch.zeros(1, 1, 1),
            action_is_pad=torch.zeros(1, 1, dtype=torch.bool),
            relative_margin=margin,
        )
