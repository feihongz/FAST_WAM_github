import math

import pytest
import torch

from fastwam.gating.metrics import compute_gate_binary_metrics


def test_gate_metrics_perfect_ranking_and_serializable_output():
    metrics = compute_gate_binary_metrics(
        logits=torch.tensor([-4.0, -2.0, 2.0, 4.0]),
        labels=torch.tensor([0, 0, 1, 1]),
        num_calibration_bins=4,
    )

    assert metrics.auroc == 1.0
    assert metrics.auprc == 1.0
    assert metrics.positive_rate == 0.5
    assert metrics.predicted_positive_rate == 0.5
    assert metrics.num_examples == 4
    assert math.isfinite(metrics.bce)
    assert math.isfinite(metrics.expected_calibration_error)
    assert metrics.to_dict()["num_examples"] == 4


def test_gate_metrics_group_ties_and_match_average_precision_definition():
    tied = compute_gate_binary_metrics(
        logits=torch.zeros(4),
        labels=torch.tensor([0, 1, 0, 1], dtype=torch.bool),
    )
    reversed_ranking = compute_gate_binary_metrics(
        logits=torch.tensor([4.0, 3.0, 2.0, 1.0]),
        labels=torch.tensor([0, 0, 1, 1]),
    )

    assert tied.auroc == 0.5
    assert tied.auprc == 0.5
    assert reversed_ranking.auroc == 0.0
    assert reversed_ranking.auprc == pytest.approx(5.0 / 12.0)


@pytest.mark.parametrize("labels", [torch.zeros(3), torch.ones(3)])
def test_gate_metrics_report_undefined_ranking_for_single_class(labels):
    metrics = compute_gate_binary_metrics(
        logits=torch.tensor([-1.0, 0.0, 1.0]),
        labels=labels,
    )

    assert metrics.auroc is None
    assert metrics.auprc is None


@pytest.mark.parametrize(
    ("logits", "labels", "message"),
    [
        (torch.empty(0), torch.empty(0), "at least one"),
        (torch.zeros(2, 1), torch.zeros(2), r"shape \[N\]"),
        (torch.tensor([0.0, torch.nan]), torch.zeros(2), "finite"),
        (torch.zeros(2), torch.tensor([0, 2]), "zero or one"),
    ],
)
def test_gate_metrics_fail_closed(logits, labels, message):
    with pytest.raises((TypeError, ValueError), match=message):
        compute_gate_binary_metrics(logits=logits, labels=labels)
