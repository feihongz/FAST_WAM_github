"""Dependency-free binary metrics for the lightweight Stage 2 Gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class GateBinaryMetrics:
    bce: float
    auroc: float | None
    auprc: float | None
    positive_rate: float
    predicted_positive_rate: float
    expected_calibration_error: float
    num_examples: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validated_binary_inputs(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(logits, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TypeError("Gate metric logits and labels must be torch.Tensor values")
    if logits.ndim != 1 or labels.ndim != 1 or logits.shape != labels.shape:
        raise ValueError("Gate metric logits and labels must have matching shape [N]")
    if logits.numel() == 0:
        raise ValueError("Gate metrics require at least one example")
    if not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError("Gate metric logits must be finite floating-point values")
    if labels.dtype == torch.bool:
        binary = labels.to(dtype=torch.float64, device="cpu")
    elif labels.is_floating_point() or labels.dtype in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        binary = labels.detach().to(dtype=torch.float64, device="cpu")
        if not torch.isfinite(binary).all() or not bool(
            ((binary == 0.0) | (binary == 1.0)).all().item()
        ):
            raise ValueError("Gate metric labels must contain only zero or one")
    else:
        raise TypeError("Gate metric labels must be bool or numeric binary values")
    return logits.detach().to(dtype=torch.float64, device="cpu"), binary


def _grouped_binary_curve(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    order = torch.argsort(probabilities, descending=True, stable=True)
    scores = probabilities[order]
    ordered_labels = labels[order]
    group_end = torch.ones(scores.shape[0], dtype=torch.bool)
    if scores.shape[0] > 1:
        group_end[:-1] = scores[:-1] != scores[1:]
    indices = torch.nonzero(group_end, as_tuple=False).flatten()
    true_positive = ordered_labels.cumsum(dim=0)[indices]
    false_positive = (1.0 - ordered_labels).cumsum(dim=0)[indices]
    return true_positive, false_positive, indices


def _auroc_auprc(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[float | None, float | None]:
    positives = float(labels.sum().item())
    negatives = float(labels.numel() - positives)
    if positives == 0.0 or negatives == 0.0:
        return None, None
    true_positive, false_positive, _ = _grouped_binary_curve(
        probabilities,
        labels,
    )
    tpr = torch.cat([torch.zeros(1, dtype=torch.float64), true_positive / positives])
    fpr = torch.cat([torch.zeros(1, dtype=torch.float64), false_positive / negatives])
    auroc = float(torch.trapezoid(tpr, fpr).item())

    recall = true_positive / positives
    precision = true_positive / (true_positive + false_positive)
    previous_recall = torch.cat(
        [torch.zeros(1, dtype=torch.float64), recall[:-1]]
    )
    auprc = float(((recall - previous_recall) * precision).sum().item())
    return auroc, auprc


def _expected_calibration_error(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_bins: int,
) -> float:
    if isinstance(num_bins, bool) or not isinstance(num_bins, int) or num_bins <= 0:
        raise ValueError("num_calibration_bins must be a positive integer")
    bin_indices = torch.clamp(
        torch.floor(probabilities * num_bins).to(dtype=torch.int64),
        max=num_bins - 1,
    )
    total = probabilities.numel()
    ece = torch.zeros((), dtype=torch.float64)
    for bin_index in range(num_bins):
        selected = bin_indices == bin_index
        count = int(selected.sum().item())
        if count == 0:
            continue
        confidence = probabilities[selected].mean()
        accuracy = labels[selected].mean()
        ece = ece + abs(confidence - accuracy) * (count / total)
    return float(ece.item())


def compute_gate_binary_metrics(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
    num_calibration_bins: int = 10,
) -> GateBinaryMetrics:
    """Compute exact binary ranking and calibration metrics on CPU."""

    cutoff = float(threshold)
    if not math.isfinite(cutoff) or not 0.0 <= cutoff <= 1.0:
        raise ValueError("threshold must be finite and in [0, 1]")
    logits_cpu, labels_cpu = _validated_binary_inputs(logits, labels)
    probabilities = torch.sigmoid(logits_cpu)
    bce = float(
        F.binary_cross_entropy_with_logits(
            logits_cpu,
            labels_cpu,
            reduction="mean",
        ).item()
    )
    auroc, auprc = _auroc_auprc(probabilities, labels_cpu)
    return GateBinaryMetrics(
        bce=bce,
        auroc=auroc,
        auprc=auprc,
        positive_rate=float(labels_cpu.mean().item()),
        predicted_positive_rate=float((probabilities >= cutoff).double().mean().item()),
        expected_calibration_error=_expected_calibration_error(
            probabilities,
            labels_cpu,
            num_bins=num_calibration_bins,
        ),
        num_examples=labels_cpu.numel(),
    )


__all__ = ["GateBinaryMetrics", "compute_gate_binary_metrics"]
