"""Deterministic, tie-safe threshold calibration for the Stage 2 Gate.

The runtime decision rule is inclusive: ``probability >= threshold`` selects
the complete video-conditioned (``w``) branch.  Consequently, examples with
the same empirical probability form an indivisible score block.  This module
selects only thresholds that are actually reachable under that rule and makes
the resulting count error explicit.

All public outputs are composed exclusively of canonical JSON-safe Python
scalars, lists, and dictionaries.  No tensor, NumPy scalar, tuple, NaN, or
infinity is returned.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math
import numbers
from typing import Any


CALIBRATION_ALGORITHM = "empirical_probability_quantile_tie_safe_v1"
DECISION_RULE = "probability >= threshold => w"


@dataclass(frozen=True)
class _ScoreBlock:
    probability: float
    count: int
    positive_count: int
    selected_count_above: int
    selected_count_at_threshold: int
    selected_positive_count_at_threshold: int


@dataclass(frozen=True)
class _Candidate:
    threshold: float
    selected_count: int
    selected_positive_count: int
    boundary_block: _ScoreBlock | None


def _materialize(values: Iterable[Any], *, field: str) -> list[Any]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field} must be an iterable of scalar values")
    try:
        return list(values)
    except TypeError as exc:
        raise TypeError(f"{field} must be an iterable of scalar values") from exc


def _unbox_scalar(value: Any) -> Any:
    """Unbox NumPy/torch-style scalar objects without importing either."""

    if isinstance(value, (bool, numbers.Real)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (RuntimeError, TypeError, ValueError):
            return value
    return value


def _probability(value: Any, *, field: str) -> float:
    value = _unbox_scalar(value)
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{field} values must be real numbers")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} values must be finite and in [0, 1]")
    # Avoid emitting the non-canonical JSON spelling -0.0.
    return 0.0 if result == 0.0 else result


def _binary_label(value: Any) -> int:
    value = _unbox_scalar(value)
    if isinstance(value, bool):
        return int(value)
    if not isinstance(value, numbers.Real):
        raise TypeError("labels must contain only bool or numeric zero/one values")
    result = float(value)
    if not math.isfinite(result) or result not in (0.0, 1.0):
        raise ValueError("labels must contain only finite zero or one values")
    return int(result)


def _configured_video_steps(value: Any) -> int:
    value = _unbox_scalar(value)
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError("configured_video_steps must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError("configured_video_steps must be non-negative")
    return result


def _build_score_blocks(
    probabilities: list[float],
    labels: list[int],
) -> list[_ScoreBlock]:
    ordered = sorted(zip(probabilities, labels), key=lambda pair: pair[0], reverse=True)
    blocks: list[_ScoreBlock] = []
    selected_count = 0
    selected_positive_count = 0
    index = 0
    while index < len(ordered):
        probability = ordered[index][0]
        end = index + 1
        block_positive_count = ordered[index][1]
        while end < len(ordered) and ordered[end][0] == probability:
            block_positive_count += ordered[end][1]
            end += 1
        block_count = end - index
        count_above = selected_count
        selected_count += block_count
        selected_positive_count += block_positive_count
        blocks.append(
            _ScoreBlock(
                probability=probability,
                count=block_count,
                positive_count=block_positive_count,
                selected_count_above=count_above,
                selected_count_at_threshold=selected_count,
                selected_positive_count_at_threshold=selected_positive_count,
            )
        )
        index = end
    return blocks


def _reachable_candidates(blocks: list[_ScoreBlock]) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    # threshold=1.0 selects nothing exactly when the maximum score is below 1.
    if blocks[0].probability < 1.0:
        candidates.append(
            _Candidate(
                threshold=1.0,
                selected_count=0,
                selected_positive_count=0,
                boundary_block=None,
            )
        )
    candidates.extend(
        _Candidate(
            threshold=block.probability,
            selected_count=block.selected_count_at_threshold,
            selected_positive_count=block.selected_positive_count_at_threshold,
            boundary_block=block,
        )
        for block in blocks
    )
    return candidates


def _blocking_diagnostics(
    *,
    target_count: int,
    selected: _Candidate,
    candidates: list[_Candidate],
    blocks_by_selected_count: dict[int, _ScoreBlock],
) -> dict[str, Any]:
    reachable_counts = sorted(candidate.selected_count for candidate in candidates)
    if target_count in reachable_counts:
        lower_count: int | None = target_count
        upper_count: int | None = target_count
        blocking_block = None
        chosen_side = "exact"
    else:
        lower_count = max(
            (count for count in reachable_counts if count < target_count),
            default=None,
        )
        upper_count = min(
            (count for count in reachable_counts if count > target_count),
            default=None,
        )
        blocking_block = (
            blocks_by_selected_count.get(upper_count)
            if upper_count is not None
            else None
        )
        chosen_side = "lower" if selected.selected_count == lower_count else "upper"

    lower_for_block = 0 if lower_count is None else lower_count
    target_inside_probability_block = bool(
        blocking_block is not None
        and upper_count is not None
        and lower_for_block < target_count < upper_count
    )
    boundary = selected.boundary_block
    return {
        "lower_reachable_count": lower_count,
        "upper_reachable_count": upper_count,
        "blocking_probability": (
            None if blocking_block is None else blocking_block.probability
        ),
        "blocking_block_size": 0 if blocking_block is None else blocking_block.count,
        "target_inside_probability_block": target_inside_probability_block,
        "target_inside_tie_block": bool(
            target_inside_probability_block
            and blocking_block is not None
            and blocking_block.count > 1
        ),
        "chosen_side": chosen_side,
        "zero_target_blocked_by_probability_one": bool(
            target_count == 0 and 0 not in reachable_counts
        ),
        "threshold_is_observed_probability": boundary is not None,
        "selected_boundary_probability": (
            None if boundary is None else boundary.probability
        ),
        "selected_boundary_block_size": 0 if boundary is None else boundary.count,
        "selected_count_above_boundary": (
            None if boundary is None else boundary.selected_count_above
        ),
    }


def calibrate_probability_thresholds(
    probabilities: Iterable[Any],
    labels: Iterable[Any],
    target_with_rates: Iterable[Any],
    *,
    configured_video_steps: int = 10,
) -> dict[str, Any]:
    """Calibrate inclusive Gate thresholds against empirical target with-rates.

    For a target rate ``r`` and ``N`` examples, the target count is exactly
    ``floor(r * N + 0.5)``.  Among reachable empirical thresholds, selection is
    lexicographic by:

    1. smallest absolute selected-count error;
    2. fewer selected video queries;
    3. higher threshold.

    Equal probabilities are never split.  The returned score-block ledger is
    sorted by descending finite probability and makes every reachable count
    auditable under the inclusive runtime decision rule.
    """

    probability_values = _materialize(probabilities, field="probabilities")
    label_values = _materialize(labels, field="labels")
    target_values = _materialize(target_with_rates, field="target_with_rates")
    if not probability_values:
        raise ValueError("probabilities and labels require at least one example")
    if len(probability_values) != len(label_values):
        raise ValueError("probabilities and labels must have equal length")
    if not target_values:
        raise ValueError("target_with_rates requires at least one target")

    normalized_probabilities = [
        _probability(value, field="probabilities") for value in probability_values
    ]
    normalized_labels = [_binary_label(value) for value in label_values]
    normalized_targets = [
        _probability(value, field="target_with_rates") for value in target_values
    ]
    video_steps = _configured_video_steps(configured_video_steps)

    blocks = _build_score_blocks(normalized_probabilities, normalized_labels)
    candidates = _reachable_candidates(blocks)
    blocks_by_selected_count = {
        block.selected_count_at_threshold: block for block in blocks
    }
    num_examples = len(normalized_probabilities)
    positive_count = sum(normalized_labels)
    negative_count = num_examples - positive_count

    calibrations: list[dict[str, Any]] = []
    for target_rate in normalized_targets:
        target_count = int(math.floor(target_rate * num_examples + 0.5))
        selected = min(
            candidates,
            key=lambda candidate: (
                abs(candidate.selected_count - target_count),
                candidate.selected_count,
                -candidate.threshold,
            ),
        )
        selected_count = selected.selected_count
        true_positive_count = selected.selected_positive_count
        false_positive_count = selected_count - true_positive_count
        false_negative_count = positive_count - true_positive_count
        true_negative_count = negative_count - false_positive_count
        actual_with_rate = selected_count / num_examples
        calibrations.append(
            {
                "target_with_rate": target_rate,
                "target_count": target_count,
                "threshold": selected.threshold,
                "selected_count": selected_count,
                "actual_with_rate": actual_with_rate,
                "count_error": abs(selected_count - target_count),
                "rate_error": abs(actual_with_rate - target_rate),
                "exact_target": selected_count == target_count,
                "expected_video_steps_per_query": video_steps * actual_with_rate,
                "true_positive_count": true_positive_count,
                "false_positive_count": false_positive_count,
                "false_negative_count": false_negative_count,
                "true_negative_count": true_negative_count,
                "precision": (
                    true_positive_count / selected_count
                    if selected_count > 0
                    else None
                ),
                "recall": (
                    true_positive_count / positive_count
                    if positive_count > 0
                    else None
                ),
                "block_diagnostics": _blocking_diagnostics(
                    target_count=target_count,
                    selected=selected,
                    candidates=candidates,
                    blocks_by_selected_count=blocks_by_selected_count,
                ),
            }
        )

    tied_blocks = [block for block in blocks if block.count > 1]
    reachable_counts = sorted({candidate.selected_count for candidate in candidates})
    return {
        "algorithm": CALIBRATION_ALGORITHM,
        "decision_rule": DECISION_RULE,
        "num_examples": num_examples,
        "configured_video_steps": video_steps,
        "label_statistics": {
            "positive_count": positive_count,
            "negative_count": negative_count,
            "positive_rate": positive_count / num_examples,
        },
        "probability_block_diagnostics": {
            "num_unique_probabilities": len(blocks),
            "num_tied_blocks": len(tied_blocks),
            "num_samples_in_tied_blocks": sum(block.count for block in tied_blocks),
            "largest_block_size": max(block.count for block in blocks),
            "num_reachable_selected_counts": len(reachable_counts),
            "minimum_reachable_selected_count": reachable_counts[0],
            "maximum_reachable_selected_count": reachable_counts[-1],
            "zero_selected_reachable": 0 in reachable_counts,
        },
        "score_blocks": [
            {
                "probability": block.probability,
                "count": block.count,
                "positive_count": block.positive_count,
                "negative_count": block.count - block.positive_count,
                "selected_count_above": block.selected_count_above,
                "selected_count_at_threshold": block.selected_count_at_threshold,
            }
            for block in blocks
        ],
        "calibrations": calibrations,
    }


__all__ = [
    "CALIBRATION_ALGORITHM",
    "DECISION_RULE",
    "calibrate_probability_thresholds",
]
