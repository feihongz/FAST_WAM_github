import json

import pytest

from fastwam.gating.calibration import (
    CALIBRATION_ALGORITHM,
    calibrate_probability_thresholds,
)


def _calibration(report, index=0):
    return report["calibrations"][index]


def test_calibration_uses_inclusive_threshold_and_reports_metrics():
    report = calibrate_probability_thresholds(
        probabilities=[0.6, 0.9, 0.7, 0.8],
        labels=[0, 1, 1, 0],
        target_with_rates=[0.5],
        configured_video_steps=10,
    )

    result = _calibration(report)
    assert report["algorithm"] == CALIBRATION_ALGORITHM
    assert report["decision_rule"] == "probability >= threshold => w"
    assert result["target_count"] == 2
    assert result["threshold"] == 0.8
    assert result["selected_count"] == 2
    assert result["actual_with_rate"] == 0.5
    assert result["expected_video_steps_per_query"] == 5.0
    assert result["exact_target"] is True
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["true_positive_count"] == 1
    assert result["false_positive_count"] == 1
    assert result["false_negative_count"] == 1
    assert result["true_negative_count"] == 1


def test_target_count_uses_explicit_round_half_up_formula():
    report = calibrate_probability_thresholds(
        probabilities=[0.9, 0.8, 0.7],
        labels=[1, 0, 1],
        target_with_rates=[0.5],
    )

    result = _calibration(report)
    assert result["target_count"] == 2  # floor(0.5 * 3 + 0.5)
    assert result["threshold"] == 0.8
    assert result["selected_count"] == 2


def test_tie_block_is_not_split_and_equal_error_prefers_less_video():
    report = calibrate_probability_thresholds(
        probabilities=[0.5, 0.9, 0.5, 0.1],
        labels=[1, 1, 0, 0],
        target_with_rates=[0.5],
    )

    result = _calibration(report)
    assert result["target_count"] == 2
    assert result["selected_count"] == 1
    assert result["threshold"] == 0.9
    assert result["count_error"] == 1
    assert result["exact_target"] is False
    assert result["block_diagnostics"] == {
        "lower_reachable_count": 1,
        "upper_reachable_count": 3,
        "blocking_probability": 0.5,
        "blocking_block_size": 2,
        "target_inside_probability_block": True,
        "target_inside_tie_block": True,
        "chosen_side": "lower",
        "zero_target_blocked_by_probability_one": False,
        "threshold_is_observed_probability": True,
        "selected_boundary_probability": 0.9,
        "selected_boundary_block_size": 1,
        "selected_count_above_boundary": 0,
    }


def test_score_blocks_are_sorted_and_deterministic_for_tied_input():
    probabilities = [0.2, 0.8, 0.5, 0.8, 0.2]
    labels = [0, 1, 1, 0, 1]
    kwargs = {"target_with_rates": [0.2, 0.6, 1.0]}

    report = calibrate_probability_thresholds(probabilities, labels, **kwargs)
    reversed_report = calibrate_probability_thresholds(
        list(reversed(probabilities)),
        list(reversed(labels)),
        **kwargs,
    )

    assert report == reversed_report
    assert report["score_blocks"] == [
        {
            "probability": 0.8,
            "count": 2,
            "positive_count": 1,
            "negative_count": 1,
            "selected_count_above": 0,
            "selected_count_at_threshold": 2,
        },
        {
            "probability": 0.5,
            "count": 1,
            "positive_count": 1,
            "negative_count": 0,
            "selected_count_above": 2,
            "selected_count_at_threshold": 3,
        },
        {
            "probability": 0.2,
            "count": 2,
            "positive_count": 1,
            "negative_count": 1,
            "selected_count_above": 3,
            "selected_count_at_threshold": 5,
        },
    ]
    assert report["probability_block_diagnostics"] == {
        "num_unique_probabilities": 3,
        "num_tied_blocks": 2,
        "num_samples_in_tied_blocks": 4,
        "largest_block_size": 2,
        "num_reachable_selected_counts": 4,
        "minimum_reachable_selected_count": 0,
        "maximum_reachable_selected_count": 5,
        "zero_selected_reachable": True,
    }


def test_zero_selection_boundary_and_higher_threshold_are_canonical():
    report = calibrate_probability_thresholds(
        probabilities=[0.7, 0.2],
        labels=[1, 0],
        target_with_rates=[0.0, 1.0],
    )

    zero, all_examples = report["calibrations"]
    assert zero["threshold"] == 1.0
    assert zero["selected_count"] == 0
    assert zero["precision"] is None
    assert zero["recall"] == 0.0
    assert zero["block_diagnostics"]["threshold_is_observed_probability"] is False
    # Threshold 0.2 and threshold 0.0 both select all examples.  The canonical
    # representative is the higher reachable threshold, 0.2.
    assert all_examples["threshold"] == 0.2
    assert all_examples["selected_count"] == 2


def test_probability_one_reports_that_zero_count_is_not_reachable():
    report = calibrate_probability_thresholds(
        probabilities=[1.0, 0.4],
        labels=[1, 0],
        target_with_rates=[0.0],
    )

    result = _calibration(report)
    assert result["threshold"] == 1.0
    assert result["selected_count"] == 1
    assert result["exact_target"] is False
    assert result["block_diagnostics"]["lower_reachable_count"] is None
    assert result["block_diagnostics"]["upper_reachable_count"] == 1
    assert result["block_diagnostics"]["zero_target_blocked_by_probability_one"] is True
    assert report["probability_block_diagnostics"]["zero_selected_reachable"] is False


def test_no_positive_labels_make_recall_explicitly_undefined():
    report = calibrate_probability_thresholds(
        probabilities=[0.8, 0.2],
        labels=[False, 0],
        target_with_rates=[0.5],
    )

    result = _calibration(report)
    assert result["precision"] == 0.0
    assert result["recall"] is None


@pytest.mark.parametrize(
    ("probabilities", "labels", "targets", "error", "match"),
    [
        ([], [], [0.5], ValueError, "at least one example"),
        ([0.5], [], [0.5], ValueError, "equal length"),
        ([float("nan")], [1], [0.5], ValueError, "finite.*\\[0, 1\\]"),
        ([float("inf")], [1], [0.5], ValueError, "finite.*\\[0, 1\\]"),
        ([-0.1], [1], [0.5], ValueError, "finite.*\\[0, 1\\]"),
        ([1.1], [1], [0.5], ValueError, "finite.*\\[0, 1\\]"),
        ([True], [1], [0.5], TypeError, "real numbers"),
        ([0.5], [2], [0.5], ValueError, "zero or one"),
        ([0.5], [float("nan")], [0.5], ValueError, "finite zero or one"),
        ([0.5], [1], [float("nan")], ValueError, "finite.*\\[0, 1\\]"),
        ([0.5], [1], [], ValueError, "at least one target"),
    ],
)
def test_calibration_rejects_invalid_inputs(
    probabilities, labels, targets, error, match
):
    with pytest.raises(error, match=match):
        calibrate_probability_thresholds(probabilities, labels, targets)


@pytest.mark.parametrize("configured_video_steps", [True, 1.5, -1])
def test_calibration_rejects_invalid_video_step_count(configured_video_steps):
    error = TypeError if configured_video_steps is True or configured_video_steps == 1.5 else ValueError
    with pytest.raises(error, match="configured_video_steps"):
        calibrate_probability_thresholds(
            [0.5],
            [1],
            [0.5],
            configured_video_steps=configured_video_steps,
        )


def test_calibration_output_is_canonical_json_safe():
    report = calibrate_probability_thresholds(
        probabilities=(value for value in [0.0, 0.4, 0.4, 1.0]),
        labels=(value for value in [False, True, 1.0, 0]),
        target_with_rates=(value for value in [0.0, 0.25, 0.75, 1.0]),
    )

    encoded = json.dumps(report, allow_nan=False, sort_keys=True)
    assert json.loads(encoded) == report
