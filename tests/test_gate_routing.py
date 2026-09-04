import inspect
import json
import random

import pytest
import torch

from fastwam.gating.routing import (
    BinaryVideoRouter,
    route_with_video_gate,
    summarize_routing_telemetry,
)
from fastwam.models.video_gate import BinaryVideoGate


PROPRIO_DIM = 8


def _inputs() -> dict[str, torch.Tensor]:
    return {
        "input_image": torch.zeros(1, 3, 16, 24),
        "context": torch.zeros(1, 3, 4096),
        "context_mask": torch.tensor([[True, True, False]]),
        "proprio": torch.zeros(1, PROPRIO_DIM),
    }


def _frozen_gate(logit: float) -> BinaryVideoGate:
    gate = BinaryVideoGate(
        proprio_dim=PROPRIO_DIM,
        cnn_channels=(2, 2, 2),
        context_feature_dim=2,
        proprio_hidden_dim=2,
        proprio_feature_dim=2,
        fusion_hidden_dim=2,
    )
    with torch.no_grad():
        for parameter in gate.parameters():
            parameter.zero_()
        gate.logit_head[-1].bias.fill_(logit)
    gate.eval().requires_grad_(False)
    return gate


def _decision(
    mode: str,
    *,
    gate_latency_s: float,
    configured_video_steps: int = 10,
    actual_video_steps: int | None = None,
) -> dict:
    if actual_video_steps is None:
        actual_video_steps = configured_video_steps if mode == "w" else 0
    return {
        "logit": 0.0,
        "probability": 0.5,
        "selected_mode": mode,
        "gate_latency_s": gate_latency_s,
        "configured_video_steps": configured_video_steps,
        "actual_video_steps": actual_video_steps,
    }


def test_route_is_current_only_json_safe_and_threshold_inclusive():
    gate = _frozen_gate(0.0)
    ticks = iter((10.0, 10.125))

    decision = route_with_video_gate(
        gate,
        **_inputs(),
        threshold=0.5,
        configured_video_steps=10,
        clock=lambda: next(ticks),
    )

    assert set(decision) == {
        "logit",
        "probability",
        "selected_mode",
        "gate_latency_s",
        "configured_video_steps",
        "actual_video_steps",
    }
    assert decision == {
        "logit": 0.0,
        "probability": 0.5,
        "selected_mode": "w",
        "gate_latency_s": 0.125,
        "configured_video_steps": 10,
        "actual_video_steps": 10,
    }
    assert json.loads(json.dumps(decision)) == decision
    parameters = inspect.signature(route_with_video_gate).parameters
    assert all(
        forbidden not in parameters
        for forbidden in ("future_video", "target_action", "action_error")
    )


def test_route_below_threshold_selects_wo_and_executes_no_video_steps():
    decision = route_with_video_gate(
        _frozen_gate(-2.0),
        **_inputs(),
        threshold=0.5,
        configured_video_steps=10,
        clock=lambda: 1.0,
    )

    assert decision["probability"] < 0.5
    assert decision["selected_mode"] == "wo"
    assert decision["configured_video_steps"] == 10
    assert decision["actual_video_steps"] == 0


@pytest.mark.parametrize("threshold", [-0.01, 1.01, float("nan"), float("inf")])
def test_route_rejects_threshold_outside_closed_unit_interval(threshold):
    with pytest.raises(ValueError, match=r"threshold.*\[0, 1\]"):
        route_with_video_gate(
            _frozen_gate(0.0),
            **_inputs(),
            threshold=threshold,
            configured_video_steps=10,
        )


@pytest.mark.parametrize("threshold", [True, "0.5", None])
def test_route_rejects_non_numeric_threshold(threshold):
    with pytest.raises(TypeError, match="threshold"):
        route_with_video_gate(
            _frozen_gate(0.0),
            **_inputs(),
            threshold=threshold,
            configured_video_steps=10,
        )


def test_route_requires_eval_frozen_gate_and_one_query():
    training_gate = _frozen_gate(0.0).train()
    with pytest.raises(ValueError, match="eval mode"):
        route_with_video_gate(
            training_gate,
            **_inputs(),
            threshold=0.5,
            configured_video_steps=10,
        )

    trainable_gate = _frozen_gate(0.0)
    trainable_gate.logit_head[-1].bias.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        route_with_video_gate(
            trainable_gate,
            **_inputs(),
            threshold=0.5,
            configured_video_steps=10,
        )

    batched = _inputs()
    batched = {
        name: value.expand(2, *value.shape[1:]).clone()
        for name, value in batched.items()
    }
    with pytest.raises(ValueError, match="exactly one current query"):
        route_with_video_gate(
            _frozen_gate(0.0),
            **batched,
            threshold=0.5,
            configured_video_steps=10,
        )


def test_routing_summary_reports_overall_and_per_route_telemetry():
    decisions = [
        _decision("w", gate_latency_s=1.0),
        _decision("wo", gate_latency_s=3.0),
        _decision("w", gate_latency_s=5.0),
    ]

    summary = summarize_routing_telemetry(
        decisions,
        policy_latencies_s=[2.0, 4.0, 8.0],
    )

    assert summary["counts"] == {"total": 3, "wo": 1, "w": 2}
    assert summary["with_rate"] == pytest.approx(2 / 3)
    assert summary["effective_video_steps"] == {
        "total": 20,
        "mean": pytest.approx(20 / 3),
    }
    assert summary["latency_s"]["gate"] == {
        "mean": 3.0,
        "p50": 3.0,
        "p95": pytest.approx(4.8),
    }
    assert summary["latency_s"]["policy"] == {
        "mean": pytest.approx(14 / 3),
        "p50": 4.0,
        "p95": pytest.approx(7.6),
    }
    assert summary["by_route"]["wo"] == {
        "count": 1,
        "effective_video_steps": {"total": 0, "mean": 0.0},
        "latency_s": {
            "gate": {"mean": 3.0, "p50": 3.0, "p95": 3.0},
            "policy": {"mean": 4.0, "p50": 4.0, "p95": 4.0},
        },
    }
    assert summary["by_route"]["w"]["count"] == 2
    assert summary["by_route"]["w"]["effective_video_steps"] == {
        "total": 20,
        "mean": 10.0,
    }
    assert summary["by_route"]["w"]["latency_s"]["gate"] == {
        "mean": 3.0,
        "p50": 3.0,
        "p95": pytest.approx(4.8),
    }
    assert summary["by_route"]["w"]["latency_s"]["policy"] == {
        "mean": 5.0,
        "p50": 5.0,
        "p95": pytest.approx(7.7),
    }
    json.dumps(summary, allow_nan=False)


def test_routing_summary_is_total_for_empty_routes_and_rejects_bad_records():
    empty = summarize_routing_telemetry([], policy_latencies_s=[])
    assert empty == {
        "counts": {"total": 0, "wo": 0, "w": 0},
        "with_rate": 0.0,
        "effective_video_steps": {"total": 0, "mean": 0.0},
        "latency_s": {
            "gate": {"mean": None, "p50": None, "p95": None},
            "policy": {"mean": None, "p50": None, "p95": None},
        },
        "by_route": {
            route: {
                "count": 0,
                "effective_video_steps": {"total": 0, "mean": 0.0},
                "latency_s": {
                    "gate": {"mean": None, "p50": None, "p95": None},
                    "policy": {"mean": None, "p50": None, "p95": None},
                },
            }
            for route in ("wo", "w")
        },
    }

    with pytest.raises(ValueError, match="equal length"):
        summarize_routing_telemetry(
            [_decision("wo", gate_latency_s=0.1)],
            policy_latencies_s=[],
        )
    with pytest.raises(ValueError, match="finite and non-negative"):
        summarize_routing_telemetry(
            [_decision("wo", gate_latency_s=0.1)],
            policy_latencies_s=[float("nan")],
        )
    with pytest.raises(ValueError, match="video steps.*'wo'"):
        summarize_routing_telemetry(
            [
                _decision(
                    "wo",
                    gate_latency_s=0.1,
                    actual_video_steps=1,
                )
            ],
            policy_latencies_s=[0.2],
        )


@pytest.mark.parametrize(
    ("inference_mode", "expected_steps"),
    [("wo", 0), ("w", 7)],
)
def test_static_router_uses_inference_mode_and_null_gate_telemetry(
    inference_mode,
    expected_steps,
):
    router = BinaryVideoRouter(
        routing_mode="static",
        inference_mode=inference_mode,
        configured_video_steps=7,
    )

    decision = router.route()

    assert decision == {
        "logit": None,
        "probability": None,
        "selected_mode": inference_mode,
        "gate_latency_s": None,
        "configured_video_steps": 7,
        "actual_video_steps": expected_steps,
    }
    json.dumps(decision, allow_nan=False)


def test_gate_router_routes_one_query_with_exactly_one_gate_forward():
    gate = _frozen_gate(0.0)
    forward_calls = 0

    def count_forward(_module, _args, _output):
        nonlocal forward_calls
        forward_calls += 1

    handle = gate.register_forward_hook(count_forward)
    try:
        router = BinaryVideoRouter(
            routing_mode="gate",
            gate=gate,
            threshold=0.5,
            configured_video_steps=10,
            clock=lambda: 1.0,
        )
        decision = router.route(**_inputs())
    finally:
        handle.remove()

    assert forward_calls == 1
    assert decision["selected_mode"] == "w"
    assert decision["actual_video_steps"] == 10


def test_gate_router_requires_all_four_current_only_inputs():
    router = BinaryVideoRouter(
        routing_mode="gate",
        gate=_frozen_gate(0.0),
        threshold=0.5,
        configured_video_steps=10,
    )

    with pytest.raises(ValueError, match="context_mask"):
        router.route(
            input_image=_inputs()["input_image"],
            context=_inputs()["context"],
            proprio=_inputs()["proprio"],
        )


def test_random_router_is_seeded_private_and_draws_once_per_query():
    global_state = random.getstate()
    first = BinaryVideoRouter(
        routing_mode="random",
        random_seed=123,
        random_video_probability=0.5,
        configured_video_steps=10,
    )
    second = BinaryVideoRouter(
        routing_mode="random",
        random_seed=123,
        random_video_probability=0.5,
        configured_video_steps=10,
    )

    first_modes = [first.route()["selected_mode"] for _ in range(12)]
    second_modes = [second.route()["selected_mode"] for _ in range(12)]

    assert first_modes == second_modes
    assert {"wo", "w"}.issubset(first_modes)
    assert random.getstate() == global_state
    extra_decision = first.route()
    assert extra_decision["logit"] is None
    assert extra_decision["probability"] is None
    assert extra_decision["gate_latency_s"] is None


@pytest.mark.parametrize(
    ("probability", "expected_mode"),
    [(0.0, "wo"), (1.0, "w")],
)
def test_random_router_probability_boundaries(probability, expected_mode):
    router = BinaryVideoRouter(
        routing_mode="random",
        random_seed=0,
        random_video_probability=probability,
        configured_video_steps=10,
    )
    assert {router.route()["selected_mode"] for _ in range(20)} == {
        expected_mode
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"routing_mode": "unknown", "configured_video_steps": 10},
        {"routing_mode": "static", "configured_video_steps": 10},
        {
            "routing_mode": "gate",
            "configured_video_steps": 10,
            "gate": None,
        },
        {
            "routing_mode": "random",
            "configured_video_steps": 10,
            "random_video_probability": -0.1,
        },
        {
            "routing_mode": "random",
            "configured_video_steps": 10,
            "random_video_probability": 1.1,
        },
    ],
)
def test_router_rejects_invalid_mode_specific_configuration(kwargs):
    with pytest.raises((TypeError, ValueError)):
        BinaryVideoRouter(**kwargs)


def test_summary_accepts_static_and_random_null_gate_telemetry():
    decisions = [
        BinaryVideoRouter(
            routing_mode="static",
            inference_mode="wo",
            configured_video_steps=10,
        ).route(),
        BinaryVideoRouter(
            routing_mode="static",
            inference_mode="w",
            configured_video_steps=10,
        ).route(),
    ]

    summary = summarize_routing_telemetry(
        decisions,
        policy_latencies_s=[0.1, 0.3],
    )

    assert summary["counts"] == {"total": 2, "wo": 1, "w": 1}
    assert summary["with_rate"] == 0.5
    assert summary["effective_video_steps"] == {"total": 10, "mean": 5.0}
    assert summary["latency_s"]["gate"] == {
        "mean": None,
        "p50": None,
        "p95": None,
    }
    assert summary["latency_s"]["policy"] == {
        "mean": 0.2,
        "p50": 0.2,
        "p95": pytest.approx(0.29),
    }
    json.dumps(summary, allow_nan=False)


def test_summary_rejects_partially_null_gate_telemetry():
    bad = _decision("wo", gate_latency_s=0.1)
    bad["logit"] = None
    with pytest.raises(ValueError, match="all Gate telemetry fields or none"):
        summarize_routing_telemetry([bad], policy_latencies_s=[0.2])
