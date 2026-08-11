import pytest

from experiments.libero.gate.router import (
    GateRequest,
    GateRouter,
    validate_gate_eval_settings,
)


def _request(*, episode_idx: int = 0, replan_idx: int = 0) -> GateRequest:
    return GateRequest(
        task_key="libero_10/3",
        episode_idx=episode_idx,
        replan_idx=replan_idx,
    )


def test_fixed_routes_match_prefix_endpoints():
    fixed_0 = GateRouter(mode="fixed_0", num_inference_steps=10)
    fixed_full = GateRouter(mode="fixed_full", num_inference_steps=10)

    assert fixed_0.select(_request()).selected_n == 0
    assert fixed_full.select(_request()).selected_n == 10


def test_random_route_is_deterministic_per_replan():
    router = GateRouter(
        mode="random",
        num_inference_steps=10,
        full_probability=0.3,
        seed=123,
    )
    first = router.select(_request(episode_idx=4, replan_idx=7))
    second = router.select(_request(episode_idx=4, replan_idx=7))

    assert first == second
    assert first.selected_n in {0, 10}


def test_random_route_matches_requested_long_run_fraction():
    router = GateRouter(
        mode="random",
        num_inference_steps=10,
        full_probability=0.3,
        seed=42,
    )
    decisions = [
        router.select(_request(episode_idx=i // 20, replan_idx=i % 20))
        for i in range(5000)
    ]
    full_fraction = sum(item.selected_n == 10 for item in decisions) / len(decisions)

    assert full_fraction == pytest.approx(0.3, abs=0.02)


def test_random_probability_endpoints_are_exact():
    never = GateRouter(mode="random", num_inference_steps=10, full_probability=0.0)
    always = GateRouter(mode="random", num_inference_steps=10, full_probability=1.0)

    for replan_idx in range(100):
        request = _request(replan_idx=replan_idx)
        assert never.select(request).selected_n == 0
        assert always.select(request).selected_n == 10


def test_learned_mode_is_reserved_until_selector_is_wired():
    with pytest.raises(ValueError, match="requires a learned_selector"):
        GateRouter(mode="learned", num_inference_steps=10)


def test_learned_selector_uses_threshold():
    router = GateRouter(
        mode="learned",
        num_inference_steps=10,
        threshold=0.25,
        learned_selector=lambda request: 0.4 if request.replan_idx == 1 else 0.1,
    )

    assert router.select(_request(replan_idx=0)).selected_n == 0
    assert router.select(_request(replan_idx=1)).selected_n == 10


@pytest.mark.parametrize("probability", [-0.1, 1.1])
def test_invalid_random_probability_is_rejected(probability):
    with pytest.raises(ValueError, match="full_probability"):
        GateRouter(
            mode="random",
            num_inference_steps=10,
            full_probability=probability,
        )


def test_gate_requires_prefix_inference():
    with pytest.raises(ValueError, match="requires inference_mode=prefix"):
        validate_gate_eval_settings(
            enabled=True,
            inference_mode="wo",
            visualize_future_video=False,
        )


def test_gate_rejects_future_video_visualization():
    with pytest.raises(ValueError, match="incompatible with Gate routing"):
        validate_gate_eval_settings(
            enabled=True,
            inference_mode="prefix",
            visualize_future_video=True,
        )


def test_disabled_gate_preserves_existing_eval_modes():
    validate_gate_eval_settings(
        enabled=False,
        inference_mode="w",
        visualize_future_video=True,
    )
