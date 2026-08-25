import pytest
import torch
from fastwam.alignment.rollout import RolloutStep, perturb_with_shared_noise, shared_action_noise, validate_solver_panel

def test_shared_noise_is_reproducible_and_reused():
    a = shared_action_noise(shape=(2, 3), seed=7)
    b = shared_action_noise(shape=(2, 3), seed=7)
    assert torch.equal(a, b)
    assert torch.equal(perturb_with_shared_noise(torch.zeros_like(a), a, 0.5), a * 0.5)

def test_perturb_matches_flow_matching_scheduler_for_nonzero_latent():
    latent = torch.full((2,), 2.0)
    noise = torch.full((2,), 10.0)
    assert torch.equal(
        perturb_with_shared_noise(latent, noise, 0.25),
        torch.full((2,), 4.0),
    )

def test_solver_panel_and_cache_step_contract():
    validate_solver_panel(torch.linspace(1.0, 0.0, 10))
    assert RolloutStep("sample-1", 9, 0.25, 3, "basehash").k == 9

def test_solver_contract_rejects_bad_inputs():
    with pytest.raises(ValueError):
        validate_solver_panel(torch.arange(9, dtype=torch.float32))
    with pytest.raises(ValueError):
        RolloutStep("sample-1", 10, 0.25, 3, "basehash")
    with pytest.raises(ValueError):
        perturb_with_shared_noise(torch.zeros(2), torch.zeros(3), 1.0)
    with pytest.raises(ValueError):
        perturb_with_shared_noise(torch.zeros(2), torch.zeros(2), 1.1)
