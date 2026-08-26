import pytest
import torch

from fastwam.gating.inference import run_paired_action_rollouts
from fastwam.gating.labels import paired_gate_label_statistics
from fastwam.models.wan22.fastwam import FastWAM
from fastwam.models.wan22.fastwam_joint import FastWAMJoint
from fastwam.models.wan22.fastwam_unified_shared import FastWAMUnifiedShared


class SpyAlignedModel(torch.nn.Module):
    def __init__(self, *, action_dim: int = 3):
        super().__init__()
        self.action_dim = action_dim
        self.calls: list[dict] = []
        self.output_mutation: str | None = None

    def infer_action_mode(self, **kwargs):
        self.calls.append(dict(kwargs, grad_enabled=torch.is_grad_enabled()))
        horizon = kwargs["action_horizon"]
        action = torch.full(
            (horizon, self.action_dim),
            2.0 if kwargs["inference_mode"] == "wo" else 1.0,
        )
        if self.output_mutation == "shape":
            action = action[:-1]
        elif self.output_mutation == "nan":
            action[0, 0] = torch.nan
        return {"action": action}


def _sample():
    return {
        "video": torch.randn(3, 5, 16, 32),
        "action": torch.zeros(4, 3),
        "proprio": torch.randn(4, 8),
        "context": torch.randn(6, 16),
        "context_mask": torch.tensor([True, True, True, False, False, False]),
        "gate_context_mask": torch.tensor(
            [True, True, True, False, False, False]
        ),
        "future_secret": torch.tensor([123.0]),
    }


def test_paired_runner_dispatches_exact_same_seed_wo_and_w_calls():
    model = SpyAlignedModel()
    sample = _sample()
    seeds = [7, 19, 23]

    result = run_paired_action_rollouts(model, sample, seeds=seeds)

    assert result.action_wo.shape == (3, 1, 4, 3)
    assert result.action_w.shape == (3, 1, 4, 3)
    assert result.seeds == tuple(seeds)
    assert len(model.calls) == 2 * len(seeds)
    for pair_index, seed in enumerate(seeds):
        wo = model.calls[2 * pair_index]
        w = model.calls[2 * pair_index + 1]
        assert wo["inference_mode"] == "wo"
        assert w["inference_mode"] == "w"
        assert wo["seed"] == w["seed"] == seed
        assert "num_video_frames" not in wo
        assert w["num_video_frames"] == sample["video"].shape[1]
        assert "action" not in wo and "action" not in w
        assert "future_secret" not in wo and "future_secret" not in w
        assert wo["input_image"] is w["input_image"]
        assert torch.equal(wo["input_image"], sample["video"][:, 0])
        assert torch.equal(wo["proprio"], sample["proprio"][0])
        assert wo["context_mask"] is sample["context_mask"]
        assert wo["grad_enabled"] is False and w["grad_enabled"] is False


def test_paired_runner_output_feeds_padding_aware_label_math():
    sample = _sample()
    rollouts = run_paired_action_rollouts(
        SpyAlignedModel(),
        sample,
        seeds=[3, 5],
    )

    stats = paired_gate_label_statistics(
        action_wo=rollouts.action_wo,
        action_w=rollouts.action_w,
        target_action=sample["action"].unsqueeze(0),
        action_is_pad=torch.zeros(1, 4, dtype=torch.bool),
    )

    assert stats.e0.item() == 4.0
    assert stats.e10.item() == 1.0
    assert stats.label.item() is True


def test_real_wo_and_w_implementations_sample_identical_initial_action_noise(
    monkeypatch,
):
    class StopAfterNoise(RuntimeError):
        pass

    class TinyNoiseProbe:
        def __init__(self):
            self.video_expert = type(
                "VideoExpert",
                (),
                {"video_attention_mask_mode": "first_frame_causal"},
            )()
            self.action_expert = type("ActionExpert", (), {"action_dim": 3})()
            self.vae = type(
                "VAE",
                (),
                {
                    "temporal_downsample_factor": 4,
                    "upsampling_factor": 8,
                    "model": type("VAEModel", (), {"z_dim": 2})(),
                },
            )()
            self.device = torch.device("cpu")
            self.torch_dtype = torch.float32

        def eval(self):
            return self

        def _check_resize_height_width(self, height, width, frames):
            return height, width, frames

        def _encode_input_image_latents_tensor(self, **kwargs):
            del kwargs
            raise StopAfterNoise

    original_randn = torch.randn
    samples = []

    def recording_randn(*args, **kwargs):
        value = original_randn(*args, **kwargs)
        samples.append(value.detach().clone())
        return value

    monkeypatch.setattr(torch, "randn", recording_randn)
    model = TinyNoiseProbe()
    common = {
        "prompt": None,
        "input_image": torch.zeros(3, 16, 16),
        "action_horizon": 4,
        "seed": 12345,
        "rand_device": "cpu",
    }
    with pytest.raises(StopAfterNoise):
        FastWAM.infer_action(model, **common)
    wo_action_noise = samples[-1]
    samples.clear()
    with pytest.raises(StopAfterNoise):
        FastWAMJoint.infer_action(model, **common, num_video_frames=5)
    assert len(samples) == 2
    w_action_noise = samples[-1]

    assert torch.equal(wo_action_noise, w_action_noise)


@pytest.mark.parametrize(
    ("seeds", "steps", "message"),
    [
        ([1], 10, "2--4"),
        ([1, 1], 10, "unique"),
        ([1, 2, 3, 4, 5], 10, "2--4"),
        ([1, 2], 9, "exactly 10"),
    ],
)
def test_paired_runner_rejects_invalid_formal_rollout_contract(
    seeds,
    steps,
    message,
):
    with pytest.raises((TypeError, ValueError), match=message):
        run_paired_action_rollouts(
            SpyAlignedModel(),
            _sample(),
            seeds=seeds,
            num_inference_steps=steps,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("future_frames", r"T % 4"),
        ("spatial", "multiples of 16"),
        ("horizon", "horizons must match"),
        ("transition_ratio", "divisible by video transitions"),
        ("mask", r"shape \[L\]"),
        ("non_finite", "non-finite"),
    ],
)
def test_paired_runner_rejects_invalid_sample(mutation, message):
    sample = _sample()
    if mutation == "future_frames":
        sample["video"] = sample["video"][:, :4]
    elif mutation == "spatial":
        sample["video"] = sample["video"][:, :, :, :31]
    elif mutation == "horizon":
        sample["proprio"] = sample["proprio"][:-1]
    elif mutation == "transition_ratio":
        sample["action"] = torch.zeros(5, 3)
        sample["proprio"] = torch.zeros(5, 8)
    elif mutation == "mask":
        sample["context_mask"] = sample["context_mask"][:-1]
    elif mutation == "non_finite":
        sample["context"][0, 0] = torch.inf

    with pytest.raises((TypeError, ValueError), match=message):
        run_paired_action_rollouts(
            SpyAlignedModel(),
            sample,
            seeds=[1, 2],
        )


@pytest.mark.parametrize(("mutation", "message"), [("shape", "shape"), ("nan", "non-finite")])
def test_paired_runner_rejects_invalid_model_output(mutation, message):
    model = SpyAlignedModel()
    model.output_mutation = mutation

    with pytest.raises((TypeError, ValueError), match=message):
        run_paired_action_rollouts(model, _sample(), seeds=[1, 2])


@pytest.mark.parametrize("raise_error", [False, True])
def test_wo_dispatch_forces_wo_and_restores_previous_mode(monkeypatch, raise_error):
    model = torch.nn.Module.__new__(FastWAMUnifiedShared)
    torch.nn.Module.__init__(model)
    model._unified_inference_mode = "w"

    def fake_infer_action(instance, **kwargs):
        assert instance._unified_inference_mode == "wo"
        if raise_error:
            raise RuntimeError("expected test failure")
        return kwargs

    monkeypatch.setattr(FastWAM, "infer_action", fake_infer_action)
    if raise_error:
        with pytest.raises(RuntimeError, match="expected test failure"):
            model.infer_action_without_video(marker=7)
    else:
        assert model.infer_action_without_video(marker=7) == {"marker": 7}
    assert model._unified_inference_mode == "w"
