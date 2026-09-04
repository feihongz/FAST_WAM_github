"""CPU-only execution tests for LIBERO's dynamic video-routing evaluator.

These tests deliberately invoke the evaluator's real helper functions while
replacing only the simulator/model-dependent pieces.  This keeps the routing
contract testable without launching MuJoCo or loading a WAN checkpoint.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
LIBERO_ROOT = Path("/root/feihong/FastWAM/third_party/LIBERO")

# ``eval_libero_single.py`` historically imports ``action_ensembler`` as a
# script-local module.  Preserve its real runtime import layout in this test
# instead of changing evaluator production code for a unit-test convenience.
os.environ.setdefault("FASTWAM_LIBERO_ROOT", str(LIBERO_ROOT))
LIBERO_EVAL_DIR = str(REPO_ROOT / "experiments" / "libero")
if LIBERO_EVAL_DIR not in sys.path:
    sys.path.insert(0, LIBERO_EVAL_DIR)

from experiments.libero import eval_libero_single as evaluator
from fastwam.gating.eval_runtime import BoundPromptContext
from fastwam.gating.routing import BinaryVideoRouter
from fastwam.models.video_gate import BinaryVideoGate


class _ToUnitInterval:
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return image.to(dtype=torch.float32).div(255.0)


class _ScaleAndOffset:
    def __init__(self, scale: float, offset: float) -> None:
        self.scale = scale
        self.offset = offset

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return image.mul(self.scale).add(self.offset)


class _Processor:
    """Minimum processor surface used by the two evaluator helpers."""

    def __init__(self) -> None:
        self.num_output_cameras = 2
        self.shape_meta = {
            "images": [
                {"key": "image", "shape": [3, 4, 4]},
                {"key": "wrist_image", "shape": [3, 4, 4]},
            ]
        }
        self.val_transforms = {
            "image": [_ToUnitInterval(), _ScaleAndOffset(0.5, 0.25)],
            "wrist_image": [_ToUnitInterval(), _ScaleAndOffset(0.75, 0.0)],
        }


class _SpyModel(torch.nn.Module):
    def __init__(self, *, action_dim: int = 3) -> None:
        super().__init__()
        self.torch_dtype = torch.float32
        self.action_dim = action_dim
        self.calls: list[dict] = []

    def infer_action_mode(self, **kwargs):
        self.calls.append(dict(kwargs))
        return {
            "action": torch.full(
                (kwargs["action_horizon"], self.action_dim),
                0.5,
                dtype=torch.float32,
            )
        }


def _cfg() -> object:
    return OmegaConf.create(
        {
            "seed": 17,
            "data": {
                "train": {
                    "concat_multi_camera": "horizontal",
                    "video_size": [4, 8],
                    "num_frames": 5,
                    "action_video_freq_ratio": 2,
                }
            },
            "EVALUATION": {
                "inference_mode": "wo",
                "num_inference_steps": 10,
                "negative_prompt": "",
                "text_cfg_scale": 1.0,
                "sigma_shift": None,
                "rand_device": "cpu",
                "tiled": False,
                "visualize_future_video": False,
                "timing_enabled": False,
            },
        }
    )


def _images() -> dict[str, np.ndarray]:
    return {
        "image": np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3),
        "wrist_image": np.full((4, 4, 3), 128, dtype=np.uint8),
    }


def _manual_gate_image(
    imgs: dict[str, np.ndarray],
    processor: _Processor,
) -> torch.Tensor:
    camera_tensors = []
    for meta in processor.shape_meta["images"]:
        image = torch.from_numpy(np.ascontiguousarray(imgs[meta["key"]])).permute(
            2, 0, 1
        ).unsqueeze(0)
        for transform in processor.val_transforms[meta["key"]]:
            image = transform(image)
        camera_tensors.append(image)
    return torch.cat(camera_tensors, dim=-1).mul(2.0).sub(1.0)


def _patch_model_input_and_action_decode(monkeypatch, imgs: dict[str, np.ndarray]) -> None:
    monkeypatch.setattr(
        evaluator,
        "_obs_to_model_input",
        lambda *args, **kwargs: (
            torch.zeros((1, 3, 4, 8), dtype=torch.float32),
            torch.zeros((1, 8), dtype=torch.float32),
            imgs,
        ),
    )
    monkeypatch.setattr(
        evaluator,
        "_denormalize_action",
        lambda action, processor: action.unsqueeze(0)
        if action.ndim == 2
        else action,
    )


@pytest.mark.parametrize("mode", ["wo", "w"])
def test_static_route_executes_exactly_one_matching_action_branch(
    monkeypatch,
    mode: str,
) -> None:
    """Static evaluation remains a single branch dispatch with correct NFE."""

    imgs = _images()
    _patch_model_input_and_action_decode(monkeypatch, imgs)
    model = _SpyModel()
    router = BinaryVideoRouter(
        routing_mode="static",
        inference_mode=mode,
        configured_video_steps=10,
    )

    _, returned_imgs, predicted_video, metrics = evaluator._predict_action_chunk(
        {},
        "move object",
        model,
        _Processor(),
        _cfg(),
        action_horizon=2,
        input_w=8,
        input_h=4,
        model_device="cpu",
        router=router,
        prompt_context=None,
    )

    assert returned_imgs is imgs
    assert predicted_video is None
    assert len(model.calls) == 1
    call = model.calls[0]
    assert call["inference_mode"] == mode
    assert call["prompt"] == evaluator.DEFAULT_PROMPT.format(task="move object")
    if mode == "wo":
        assert "num_video_frames" not in call
        assert metrics["actual_video_steps"] == 0
    else:
        assert call["num_video_frames"] == 3
        assert metrics["actual_video_steps"] == 10


def test_gate_execution_uses_original_mask_for_gate_and_all_ones_for_wam(
    monkeypatch,
) -> None:
    """Cached text preserves padding for Gate while WAM receives its contract mask."""

    imgs = _images()
    processor = _Processor()
    _patch_model_input_and_action_decode(monkeypatch, imgs)
    model = _SpyModel()

    gate = BinaryVideoGate(
        proprio_dim=8,
        context_dim=4,
        cnn_channels=(2, 2, 2),
        context_feature_dim=2,
        proprio_hidden_dim=2,
        proprio_feature_dim=2,
        fusion_hidden_dim=2,
    )
    with torch.no_grad():
        for parameter in gate.parameters():
            parameter.zero_()
        gate.logit_head[-1].bias.fill_(10.0)
    gate.eval().requires_grad_(False)

    seen_gate_inputs: dict[str, torch.Tensor] = {}

    def capture_gate_inputs(module, args, kwargs):
        del module, args
        seen_gate_inputs.update(
            {key: value.detach().clone() for key, value in kwargs.items()}
        )

    hook = gate.register_forward_pre_hook(capture_gate_inputs, with_kwargs=True)
    try:
        context = torch.tensor(
            [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]]
        )
        original_gate_mask = torch.tensor([[True, False, True]])
        wam_mask = torch.ones_like(original_gate_mask)
        prompt_context = BoundPromptContext(
            context=context,
            gate_context_mask=original_gate_mask,
            model_context_mask=wam_mask,
            identity={"test": True},
        )
        router = BinaryVideoRouter(
            routing_mode="gate",
            configured_video_steps=10,
            gate=gate,
            threshold=0.5,
        )

        _, _, _, metrics = evaluator._predict_action_chunk(
            {},
            "move object",
            model,
            processor,
            _cfg(),
            action_horizon=2,
            input_w=8,
            input_h=4,
            model_device="cpu",
            router=router,
            prompt_context=prompt_context,
        )
    finally:
        hook.remove()

    assert metrics["selected_mode"] == "w"
    assert torch.equal(seen_gate_inputs["context"], context)
    assert torch.equal(seen_gate_inputs["context_mask"], original_gate_mask)
    assert torch.equal(seen_gate_inputs["input_image"], _manual_gate_image(imgs, processor))
    assert len(model.calls) == 1
    wam_call = model.calls[0]
    assert wam_call["prompt"] is None
    assert wam_call["context"] is context
    assert wam_call["context_mask"] is wam_mask
    assert torch.equal(wam_call["context_mask"], torch.ones_like(original_gate_mask))
    assert wam_call["inference_mode"] == "w"
    assert wam_call["num_video_frames"] == 3


def test_gate_image_preprocessing_matches_processor_val_transform_chain() -> None:
    """The Gate sees the exact per-camera validation transform result."""

    imgs = _images()
    processor = _Processor()

    actual = evaluator._obs_to_gate_input(
        imgs,
        cfg=_cfg(),
        processor=processor,
        device="cpu",
    )

    expected = _manual_gate_image(imgs, processor)
    assert actual.dtype == torch.float32
    assert actual.shape == (1, 3, 4, 8)
    assert torch.equal(actual, expected)
