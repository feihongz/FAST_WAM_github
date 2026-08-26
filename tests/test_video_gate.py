import inspect
import json

import pytest
import torch

from fastwam.models.video_gate import BinaryVideoGate, VideoGateConfig


PROPRIO_DIM = 16


def _inputs(
    *,
    batch_size: int = 2,
    sequence_length: int = 5,
) -> dict[str, torch.Tensor]:
    return {
        "input_image": torch.randn(batch_size, 3, 32, 40),
        "context": torch.randn(batch_size, sequence_length, 4096),
        "context_mask": torch.ones(
            batch_size,
            sequence_length,
            dtype=torch.bool,
        ),
        "proprio": torch.randn(batch_size, PROPRIO_DIM),
    }


def test_video_gate_output_config_and_parameter_count():
    gate = BinaryVideoGate(proprio_dim=PROPRIO_DIM)

    logit = gate(**_inputs())
    serialized = json.loads(json.dumps(gate.config()))
    restored = BinaryVideoGate.from_config(serialized)

    assert logit.shape == (2,)
    assert restored.config() == serialized
    assert gate.gate_config == VideoGateConfig(proprio_dim=PROPRIO_DIM)
    assert gate.parameter_count() == sum(
        parameter.numel() for parameter in gate.parameters()
    )
    assert gate.parameter_count(trainable_only=True) == gate.parameter_count()
    assert gate.parameter_count() == 659_489
    assert 625_000 <= gate.parameter_count() <= 700_000
    logit.sum().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in gate.parameters()
    )


def test_video_gate_uses_boolean_masked_context_mean():
    torch.manual_seed(19)
    gate = BinaryVideoGate(proprio_dim=PROPRIO_DIM).eval()
    inputs = _inputs(batch_size=2, sequence_length=4)
    inputs["context_mask"] = torch.tensor(
        [[True, True, False, False], [True, False, True, False]]
    )
    changed = inputs["context"].clone()
    changed[~inputs["context_mask"]] = 1_000_000.0

    with torch.no_grad():
        original_logit = gate(**inputs)
        changed_logit = gate(**{**inputs, "context": changed})

    assert torch.equal(original_logit, changed_logit)

    single = _inputs(batch_size=1, sequence_length=3)
    single["context"][:, 1] = single["context"][:, 0]
    single["context_mask"] = torch.tensor([[True, False, False]])
    duplicated_mask = torch.tensor([[True, True, False]])
    with torch.no_grad():
        single_logit = gate(**single)
        duplicated_logit = gate(
            **{**single, "context_mask": duplicated_mask}
        )

    assert torch.equal(single_logit, duplicated_logit)


def test_video_gate_casts_cached_bfloat16_context_to_parameter_dtype():
    gate = BinaryVideoGate(proprio_dim=PROPRIO_DIM).eval()
    inputs = _inputs(batch_size=2)
    inputs["context"] = inputs["context"].to(dtype=torch.bfloat16)

    with torch.no_grad():
        logits = gate(**inputs)

    assert logits.dtype == gate.context_encoder.weight.dtype == torch.float32
    assert torch.isfinite(logits).all()


def test_video_gate_rejects_temporal_or_non_rgb_image():
    gate = BinaryVideoGate(proprio_dim=PROPRIO_DIM)
    inputs = _inputs()

    with pytest.raises(ValueError, match=r"input_image.*\[B,3,H,W\]"):
        gate(
            **{
                **inputs,
                "input_image": torch.randn(2, 2, 3, 32, 40),
            }
        )
    with pytest.raises(ValueError, match=r"input_image.*\[B,3,H,W\]"):
        gate(
            **{
                **inputs,
                "input_image": torch.randn(2, 6, 32, 40),
            }
        )


def test_video_gate_rejects_invalid_context_and_mask_contract():
    gate = BinaryVideoGate(proprio_dim=PROPRIO_DIM)
    inputs = _inputs()

    with pytest.raises(ValueError, match="context hidden dimension"):
        gate(**{**inputs, "context": torch.randn(2, 5, 4095)})
    with pytest.raises(ValueError, match=r"context_mask.*\[B,L\]"):
        gate(**{**inputs, "context_mask": torch.ones(2, 4, dtype=torch.bool)})
    with pytest.raises(ValueError, match="context_mask must have bool dtype"):
        gate(**{**inputs, "context_mask": torch.ones(2, 5)})
    with pytest.raises(ValueError, match="at least one token"):
        gate(
            **{
                **inputs,
                "context_mask": torch.tensor(
                    [
                        [True, True, True, True, True],
                        [False, False, False, False, False],
                    ]
                ),
            }
        )


def test_video_gate_rejects_invalid_proprio_shape_dimension_and_batch():
    gate = BinaryVideoGate(proprio_dim=PROPRIO_DIM)
    inputs = _inputs()

    with pytest.raises(ValueError, match=r"proprio.*\[B,D\]"):
        gate(**{**inputs, "proprio": torch.randn(2, 1, PROPRIO_DIM)})
    with pytest.raises(ValueError, match="proprio dimension"):
        gate(**{**inputs, "proprio": torch.randn(2, PROPRIO_DIM - 1)})
    with pytest.raises(ValueError, match="batch sizes must match"):
        gate(**{**inputs, "proprio": torch.randn(1, PROPRIO_DIM)})
    with pytest.raises(ValueError, match="dimensions must be positive"):
        BinaryVideoGate(proprio_dim=0)


def test_video_gate_forward_api_has_no_future_or_generated_inputs():
    parameters = inspect.signature(BinaryVideoGate.forward).parameters
    assert tuple(parameters) == (
        "self",
        "input_image",
        "context",
        "context_mask",
        "proprio",
    )
    assert all(
        "future" not in name and "error" not in name and "video_quality" not in name
        for name in parameters
    )
    assert not any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )

    with pytest.raises(TypeError, match="future_video"):
        BinaryVideoGate(proprio_dim=PROPRIO_DIM)(
            **_inputs(),
            future_video=torch.randn(2, 3, 3, 32, 40),
        )
