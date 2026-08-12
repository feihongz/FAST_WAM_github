from __future__ import annotations

import hashlib
import json

import pytest
import torch

from experiments.libero.gate.demo_utility import (
    LIBERO_DATASET_TO_SUITE,
    collect_paired_utility,
    extract_current_state,
    parse_sample_identity,
    stable_sample_seed,
    valid_chunk_mse,
)


def _metadata(**overrides):
    result = {
        "dataset_name": "libero_10_no_noops_lerobot",
        "episode_index": 7,
        "frame_index": 13,
        "task_index": 2,
        "task": "put the mug on the plate",
    }
    result.update(overrides)
    return result


def _sample(*, future_value: float = 0.0):
    video = torch.zeros(3, 9, 8, 12)
    video[:, 0] = 0.25
    video[:, 1:] = future_value
    return {
        "video": video,
        "proprio": torch.arange(32, dtype=torch.float32).reshape(4, 8),
        "context": torch.arange(30, dtype=torch.float32).reshape(5, 6),
        "context_mask": torch.tensor([True, True, True, False, False]),
        "action": torch.ones(4, 2),
        "action_is_pad": torch.tensor([False, False, False, True]),
        "metadata": _metadata(),
    }


@pytest.mark.parametrize(
    ("dataset_name", "suite"),
    list(LIBERO_DATASET_TO_SUITE.items()),
)
def test_explicit_libero_dataset_suite_mapping(dataset_name, suite):
    identity = parse_sample_identity(_metadata(dataset_name=dataset_name))

    assert identity.dataset_name == dataset_name
    assert identity.dataset_id == dataset_name
    assert identity.suite == suite
    assert identity.sample_id == f"{dataset_name}/episode_000007/frame_000013"


def test_identity_validates_dataset_suite_task_and_sample_id():
    identity = parse_sample_identity(
        {
            **_metadata(),
            "dataset_id": "/another/root/libero_10_no_noops_lerobot",
            "suite": "libero_10",
        },
        task_by_index={2: "put the mug on the plate"},
    )
    assert identity.task_index == 2
    assert identity.dataset_id == "libero_10_no_noops_lerobot"

    with pytest.raises(ValueError, match="unknown LIBERO dataset"):
        parse_sample_identity(_metadata(dataset_name="libero_typo_no_noops_lerobot"))
    with pytest.raises(ValueError, match="suite mismatch"):
        parse_sample_identity(_metadata(suite="libero_goal"))
    with pytest.raises(ValueError, match="task string/index mismatch"):
        parse_sample_identity(
            _metadata(),
            task_by_index={2: "a different task"},
        )
    with pytest.raises(ValueError, match="sample_id mismatch"):
        parse_sample_identity(_metadata(sample_id="wrong/id"))


def test_stable_sample_seed_is_sha256_based_and_path_independent():
    identity = parse_sample_identity(_metadata())
    payload = "\0".join(
        ["123", "libero_10_no_noops_lerobot", "7", "13"]
    ).encode("utf-8")
    expected = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)

    first = stable_sample_seed(123, identity)
    second = stable_sample_seed(123, "libero_10_no_noops_lerobot", 7, 13)
    rooted = stable_sample_seed(
        123, "/different/root/libero_10_no_noops_lerobot", 7, 13
    )

    assert first == expected
    assert second == expected
    assert rooted == expected
    assert stable_sample_seed(123, identity) == first
    assert stable_sample_seed(124, identity) != first


def test_extract_current_state_never_depends_on_future_frames():
    first = extract_current_state(_sample(future_value=-1.0))
    second = extract_current_state(_sample(future_value=999.0))

    assert first.input_image.shape == (3, 8, 12)
    assert torch.equal(first.input_image, second.input_image)
    assert torch.equal(first.proprio, torch.arange(8, dtype=torch.float32))
    assert first.context.shape == (5, 6)
    assert first.context_mask.dtype == torch.bool
    assert first.source_video_frames == 9

    source = _sample()
    extracted = extract_current_state(source)
    source["video"][:, 0].fill_(123)
    assert torch.all(extracted.input_image == 0.25)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("video", torch.zeros(9, 3, 8, 12), r"\[3,T,H,W\]"),
        ("proprio", torch.zeros(8), r"\[T,D\]"),
        ("context", torch.zeros(5), r"\[L,D\]"),
        ("context_mask", torch.ones(4, dtype=torch.bool), "matching context"),
        ("action", torch.zeros(4), r"\[T,A\]"),
        ("action_is_pad", torch.zeros(3, dtype=torch.bool), "matching action"),
    ],
)
def test_extract_current_state_rejects_bad_shapes(field, value, message):
    sample = _sample()
    sample[field] = value
    with pytest.raises(ValueError, match=message):
        extract_current_state(sample)


def test_valid_chunk_mse_is_per_element_and_excludes_padding():
    prediction = torch.tensor([[1.0, 3.0], [100.0, 100.0], [2.0, 2.0]])
    target = torch.zeros_like(prediction)
    action_is_pad = torch.tensor([False, True, False])

    mse, valid_length = valid_chunk_mse(prediction, target, action_is_pad)

    assert mse == pytest.approx((1.0 + 9.0 + 4.0 + 4.0) / 4.0)
    assert valid_length == 2


def test_valid_chunk_mse_rejects_all_padding_and_shape_mismatch():
    with pytest.raises(ValueError, match="all-padding"):
        valid_chunk_mse(torch.zeros(2, 3), torch.zeros(2, 3), torch.ones(2, dtype=torch.bool))
    with pytest.raises(ValueError, match="identical"):
        valid_chunk_mse(torch.zeros(2, 3), torch.zeros(3, 3), torch.zeros(2, dtype=torch.bool))
    with pytest.raises(ValueError, match=r"must be \[T\]"):
        valid_chunk_mse(torch.zeros(2, 3), torch.zeros(2, 3), torch.zeros(3, dtype=torch.bool))


class _FakePrefixModel:
    def __init__(
        self,
        *,
        reported_prefix_offset: int = 0,
        omit_route_key: str | None = None,
        nonfinite: bool = False,
        reported_custom_prefix=True,
    ):
        self.calls = []
        self.reported_prefix_offset = reported_prefix_offset
        self.omit_route_key = omit_route_key
        self.nonfinite = nonfinite
        self.reported_custom_prefix = reported_custom_prefix

    def infer_action_mode(self, **kwargs):
        self.calls.append(
            {
                "kwargs": dict(kwargs),
                "inference_mode_enabled": torch.is_inference_mode_enabled(),
            }
        )
        prefix = kwargs["video_prefix_steps"]
        action = torch.zeros(kwargs["action_horizon"], 2)
        if prefix == kwargs["num_inference_steps"]:
            action.fill_(1.0)
        if self.nonfinite:
            action[0, 0] = torch.nan
        result = {
            "action": action,
            "video_prefix_steps": prefix + self.reported_prefix_offset,
            "num_inference_steps": kwargs["num_inference_steps"],
            "force_custom_prefix": self.reported_custom_prefix,
        }
        if self.omit_route_key is not None:
            del result[self.omit_route_key]
        return result


def test_paired_collection_changes_only_prefix_and_is_json_safe():
    sample = _sample(future_value=456.0)
    model = _FakePrefixModel()

    record = collect_paired_utility(
        model,
        sample,
        base_seed=123,
        num_inference_steps=10,
        full_prefix_steps=10,
        rand_device="cpu",
    )

    assert len(model.calls) == 2
    n0_call, nfull_call = model.calls
    assert n0_call["inference_mode_enabled"] is True
    assert nfull_call["inference_mode_enabled"] is True
    assert n0_call["kwargs"]["video_prefix_steps"] == 0
    assert nfull_call["kwargs"]["video_prefix_steps"] == 10

    n0_kwargs = n0_call["kwargs"]
    nfull_kwargs = nfull_call["kwargs"]
    assert set(n0_kwargs) == set(nfull_kwargs)
    for key in n0_kwargs:
        if key == "video_prefix_steps":
            continue
        if isinstance(n0_kwargs[key], torch.Tensor):
            assert n0_kwargs[key] is nfull_kwargs[key]
        else:
            assert n0_kwargs[key] == nfull_kwargs[key]

    assert n0_kwargs["inference_mode"] == "prefix"
    assert n0_kwargs["prompt"] is None
    assert n0_kwargs["input_image"].shape == (3, 8, 12)
    assert torch.all(n0_kwargs["input_image"] == 0.25)
    assert n0_kwargs["proprio"].shape == (8,)
    assert n0_kwargs["context"].shape == (5, 6)
    assert n0_kwargs["rand_device"] == "cpu"
    assert n0_kwargs["seed"] == stable_sample_seed(123, parse_sample_identity(_metadata()))

    assert record.e0 == pytest.approx(1.0)
    assert record.efull == pytest.approx(0.0)
    assert record.utility == pytest.approx(1.0)
    assert record.valid_length == 3
    assert record.target_action_shape == (4, 2)
    assert record.n0_route["video_prefix_steps"] == 0
    assert record.nfull_route["video_prefix_steps"] == 10
    assert record.to_dict()["task_id"] == record.identity.task_index
    assert record.to_dict()["episode_id"] == record.identity.episode_index
    assert record.to_dict()["task_id_source"] == "lerobot_task_index"
    assert record.n0_latency_ms >= 0
    assert record.nfull_latency_ms >= 0
    json.dumps(record.to_dict(), sort_keys=True, allow_nan=False)


def test_paired_collection_result_is_unchanged_when_only_future_video_changes():
    first_model = _FakePrefixModel()
    second_model = _FakePrefixModel()

    first = collect_paired_utility(first_model, _sample(future_value=-1.0))
    second = collect_paired_utility(second_model, _sample(future_value=999.0))

    assert first.e0 == second.e0
    assert first.efull == second.efull
    assert first.utility == second.utility
    assert first.input_hashes == second.input_hashes
    assert torch.equal(
        first_model.calls[0]["kwargs"]["input_image"],
        second_model.calls[0]["kwargs"]["input_image"],
    )


def test_input_hashes_change_when_current_condition_or_valid_gt_changes():
    baseline_sample = _sample()
    changed_image_sample = _sample()
    changed_image_sample["video"][:, 0].add_(0.5)
    changed_gt_sample = _sample()
    changed_gt_sample["action"][0, 0].add_(0.5)

    baseline = collect_paired_utility(_FakePrefixModel(), baseline_sample)
    changed_image = collect_paired_utility(_FakePrefixModel(), changed_image_sample)
    changed_gt = collect_paired_utility(_FakePrefixModel(), changed_gt_sample)

    assert baseline.input_hashes["input_image"] != changed_image.input_hashes["input_image"]
    assert baseline.input_hashes["combined"] != changed_image.input_hashes["combined"]
    assert (
        baseline.input_hashes["valid_target_action"]
        != changed_gt.input_hashes["valid_target_action"]
    )


def test_paired_collection_requires_full_endpoint_before_model_call():
    model = _FakePrefixModel()
    with pytest.raises(ValueError, match="full_prefix_steps == num_inference_steps"):
        collect_paired_utility(
            model,
            _sample(),
            num_inference_steps=10,
            full_prefix_steps=9,
        )
    assert model.calls == []


def test_paired_collection_rejects_nonstable_explicit_seed():
    model = _FakePrefixModel()
    with pytest.raises(ValueError, match="explicit seed must equal"):
        collect_paired_utility(model, _sample(), base_seed=42, seed=123)
    assert model.calls == []


@pytest.mark.parametrize(
    "model",
    [
        _FakePrefixModel(reported_prefix_offset=1),
        _FakePrefixModel(omit_route_key="num_inference_steps"),
        _FakePrefixModel(omit_route_key="force_custom_prefix"),
        _FakePrefixModel(reported_custom_prefix="false"),
    ],
)
def test_paired_collection_rejects_wrong_or_missing_route_metadata(model):
    with pytest.raises(AssertionError, match="route|metadata"):
        collect_paired_utility(model, _sample())


def test_paired_collection_rejects_nonfinite_prediction():
    with pytest.raises(ValueError, match="non-finite"):
        collect_paired_utility(_FakePrefixModel(nonfinite=True), _sample())
