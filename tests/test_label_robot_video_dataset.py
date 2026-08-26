from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from scripts import generate_gate_labels as generate_cli
from fastwam.alignment.text_cache_index import (
    TextCacheIndexIdentity,
    text_cache_contract_sha256,
)
from fastwam.datasets.lerobot import base_lerobot_dataset as base_module
from fastwam.datasets.lerobot import label_robot_video_dataset as label_module
from fastwam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset
from fastwam.datasets.lerobot.label_robot_video_dataset import (
    LabelRobotVideoDataset,
)
from fastwam.datasets.lerobot.robot_video_dataset import RobotVideoDataset
from fastwam.gating.inference import run_paired_action_rollouts
from fastwam.gating.labels import paired_gate_label_statistics


_SHAPE_META = {
    "images": [
        {"key": "cam_high", "raw_shape": [3, 8, 8], "shape": [3, 8, 8]},
        {"key": "cam_left_wrist", "raw_shape": [3, 8, 8], "shape": [3, 8, 8]},
        {"key": "cam_right_wrist", "raw_shape": [3, 8, 8], "shape": [3, 8, 8]},
    ],
    "state": [{"key": "default", "raw_shape": 14, "shape": 14}],
    "action": [{"key": "default", "raw_shape": 14, "shape": 14}],
}


class _FakeMetadata:
    fps = 10
    total_episodes = 1

    def __init__(self, repo_id, root):
        del root
        self.repo_id = repo_id


class _FakeChild:
    def __init__(self) -> None:
        self.episode_data_index = {
            "from": torch.tensor([0]),
            "to": torch.tensor([2]),
        }


class _TimestampCaptureMulti:
    last_delta_timestamps = None

    def __init__(self, *, delta_timestamps, **kwargs) -> None:
        del kwargs
        type(self).last_delta_timestamps = deepcopy(delta_timestamps)
        self._datasets = [_FakeChild()]
        self.num_frames = 2


def test_independent_label_query_requests_t0_images_state_and_action32(
    monkeypatch,
):
    monkeypatch.setattr(base_module, "LeRobotDatasetMetadata", _FakeMetadata)
    monkeypatch.setattr(
        base_module, "MultiLeRobotDataset", _TimestampCaptureMulti
    )

    dataset = BaseLerobotDataset(
        dataset_dirs=["/data/robotwin"],
        shape_meta=deepcopy(_SHAPE_META),
        obs_size=1,
        action_size=32,
        val_set_proportion=0.0,
        is_training_set=True,
        global_sample_stride=4,
        strict_data_mode=True,
        allow_independent_action_horizon=True,
    )

    timestamps = _TimestampCaptureMulti.last_delta_timestamps
    for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
        assert timestamps[f"observation.images.{camera}"] == [0.0]
    assert timestamps["observation.state"] == [0.0]
    assert timestamps["action"] == pytest.approx(
        [(step * 4) / 10 for step in range(32)]
    )
    assert dataset.obs_size == 1
    assert dataset.action_size == 32
    assert dataset.allow_independent_action_horizon is True

    with pytest.raises(AssertionError, match="action_size should be obs_size"):
        BaseLerobotDataset(
            dataset_dirs=["/data/robotwin"],
            shape_meta=deepcopy(_SHAPE_META),
            obs_size=1,
            action_size=32,
            val_set_proportion=0.0,
            strict_data_mode=True,
        )
    with pytest.raises(ValueError, match="strict_data_mode"):
        BaseLerobotDataset(
            dataset_dirs=["/data/robotwin"],
            shape_meta=deepcopy(_SHAPE_META),
            obs_size=1,
            action_size=32,
            val_set_proportion=0.0,
            allow_independent_action_horizon=True,
        )


class _IdentityTransform:
    def __call__(self, value):
        return value


class _SizedNamespace(SimpleNamespace):
    def __len__(self) -> int:
        return 123


class _LabelProcessor:
    def __init__(self) -> None:
        self.num_obs_steps = 33
        self.mode = "train"

    def train(self):
        self.mode = "train"
        return self

    def eval(self):
        self.mode = "eval"
        return self


class _FullBase:
    def __init__(self, processor, sample) -> None:
        self.dataset_dirs = ["/data/robotwin"]
        self.shape_meta = deepcopy(_SHAPE_META)
        self.val_set_proportion = 0.01
        self.is_training_set = True
        self.seed = 42
        self.global_sample_stride = 1
        self.video_backend = "torchcodec"
        self.strict_data_mode = True
        self.processor = processor
        self.action_size = 32
        self.sample = sample

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        assert index == 0
        return self.sample


class _FakeLabelBase:
    init_kwargs = None
    sample = None

    def __init__(self, **kwargs) -> None:
        type(self).init_kwargs = dict(kwargs)
        self.__dict__.update(kwargs)
        self.processor = None

    def _set_return_images(self, flag: bool) -> None:
        self.return_images = flag

    def set_processor(self, processor):
        self.processor = processor
        processor.train()
        return self

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        assert index == 0
        assert self.processor.num_obs_steps == 1
        return type(self).sample


def _source_and_label_sample(monkeypatch):
    processor = _LabelProcessor()
    t0 = torch.linspace(0.0, 1.0, 3 * 1 * 3 * 8 * 8).reshape(
        3, 1, 3, 8, 8
    )
    future = t0.expand(-1, 32, -1, -1, -1).clone() + 0.125
    full_pixels = torch.cat([t0, future], dim=1)
    action = torch.linspace(-1.0, 1.0, 32 * 14).reshape(32, 14)
    current_proprio = torch.linspace(-0.5, 0.5, 14).reshape(1, 14)
    future_proprio = torch.full((32, 14), 0.75)
    identity = {
        "global_sample_index": 0,
        "dataset_index": 0,
        "episode_index": 0,
        "frame_index": 0,
        "dataset_frame_index": 0,
    }
    action_is_pad = torch.zeros(32, dtype=torch.bool)
    action_is_pad[-2:] = True
    action_dim_is_pad = torch.zeros(14, dtype=torch.bool)
    full_sample = {
        "pixel_values": full_pixels,
        "image_is_pad": torch.zeros(33, dtype=torch.bool),
        "action": action,
        "proprio": torch.cat([current_proprio, future_proprio], dim=0),
        "instruction": "put the red block in the bowl",
        "action_is_pad": action_is_pad,
        "action_dim_is_pad": action_dim_is_pad,
        "proprio_is_pad": torch.zeros(33, dtype=torch.bool),
        "sample_identity": identity,
    }
    _FakeLabelBase.sample = {
        "pixel_values": t0,
        "image_is_pad": torch.zeros(1, dtype=torch.bool),
        "action": action,
        "proprio": current_proprio,
        "instruction": full_sample["instruction"],
        "action_is_pad": action_is_pad,
        "action_dim_is_pad": action_dim_is_pad,
        "proprio_is_pad": torch.zeros(1, dtype=torch.bool),
        "sample_identity": identity,
    }

    source = RobotVideoDataset.__new__(RobotVideoDataset)
    source.lerobot_dataset = _FullBase(processor, full_sample)
    source.strict_data_mode = True
    source.skip_padding_as_possible = False
    source.max_padding_retry = 0
    source.num_frames = 33
    source.action_video_freq_ratio = 4
    source.video_sample_indices = list(range(0, 33, 4))
    source.concat_multi_camera = "robotwin"
    source.override_instruction = None
    source.text_embedding_cache_dir = "/fake/text-cache"
    source.context_len = 3
    source.text_context_cache_max_entries = 16
    source._text_cache_index_descriptor_path = None
    source._text_cache_index = None
    source._text_cache_index_pid = None
    source.resize_transform = _IdentityTransform()
    source.crop_transform = _IdentityTransform()
    source.normalize_transform = _IdentityTransform()

    context = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    true_mask = torch.tensor([True, False, True])

    def cached_context(_dataset, _prompt):
        return context.clone(), true_mask.clone()

    monkeypatch.setattr(
        RobotVideoDataset,
        "_get_cached_text_context",
        cached_context,
    )
    monkeypatch.setattr(label_module, "BaseLerobotDataset", _FakeLabelBase)
    return source


class _RolloutModel:
    def eval(self):
        return self

    def infer_action_mode(self, **kwargs):
        value = (
            kwargs["input_image"].float().mean()
            + kwargs["proprio"].float().mean()
            + kwargs["context"][kwargs["context_mask"]].float().mean()
            + float(kwargs["seed"] % 101) / 1000.0
        )
        if kwargs["inference_mode"] == "w":
            value = value - 0.2
        return {
            "action": torch.full(
                (kwargs["action_horizon"], 14),
                value,
                dtype=torch.float32,
            )
        }


def _label_statistics(sample, rollouts):
    return paired_gate_label_statistics(
        action_wo=rollouts.action_wo,
        action_w=rollouts.action_w,
        target_action=sample["action"].unsqueeze(0),
        action_is_pad=sample["action_is_pad"].unsqueeze(0),
        action_dim_is_pad=sample["action_dim_is_pad"],
    )


def test_label_only_matches_full_inputs_rollouts_and_labels(monkeypatch):
    source = _source_and_label_sample(monkeypatch)
    full = source[0]
    label_dataset = source.label_only()
    optimized = label_dataset[0]

    assert isinstance(label_dataset, LabelRobotVideoDataset)
    assert _FakeLabelBase.init_kwargs["obs_size"] == 1
    assert _FakeLabelBase.init_kwargs["action_size"] == 32
    assert _FakeLabelBase.init_kwargs["allow_independent_action_horizon"] is True
    assert label_dataset.processor is not source.lerobot_dataset.processor
    assert label_dataset.processor.num_obs_steps == 1
    assert source.lerobot_dataset.processor.num_obs_steps == 33

    assert optimized["video"].shape == (3, 9, 384, 320)
    assert optimized["video"].stride(1) == 0
    assert optimized["proprio"].shape == (32, 14)
    assert optimized["proprio"].stride(0) == 0
    torch.testing.assert_close(optimized["video"][:, 0], full["video"][:, 0])
    torch.testing.assert_close(optimized["action"], full["action"])
    torch.testing.assert_close(optimized["proprio"][0], full["proprio"][0])
    torch.testing.assert_close(optimized["context"], full["context"])
    torch.testing.assert_close(optimized["context_mask"], full["context_mask"])
    torch.testing.assert_close(
        optimized["gate_context_mask"], full["gate_context_mask"]
    )
    assert optimized["sample_identity"] == full["sample_identity"]

    model = _RolloutModel()
    kwargs = {
        "seeds": (7, 11),
        "num_inference_steps": 10,
        "rand_device": "cpu",
    }
    full_rollouts = run_paired_action_rollouts(model, full, **kwargs)
    optimized_rollouts = run_paired_action_rollouts(model, optimized, **kwargs)
    assert full_rollouts.seeds == optimized_rollouts.seeds
    assert full_rollouts.action_horizon == optimized_rollouts.action_horizon == 32
    assert full_rollouts.num_video_frames == optimized_rollouts.num_video_frames == 9
    torch.testing.assert_close(full_rollouts.action_wo, optimized_rollouts.action_wo)
    torch.testing.assert_close(full_rollouts.action_w, optimized_rollouts.action_w)

    full_statistics = _label_statistics(full, full_rollouts)
    optimized_statistics = _label_statistics(optimized, optimized_rollouts)
    for field in ("e0", "e10", "relative_gain", "label", "sample_weight"):
        torch.testing.assert_close(
            getattr(full_statistics, field),
            getattr(optimized_statistics, field),
        )


def test_label_only_reverifies_v2_binding_and_fails_closed(
    tmp_path, monkeypatch
):
    source = _source_and_label_sample(monkeypatch)
    source._text_cache_index_descriptor_path = "/verified/index.json"
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    contract_sha = text_cache_contract_sha256(
        context_len=3,
        prompt_template="Instruction: {task}",
        filename_suffix=".t5_len3.wan22ti2v5b.pt",
    )
    expected_identity = TextCacheIndexIdentity(
        descriptor_file_sha256="a" * 64,
        descriptor_size_bytes=101,
        descriptor_sha256="b" * 64,
        index_sha256="c" * 64,
        index_size_bytes=4096,
        record_count=17,
        prompt_set_sha256="d" * 64,
        contract_sha256=contract_sha,
        cache_root=str(cache_root.resolve()),
        context_len=3,
        prompt_template="Instruction: {task}",
        filename_suffix=".t5_len3.wan22ti2v5b.pt",
        index_relative_path="index.bin",
    )
    source._text_cache_index_expected_identity = expected_identity
    calls = []

    def verified_bind(dataset, descriptor_path, identity):
        calls.append((dataset, descriptor_path, identity))
        dataset._text_cache_index_descriptor_path = descriptor_path
        dataset._text_cache_index_expected_identity = identity

    monkeypatch.setattr(
        label_module, "_bind_dataset_text_cache_index", verified_bind
    )
    label_dataset = source.label_only()
    assert calls == [
        (label_dataset, "/verified/index.json", expected_identity)
    ]
    assert label_dataset._text_cache_index_descriptor_path == "/verified/index.json"
    assert label_dataset._text_cache_index_expected_identity == expected_identity

    def rejected_bind(_dataset, _descriptor_path, _identity):
        raise ValueError("index SHA256 mismatch")

    monkeypatch.setattr(
        label_module, "_bind_dataset_text_cache_index", rejected_bind
    )
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        source.label_only()

    source._text_cache_index_descriptor_path = None
    source._text_cache_index = object()
    with pytest.raises(RuntimeError, match="without a verified descriptor"):
        source.label_only()


def test_generate_entrypoint_accepts_only_contract_preserving_label_view():
    label_base = SimpleNamespace(
        obs_size=1,
        action_size=32,
        allow_independent_action_horizon=True,
    )
    label_dataset = _SizedNamespace(
        lerobot_dataset=label_base,
        strict_data_mode=True,
        skip_padding_as_possible=False,
        num_video_frames=9,
        _text_cache_index_descriptor_path="/verified/index.json",
    )
    source = _SizedNamespace(
        lerobot_dataset=SimpleNamespace(action_size=32),
        video_sample_indices=list(range(0, 33, 4)),
        _text_cache_index_descriptor_path="/verified/index.json",
        label_only=lambda: label_dataset,
    )
    assert generate_cli._build_label_only_dataset(source) is label_dataset

    label_dataset._text_cache_index_descriptor_path = None
    with pytest.raises(RuntimeError, match="text cache binding"):
        generate_cli._build_label_only_dataset(source)
