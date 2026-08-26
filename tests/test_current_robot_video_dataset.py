from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from fastwam.datasets.lerobot import base_lerobot_dataset as base_module
from fastwam.datasets.lerobot import current_robot_video_dataset as current_module
from fastwam.datasets.lerobot import robot_video_dataset as robot_module
from fastwam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset
from fastwam.datasets.lerobot.current_robot_video_dataset import (
    CurrentRobotVideoDataset,
)
from fastwam.datasets.lerobot.robot_video_dataset import (
    DEFAULT_PROMPT,
    RobotVideoDataset,
)


_SHAPE_META = {
    "images": [
        {
            "key": "default",
            "raw_shape": [3, 2, 2],
            "shape": [3, 2, 2],
        }
    ],
    "state": [{"key": "default", "raw_shape": 2, "shape": 2}],
    "action": [{"key": "default", "raw_shape": 2, "shape": 2}],
}


class _FakeMetadata:
    fps = 10
    total_episodes = 1

    def __init__(self, repo_id, root):
        del root
        self.repo_id = repo_id


class _FakeHFDataset:
    def __init__(self) -> None:
        self.column_names = [
            "action",
            "action.gripper",
            "observation.state",
            "timestamp",
        ]
        self.remove_calls: list[tuple[str, ...]] = []

    def remove_columns(self, names):
        self.remove_calls.append(tuple(names))
        self.column_names = [
            name for name in self.column_names if name not in set(names)
        ]
        return self


class _FakeLeRobotChild:
    def __init__(self) -> None:
        self.hf_dataset = _FakeHFDataset()
        self.episode_data_index = {
            "from": torch.tensor([0]),
            "to": torch.tensor([2]),
        }


class _FakeMultiDataset:
    last_delta_timestamps = None
    last_instance = None

    def __init__(self, *, delta_timestamps, **kwargs) -> None:
        del kwargs
        type(self).last_delta_timestamps = deepcopy(delta_timestamps)
        type(self).last_instance = self
        self._datasets = [_FakeLeRobotChild()]
        self.num_frames = 2
        self.num_episodes = 1

    def set_during_training(self, flag: bool) -> None:
        self.during_training = flag

    def __getitem__(self, index: int):
        return {
            "dataset_index": torch.tensor(0),
            "episode_index": torch.tensor(0),
            "frame_index": torch.tensor(index),
            "index": torch.tensor(index),
            "task": "move the block",
            "observation.state": torch.tensor([[1.0, 2.0]]),
            "observation.images": torch.ones(1, 3, 2, 2),
            "observation.state_is_pad": torch.tensor([False]),
            "observation.images_is_pad": torch.tensor([False]),
        }


def test_current_only_base_queries_only_t0_and_strips_action_columns(
    monkeypatch,
):
    monkeypatch.setattr(
        base_module, "LeRobotDatasetMetadata", _FakeMetadata
    )
    monkeypatch.setattr(
        base_module, "MultiLeRobotDataset", _FakeMultiDataset
    )

    dataset = BaseLerobotDataset(
        dataset_dirs=["/data/libero"],
        shape_meta=deepcopy(_SHAPE_META),
        obs_size=1,
        action_size=0,
        val_set_proportion=0.0,
        is_training_set=True,
        global_sample_stride=4,
        strict_data_mode=True,
        load_actions=False,
    )

    assert _FakeMultiDataset.last_delta_timestamps == {
        "observation.images": [0.0],
        "observation.state": [0.0],
    }
    child = _FakeMultiDataset.last_instance._datasets[0]
    assert child.hf_dataset.remove_calls == [
        ("action", "action.gripper")
    ]
    assert all(
        name != "action" and not name.startswith("action.")
        for name in child.hf_dataset.column_names
    )

    sample = dataset[0]
    assert "action" not in sample
    assert "action_is_pad" not in sample
    assert sample["sample_identity"] == {
        "global_sample_index": 0,
        "dataset_index": 0,
        "episode_index": 0,
        "frame_index": 0,
        "dataset_frame_index": 0,
    }


class _IdentityTransform:
    def __call__(self, value):
        return value


class _CurrentProcessor:
    def __init__(self) -> None:
        self.num_obs_steps = 2
        self.mode = None
        self.preprocess_calls = 0

    def train(self):
        self.mode = "train"
        return self

    def eval(self):
        self.mode = "eval"
        return self

    def preprocess(self, sample):
        self.preprocess_calls += 1
        assert self.num_obs_steps == 1
        assert "action" not in sample
        assert "future" not in sample
        return {
            "pixel_values": sample["pixel_values"],
            "proprio": sample["state"] * 2.0 + 1.0,
            "instruction": sample["instruction"],
            "sample_identity": dict(sample["sample_identity"]),
        }


class _FakeCurrentBase:
    init_kwargs = None

    def __init__(self, **kwargs) -> None:
        type(self).init_kwargs = dict(kwargs)
        self.strict_data_mode = kwargs["strict_data_mode"]
        self.processor = None
        self.raw_sample = {
            "pixel_values": torch.tensor(
                [
                    [[[[0.0, 0.1], [0.2, 0.3]]] * 3],
                    [[[[0.4, 0.5], [0.6, 0.7]]] * 3],
                ],
                dtype=torch.float32,
            ).reshape(2, 1, 3, 2, 2),
            "state": torch.tensor([[0.25, -0.5]], dtype=torch.float32),
            "instruction": "move the block",
            "sample_identity": {
                "global_sample_index": 0,
                "dataset_index": 0,
                "episode_index": 0,
                "frame_index": 0,
                "dataset_frame_index": 0,
            },
        }

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
        return self.processor.preprocess(dict(self.raw_sample))


class _FullBase:
    def __init__(self, processor, sample):
        self.dataset_dirs = ["/data/libero"]
        self.shape_meta = deepcopy(_SHAPE_META)
        self.val_set_proportion = 0.0
        self.is_training_set = True
        self.seed = 42
        self.global_sample_stride = 4
        self.video_backend = "torchcodec"
        self.strict_data_mode = True
        self.processor = processor
        self.sample = sample

    def __len__(self):
        return 1

    def __getitem__(self, index):
        assert index == 0
        return self.sample


def _full_source() -> tuple[RobotVideoDataset, _CurrentProcessor]:
    processor = _CurrentProcessor()
    current_pixels = _FakeCurrentBase(
        strict_data_mode=True
    ).raw_sample["pixel_values"]
    future_pixels = current_pixels + 0.25
    full_pixels = torch.cat([current_pixels, future_pixels], dim=1)
    current_state = torch.tensor([[0.25, -0.5]], dtype=torch.float32)
    full_sample = {
        "pixel_values": full_pixels,
        "image_is_pad": torch.tensor([False, False]),
        "action": torch.zeros(1, 2),
        "proprio": torch.cat(
            [current_state * 2.0 + 1.0, torch.zeros_like(current_state)],
            dim=0,
        ),
        "instruction": "move the block",
        "action_is_pad": torch.tensor([False]),
        "action_dim_is_pad": torch.tensor([False, False]),
        "proprio_is_pad": torch.tensor([False, False]),
        "sample_identity": {
            "global_sample_index": 0,
            "dataset_index": 0,
            "episode_index": 0,
            "frame_index": 0,
            "dataset_frame_index": 0,
        },
    }

    source = RobotVideoDataset.__new__(RobotVideoDataset)
    source.lerobot_dataset = _FullBase(processor, full_sample)
    source.strict_data_mode = True
    source.skip_padding_as_possible = False
    source.max_padding_retry = 0
    source.video_sample_indices = [0, 1]
    source.concat_multi_camera = "horizontal"
    source.override_instruction = None
    source.text_embedding_cache_dir = "/fake/text-cache"
    source.context_len = 3
    source.resize_transform = _IdentityTransform()
    source.crop_transform = _IdentityTransform()
    source.normalize_transform = _IdentityTransform()
    return source, processor


def test_current_only_matches_full_t0_and_caches_text_without_aliasing(
    monkeypatch,
):
    source, full_processor = _full_source()
    context = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    true_mask = torch.tensor([True, False, True])

    source._get_cached_text_context = lambda prompt: (
        context.clone(),
        true_mask.clone(),
    )
    full = source[0]

    monkeypatch.setattr(
        current_module, "BaseLerobotDataset", _FakeCurrentBase
    )
    monkeypatch.setattr(current_module.os.path, "isfile", lambda path: True)
    load_calls = []

    def fake_load(path, *, map_location):
        load_calls.append((path, map_location))
        return {"context": context.clone(), "mask": true_mask.clone()}

    monkeypatch.setattr(current_module.torch, "load", fake_load)
    current = source.current_only()

    assert isinstance(current, CurrentRobotVideoDataset)
    assert current.processor is not full_processor
    assert current.processor.num_obs_steps == 1
    assert full_processor.num_obs_steps == 2
    assert _FakeCurrentBase.init_kwargs["obs_size"] == 1
    assert _FakeCurrentBase.init_kwargs["action_size"] == 0
    assert _FakeCurrentBase.init_kwargs["load_actions"] is False
    assert _FakeCurrentBase.init_kwargs["strict_data_mode"] is True

    torch.manual_seed(123)
    first = current[0]
    first["context"].fill_(99.0)
    first["context_mask"].fill_(False)
    torch.manual_seed(123)
    second = current[0]

    assert len(load_calls) == 1
    assert set(second) == {
        "input_image",
        "context",
        "context_mask",
        "proprio",
        "sample_identity",
    }
    torch.testing.assert_close(second["input_image"], full["video"][:, 0])
    torch.testing.assert_close(second["proprio"], full["proprio"][0])
    torch.testing.assert_close(second["context"], full["context"])
    torch.testing.assert_close(second["context_mask"], full["gate_context_mask"])
    assert second["context_mask"].tolist() == [True, False, True]
    assert {
        "video",
        "action",
        "future",
        "prompt",
        "context_mask_wan",
        "image_is_pad",
        "proprio_is_pad",
    }.isdisjoint(second)


def test_current_only_requires_strict_source_and_never_random_fallback(
    monkeypatch,
):
    source, _ = _full_source()
    source.strict_data_mode = False
    monkeypatch.setattr(
        current_module, "BaseLerobotDataset", _FakeCurrentBase
    )

    with pytest.raises(ValueError, match="strict_data_mode"):
        source.current_only()


def test_full_prompt_used_by_current_loader():
    prompt = DEFAULT_PROMPT.format(task="move the block")
    assert "move the block" in prompt


def test_pretrained_stats_can_disable_workdir_copy(monkeypatch):
    monkeypatch.setattr(
        robot_module,
        "BaseLerobotDataset",
        _FakeCurrentBase,
    )
    monkeypatch.setattr(
        robot_module,
        "load_dataset_stats_from_json",
        lambda _path: {"verified": True},
    )
    monkeypatch.setattr(
        robot_module,
        "save_dataset_stats_to_json",
        lambda *_args, **_kwargs: pytest.fail(
            "save_stats_copy=false must not write a stats copy"
        ),
    )

    class Processor:
        def set_normalizer_from_stats(self, stats):
            self.stats = stats

        def train(self):
            return self

    processor = Processor()
    dataset = RobotVideoDataset(
        dataset_dirs=["/data/libero"],
        shape_meta=OmegaConf.create(_SHAPE_META),
        num_frames=5,
        action_video_freq_ratio=1,
        processor=processor,
        pretrained_norm_stats="/verified/stats.json",
        strict_data_mode=True,
        save_stats_copy=False,
    )

    assert dataset.save_stats_copy is False
    assert processor.stats == {"verified": True}
