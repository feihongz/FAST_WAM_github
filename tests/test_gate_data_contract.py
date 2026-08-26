from __future__ import annotations

import pytest
import torch

from fastwam.datasets.lerobot.base_lerobot_dataset import (
    BaseLerobotDataset,
    _build_sample_identity,
)
from fastwam.datasets.lerobot.robot_video_dataset import RobotVideoDataset


def _raw_identity_sample() -> dict[str, torch.Tensor]:
    return {
        "dataset_index": torch.tensor(2),
        "episode_index": torch.tensor(17),
        "frame_index": torch.tensor(23),
        "index": torch.tensor(101),
    }


def test_build_sample_identity_uses_current_frame_anchor():
    identity = _build_sample_identity(303, _raw_identity_sample())

    assert identity == {
        "global_sample_index": 303,
        "dataset_index": 2,
        "episode_index": 17,
        "frame_index": 23,
        "dataset_frame_index": 101,
    }


@pytest.mark.parametrize("field", ["dataset_index", "episode_index", "frame_index", "index"])
def test_build_sample_identity_rejects_missing_or_non_scalar_fields(field):
    sample = _raw_identity_sample()
    del sample[field]
    with pytest.raises(KeyError, match="identity fields"):
        _build_sample_identity(0, sample)

    sample = _raw_identity_sample()
    sample[field] = torch.tensor([1, 2])
    with pytest.raises(ValueError, match="must be scalar"):
        _build_sample_identity(0, sample)


def test_base_dataset_restores_identity_after_processor():
    raw = {
        **_raw_identity_sample(),
        "task": "pick object",
        "action_is_pad": torch.tensor([False]),
        "state_is_pad": torch.tensor([False, False]),
        "image_is_pad": torch.tensor([False, False]),
    }

    class Multi:
        num_frames = 1

        def __getitem__(self, index):
            assert index == 0
            return dict(raw)

    class Processor:
        def preprocess(self, sample):
            return {"idx": sample["idx"], "processed": True}

    dataset = BaseLerobotDataset.__new__(BaseLerobotDataset)
    dataset.strict_data_mode = True
    dataset.multi_dataset = Multi()
    dataset.state_meta = [{"key": "default", "lerobot_key": "state"}]
    dataset.action_meta = [{"key": "default", "lerobot_key": "action"}]
    dataset.image_meta = [{"key": "default", "lerobot_key": "image"}]
    dataset._get_state = lambda *args: torch.zeros(2, 1)
    dataset._get_action = lambda *args: torch.zeros(1, 1)
    dataset._get_image = lambda *args: torch.zeros(2, 3, 4, 4)
    dataset._get_additional_data = lambda sample, _: sample
    dataset.processor = Processor()

    sample = dataset[0]

    assert sample["processed"] is True
    assert sample["sample_identity"] == {
        "global_sample_index": 0,
        "dataset_index": 2,
        "episode_index": 17,
        "frame_index": 23,
        "dataset_frame_index": 101,
    }


def test_robot_video_dataset_preserves_identity_masks_and_current_only_inputs():
    identity = {
        "global_sample_index": 9,
        "dataset_index": 1,
        "episode_index": 4,
        "frame_index": 5,
        "dataset_frame_index": 45,
    }
    lower_sample = {
        "pixel_values": torch.zeros(2, 3, 4, 4),
        "image_is_pad": torch.tensor([False, False]),
        "action": torch.zeros(1, 3),
        "action_is_pad": torch.tensor([False]),
        "action_dim_is_pad": torch.tensor([False, False, True]),
        "proprio": torch.zeros(2, 2),
        "proprio_is_pad": torch.tensor([False, False]),
        "instruction": "pick object",
        "sample_identity": identity,
    }
    dataset = RobotVideoDataset.__new__(RobotVideoDataset)
    dataset.lerobot_dataset = [lower_sample]
    dataset.max_padding_retry = 0
    dataset.skip_padding_as_possible = False
    dataset.video_sample_indices = [0, 1]
    dataset.concat_multi_camera = "horizontal"
    dataset.resize_transform = torch.nn.Identity()
    dataset.crop_transform = torch.nn.Identity()
    dataset.normalize_transform = torch.nn.Identity()
    dataset.override_instruction = None
    raw_mask = torch.tensor([True, True, False, False])
    dataset._get_cached_text_context = lambda _: (
        torch.randn(4, 8),
        raw_mask.clone(),
    )

    sample = dataset._get(0)

    assert sample["video"].shape == (3, 2, 4, 4)
    assert sample["sample_identity"] == identity
    assert sample["sample_identity"] is not identity
    assert torch.equal(sample["action_dim_is_pad"], lower_sample["action_dim_is_pad"])
    assert torch.equal(sample["gate_context_mask"], raw_mask)
    assert sample["context_mask"].all()
    assert torch.count_nonzero(sample["context"][~raw_mask]) == 0
