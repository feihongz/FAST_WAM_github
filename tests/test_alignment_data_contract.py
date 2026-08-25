from __future__ import annotations

import pytest
import torch

from fastwam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset
from fastwam.datasets.lerobot.lerobot.datasets import video_utils
from fastwam.datasets.lerobot.robot_video_dataset import RobotVideoDataset


def test_torchcodec_failure_is_fail_closed_when_fallback_disabled(monkeypatch):
    fallback_calls = []

    def fail_torchcodec(*args, **kwargs):
        raise OSError("decoder failed")

    def record_fallback(*args, **kwargs):
        fallback_calls.append((args, kwargs))
        return torch.zeros(1)

    monkeypatch.setattr(
        video_utils,
        "decode_video_frames_torchcodec",
        fail_torchcodec,
    )
    monkeypatch.setattr(
        video_utils,
        "decode_video_frames_torchvision",
        record_fallback,
    )

    with pytest.raises(RuntimeError, match="torchcodec video decode failed"):
        video_utils.decode_video_frames(
            "broken.mp4",
            [0.0],
            1.0e-4,
            backend="torchcodec",
            allow_fallback=False,
        )

    assert fallback_calls == []


def test_torchcodec_failure_uses_legacy_pyav_fallback_by_default(monkeypatch):
    fallback_calls = []
    expected = torch.tensor([42.0])

    def fail_torchcodec(*args, **kwargs):
        raise OSError("decoder failed")

    def record_fallback(*args, **kwargs):
        fallback_calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(
        video_utils,
        "decode_video_frames_torchcodec",
        fail_torchcodec,
    )
    monkeypatch.setattr(
        video_utils,
        "decode_video_frames_torchvision",
        record_fallback,
    )

    with pytest.warns(UserWarning, match="falling back"):
        actual = video_utils.decode_video_frames(
            "broken.mp4",
            [0.25],
            2.0e-4,
            backend="torchcodec",
        )

    assert actual is expected
    assert len(fallback_calls) == 1
    args, kwargs = fallback_calls[0]
    assert args == ("broken.mp4", [0.25], 2.0e-4)
    assert kwargs == {"backend": "pyav"}


def test_robot_video_dataset_strict_mode_never_randomly_replaces_index(
    monkeypatch,
):
    dataset = RobotVideoDataset.__new__(RobotVideoDataset)
    dataset.strict_data_mode = True
    requested = []

    def fail_requested_index(index):
        requested.append(index)
        raise LookupError("requested sample is corrupt")

    def forbid_random_replacement(*args, **kwargs):
        raise AssertionError("strict mode attempted a random replacement")

    dataset._get = fail_requested_index
    monkeypatch.setattr(
        "fastwam.datasets.lerobot.robot_video_dataset.np.random.randint",
        forbid_random_replacement,
    )

    with pytest.raises(LookupError, match="requested sample is corrupt"):
        dataset[7]

    assert requested == [7]


def test_base_lerobot_dataset_strict_mode_is_fail_closed(monkeypatch):
    class FailingMultiDataset:
        num_frames = 10

        def __init__(self):
            self.requested = []

        def __getitem__(self, index):
            self.requested.append(index)
            raise OSError("frame decode failed")

    dataset = BaseLerobotDataset.__new__(BaseLerobotDataset)
    dataset.strict_data_mode = True
    dataset.multi_dataset = FailingMultiDataset()

    def forbid_random_replacement(*args, **kwargs):
        raise AssertionError("strict mode attempted a random replacement")

    monkeypatch.setattr(
        "fastwam.datasets.lerobot.base_lerobot_dataset.np.random.randint",
        forbid_random_replacement,
    )

    with pytest.raises(RuntimeError, match="requested index 4"):
        dataset[4]

    assert dataset.multi_dataset.requested == [4]
