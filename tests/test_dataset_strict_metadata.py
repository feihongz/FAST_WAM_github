import copy
import json
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

from fastwam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset
from fastwam.datasets.lerobot.robot_video_dataset import RobotVideoDataset


class _FakeSourceDataset:
    def __init__(self, num_frames, tasks):
        self.num_frames = num_frames
        self.meta = SimpleNamespace(tasks=tasks)


class _FakeMultiDataset:
    def __init__(self, records, populations, *, failures=None):
        self.records = records
        self.num_frames = len(records)
        self._datasets = [
            _FakeSourceDataset(populations[0], {0: "spatial-zero", np.int64(1): "spatial-one"}),
            _FakeSourceDataset(populations[1], {0: "object-zero"}),
        ]
        self.repo_index_to_id = {
            0: "/datasets/libero_spatial",
            1: "/datasets/libero_object",
        }
        self.failures = dict(failures or {})
        self.calls = []

    def __getitem__(self, idx):
        self.calls.append(idx)
        failure = self.failures.get(idx)
        if failure is not None:
            if isinstance(failure, list):
                if failure:
                    raise failure.pop(0)
            else:
                raise failure
        return copy.deepcopy(self.records[idx])


class _DropRawFieldsProcessor:
    def __init__(self, error=None):
        self.error = error

    def preprocess(self, sample):
        if self.error is not None:
            raise self.error
        return {"idx": sample["idx"], "processed": True}


def _raw_record(index, dataset_index=0):
    return {
        "task": f"task-{index}",
        "task_index": torch.tensor(index + 10, dtype=torch.int64),
        "episode_index": np.int64(index + 20),
        "frame_index": torch.tensor(index + 30, dtype=torch.int64),
        "timestamp": torch.tensor(index + 0.25, dtype=torch.float32),
        "index": np.int64(index + 1000),
        "dataset_index": torch.tensor(dataset_index, dtype=torch.int64),
        "observation.state": torch.zeros(3, 1),
        "action": torch.zeros(2, 1),
        "observation.images.camera": torch.zeros(3, 3, 2, 2),
        "observation.state_is_pad": torch.zeros(3, dtype=torch.bool),
        "action_is_pad": torch.zeros(2, dtype=torch.bool),
        "observation.images.camera_is_pad": torch.zeros(3, dtype=torch.bool),
    }


def _base_dataset(
    *,
    strict_getitem=False,
    return_metadata=False,
    failures=None,
    processor=None,
    records=None,
):
    if records is None:
        records = [_raw_record(0), _raw_record(1), _raw_record(2, dataset_index=1)]
    dataset = BaseLerobotDataset.__new__(BaseLerobotDataset)
    dataset.multi_dataset = _FakeMultiDataset(records, [2, 1], failures=failures)
    dataset.strict_getitem = strict_getitem
    dataset.return_metadata = return_metadata
    dataset.processor = processor
    dataset.state_meta = [
        {"key": "state", "lerobot_key": "observation.state", "raw_shape": 1}
    ]
    dataset.action_meta = [{"key": "action", "lerobot_key": "action", "raw_shape": 1}]
    dataset.image_meta = [
        {
            "key": "camera",
            "lerobot_key": "observation.images.camera",
            "raw_shape": [3, 2, 2],
        }
    ]
    return dataset


def test_strict_read_error_is_raised_once_without_random_replacement():
    source_error = ValueError("source decode failed")
    dataset = _base_dataset(strict_getitem=True, failures={0: source_error})

    with pytest.raises(ValueError) as exc_info:
        dataset[0]

    assert exc_info.value is source_error
    assert dataset.multi_dataset.calls == [0]


def test_strict_negative_index_is_rejected_without_source_read():
    dataset = _base_dataset(strict_getitem=True, return_metadata=True)

    with pytest.raises(IndexError, match="non-negative index"):
        dataset[-1]

    assert dataset.multi_dataset.calls == []


def test_strict_processing_error_is_preserved():
    processing_error = RuntimeError("processor failed")
    dataset = _base_dataset(
        strict_getitem=True,
        processor=_DropRawFieldsProcessor(error=processing_error),
    )

    with pytest.raises(RuntimeError) as exc_info:
        dataset[1]

    assert exc_info.value is processing_error
    assert dataset.multi_dataset.calls == [1]


def test_legacy_mode_retries_with_random_source_index():
    source_error = ValueError("transient source failure")
    dataset = _base_dataset(
        return_metadata=True,
        failures={0: [source_error]},
        processor=_DropRawFieldsProcessor(),
    )

    with mock.patch(
        "fastwam.datasets.lerobot.base_lerobot_dataset.np.random.randint",
        return_value=1,
    ):
        sample = dataset[0]

    assert dataset.multi_dataset.calls == [0, 1]
    assert sample["metadata"]["requested_sample_idx"] == 0
    assert sample["metadata"]["source_sample_idx"] == 1


def test_metadata_is_captured_before_processor_and_is_json_safe():
    dataset = _base_dataset(
        strict_getitem=True,
        return_metadata=True,
        processor=_DropRawFieldsProcessor(),
    )

    sample = dataset[2]

    assert sample["processed"] is True
    assert sample["metadata"] == {
        "requested_sample_idx": 2,
        "source_sample_idx": 2,
        "dataset_index": 1,
        "dataset_id": "/datasets/libero_object",
        "dataset_name": "libero_object",
        "episode_index": 22,
        "frame_index": 32,
        "task_index": 12,
        "task": "task-2",
        "timestamp": pytest.approx(2.25),
        "source_index": 1002,
    }
    json.dumps(sample["metadata"], allow_nan=False)
    assert all(
        value is None or isinstance(value, (str, bool, int, float))
        for value in sample["metadata"].values()
    )


def test_default_schema_does_not_include_metadata():
    dataset = _base_dataset()

    assert "metadata" not in dataset[0]


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        (lambda record: record.pop("task_index"), KeyError),
        (
            lambda record: record.__setitem__("frame_index", torch.tensor([1, 2])),
            ValueError,
        ),
    ],
)
def test_required_or_non_scalar_metadata_fails_fast(mutation, expected_error):
    records = [_raw_record(0), _raw_record(1), _raw_record(2, dataset_index=1)]
    mutation(records[0])
    dataset = _base_dataset(
        strict_getitem=True,
        return_metadata=True,
        records=records,
    )

    with pytest.raises(expected_error):
        dataset[0]

    assert dataset.multi_dataset.calls == [0]


def test_strict_metadata_rejects_requested_source_mismatch():
    dataset = _base_dataset(strict_getitem=True, return_metadata=True)
    raw = dataset.multi_dataset[1]

    with pytest.raises(AssertionError, match="requested=0, source=1"):
        dataset._build_metadata(0, 1, raw)


def test_dataset_index_ranges_are_json_safe_and_do_not_load_samples():
    dataset = _base_dataset()

    ranges = dataset.dataset_index_ranges()

    assert ranges == [
        {
            "dataset_index": 0,
            "dataset_id": "/datasets/libero_spatial",
            "dataset_name": "libero_spatial",
            "start": 0,
            "stop": 2,
            "population": 2,
        },
        {
            "dataset_index": 1,
            "dataset_id": "/datasets/libero_object",
            "dataset_name": "libero_object",
            "start": 2,
            "stop": 3,
            "population": 1,
        },
    ]
    assert dataset.multi_dataset.calls == []
    json.dumps(ranges, allow_nan=False)


def test_dataset_index_ranges_prefer_ordered_ds_names():
    dataset = _base_dataset()
    dataset.multi_dataset.ds_names = [
        "/primary/libero_spatial",
        "/primary/libero_object",
    ]
    dataset.multi_dataset.repo_index_to_id = None

    ranges = dataset.dataset_index_ranges()

    assert [item["dataset_id"] for item in ranges] == dataset.multi_dataset.ds_names
    assert dataset.multi_dataset.calls == []


def test_robot_dataset_delegates_index_ranges():
    base = _base_dataset()
    dataset = RobotVideoDataset.__new__(RobotVideoDataset)
    dataset.lerobot_dataset = base

    assert dataset.dataset_index_ranges() == base.dataset_index_ranges()


def test_dataset_task_table_is_local_json_safe_and_does_not_load_samples():
    dataset = _base_dataset()

    assert dataset.dataset_task_table(0) == {0: "spatial-zero", 1: "spatial-one"}
    assert dataset.dataset_task_table(1) == {0: "object-zero"}
    assert dataset.multi_dataset.calls == []
    json.dumps(dataset.dataset_task_table(0), allow_nan=False)
    with pytest.raises(IndexError, match="out of bounds"):
        dataset.dataset_task_table(2)


def test_robot_dataset_delegates_task_table():
    base = _base_dataset()
    dataset = RobotVideoDataset.__new__(RobotVideoDataset)
    dataset.lerobot_dataset = base

    assert dataset.dataset_task_table(1) == {0: "object-zero"}


def test_robot_default_schema_does_not_include_metadata():
    dataset = RobotVideoDataset.__new__(RobotVideoDataset)
    dataset.strict_getitem = False
    dataset.return_metadata = False
    dataset._get = mock.Mock(return_value={"value": 1})

    assert dataset[0] == {"value": 1}
    assert "metadata" not in dataset[0]


def test_robot_legacy_fallback_preserves_original_request_and_actual_source():
    dataset = RobotVideoDataset.__new__(RobotVideoDataset)
    dataset.strict_getitem = False
    dataset.return_metadata = True
    dataset.lerobot_dataset = list(range(10))
    replacement = {
        "metadata": {
            "requested_sample_idx": 7,
            "source_sample_idx": 7,
        }
    }
    dataset._get = mock.Mock(side_effect=[RuntimeError("decode failed"), replacement])

    with mock.patch(
        "fastwam.datasets.lerobot.robot_video_dataset.np.random.randint",
        return_value=7,
    ):
        sample = dataset[5]

    assert dataset._get.call_args_list == [mock.call(5), mock.call(7)]
    assert sample["metadata"]["requested_sample_idx"] == 5
    assert sample["metadata"]["source_sample_idx"] == 7


def test_robot_strict_mode_rejects_padding_retries_before_source_creation():
    with pytest.raises(ValueError, match="incompatible with skip_padding_as_possible"):
        RobotVideoDataset(
            dataset_dirs=["unused"],
            shape_meta={},
            num_frames=5,
            skip_padding_as_possible=True,
            strict_getitem=True,
        )


def test_robot_strict_processing_error_is_not_replaced():
    source_error = RuntimeError("robot processing failed")
    dataset = RobotVideoDataset.__new__(RobotVideoDataset)
    dataset.strict_getitem = True
    dataset.return_metadata = False
    dataset._get = mock.Mock(side_effect=source_error)

    with pytest.raises(RuntimeError) as exc_info:
        dataset[4]

    assert exc_info.value is source_error
    dataset._get.assert_called_once_with(4)
