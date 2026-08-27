from __future__ import annotations

import hashlib
import os

import pytest
import torch

import scripts.compare_stage3_adapter_exports as comparator
from fastwam.models.wan22.video_action_alignment import (
    VideoActionResidualAdapter,
)
from scripts.compare_stage3_adapter_exports import (
    compare_stage3_adapter_exports,
)


BASE_SHA = "a" * 64
DATA_SHA = "b" * 64
GIT_COMMIT = "c" * 40
CONTRACT_SHA = "d" * 64
ASSET_SHA = "e" * 64
ALIGNMENT_CONFIG = {
    "action_hidden_dim": 4,
    "video_hidden_dim": 6,
    "action_dim": 2,
    "bottleneck_dim": 4,
    "num_heads": 2,
    "ffn_multiplier": 2,
    "drop_first_video_frame": True,
    "zero_init_output": True,
}


def _adapter_state(*, value: float = 1.0, config: dict | None = None) -> dict:
    adapter = VideoActionResidualAdapter(**(config or ALIGNMENT_CONFIG))
    return {
        name: torch.full_like(tensor, value + index / 100.0)
        for index, (name, tensor) in enumerate(adapter.state_dict().items())
    }


def _payload(*, value: float = 1.0, step: int = 2) -> dict:
    return {
        "schema_version": 2,
        "kind": "stage3_alignment_export",
        "base_checkpoint": "/formal/base.pt",
        "base_checkpoint_sha256": BASE_SHA,
        "data_manifest_sha256": DATA_SHA,
        "alignment_config": dict(ALIGNMENT_CONFIG),
        "global_step": step,
        "git_commit": GIT_COMMIT,
        "training_contract_sha256": CONTRACT_SHA,
        "asset_identities": {
            "vae": {
                "path": "/formal/vae.pt",
                "sha256": ASSET_SHA,
                "size_bytes": 123,
            },
            "normalization_stats": {
                "path": "/formal/dataset_stats.json",
                "sha256": "f" * 64,
                "size_bytes": 456,
            },
        },
        "adapter": _adapter_state(value=value),
    }


def _write(path, payload) -> None:
    torch.save(payload, path)


def _write_pair(tmp_path, left_payload=None, right_payload=None):
    left = tmp_path / "left.pt"
    right = tmp_path / "right.pt"
    _write(left, _payload() if left_payload is None else left_payload)
    _write(right, _payload() if right_payload is None else right_payload)
    return left, right


def _first_tensor(payload: dict) -> str:
    return next(iter(payload["adapter"]))


def test_compare_stage3_adapter_exports_accepts_exact_equivalence(tmp_path):
    left, right = _write_pair(tmp_path)

    receipt = compare_stage3_adapter_exports(left, right, expected_step=2)

    expected_parameter_count = sum(
        tensor.numel() for tensor in _payload()["adapter"].values()
    )
    assert receipt["status"] == "ok"
    assert receipt["global_step"] == 2
    assert receipt["tensor_count"] == len(_payload()["adapter"])
    assert receipt["parameter_count"] == expected_parameter_count
    assert receipt["left"]["size_bytes"] == len(left.read_bytes())
    assert receipt["left"]["sha256"] == hashlib.sha256(left.read_bytes()).hexdigest()
    assert receipt["left"]["st_dev"] == left.stat().st_dev
    assert receipt["left"]["st_ino"] == left.stat().st_ino


def test_compare_stage3_adapter_exports_rejects_metadata_drift(tmp_path):
    drifted = _payload()
    drifted["training_contract_sha256"] = "f" * 64
    left, right = _write_pair(tmp_path, right_payload=drifted)

    with pytest.raises(ValueError, match="metadata differs"):
        compare_stage3_adapter_exports(left, right)


def test_compare_stage3_adapter_exports_rejects_tensor_drift(tmp_path):
    drifted = _payload()
    name = _first_tensor(drifted)
    drifted["adapter"][name] = drifted["adapter"][name].clone()
    drifted["adapter"][name].reshape(-1)[0] += 1.0
    left, right = _write_pair(tmp_path, right_payload=drifted)

    with pytest.raises(ValueError, match="tensor values differ"):
        compare_stage3_adapter_exports(left, right)


def test_compare_stage3_adapter_exports_rejects_wrong_step_and_top_schema(tmp_path):
    left, right = _write_pair(tmp_path)
    with pytest.raises(ValueError, match="expected_step"):
        compare_stage3_adapter_exports(left, right, expected_step=3)

    invalid = _payload()
    invalid["unexpected"] = True
    _write(right, invalid)
    with pytest.raises(ValueError, match="schema is invalid"):
        compare_stage3_adapter_exports(left, right)


def test_compare_stage3_adapter_exports_rejects_non_integer_schema_version(tmp_path):
    left_payload = _payload()
    right_payload = _payload()
    left_payload["schema_version"] = right_payload["schema_version"] = 2.0
    left, right = _write_pair(tmp_path, left_payload, right_payload)

    with pytest.raises(ValueError, match="header is invalid"):
        compare_stage3_adapter_exports(left, right)


def test_compare_stage3_adapter_exports_rejects_same_resolved_path(tmp_path):
    source = tmp_path / "source.pt"
    _write(source, _payload())

    with pytest.raises(ValueError, match="distinct resolved paths"):
        compare_stage3_adapter_exports(source, source)


def test_compare_stage3_adapter_exports_rejects_same_inode(tmp_path):
    left = tmp_path / "left.pt"
    right = tmp_path / "right.pt"
    _write(left, _payload())
    os.link(left, right)

    with pytest.raises(ValueError, match="same inode"):
        compare_stage3_adapter_exports(left, right)


@pytest.mark.parametrize(
    "field",
    [
        "base_checkpoint_sha256",
        "data_manifest_sha256",
        "training_contract_sha256",
    ],
)
def test_compare_stage3_adapter_exports_rejects_common_fake_sha(tmp_path, field):
    left_payload = _payload()
    right_payload = _payload()
    left_payload[field] = right_payload[field] = "NOT-A-SHA"
    left, right = _write_pair(tmp_path, left_payload, right_payload)

    with pytest.raises(ValueError, match="64 lowercase hex"):
        compare_stage3_adapter_exports(left, right)


def test_compare_stage3_adapter_exports_rejects_common_missing_parameter(tmp_path):
    left_payload = _payload()
    right_payload = _payload()
    name = _first_tensor(left_payload)
    del left_payload["adapter"][name]
    del right_payload["adapter"][name]
    left, right = _write_pair(tmp_path, left_payload, right_payload)

    with pytest.raises(ValueError, match="tensor names are incomplete"):
        compare_stage3_adapter_exports(left, right)


@pytest.mark.parametrize("dtype", [torch.int64, torch.bool, torch.complex64])
def test_compare_stage3_adapter_exports_rejects_non_fp32_tensor(tmp_path, dtype):
    left_payload = _payload()
    right_payload = _payload()
    name = _first_tensor(left_payload)
    shape = left_payload["adapter"][name].shape
    left_payload["adapter"][name] = torch.ones(shape, dtype=dtype)
    right_payload["adapter"][name] = torch.ones(shape, dtype=dtype)
    left, right = _write_pair(tmp_path, left_payload, right_payload)

    with pytest.raises(ValueError, match="dense CPU FP32"):
        compare_stage3_adapter_exports(left, right)


def test_compare_stage3_adapter_exports_rejects_noncontiguous_tensor(tmp_path):
    left_payload = _payload()
    right_payload = _payload()
    name = "action_proj.weight"
    left_payload["adapter"][name] = left_payload["adapter"][name].T
    right_payload["adapter"][name] = right_payload["adapter"][name].T
    left, right = _write_pair(tmp_path, left_payload, right_payload)

    with pytest.raises(ValueError, match="contiguous"):
        compare_stage3_adapter_exports(left, right)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_compare_stage3_adapter_exports_rejects_nonfinite_tensor(tmp_path, value):
    left_payload = _payload()
    right_payload = _payload()
    name = _first_tensor(left_payload)
    left_payload["adapter"][name].reshape(-1)[0] = value
    right_payload["adapter"][name].reshape(-1)[0] = value
    left, right = _write_pair(tmp_path, left_payload, right_payload)

    with pytest.raises(ValueError, match="non-finite"):
        compare_stage3_adapter_exports(left, right)


def test_compare_stage3_adapter_exports_rejects_common_shape_drift(tmp_path):
    left_payload = _payload()
    right_payload = _payload()
    name = _first_tensor(left_payload)
    left_payload["adapter"][name] = torch.ones(1, dtype=torch.float32)
    right_payload["adapter"][name] = torch.ones(1, dtype=torch.float32)
    left, right = _write_pair(tmp_path, left_payload, right_payload)

    with pytest.raises(ValueError, match="tensor shape is invalid"):
        compare_stage3_adapter_exports(left, right)


def test_compare_stage3_adapter_exports_rejects_parameter_name_drift(tmp_path):
    drifted = _payload()
    name = _first_tensor(drifted)
    drifted["adapter"][f"wrong.{name}"] = drifted["adapter"].pop(name)
    left, right = _write_pair(tmp_path, right_payload=drifted)

    with pytest.raises(ValueError, match="tensor names are incomplete"):
        compare_stage3_adapter_exports(left, right)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda config: config.pop("action_dim"),
        lambda config: config.__setitem__("action_hidden_dim", True),
        lambda config: config.__setitem__("bottleneck_dim", 3),
    ],
    ids=["missing-field", "bool-dimension", "not-divisible"],
)
def test_compare_stage3_adapter_exports_rejects_invalid_config(tmp_path, mutate):
    left_payload = _payload()
    right_payload = _payload()
    mutate(left_payload["alignment_config"])
    mutate(right_payload["alignment_config"])
    left, right = _write_pair(tmp_path, left_payload, right_payload)

    with pytest.raises(ValueError, match="alignment_config"):
        compare_stage3_adapter_exports(left, right)


def test_compare_stage3_adapter_exports_limits_logical_parameter_count(tmp_path):
    left_payload = _payload()
    right_payload = _payload()
    huge_dimension = comparator.MAX_ADAPTER_PARAMETER_COUNT // 4 + 1
    left_payload["alignment_config"]["action_hidden_dim"] = huge_dimension
    right_payload["alignment_config"]["action_hidden_dim"] = huge_dimension
    left, right = _write_pair(tmp_path, left_payload, right_payload)

    with pytest.raises(ValueError, match="parameter count exceeds"):
        compare_stage3_adapter_exports(left, right)


def test_compare_stage3_adapter_exports_rejects_invalid_asset_identity(tmp_path):
    left_payload = _payload()
    right_payload = _payload()
    left_payload["asset_identities"]["vae"]["size_bytes"] = 0
    right_payload["asset_identities"]["vae"]["size_bytes"] = 0
    left, right = _write_pair(tmp_path, left_payload, right_payload)

    with pytest.raises(ValueError, match="positive integer"):
        compare_stage3_adapter_exports(left, right)


def test_compare_stage3_adapter_exports_rejects_common_missing_asset(tmp_path):
    left_payload = _payload()
    right_payload = _payload()
    del left_payload["asset_identities"]["normalization_stats"]
    del right_payload["asset_identities"]["normalization_stats"]
    left, right = _write_pair(tmp_path, left_payload, right_payload)

    with pytest.raises(ValueError, match="exactly vae and normalization_stats"):
        compare_stage3_adapter_exports(left, right)


def test_compare_stage3_adapter_exports_validates_expected_identities(tmp_path):
    left, right = _write_pair(tmp_path)

    receipt = compare_stage3_adapter_exports(
        left,
        right,
        expected_base_checkpoint_sha256=BASE_SHA,
        expected_data_manifest_sha256=DATA_SHA,
        expected_training_contract_sha256=CONTRACT_SHA,
        expected_git_commit=GIT_COMMIT,
    )
    assert receipt["base_checkpoint_sha256"] == BASE_SHA

    with pytest.raises(ValueError, match="differs from expectation"):
        compare_stage3_adapter_exports(
            left,
            right,
            expected_data_manifest_sha256="f" * 64,
        )
    with pytest.raises(ValueError, match="64 lowercase hex"):
        compare_stage3_adapter_exports(
            left,
            right,
            expected_training_contract_sha256="INVALID",
        )
    with pytest.raises(ValueError, match="git_commit differs"):
        compare_stage3_adapter_exports(
            left,
            right,
            expected_git_commit="1" * 40,
        )
    with pytest.raises(ValueError, match="expected_git_commit"):
        compare_stage3_adapter_exports(
            left,
            right,
            expected_git_commit="INVALID",
        )


def test_compare_stage3_adapter_exports_enforces_maximum_size(tmp_path, monkeypatch):
    left, right = _write_pair(tmp_path)
    monkeypatch.setattr(comparator, "MAX_EXPORT_SIZE_BYTES", 16)

    with pytest.raises(ValueError, match="maximum allowed size"):
        compare_stage3_adapter_exports(left, right)


def test_compare_stage3_adapter_exports_rejects_non_regular_file_without_blocking(
    tmp_path,
):
    fifo = tmp_path / "export.fifo"
    other = tmp_path / "other.pt"
    os.mkfifo(fifo)
    _write(other, _payload())

    with pytest.raises(ValueError, match="not a regular file"):
        compare_stage3_adapter_exports(fifo, other)
