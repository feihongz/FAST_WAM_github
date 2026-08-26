from inspect import signature
from pathlib import Path

from accelerate import Accelerator
from hydra import compose, initialize_config_dir
import pytest

from fastwam.alignment.runtime import (
    _resolve_data_identity,
    _resolved_config,
)
from fastwam.datasets.lerobot.robot_video_dataset import RobotVideoDataset
from fastwam.utils.config_resolvers import register_default_resolvers


REPO_ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDER = "REPLACE_AFTER_IDENTITY_BUILD"
TASK_NAME = "robotwin_stage3_alignment_3cam384_1e-4"
ROBOTWIN_ENVIRONMENT_OVERRIDES = (
    "FASTWAM_ROBOTWIN_STAGE3_BASE_CHECKPOINT",
    "FASTWAM_ROBOTWIN_STAGE3_BASE_SHA256",
    "FASTWAM_ROBOTWIN_STAGE3_VAE",
    "FASTWAM_ROBOTWIN_STATS",
    "FASTWAM_ROBOTWIN_STAGE3_DATA_MANIFEST",
    "FASTWAM_ROBOTWIN_STAGE3_DATA_MANIFEST_SHA256",
    "FASTWAM_ROBOTWIN_TEXT_CACHE_INDEX_DESCRIPTOR",
    "FASTWAM_ROBOTWIN_STAGE3_EXPECTED_DATASET_LENGTH",
    "FASTWAM_ROBOTWIN_STAGE3_EXPECTED_DATASET_EPISODES",
)


def _compose_robotwin_stage3(monkeypatch) -> dict:
    for name in ROBOTWIN_ENVIRONMENT_OVERRIDES:
        monkeypatch.delenv(name, raising=False)
    register_default_resolvers()
    with initialize_config_dir(
        config_dir=str(REPO_ROOT / "configs"),
        version_base="1.3",
    ):
        config = compose(
            config_name="train_stage3_alignment",
            overrides=[f"task={TASK_NAME}"],
        )
    return _resolved_config(config)


def test_robotwin_stage3_train_fields_match_dataset_constructor(monkeypatch):
    train = _compose_robotwin_stage3(monkeypatch)["data"]["train"]
    configured_fields = set(train) - {"_target_"}
    supported_fields = set(signature(RobotVideoDataset.__init__).parameters) - {
        "self"
    }

    assert configured_fields <= supported_fields, sorted(
        configured_fields - supported_fields
    )


def test_robotwin_stage3_hydra_contract_resolves(monkeypatch):
    resolved = _compose_robotwin_stage3(monkeypatch)
    train = resolved["data"]["train"]

    assert resolved["model"]["_target_"] == (
        "fastwam.runtime.create_fastwam_unified_aligned"
    )
    assert resolved["model"]["skip_dit_load_from_pretrain"] is True
    assert resolved["model"]["redirect_common_files"] is False
    assert resolved["model"]["action_dit_config"]["action_dim"] == 14
    assert resolved["model"]["proprio_dim"] == 14

    assert train["video_backend"] == "torchcodec"
    assert train["strict_data_mode"] is True
    assert train["save_stats_copy"] is False
    assert train["concat_multi_camera"] == "robotwin"
    assert train["val_set_proportion"] == pytest.approx(0.01)
    assert train["is_training_set"] is True
    assert train["seed"] == 42
    assert resolved["data"]["val"] is None

    image_keys = [row["key"] for row in train["shape_meta"]["images"]]
    assert image_keys == ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    assert train["processor"]["num_output_cameras"] == 3
    assert train["processor"]["action_output_dim"] == 14
    assert train["processor"]["proprio_output_dim"] == 14
    assert train["shape_meta"]["action"] == [
        {"key": "default", "raw_shape": 14, "shape": 14}
    ]
    assert train["shape_meta"]["state"] == [
        {"key": "default", "raw_shape": 14, "shape": 14}
    ]
    assert train["num_frames"] == 33
    assert train["action_video_freq_ratio"] == 4
    assert train["video_size"] == [384, 320]

    assert resolved["assets"]["normalization_stats"] == {
        "path": "/root/feihong/FastWAM/datasets/robotwin2.0/dataset_stats.json",
        "expected_sha256": (
            "7a02c46cfc8c5e746c0afbe41fca73f723eda34cbc083f8ca54f76d8f7468095"
        ),
    }
    assert resolved["assets"]["vae"]["expected_sha256"] == (
        "20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36"
    )
    assert resolved["data_manifest"]["schema_version"] == 2
    assert resolved["data_manifest"]["text_cache_index_descriptor_path"] == (
        "/root/feihong/FastWAM/formal_runs/contracts/stage3/"
        "robotwin_train_6011575f_27225e/robotwin_text_cache_index.json"
    )


def test_robotwin_stage3_locks_base_train_split_and_legal_8gpu_batch(monkeypatch):
    resolved = _compose_robotwin_stage3(monkeypatch)
    train = resolved["data"]["train"]
    runtime = resolved["runtime"]
    training = resolved["training"]

    assert int(27_500 * (1.0 - train["val_set_proportion"])) == 27_225
    assert int(runtime["expected_dataset_episodes"]) == 27_225
    assert int(runtime["expected_dataset_length"]) == 6_011_575

    world_size = 8
    batch_size = int(training["batch_size"])
    accumulation = int(training["gradient_accumulation_steps"])
    micro_batches = int(runtime["expected_dataset_length"]) // (
        world_size * batch_size
    )
    assert (batch_size, accumulation) == (2, 3)
    assert micro_batches == 375_723
    assert micro_batches % accumulation == 0
    assert batch_size * world_size * accumulation == 48
    assert int(runtime["expected_dataset_length"]) - (
        micro_batches * world_size * batch_size
    ) == 7


def test_robotwin_stage3_manifest_placeholder_fails_closed(monkeypatch):
    resolved = _compose_robotwin_stage3(monkeypatch)
    assert resolved["base"]["expected_sha256"] == (
        "368a99ca9575a78d01f4cdcdee8820ec74d30c4528cf7aff07b83361a17cbbda"
    )
    assert resolved["data_manifest"]["expected_sha256"] == PLACEHOLDER

    accelerator = Accelerator(
        cpu=True,
        step_scheduler_with_optimizer=False,
    )
    with pytest.raises(
        RuntimeError,
        match="requires data_manifest.path and data_manifest.expected_sha256",
    ):
        _resolve_data_identity(
            accelerator,
            object(),
            manifest_config=dict(resolved["data_manifest"]),
            normalization_stats_path=resolved["assets"]["normalization_stats"][
                "path"
            ],
        )
