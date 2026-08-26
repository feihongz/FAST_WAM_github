from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationResolutionError
import pytest

from fastwam.gating.contracts import require_sha256
from scripts.generate_gate_labels import _resolved_config


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_NAME = "robotwin_stage2_gate_labels_3cam384"
MANIFEST_SHA256 = "1190b75b1ef19a7abd949bdff5679da59afa7e51a043eeb43663cf2c4495173c"
ADAPTER_PLACEHOLDER = "REPLACE_AFTER_STAGE3_ADAPTER_EXPORT"


def _compose(monkeypatch) -> dict:
    for name in (
        "FASTWAM_ROBOTWIN_STAGE2_LABEL_JOB",
        "FASTWAM_ROBOTWIN_STAGE3_BASE_CHECKPOINT",
        "FASTWAM_ROBOTWIN_STAGE3_BASE_SHA256",
        "FASTWAM_ROBOTWIN_STAGE3_ADAPTER",
        "FASTWAM_ROBOTWIN_STAGE3_ADAPTER_SHA256",
        "FASTWAM_ROBOTWIN_STAGE3_VAE",
        "FASTWAM_ROBOTWIN_STATS",
        "FASTWAM_ROBOTWIN_STAGE3_DATA_MANIFEST",
        "FASTWAM_ROBOTWIN_STAGE3_DATA_MANIFEST_SHA256",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(
        "FASTWAM_ROBOTWIN_STAGE2_LABEL_JOB",
        "/durable/formal_runs/robotwin/stage2/label_job",
    )
    with initialize_config_dir(
        config_dir=str(REPO_ROOT / "configs"),
        version_base="1.3",
    ):
        config = compose(
            config_name="generate_gate_labels",
            overrides=[f"task={TASK_NAME}"],
        )
    return _resolved_config(config)


def test_robotwin_stage2_label_task_locks_formal_contract(monkeypatch):
    resolved = _compose(monkeypatch)
    train = resolved["data"]["train"]

    assert resolved["output_dir"] == (
        "/durable/formal_runs/robotwin/stage2/label_job"
    )
    assert resolved["model"]["_target_"] == (
        "fastwam.runtime.create_fastwam_unified_aligned"
    )
    assert resolved["model"]["action_dit_config"]["action_dim"] == 14
    assert resolved["model"]["proprio_dim"] == 14
    assert train["seed"] == 42
    assert train["val_set_proportion"] == pytest.approx(0.01)
    assert train["is_training_set"] is True
    assert train["strict_data_mode"] is True
    assert train["video_backend"] == "torchcodec"
    assert train["save_stats_copy"] is False
    assert resolved["data"]["val"] is None

    assert resolved["base"]["expected_sha256"] == (
        "368a99ca9575a78d01f4cdcdee8820ec74d30c4528cf7aff07b83361a17cbbda"
    )
    assert resolved["assets"]["vae"]["expected_sha256"] == (
        "20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36"
    )
    assert resolved["assets"]["normalization_stats"]["expected_sha256"] == (
        "7a02c46cfc8c5e746c0afbe41fca73f723eda34cbc083f8ca54f76d8f7468095"
    )
    assert resolved["data_manifest"]["expected_sha256"] == MANIFEST_SHA256
    assert resolved["adapter"] == {
        "checkpoint": ADAPTER_PLACEHOLDER,
        "expected_sha256": ADAPTER_PLACEHOLDER,
    }

    assert resolved["episode_split"] == {
        "path": "/durable/formal_runs/robotwin/stage2/label_job/episode_split.json",
        "validation_fraction": 0.1,
        "split_seed": 42,
        "expected_assignment_sha256": "",
    }
    assert resolved["labeling"] == {
        "base_seed": 42,
        "num_seed_pairs": 2,
        "relative_margin": 0.05,
        "relative_gain_epsilon": 1.0e-12,
        "num_inference_steps": 10,
        "sigma_shift": None,
        "rand_device": "cpu",
        "tiled": False,
        "num_shards": 64,
        "chunk_size": 64,
        "shard_indices": None,
        "contract_file": "label_contract.json",
        "runtime_config_file": "label_runtime_config.json",
    }
    assert resolved["runtime"]["required_environment"] == {
        "DIFFSYNTH_MODEL_BASE_PATH": "/root/feihong/FastWAM/checkpoints",
        "DIFFSYNTH_SKIP_DOWNLOAD": "true",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }


def test_robotwin_stage2_label_task_locks_manifest_and_adapter_fails_closed(
    monkeypatch,
):
    resolved = _compose(monkeypatch)
    assert require_sha256(
        resolved["data_manifest"]["expected_sha256"],
        field="manifest",
    ) == MANIFEST_SHA256
    with pytest.raises(ValueError, match="64 lowercase hex"):
        require_sha256(
            resolved["adapter"]["expected_sha256"],
            field="adapter",
        )


def test_robotwin_stage2_label_task_requires_durable_output_env(monkeypatch):
    monkeypatch.delenv("FASTWAM_ROBOTWIN_STAGE2_LABEL_JOB", raising=False)
    with initialize_config_dir(
        config_dir=str(REPO_ROOT / "configs"),
        version_base="1.3",
    ):
        config = compose(
            config_name="generate_gate_labels",
            overrides=[f"task={TASK_NAME}"],
        )
    with pytest.raises(
        InterpolationResolutionError,
        match="FASTWAM_ROBOTWIN_STAGE2_LABEL_JOB",
    ):
        OmegaConf.to_container(config, resolve=True)
