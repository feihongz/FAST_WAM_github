from __future__ import annotations

from inspect import signature
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationResolutionError
import pytest

from fastwam.datasets.lerobot.robot_video_dataset import RobotVideoDataset
from fastwam.gating.contracts import require_sha256
from scripts.generate_gate_labels import _resolved_config


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_NAME = "libero_stage2_gate_labels_2cam224"
MANIFEST_SHA256 = "08da49109a57b55c67f3fa4ac31fbfa44e44dd541a194a5d3420838537d0d320"
SELECTION_DIR = (
    "/root/feihong/FastWAM/formal_runs/contracts/stage2/"
    "libero_nested64_stratified_v2_426b635d"
)
SELECTION_SHA256 = (
    "426b635d637a0f3e5d31dd13612ff5ad786fd5cfe9ce27b0e8689854d9aa9e9b"
)
FORMAL_COVERAGE_SHA256 = (
    "d114ac25b61ab30f18185c9ea69a33d537b5196b145a8c5c3d6f6fd9d884708f"
)
SPLIT_ASSIGNMENT_SHA256 = (
    "a77efa24249dab8cfacbc228b1da341947240b36fa77d90182701c07bdcf7787"
)
FINAL_ADAPTER_PATH = (
    "/root/feihong/FastWAM/formal_runs/stage3/full/"
    "libero_stage3_alignment_2cam224_1e-4/2026-08-30_10-29-08/"
    "checkpoints/exports/step_030000.pt"
)
FINAL_ADAPTER_SHA256 = (
    "cbc593bc6ce99c0249a65e5c7cef754c9a1d7ea602f81fdae2b8cb158a25858c"
)
LIBERO_ENVIRONMENT_OVERRIDES = (
    "FASTWAM_LIBERO_STAGE2_LABEL_JOB",
    "FASTWAM_LIBERO_STAGE3_BASE_CHECKPOINT",
    "FASTWAM_LIBERO_STAGE3_BASE_SHA256",
    "FASTWAM_LIBERO_STAGE3_ADAPTER",
    "FASTWAM_LIBERO_STAGE3_ADAPTER_SHA256",
    "FASTWAM_LIBERO_STAGE3_VAE",
    "FASTWAM_LIBERO_STATS",
    "FASTWAM_LIBERO_STAGE3_DATA_MANIFEST",
    "FASTWAM_LIBERO_STAGE3_DATA_MANIFEST_SHA256",
    "FASTWAM_LIBERO_STAGE2_SELECTION_DIR",
    "FASTWAM_LIBERO_STAGE2_SELECTION_SHA256",
    "FASTWAM_LIBERO_STAGE2_COVERAGE_SHA256",
)


def _compose(monkeypatch) -> dict:
    for name in LIBERO_ENVIRONMENT_OVERRIDES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(
        "FASTWAM_LIBERO_STAGE2_LABEL_JOB",
        "/durable/formal_runs/libero/stage2/label_job",
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


def test_libero_stage2_label_task_locks_formal_contract(monkeypatch):
    resolved = _compose(monkeypatch)
    train = resolved["data"]["train"]

    assert resolved["output_dir"] == (
        "/durable/formal_runs/libero/stage2/label_job"
    )
    assert resolved["model"]["_target_"] == (
        "fastwam.runtime.create_fastwam_unified_aligned"
    )
    assert resolved["model"]["action_dit_config"]["action_dim"] == 7
    assert resolved["model"]["proprio_dim"] == 8
    assert train["seed"] == 42
    assert train["val_set_proportion"] == pytest.approx(0.0)
    assert train["is_training_set"] is True
    assert train["strict_data_mode"] is True
    assert train["video_backend"] == "torchcodec"
    assert train["save_stats_copy"] is False
    assert train["processor"]["num_output_cameras"] == 2
    assert train["processor"]["action_output_dim"] == 7
    assert train["processor"]["proprio_output_dim"] == 8
    assert len(train["dataset_dirs"]) == 4
    assert resolved["data"]["val"] is None

    configured_fields = set(train) - {"_target_"}
    supported_fields = set(signature(RobotVideoDataset.__init__).parameters) - {
        "self"
    }
    assert configured_fields <= supported_fields, sorted(
        configured_fields - supported_fields
    )

    assert resolved["base"]["expected_sha256"] == (
        "17a5588cc2b8d162219c9daf818614f614ee4a7921933a4a26c5d678111330e9"
    )
    assert resolved["assets"]["vae"]["expected_sha256"] == (
        "20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36"
    )
    assert resolved["assets"]["normalization_stats"] == {
        "path": (
            "/root/feihong/FastWAM/formal_runs/FAST_WAM_github/"
            "libero_unified_shared_2cam224_1e-4/2026-07-01_00-44-20/"
            "dataset_stats.json"
        ),
        "expected_sha256": (
            "30f81ad7d5076e97323e3328bce003e01a04cb21327b5bacd21bb72846768638"
        ),
    }
    assert resolved["data_manifest"] == {
        "path": (
            "/root/feihong/FastWAM/formal_runs/contracts/stage3/"
            "libero_current_273465f_1693e/libero_stage3_data_manifest.json"
        ),
        "expected_sha256": MANIFEST_SHA256,
    }
    assert resolved["adapter"] == {
        "checkpoint": FINAL_ADAPTER_PATH,
        "expected_sha256": FINAL_ADAPTER_SHA256,
    }
    assert resolved["label_selection"] == {
        "directory": SELECTION_DIR,
        "expected_sha256": SELECTION_SHA256,
    }
    assert resolved["label_coverage"] == {
        "tier": "formal",
        "expected_sha256": FORMAL_COVERAGE_SHA256,
    }

    assert resolved["episode_split"] == {
        "path": f"{SELECTION_DIR}/episode_split.json",
        "validation_fraction": 0.1,
        "split_seed": 42,
        "expected_assignment_sha256": SPLIT_ASSIGNMENT_SHA256,
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


def test_libero_stage2_label_task_locks_manifest_and_final_adapter(
    monkeypatch,
):
    resolved = _compose(monkeypatch)
    assert require_sha256(
        resolved["data_manifest"]["expected_sha256"],
        field="manifest",
    ) == MANIFEST_SHA256
    assert resolved["adapter"]["checkpoint"] == FINAL_ADAPTER_PATH
    assert require_sha256(
        resolved["adapter"]["expected_sha256"],
        field="adapter",
    ) == FINAL_ADAPTER_SHA256


def test_libero_stage2_label_task_requires_durable_output_env(monkeypatch):
    for name in LIBERO_ENVIRONMENT_OVERRIDES:
        monkeypatch.delenv(name, raising=False)
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
        match="FASTWAM_LIBERO_STAGE2_LABEL_JOB",
    ):
        OmegaConf.to_container(config, resolve=True)
