from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationResolutionError
import pytest

from fastwam.gating.artifacts import canonical_json_sha256
from scripts import generate_gate_labels as generate_cli
from scripts import train_video_gate as train_cli


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_NAME = "libero_stage2_gate_2cam224"
LABEL_TASK_NAME = "libero_stage2_gate_labels_2cam224"
DATA_MANIFEST_SHA256 = (
    "08da49109a57b55c67f3fa4ac31fbfa44e44dd541a194a5d3420838537d0d320"
)
SELECTION_SHA256 = (
    "426b635d637a0f3e5d31dd13612ff5ad786fd5cfe9ce27b0e8689854d9aa9e9b"
)
COVERAGE_SHA256 = (
    "d114ac25b61ab30f18185c9ea69a33d537b5196b145a8c5c3d6f6fd9d884708f"
)
CONTRACT_SHA256 = (
    "0e089ebc97b0532484a7dacf526cfc8e2c68894e637fe7d9e483fa566e46ff17"
)
LABEL_MANIFEST_SHA256 = (
    "d6dc98a6a36c30150db30000c86d07c7a1e7d90b1dc5d1a5a60e02126c22b3e0"
)
DATA_CONFIG_SHA256 = (
    "44dc596c6700e02e69ba12823ed899d12d25c6980263f9bf3ac85cb73d53daa4"
)


def _compose(config_name: str, task_name: str):
    with initialize_config_dir(
        config_dir=str(REPO_ROOT / "configs"),
        version_base="1.3",
    ):
        return compose(
            config_name=config_name,
            overrides=[f"task={task_name}"],
        )


def test_libero_gate_task_pins_formal_merged_artifact_and_small_model_contract(
    monkeypatch,
):
    monkeypatch.setenv("FASTWAM_LIBERO_STAGE2_GATE_RUN", "/durable/gate")
    config = _compose("train_video_gate", TASK_NAME)
    resolved = train_cli._resolved_config(config)

    assert resolved["output_dir"] == "/durable/gate"
    assert resolved["data_manifest"]["expected_sha256"] == DATA_MANIFEST_SHA256
    assert resolved["label_selection"]["expected_sha256"] == SELECTION_SHA256
    assert resolved["label_coverage"] == {
        "tier": "formal",
        "expected_sha256": COVERAGE_SHA256,
    }
    assert resolved["label_contract"]["expected_sha256"] == CONTRACT_SHA256
    assert resolved["label_manifest"]["expected_sha256"] == (
        LABEL_MANIFEST_SHA256
    )
    assert resolved["episode_split"]["expected_assignment_sha256"] == (
        "a77efa24249dab8cfacbc228b1da341947240b36fa77d90182701c07bdcf7787"
    )
    assert resolved["source_identities"] == {
        "base_checkpoint_sha256": (
            "17a5588cc2b8d162219c9daf818614f614ee4a7921933a4a26c5d678111330e9"
        ),
        "adapter_checkpoint_sha256": (
            "cbc593bc6ce99c0249a65e5c7cef754c9a1d7ea602f81fdae2b8cb158a25858c"
        ),
    }
    assert resolved["gate"]["proprio_dim"] == 8
    assert resolved["gate"]["context_dim"] == 4096
    assert resolved["training"] == {
        "seed": 42,
        "batch_size": 64,
        "num_workers": 0,
        "pin_memory": True,
        "shuffle": True,
        "learning_rate": pytest.approx(1.0e-4),
        "weight_decay": pytest.approx(1.0e-4),
        "max_grad_norm": pytest.approx(1.0),
        "num_epochs": 20,
        "early_stop_patience": 3,
        "min_delta": pytest.approx(0.0),
        "threshold": pytest.approx(0.5),
        "num_calibration_bins": 10,
    }


def test_libero_gate_data_exactly_reproduces_label_generation_contract(
    monkeypatch,
):
    monkeypatch.setenv("FASTWAM_LIBERO_STAGE2_GATE_RUN", "/durable/gate")
    monkeypatch.setenv(
        "FASTWAM_LIBERO_STAGE2_LABEL_JOB", "/durable/labels"
    )
    gate = train_cli._resolved_config(
        _compose("train_video_gate", TASK_NAME)
    )
    generated = generate_cli._resolved_config(
        _compose("generate_gate_labels", LABEL_TASK_NAME)
    )
    gate_data = train_cli._canonicalize_data_paths(
        gate["data"], repo_dir=REPO_ROOT
    )
    generated_data = generate_cli._canonicalize_data_paths(
        generated["data"], repo_dir=REPO_ROOT
    )

    assert gate_data == generated_data
    assert gate_data["val"] is None
    assert gate_data["train"]["seed"] == 42
    assert canonical_json_sha256(gate_data) == DATA_CONFIG_SHA256


def test_libero_gate_task_requires_explicit_durable_output(monkeypatch):
    monkeypatch.delenv("FASTWAM_LIBERO_STAGE2_GATE_RUN", raising=False)
    config = _compose("train_video_gate", TASK_NAME)
    with pytest.raises(
        InterpolationResolutionError,
        match="FASTWAM_LIBERO_STAGE2_GATE_RUN",
    ):
        OmegaConf.to_container(config, resolve=True)
