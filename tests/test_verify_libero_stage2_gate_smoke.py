from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from hydra import compose, initialize_config_dir
import pytest
import torch

from fastwam.gating.artifacts import canonical_json_sha256
from fastwam.gating.trainer import GateTrainer
from fastwam.models.video_gate import BinaryVideoGate
from scripts import train_video_gate as train_cli
from scripts.verify_libero_stage2_gate_smoke import (
    ADAPTER_SHA256,
    BASE_SHA256,
    DATA_MANIFEST_SHA256,
    EPISODE_ASSIGNMENT_SHA256,
    EXPECTED_GATE_CONFIG,
    EXPECTED_PARAMETER_COUNT,
    EXPECTED_TRAIN_NEGATIVES,
    EXPECTED_TRAIN_POSITIVES,
    EXPECTED_TRAINING_CONFIG,
    LABEL_MANIFEST_SHA256,
    verify,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
GIT_COMMIT = "1" * 40
GIT_IDENTITY = {
    "commit": GIT_COMMIT,
    "tracked_dirty": False,
    "untracked_source_files": [],
}


def _metrics(*, examples: int, batches: int) -> dict:
    return {
        "objective_bce": 0.65,
        "bce": 0.64,
        "auroc": 0.61,
        "auprc": 0.57,
        "positive_rate": 0.45,
        "predicted_positive_rate": 0.48,
        "expected_calibration_error": 0.08,
        "num_examples": examples,
        "num_batches": batches,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _training_contract() -> dict:
    with initialize_config_dir(
        config_dir=str(REPO_ROOT / "configs"),
        version_base="1.3",
    ):
        config = compose(
            config_name="train_video_gate",
            overrides=[
                "task=libero_stage2_gate_2cam224",
                "output_dir=/tmp/test-libero-gate-smoke",
                "training.num_epochs=1",
            ],
        )
    resolved = train_cli._resolved_config(config)
    data = train_cli._canonicalize_data_paths(
        resolved["data"], repo_dir=REPO_ROOT
    )
    numerical_runtime = {
        "versions": {"torch": str(torch.__version__)},
        "device": {
            "type": "cuda",
            "name": "NVIDIA H100 80GB HBM3",
        },
        "ffmpeg": {},
        "backend": {"deterministic_algorithms": True},
    }
    contract = train_cli.build_training_config_contract(
        data=data,
        gate=resolved["gate"],
        training=resolved["training"],
        runtime=resolved["runtime"],
        numerical_runtime=numerical_runtime,
    )
    assert contract["gate"] == EXPECTED_GATE_CONFIG
    assert contract["training"] == EXPECTED_TRAINING_CONFIG
    return contract


def _training_identity(training_config_sha256: str) -> dict:
    return {
        "label_manifest_sha256": LABEL_MANIFEST_SHA256,
        "adapter_checkpoint_sha256": ADAPTER_SHA256,
        "base_checkpoint_sha256": BASE_SHA256,
        "data_manifest_sha256": DATA_MANIFEST_SHA256,
        "episode_split_assignment_sha256": EPISODE_ASSIGNMENT_SHA256,
        "training_config_sha256": training_config_sha256,
        "git_identity": GIT_IDENTITY,
    }


def _populate_adamw_state(trainer: GateTrainer) -> None:
    zero_loss = sum(
        parameter.sum() * 0.0 for parameter in trainer.gate.parameters()
    )
    zero_loss.backward()
    trainer.optimizer.step()
    trainer.optimizer.zero_grad(set_to_none=True)
    for state in trainer.optimizer.state.values():
        state["step"].fill_(762)


def _build_complete_smoke(output_dir: Path) -> None:
    output_dir.mkdir()
    training_contract = _training_contract()
    training_config_sha256 = canonical_json_sha256(training_contract)
    training_identity = _training_identity(training_config_sha256)
    gate = BinaryVideoGate(**training_contract["gate"])
    assert gate.parameter_count() == EXPECTED_PARAMETER_COUNT
    val = _metrics(examples=5_408, batches=85)
    train = _metrics(examples=48_768, batches=762)
    run_identity = {
        "schema_version": 1,
        "kind": "stage2_binary_video_gate_run_identity",
        "training_config": training_contract,
        "training_config_sha256": training_config_sha256,
        "training_identity": training_identity,
    }
    summary = {
        "schema_version": 1,
        "kind": "stage2_binary_video_gate_training_summary",
        "training_identity": training_identity,
        "initial_epoch": 0,
        "final_epoch": 1,
        "global_step": 762,
        "stopped_early": False,
        "best_epoch": 1,
        "best_val_bce": 0.65,
        "best_metrics": val,
        "history_complete": True,
        "epoch_history": [
            {
                "epoch": 1,
                "global_step": 762,
                "train": train,
                "val": val,
            }
        ],
        "new_epoch_history": [
            {
                "epoch": 1,
                "global_step": 762,
                "train": train,
                "val": val,
            }
        ],
        "state_file": "training_state.pt",
        "best_file": "gate_best.pt",
        "last_file": "gate_last.pt",
    }
    _write_json(output_dir / "run_identity.json", run_identity)
    _write_json(output_dir / "summary.json", summary)

    train_labels = torch.cat(
        (
            torch.ones(EXPECTED_TRAIN_POSITIVES, dtype=torch.float32),
            torch.zeros(EXPECTED_TRAIN_NEGATIVES, dtype=torch.float32),
        )
    )
    training = training_contract["training"]
    trainer = GateTrainer(
        gate,
        train_labels=train_labels,
        training_identity=training_identity,
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
        max_grad_norm=training["max_grad_norm"],
    )
    _populate_adamw_state(trainer)
    trainer.epoch = 1
    trainer.global_step = 762
    trainer.best_epoch = 1
    trainer.best_global_step = 762
    trainer.best_val_bce = 0.65
    trainer.best_metrics = dict(val)
    trainer.epochs_without_improvement = 0
    trainer._best_gate_state = {
        name: value.detach().cpu().clone()
        for name, value in trainer.gate.state_dict().items()
    }
    trainer.save_training_state(output_dir / "training_state.pt")
    export_kwargs = {
        "label_manifest_sha256": LABEL_MANIFEST_SHA256,
        "adapter_checkpoint_sha256": ADAPTER_SHA256,
        "data_manifest_sha256": DATA_MANIFEST_SHA256,
        "episode_split_assignment_sha256": EPISODE_ASSIGNMENT_SHA256,
        "training_config_sha256": training_config_sha256,
        "git_identity": GIT_IDENTITY,
    }
    trainer.export_checkpoint(
        output_dir / "gate_best.pt", selection="best", **export_kwargs
    )
    trainer.export_checkpoint(
        output_dir / "gate_last.pt", selection="last", **export_kwargs
    )


def test_verifier_strictly_loads_both_gate_exports_and_publishes_receipt(
    tmp_path,
):
    output_dir = tmp_path / "gate_run"
    receipt = tmp_path / "receipt.json"
    _build_complete_smoke(output_dir)

    result = verify(
        output_dir=output_dir,
        expected_git_commit=GIT_COMMIT,
        receipt=receipt,
        resume_device="cpu",
    )

    assert result["status"] == "pass"
    assert result["global_step"] == 762
    assert result["parameter_count"] == EXPECTED_PARAMETER_COUNT
    assert set(result["artifact_sha256"]) == {
        "run_identity.json",
        "training_state.pt",
        "gate_best.pt",
        "gate_last.pt",
        "summary.json",
    }
    assert json.loads(receipt.read_text(encoding="utf-8")) == result
    assert verify(
        output_dir=output_dir,
        expected_git_commit=GIT_COMMIT,
        receipt=receipt,
        resume_device="cpu",
    ) == result


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.device_count() != 1
    or "H100" not in torch.cuda.get_device_name(0),
    reason="formal resume probe requires exactly one visible H100",
)
def test_verifier_formal_resume_probe_runs_on_h100(tmp_path):
    output_dir = tmp_path / "gate_run"
    receipt = tmp_path / "receipt.json"
    _build_complete_smoke(output_dir)

    environment = os.environ.copy()
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            (str(REPO_ROOT / "src"), environment.get("PYTHONPATH", "")),
        )
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "verify_libero_stage2_gate_smoke.py"),
            "--output-dir",
            str(output_dir),
            "--expected-git-commit",
            GIT_COMMIT,
            "--receipt",
            str(receipt),
        ],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(receipt.read_text(encoding="utf-8"))
    assert result["status"] == "pass"


def test_verifier_rejects_non_finite_or_incomplete_epoch_metrics(tmp_path):
    output_dir = tmp_path / "gate_run"
    _build_complete_smoke(output_dir)
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["epoch_history"][0]["val"]["auroc"] = float("nan")
    summary["new_epoch_history"][0]["val"]["auroc"] = float("nan")
    _write_json(summary_path, summary)

    with pytest.raises(ValueError, match="finite"):
        verify(
            output_dir=output_dir,
            expected_git_commit=GIT_COMMIT,
            receipt=tmp_path / "receipt.json",
            resume_device="cpu",
        )


def test_verifier_rejects_training_contract_content_without_matching_hash(
    tmp_path,
):
    output_dir = tmp_path / "gate_run"
    _build_complete_smoke(output_dir)
    identity_path = output_dir / "run_identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    identity["training_config"]["training"]["learning_rate"] = 2.0e-4
    _write_json(identity_path, identity)

    with pytest.raises(ValueError, match="training_config SHA256 mismatch"):
        verify(
            output_dir=output_dir,
            expected_git_commit=GIT_COMMIT,
            receipt=tmp_path / "receipt.json",
            resume_device="cpu",
        )


def test_verifier_rejects_training_identity_config_hash_drift(tmp_path):
    output_dir = tmp_path / "gate_run"
    _build_complete_smoke(output_dir)
    identity_path = output_dir / "run_identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    identity["training_identity"]["training_config_sha256"] = "3" * 64
    _write_json(identity_path, identity)

    with pytest.raises(ValueError, match="identity config SHA256 mismatch"):
        verify(
            output_dir=output_dir,
            expected_git_commit=GIT_COMMIT,
            receipt=tmp_path / "receipt.json",
            resume_device="cpu",
        )


def test_verifier_rejects_self_consistent_non_smoke_training_config(tmp_path):
    output_dir = tmp_path / "gate_run"
    _build_complete_smoke(output_dir)
    identity_path = output_dir / "run_identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    identity["training_config"]["training"]["learning_rate"] = 2.0e-4
    wrong_sha256 = canonical_json_sha256(identity["training_config"])
    identity["training_config_sha256"] = wrong_sha256
    identity["training_identity"]["training_config_sha256"] = wrong_sha256
    _write_json(identity_path, identity)

    with pytest.raises(ValueError, match="smoke training config mismatch"):
        verify(
            output_dir=output_dir,
            expected_git_commit=GIT_COMMIT,
            receipt=tmp_path / "receipt.json",
            resume_device="cpu",
        )


def test_verifier_rejects_incomplete_training_state(tmp_path):
    output_dir = tmp_path / "gate_run"
    _build_complete_smoke(output_dir)
    torch.save(
        {
            "kind": "stage2_binary_video_gate_training_state",
            "epoch": 1,
            "global_step": 762,
        },
        output_dir / "training_state.pt",
    )

    with pytest.raises(ValueError, match="fields do not match schema"):
        verify(
            output_dir=output_dir,
            expected_git_commit=GIT_COMMIT,
            receipt=tmp_path / "receipt.json",
            resume_device="cpu",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("optimizer_step", "AdamW step mismatch"),
        ("optimizer_lr", "optimizer config mismatch"),
    ),
)
def test_verifier_rejects_non_resumable_optimizer_state(
    tmp_path,
    mutation,
    message,
):
    output_dir = tmp_path / "gate_run"
    _build_complete_smoke(output_dir)
    state_path = output_dir / "training_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    optimizer = state["optimizer_state_dict"]
    if mutation == "optimizer_step":
        first_state = next(iter(optimizer["state"].values()))
        first_state["step"].fill_(761)
    else:
        optimizer["param_groups"][0]["lr"] = 2.0e-4
    torch.save(state, state_path)

    with pytest.raises(ValueError, match=message):
        verify(
            output_dir=output_dir,
            expected_git_commit=GIT_COMMIT,
            receipt=tmp_path / "receipt.json",
            resume_device="cpu",
        )
