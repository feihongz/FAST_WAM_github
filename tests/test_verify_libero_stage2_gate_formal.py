from __future__ import annotations

import copy
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
from scripts.verify_libero_stage2_gate_formal import (
    ADAPTER_SHA256,
    BASE_SHA256,
    DATA_MANIFEST_SHA256,
    EPISODE_ASSIGNMENT_SHA256,
    EXPECTED_GATE_CONFIG,
    EXPECTED_MAX_EPOCHS,
    EXPECTED_MIN_DELTA,
    EXPECTED_PARAMETER_COUNT,
    EXPECTED_TRAIN_BATCHES,
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
EARLY_STOP_VALUES = [0.60000, 0.59995, 0.59993, 0.59992]
FULL_VALUES = [0.70000 - 0.001 * index for index in range(EXPECTED_MAX_EPOCHS)]


def _metrics(
    *,
    examples: int,
    batches: int,
    positives: int,
    objective_bce: float,
) -> dict:
    return {
        "objective_bce": objective_bce,
        "bce": objective_bce + 0.001,
        "auroc": 0.61,
        "auprc": 0.57,
        "positive_rate": positives / examples,
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


def _training_contract(*, min_delta: float = EXPECTED_MIN_DELTA) -> dict:
    with initialize_config_dir(
        config_dir=str(REPO_ROOT / "configs"),
        version_base="1.3",
    ):
        config = compose(
            config_name="train_video_gate",
            overrides=[
                "task=libero_stage2_gate_2cam224",
                "output_dir=/tmp/test-libero-gate-formal",
                "training.num_epochs=20",
                f"training.min_delta={min_delta}",
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
        "backend": {
            "deterministic_algorithms": True,
            "deterministic_warn_only": False,
        },
    }
    contract = train_cli.build_training_config_contract(
        data=data,
        gate=resolved["gate"],
        training=resolved["training"],
        runtime=resolved["runtime"],
        numerical_runtime=numerical_runtime,
    )
    assert contract["gate"] == EXPECTED_GATE_CONFIG
    if min_delta == EXPECTED_MIN_DELTA:
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


def _history(values: list[float]) -> list[dict]:
    records = []
    for epoch, value in enumerate(values, start=1):
        records.append(
            {
                "epoch": epoch,
                "global_step": epoch * EXPECTED_TRAIN_BATCHES,
                "train": _metrics(
                    examples=48_768,
                    batches=762,
                    positives=21_925,
                    objective_bce=value + 0.02,
                ),
                "val": _metrics(
                    examples=5_408,
                    batches=85,
                    positives=2_559,
                    objective_bce=value,
                ),
            }
        )
    return records


def _replay(values: list[float], *, min_delta: float) -> dict:
    best_val_bce = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    stopped_early = False
    for epoch, value in enumerate(values, start=1):
        if value < best_val_bce - min_delta:
            best_val_bce = value
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= 3:
                stopped_early = True
                break
    assert epoch == len(values)
    return {
        "best_epoch": best_epoch,
        "best_val_bce": best_val_bce,
        "epochs_without_improvement": epochs_without_improvement,
        "stopped_early": stopped_early,
    }


def _populate_adamw_state(trainer: GateTrainer, *, step: int) -> None:
    zero_loss = sum(
        parameter.sum() * 0.0 for parameter in trainer.gate.parameters()
    )
    zero_loss.backward()
    trainer.optimizer.step()
    trainer.optimizer.zero_grad(set_to_none=True)
    for state in trainer.optimizer.state.values():
        state["step"].fill_(step)


def _build_complete_formal(
    output_dir: Path,
    *,
    values: list[float] | None = None,
    min_delta: float = EXPECTED_MIN_DELTA,
) -> dict:
    values = list(EARLY_STOP_VALUES if values is None else values)
    output_dir.mkdir()
    training_contract = _training_contract(min_delta=min_delta)
    training_config_sha256 = canonical_json_sha256(training_contract)
    training_identity = _training_identity(training_config_sha256)
    gate = BinaryVideoGate(**training_contract["gate"])
    assert gate.parameter_count() == EXPECTED_PARAMETER_COUNT
    records = _history(values)
    replay = _replay(values, min_delta=min_delta)
    best_metrics = records[replay["best_epoch"] - 1]["val"]
    final_epoch = len(records)
    global_step = final_epoch * EXPECTED_TRAIN_BATCHES
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
        "final_epoch": final_epoch,
        "global_step": global_step,
        "stopped_early": replay["stopped_early"],
        "best_epoch": replay["best_epoch"],
        "best_val_bce": replay["best_val_bce"],
        "best_metrics": best_metrics,
        "history_complete": True,
        "epoch_history": records,
        "new_epoch_history": records,
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
    _populate_adamw_state(trainer, step=global_step)
    trainer.epoch = final_epoch
    trainer.global_step = global_step
    trainer.best_epoch = replay["best_epoch"]
    trainer.best_global_step = replay["best_epoch"] * EXPECTED_TRAIN_BATCHES
    trainer.best_val_bce = replay["best_val_bce"]
    trainer.best_metrics = dict(best_metrics)
    trainer.epochs_without_improvement = replay["epochs_without_improvement"]
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
    return summary


def test_verifier_accepts_min_delta_early_stop_and_idempotent_receipt(tmp_path):
    output_dir = tmp_path / "gate_run"
    receipt = tmp_path / "receipt.json"
    _build_complete_formal(output_dir)

    result = verify(
        output_dir=output_dir,
        expected_git_commit=GIT_COMMIT,
        receipt=receipt,
        resume_device="cpu",
    )

    assert result["status"] == "pass"
    assert result["final_epoch"] == 4
    assert result["global_step"] == 4 * EXPECTED_TRAIN_BATCHES
    assert result["best_epoch"] == 1
    assert result["stopped_early"] is True
    assert result["best_val_bce"] == EARLY_STOP_VALUES[0]
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


def test_verifier_accepts_full_twenty_epoch_run(tmp_path):
    output_dir = tmp_path / "gate_run"
    receipt = tmp_path / "receipt.json"
    _build_complete_formal(output_dir, values=FULL_VALUES)

    result = verify(
        output_dir=output_dir,
        expected_git_commit=GIT_COMMIT,
        receipt=receipt,
        resume_device="cpu",
    )

    assert result["final_epoch"] == EXPECTED_MAX_EPOCHS
    assert result["global_step"] == EXPECTED_MAX_EPOCHS * EXPECTED_TRAIN_BATCHES
    assert result["best_epoch"] == EXPECTED_MAX_EPOCHS
    assert result["stopped_early"] is False


def test_verifier_rejects_internally_consistent_wrong_min_delta(tmp_path):
    output_dir = tmp_path / "gate_run"
    _build_complete_formal(output_dir, min_delta=0.0)

    with pytest.raises(ValueError, match="formal training config mismatch"):
        verify(
            output_dir=output_dir,
            expected_git_commit=GIT_COMMIT,
            receipt=tmp_path / "receipt.json",
            resume_device="cpu",
        )


def test_verifier_rejects_history_past_first_early_stop_boundary(tmp_path):
    output_dir = tmp_path / "gate_run"
    summary = _build_complete_formal(output_dir)
    extra = copy.deepcopy(summary["epoch_history"][-1])
    extra["epoch"] = 5
    extra["global_step"] = 5 * EXPECTED_TRAIN_BATCHES
    summary["epoch_history"].append(extra)
    summary["new_epoch_history"] = copy.deepcopy(summary["epoch_history"])
    summary["final_epoch"] = 5
    summary["global_step"] = 5 * EXPECTED_TRAIN_BATCHES
    _write_json(output_dir / "summary.json", summary)

    with pytest.raises(ValueError, match="continues after early-stop boundary"):
        verify(
            output_dir=output_dir,
            expected_git_commit=GIT_COMMIT,
            receipt=tmp_path / "receipt.json",
            resume_device="cpu",
        )


def test_verifier_rejects_optimizer_step_drift(tmp_path):
    output_dir = tmp_path / "gate_run"
    _build_complete_formal(output_dir)
    state_path = output_dir / "training_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    first = next(iter(state["optimizer_state_dict"]["state"].values()))
    first["step"].sub_(1)
    torch.save(state, state_path)

    with pytest.raises(ValueError, match="AdamW step mismatch"):
        verify(
            output_dir=output_dir,
            expected_git_commit=GIT_COMMIT,
            receipt=tmp_path / "receipt.json",
            resume_device="cpu",
        )


def test_verifier_rejects_best_export_progress_drift(tmp_path):
    output_dir = tmp_path / "gate_run"
    _build_complete_formal(output_dir)
    best_path = output_dir / "gate_best.pt"
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    best["epoch"] = 2
    torch.save(best, best_path)

    with pytest.raises(ValueError, match="export progress mismatch"):
        verify(
            output_dir=output_dir,
            expected_git_commit=GIT_COMMIT,
            receipt=tmp_path / "receipt.json",
            resume_device="cpu",
        )


def test_verifier_refuses_conflicting_existing_receipt(tmp_path):
    output_dir = tmp_path / "gate_run"
    receipt = tmp_path / "receipt.json"
    _build_complete_formal(output_dir)
    verify(
        output_dir=output_dir,
        expected_git_commit=GIT_COMMIT,
        receipt=receipt,
        resume_device="cpu",
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["status"] = "tampered"
    _write_json(receipt, payload)

    with pytest.raises(RuntimeError, match="existing verification receipt differs"):
        verify(
            output_dir=output_dir,
            expected_git_commit=GIT_COMMIT,
            receipt=receipt,
            resume_device="cpu",
        )


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.device_count() != 1
    or "H100" not in torch.cuda.get_device_name(0),
    reason="formal resume probe requires exactly one visible H100",
)
def test_verifier_cli_runs_formal_resume_probe_on_h100(tmp_path):
    output_dir = tmp_path / "gate_run"
    receipt = tmp_path / "receipt.json"
    _build_complete_formal(output_dir)

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
            str(REPO_ROOT / "scripts" / "verify_libero_stage2_gate_formal.py"),
            "--output-dir",
            str(output_dir),
            "--expected-git-commit",
            GIT_COMMIT,
            "--receipt",
            str(receipt),
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "formal run passed" in completed.stdout
    assert json.loads(receipt.read_text(encoding="utf-8"))["status"] == "pass"
