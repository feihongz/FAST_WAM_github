#!/usr/bin/env python3
"""Strict acceptance verifier for formal LIBERO Stage 2 Gate training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
from pathlib import Path
from typing import Any, Mapping

import torch

from fastwam.gating.artifacts import (
    canonical_json_sha256,
    publish_json_atomic_no_clobber,
)
from fastwam.gating.checkpointing import load_gate_checkpoint
from fastwam.gating.trainer import GateTrainer
from fastwam.models.video_gate import BinaryVideoGate


LABEL_MANIFEST_SHA256 = (
    "d6dc98a6a36c30150db30000c86d07c7a1e7d90b1dc5d1a5a60e02126c22b3e0"
)
ADAPTER_SHA256 = (
    "cbc593bc6ce99c0249a65e5c7cef754c9a1d7ea602f81fdae2b8cb158a25858c"
)
BASE_SHA256 = (
    "17a5588cc2b8d162219c9daf818614f614ee4a7921933a4a26c5d678111330e9"
)
DATA_MANIFEST_SHA256 = (
    "08da49109a57b55c67f3fa4ac31fbfa44e44dd541a194a5d3420838537d0d320"
)
EPISODE_ASSIGNMENT_SHA256 = (
    "a77efa24249dab8cfacbc228b1da341947240b36fa77d90182701c07bdcf7787"
)
DATA_CONFIG_SHA256 = (
    "44dc596c6700e02e69ba12823ed899d12d25c6980263f9bf3ac85cb73d53daa4"
)
EXPECTED_PARAMETER_COUNT = 658_977
EXPECTED_TRAIN_EXAMPLES = 48_768
EXPECTED_TRAIN_BATCHES = 762
EXPECTED_TRAIN_POSITIVES = 21_925
EXPECTED_TRAIN_NEGATIVES = 26_843
EXPECTED_VALIDATION_EXAMPLES = 5_408
EXPECTED_VALIDATION_BATCHES = 85
EXPECTED_VALIDATION_POSITIVES = 2_559
EXPECTED_MAX_EPOCHS = 20
EXPECTED_EARLY_STOP_PATIENCE = 3
EXPECTED_MIN_DELTA = 1.0e-4
EXPECTED_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
EXPECTED_GATE_CONFIG = {
    "proprio_dim": 8,
    "context_dim": 4096,
    "cnn_channels": [32, 64, 128],
    "context_feature_dim": 128,
    "proprio_hidden_dim": 64,
    "proprio_feature_dim": 32,
    "fusion_hidden_dim": 128,
}
EXPECTED_TRAINING_CONFIG = {
    "seed": 42,
    "batch_size": 64,
    "num_workers": 0,
    "pin_memory": True,
    "shuffle": True,
    "learning_rate": 1.0e-4,
    "weight_decay": 1.0e-4,
    "max_grad_norm": 1.0,
    "num_epochs": EXPECTED_MAX_EPOCHS,
    "early_stop_patience": EXPECTED_EARLY_STOP_PATIENCE,
    "min_delta": EXPECTED_MIN_DELTA,
    "threshold": 0.5,
    "num_calibration_bins": 10,
}
_METRIC_FIELDS = {
    "objective_bce",
    "bce",
    "auroc",
    "auprc",
    "positive_rate",
    "predicted_positive_rate",
    "expected_calibration_error",
    "num_examples",
    "num_batches",
}


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable: {path}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return payload


def _regular_file(path: Path, *, label: str) -> Path:
    try:
        metadata = os.lstat(path)
    except OSError as error:
        raise FileNotFoundError(f"{label} is missing: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"{label} must be a regular non-symlink file: {path}")
    if path.resolve(strict=True) != path:
        raise RuntimeError(f"{label} path contains a symlink: {path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _finite_metrics(
    metrics: Mapping[str, Any],
    *,
    expected_examples: int,
    expected_batches: int,
    expected_positives: int,
    label: str,
) -> None:
    if set(metrics) != _METRIC_FIELDS:
        raise ValueError(f"{label} metric fields do not match schema")
    if metrics.get("num_examples") != expected_examples:
        raise ValueError(f"{label} num_examples mismatch")
    if metrics.get("num_batches") != expected_batches:
        raise ValueError(f"{label} num_batches mismatch")
    for field in (
        "objective_bce",
        "bce",
        "auroc",
        "auprc",
        "positive_rate",
        "predicted_positive_rate",
        "expected_calibration_error",
    ):
        value = metrics.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{label} {field} must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError(f"{label} {field} must be finite")
    if float(metrics["objective_bce"]) < 0.0 or float(metrics["bce"]) < 0.0:
        raise ValueError(f"{label} BCE metrics must be non-negative")
    for field in (
        "auroc",
        "auprc",
        "positive_rate",
        "predicted_positive_rate",
        "expected_calibration_error",
    ):
        if not 0.0 <= float(metrics[field]) <= 1.0:
            raise ValueError(f"{label} {field} must be in [0, 1]")
    expected_positive_rate = expected_positives / expected_examples
    if not math.isclose(
        float(metrics["positive_rate"]),
        expected_positive_rate,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise ValueError(f"{label} positive_rate mismatch")


def _validate_training_contract(
    run_identity: Mapping[str, Any],
    training_identity: Mapping[str, Any],
) -> dict[str, Any]:
    if set(run_identity) != {
        "schema_version",
        "kind",
        "training_config",
        "training_config_sha256",
        "training_identity",
    }:
        raise ValueError("Gate run identity fields do not match schema")
    contract = run_identity.get("training_config")
    if not isinstance(contract, Mapping):
        raise TypeError("Gate run training_config must be a mapping")
    contract = dict(contract)
    if set(contract) != {
        "schema_version",
        "kind",
        "data",
        "gate",
        "training",
        "runtime",
        "dataloader_seed_algorithm",
    }:
        raise ValueError("Gate training contract fields do not match schema")
    if (
        contract.get("schema_version") != 1
        or contract.get("kind")
        != "stage2_binary_video_gate_training_contract"
        or contract.get("dataloader_seed_algorithm")
        != "base_seed_plus_zero_based_epoch_v1"
    ):
        raise ValueError("Gate training contract schema mismatch")

    computed_sha256 = canonical_json_sha256(contract)
    if run_identity.get("training_config_sha256") != computed_sha256:
        raise ValueError("Gate run training_config SHA256 mismatch")
    if training_identity.get("training_config_sha256") != computed_sha256:
        raise ValueError("Gate training identity config SHA256 mismatch")
    data = contract.get("data")
    if (
        not isinstance(data, Mapping)
        or canonical_json_sha256(data) != DATA_CONFIG_SHA256
    ):
        raise ValueError("Gate formal data config mismatch")
    if contract.get("gate") != EXPECTED_GATE_CONFIG:
        raise ValueError("Gate formal architecture config mismatch")
    if contract.get("training") != EXPECTED_TRAINING_CONFIG:
        raise ValueError("Gate formal training config mismatch")

    runtime = contract.get("runtime")
    if not isinstance(runtime, Mapping) or set(runtime) != {
        "device",
        "require_cuda",
        "deterministic_algorithms",
        "numerical_runtime",
    }:
        raise ValueError("Gate formal runtime config mismatch")
    if (
        runtime.get("device") != "cuda:0"
        or runtime.get("require_cuda") is not True
        or runtime.get("deterministic_algorithms") is not True
    ):
        raise ValueError("Gate formal CUDA/determinism config mismatch")
    numerical = runtime.get("numerical_runtime")
    if not isinstance(numerical, Mapping) or set(numerical) != {
        "versions",
        "device",
        "ffmpeg",
        "backend",
    }:
        raise ValueError("Gate formal numerical runtime identity mismatch")
    device = numerical.get("device")
    backend = numerical.get("backend")
    if (
        not isinstance(device, Mapping)
        or device.get("type") != "cuda"
        or "H100" not in str(device.get("name", ""))
        or not isinstance(backend, Mapping)
        or backend.get("deterministic_algorithms") is not True
        or backend.get("deterministic_warn_only") is not False
    ):
        raise ValueError("Gate formal numerical H100/determinism identity mismatch")
    return contract


def _validate_epoch_history(history: Any) -> dict[str, Any]:
    if not isinstance(history, list) or not history:
        raise ValueError("Gate formal epoch history must be non-empty")
    if len(history) > EXPECTED_MAX_EPOCHS:
        raise ValueError("Gate formal epoch history exceeds target epochs")

    best_val_bce = math.inf
    best_epoch = -1
    best_metrics: dict[str, Any] = {}
    epochs_without_improvement = 0
    stopped_early = False
    for index, raw_record in enumerate(history, start=1):
        if not isinstance(raw_record, Mapping) or set(raw_record) != {
            "epoch",
            "global_step",
            "train",
            "val",
        }:
            raise ValueError("Gate formal epoch record fields mismatch")
        if raw_record.get("epoch") != index:
            raise ValueError("Gate formal epoch history is not contiguous")
        if raw_record.get("global_step") != index * EXPECTED_TRAIN_BATCHES:
            raise ValueError("Gate formal epoch global_step mismatch")
        train_metrics = raw_record.get("train")
        val_metrics = raw_record.get("val")
        if not isinstance(train_metrics, Mapping) or not isinstance(
            val_metrics, Mapping
        ):
            raise TypeError("Gate formal epoch metrics must be mappings")
        _finite_metrics(
            train_metrics,
            expected_examples=EXPECTED_TRAIN_EXAMPLES,
            expected_batches=EXPECTED_TRAIN_BATCHES,
            expected_positives=EXPECTED_TRAIN_POSITIVES,
            label=f"train epoch {index}",
        )
        _finite_metrics(
            val_metrics,
            expected_examples=EXPECTED_VALIDATION_EXAMPLES,
            expected_batches=EXPECTED_VALIDATION_BATCHES,
            expected_positives=EXPECTED_VALIDATION_POSITIVES,
            label=f"validation epoch {index}",
        )
        val_bce = float(val_metrics["objective_bce"])
        if val_bce < best_val_bce - EXPECTED_MIN_DELTA:
            best_val_bce = val_bce
            best_epoch = index
            best_metrics = dict(val_metrics)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EXPECTED_EARLY_STOP_PATIENCE:
                if index != len(history):
                    raise ValueError(
                        "Gate formal history continues after early-stop boundary"
                    )
                stopped_early = True
    return {
        "best_epoch": best_epoch,
        "best_global_step": best_epoch * EXPECTED_TRAIN_BATCHES,
        "best_val_bce": best_val_bce,
        "best_metrics": best_metrics,
        "epochs_without_improvement": epochs_without_improvement,
        "final_epoch": len(history),
        "global_step": len(history) * EXPECTED_TRAIN_BATCHES,
        "stopped_early": stopped_early,
    }


def _validate_export(
    path: Path,
    *,
    training_identity: Mapping[str, Any],
    expected_epoch: int,
    expected_global_step: int,
) -> tuple[BinaryVideoGate, dict[str, Any]]:
    gate, payload = load_gate_checkpoint(
        path,
        expected_label_manifest_sha256=LABEL_MANIFEST_SHA256,
        expected_adapter_checkpoint_sha256=ADAPTER_SHA256,
        expected_base_checkpoint_sha256=BASE_SHA256,
        expected_data_manifest_sha256=DATA_MANIFEST_SHA256,
        expected_episode_split_assignment_sha256=EPISODE_ASSIGNMENT_SHA256,
        expected_training_config_sha256=training_identity[
            "training_config_sha256"
        ],
        expected_git_identity=training_identity["git_identity"],
        map_location="cpu",
    )
    if gate.parameter_count() != EXPECTED_PARAMETER_COUNT:
        raise ValueError("Gate export parameter count mismatch")
    if payload["parameter_count"] != EXPECTED_PARAMETER_COUNT:
        raise ValueError("Gate export recorded parameter count mismatch")
    if (
        payload["epoch"] != expected_epoch
        or payload["global_step"] != expected_global_step
    ):
        raise ValueError("Gate export progress mismatch")
    return gate, payload


def _assert_same_gate_state(
    actual: Mapping[str, torch.Tensor],
    expected: Mapping[str, torch.Tensor],
    *,
    label: str,
) -> None:
    if set(actual) != set(expected):
        raise ValueError(f"{label} Gate state fields mismatch")
    for name, value in actual.items():
        reference = expected[name]
        if (
            not isinstance(value, torch.Tensor)
            or not isinstance(reference, torch.Tensor)
            or value.dtype != reference.dtype
            or value.shape != reference.shape
            or not torch.equal(value.detach().cpu(), reference.detach().cpu())
        ):
            raise ValueError(f"{label} Gate tensor mismatch: {name}")


def _assert_finite_gate(gate: BinaryVideoGate, *, label: str) -> None:
    for name, parameter in gate.named_parameters():
        if not torch.isfinite(parameter).all():
            raise ValueError(f"{label} Gate parameter is non-finite: {name}")


def _optimizer_group_contract(
    optimizer: torch.optim.Optimizer,
) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in group.items() if key != "params"}
        for group in optimizer.param_groups
    ]


def _validate_optimizer_state(
    trainer: GateTrainer,
    *,
    expected_step: int,
) -> None:
    parameters = tuple(trainer.gate.parameters())
    if set(trainer.optimizer.state) != set(parameters):
        raise ValueError("Gate canonical training state optimizer coverage mismatch")
    for parameter in parameters:
        state = trainer.optimizer.state[parameter]
        if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("Gate AdamW state fields mismatch")
        step = state["step"]
        if (
            not isinstance(step, torch.Tensor)
            or step.numel() != 1
            or not torch.isfinite(step).all()
            or float(step.detach().cpu().item()) != float(expected_step)
        ):
            raise ValueError("Gate AdamW step mismatch")
        for field in ("exp_avg", "exp_avg_sq"):
            moment = state[field]
            if (
                not isinstance(moment, torch.Tensor)
                or moment.shape != parameter.shape
                or moment.dtype != parameter.dtype
                or not torch.isfinite(moment).all()
            ):
                raise ValueError(f"Gate AdamW {field} is invalid")


def _validate_training_state(
    path: Path,
    *,
    training_identity: Mapping[str, Any],
    training_contract: Mapping[str, Any],
    replay: Mapping[str, Any],
    best_gate: BinaryVideoGate,
    last_gate: BinaryVideoGate,
    resume_device: str | torch.device,
) -> None:
    training = training_contract["training"]
    device = torch.device(resume_device)
    gate = BinaryVideoGate(**dict(training_contract["gate"])).to(device=device)
    train_labels = torch.cat(
        (
            torch.ones(EXPECTED_TRAIN_POSITIVES, dtype=torch.float32),
            torch.zeros(EXPECTED_TRAIN_NEGATIVES, dtype=torch.float32),
        )
    )
    trainer = GateTrainer(
        gate,
        train_labels=train_labels,
        training_identity=training_identity,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        max_grad_norm=float(training["max_grad_norm"]),
    )
    expected_optimizer_groups = _optimizer_group_contract(trainer.optimizer)
    trainer.load_training_state(path)
    if _optimizer_group_contract(trainer.optimizer) != expected_optimizer_groups:
        raise ValueError("Gate canonical training state optimizer config mismatch")
    if (
        trainer.epoch != replay["final_epoch"]
        or trainer.global_step != replay["global_step"]
        or trainer.best_epoch != replay["best_epoch"]
        or trainer.best_global_step != replay["best_global_step"]
        or trainer.epochs_without_improvement
        != replay["epochs_without_improvement"]
        or trainer.train_positive_count != EXPECTED_TRAIN_POSITIVES
        or trainer.train_negative_count != EXPECTED_TRAIN_NEGATIVES
        or trainer.best_metrics != replay["best_metrics"]
        or not math.isclose(
            trainer.best_val_bce,
            float(replay["best_val_bce"]),
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise ValueError("Gate canonical training state progress/metrics mismatch")
    _validate_optimizer_state(trainer, expected_step=int(replay["global_step"]))

    raw_state = torch.load(path, map_location="cpu", weights_only=False)
    _assert_same_gate_state(
        trainer.gate.state_dict(), last_gate.state_dict(), label="last/resume"
    )
    _assert_same_gate_state(
        raw_state["best_gate_state_dict"],
        best_gate.state_dict(),
        label="best/resume",
    )
    _assert_finite_gate(trainer.gate, label="resumed")

    required_deterministic = bool(
        training_contract["runtime"]["deterministic_algorithms"]
    )
    if (
        device.type == "cuda"
        and required_deterministic
        and os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        != EXPECTED_CUBLAS_WORKSPACE_CONFIG
    ):
        raise RuntimeError(
            "deterministic CUDA resume probe requires "
            f"CUBLAS_WORKSPACE_CONFIG={EXPECTED_CUBLAS_WORKSPACE_CONFIG}"
        )
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(required_deterministic)
        loss = trainer.train_batch(
            {
                "input_image": torch.zeros((2, 3, 8, 8), dtype=torch.float32),
                "context": torch.zeros((2, 1, 4096), dtype=torch.float32),
                "context_mask": torch.ones((2, 1), dtype=torch.bool),
                "proprio": torch.zeros((2, 8), dtype=torch.float32),
                "label": torch.tensor([0.0, 1.0], dtype=torch.float32),
                "sample_weight": torch.ones(2, dtype=torch.float32),
                "sample_id": ("resume-probe-0", "resume-probe-1"),
            }
        )
    finally:
        torch.use_deterministic_algorithms(
            previous_deterministic,
            warn_only=previous_warn_only,
        )
    if (
        not math.isfinite(loss)
        or trainer.global_step != int(replay["global_step"]) + 1
        or trainer.epoch != replay["final_epoch"]
        or trainer.best_epoch != replay["best_epoch"]
        or trainer.best_global_step != replay["best_global_step"]
        or trainer.epochs_without_improvement
        != replay["epochs_without_improvement"]
    ):
        raise ValueError("Gate synthetic resume update failed")
    _assert_finite_gate(trainer.gate, label="post-resume-probe")
    _validate_optimizer_state(
        trainer, expected_step=int(replay["global_step"]) + 1
    )


def verify(
    *,
    output_dir: Path,
    expected_git_commit: str,
    receipt: Path,
    resume_device: str | torch.device | None = None,
) -> dict[str, Any]:
    if resume_device is None:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Gate formal verifier requires one visible CUDA device")
        device_name = torch.cuda.get_device_name(0)
        if "H100" not in device_name:
            raise RuntimeError(
                f"Gate formal verifier requires one H100, found {device_name}"
            )
        resume_device = torch.device("cuda:0")
    else:
        resume_device = torch.device(resume_device)

    output_dir = output_dir.expanduser().resolve(strict=True)
    expected_git_commit = expected_git_commit.strip().lower()
    if len(expected_git_commit) != 40 or any(
        character not in "0123456789abcdef" for character in expected_git_commit
    ):
        raise ValueError("expected_git_commit must be a full lowercase Git SHA")

    names = (
        "run_identity.json",
        "training_state.pt",
        "gate_best.pt",
        "gate_last.pt",
        "summary.json",
    )
    files = {
        name: _regular_file(output_dir / name, label=name) for name in names
    }
    run_identity = _load_json(files["run_identity.json"], label="run identity")
    summary = _load_json(files["summary.json"], label="Gate summary")

    if (
        run_identity.get("schema_version") != 1
        or run_identity.get("kind") != "stage2_binary_video_gate_run_identity"
    ):
        raise ValueError("Gate run identity schema mismatch")
    training_identity = run_identity.get("training_identity")
    if not isinstance(training_identity, Mapping):
        raise TypeError("Gate run identity lacks training_identity")
    training_identity = dict(training_identity)
    training_contract = _validate_training_contract(
        run_identity, training_identity
    )
    if training_identity.get("label_manifest_sha256") != LABEL_MANIFEST_SHA256:
        raise ValueError("Gate run label manifest identity mismatch")
    expected_sources = {
        "adapter_checkpoint_sha256": ADAPTER_SHA256,
        "base_checkpoint_sha256": BASE_SHA256,
        "data_manifest_sha256": DATA_MANIFEST_SHA256,
        "episode_split_assignment_sha256": EPISODE_ASSIGNMENT_SHA256,
    }
    for field, expected in expected_sources.items():
        if training_identity.get(field) != expected:
            raise ValueError(f"Gate run {field} mismatch")
    git_identity = training_identity.get("git_identity")
    if git_identity != {
        "commit": expected_git_commit,
        "tracked_dirty": False,
        "untracked_source_files": [],
    }:
        raise ValueError("Gate run Git identity mismatch")

    expected_summary_fields = {
        "schema_version",
        "kind",
        "training_identity",
        "initial_epoch",
        "final_epoch",
        "global_step",
        "stopped_early",
        "best_epoch",
        "best_val_bce",
        "best_metrics",
        "history_complete",
        "epoch_history",
        "new_epoch_history",
        "state_file",
        "best_file",
        "last_file",
    }
    if set(summary) != expected_summary_fields:
        raise ValueError("Gate formal summary fields do not match schema")
    if (
        summary.get("schema_version") != 1
        or summary.get("kind")
        != "stage2_binary_video_gate_training_summary"
        or summary.get("training_identity") != training_identity
        or summary.get("initial_epoch") != 0
        or summary.get("history_complete") is not True
        or summary.get("state_file") != "training_state.pt"
        or summary.get("best_file") != "gate_best.pt"
        or summary.get("last_file") != "gate_last.pt"
    ):
        raise ValueError("Gate formal summary identity/schema mismatch")
    history = summary.get("epoch_history")
    if history != summary.get("new_epoch_history"):
        raise ValueError("fresh Gate formal history/new history mismatch")
    replay = _validate_epoch_history(history)
    summary_best_val_bce = summary.get("best_val_bce")
    if (
        isinstance(summary_best_val_bce, bool)
        or not isinstance(summary_best_val_bce, (int, float))
        or summary.get("final_epoch") != replay["final_epoch"]
        or summary.get("global_step") != replay["global_step"]
        or summary.get("stopped_early") is not replay["stopped_early"]
        or summary.get("best_epoch") != replay["best_epoch"]
        or summary.get("best_metrics") != replay["best_metrics"]
        or not math.isclose(
            float(summary_best_val_bce),
            float(replay["best_val_bce"]),
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise ValueError("Gate formal summary progress/early-stop mismatch")
    if not replay["stopped_early"] and replay["final_epoch"] != EXPECTED_MAX_EPOCHS:
        raise ValueError("Gate formal run ended before 20 epochs without early stop")

    best_gate, best = _validate_export(
        files["gate_best.pt"],
        training_identity=training_identity,
        expected_epoch=int(replay["best_epoch"]),
        expected_global_step=int(replay["best_global_step"]),
    )
    last_gate, last = _validate_export(
        files["gate_last.pt"],
        training_identity=training_identity,
        expected_epoch=int(replay["final_epoch"]),
        expected_global_step=int(replay["global_step"]),
    )
    if best["best_metrics"] != replay["best_metrics"]:
        raise ValueError("best Gate export metrics differ from summary")
    if last["best_metrics"] != replay["best_metrics"]:
        raise ValueError("last Gate export metrics differ from summary")
    _validate_training_state(
        files["training_state.pt"],
        training_identity=training_identity,
        training_contract=training_contract,
        replay=replay,
        best_gate=best_gate,
        last_gate=last_gate,
        resume_device=resume_device,
    )

    result = {
        "artifact_sha256": {
            name: _sha256(path) for name, path in sorted(files.items())
        },
        "benchmark": "LIBERO",
        "best_epoch": replay["best_epoch"],
        "best_global_step": replay["best_global_step"],
        "best_metrics": replay["best_metrics"],
        "best_val_bce": replay["best_val_bce"],
        "final_epoch": replay["final_epoch"],
        "git_commit": expected_git_commit,
        "global_step": replay["global_step"],
        "kind": "libero_stage2_gate_formal_verification",
        "label_manifest_sha256": LABEL_MANIFEST_SHA256,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "schema_version": 1,
        "status": "pass",
        "stopped_early": replay["stopped_early"],
        "train_examples": EXPECTED_TRAIN_EXAMPLES,
        "validation_examples": EXPECTED_VALIDATION_EXAMPLES,
    }
    receipt = receipt.expanduser().resolve()
    published = publish_json_atomic_no_clobber(receipt, result)
    if not published:
        existing = _load_json(
            _regular_file(receipt, label="existing verification receipt"),
            label="existing verification receipt",
        )
        if existing != result:
            raise RuntimeError("existing verification receipt differs")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = verify(
        output_dir=args.output_dir,
        expected_git_commit=args.expected_git_commit,
        receipt=args.receipt,
    )
    print(
        "[verify] LIBERO Stage 2 Gate formal run passed: "
        f"epoch={result['final_epoch']} step={result['global_step']} "
        f"best_epoch={result['best_epoch']} "
        f"stopped_early={result['stopped_early']}"
    )
    print(f"[verify] receipt={args.receipt.expanduser().resolve()}")


if __name__ == "__main__":
    main()
