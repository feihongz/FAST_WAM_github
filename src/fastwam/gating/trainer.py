"""Lightweight, WAM-free optimization runtime for the Stage 2 Gate."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from fastwam.models.video_gate import BinaryVideoGate

from .checkpointing import save_gate_checkpoint
from .contracts import require_sha256
from .metrics import GateBinaryMetrics, compute_gate_binary_metrics


GATE_TRAINING_STATE_SCHEMA_VERSION = 2
GATE_TRAINING_STATE_KIND = "stage2_binary_video_gate_training_state"

_TRAINING_IDENTITY_SHA_FIELDS = frozenset(
    {
        "label_manifest_sha256",
        "adapter_checkpoint_sha256",
        "base_checkpoint_sha256",
        "data_manifest_sha256",
        "episode_split_assignment_sha256",
        "training_config_sha256",
    }
)
_TRAINING_IDENTITY_KEYS = _TRAINING_IDENTITY_SHA_FIELDS | {"git_identity"}
_GIT_IDENTITY_KEYS = {
    "commit",
    "tracked_dirty",
    "untracked_source_files",
}

_BATCH_KEYS = frozenset(
    {
        "input_image",
        "context",
        "context_mask",
        "proprio",
        "label",
        "sample_weight",
        "sample_id",
    }
)


@dataclass(frozen=True)
class GateEpochResult:
    """Metrics collected from one complete train or validation epoch."""

    objective_bce: float
    metrics: GateBinaryMetrics
    num_batches: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective_bce": self.objective_bce,
            **self.metrics.to_dict(),
            "num_batches": self.num_batches,
        }


@dataclass(frozen=True)
class GateFitResult:
    """History and stop decision from a call to :meth:`GateTrainer.fit`."""

    epochs: tuple[dict[str, Any], ...]
    stopped_early: bool
    best_epoch: int
    best_val_bce: float


@dataclass(frozen=True)
class _ValidatedBatch:
    model_inputs: dict[str, torch.Tensor]
    labels: torch.Tensor
    sample_weights: torch.Tensor
    sample_ids: tuple[str, ...]


def _binary_labels(labels: Any, *, field: str) -> torch.Tensor:
    values = torch.as_tensor(labels, device="cpu")
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError(f"{field} must have non-empty shape [N]")
    if values.dtype == torch.bool:
        return values.to(dtype=torch.float32)
    if not (
        values.is_floating_point()
        or values.dtype
        in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
    ):
        raise TypeError(f"{field} must contain bool or numeric labels")
    values = values.to(dtype=torch.float32)
    if not torch.isfinite(values).all() or not bool(
        ((values == 0.0) | (values == 1.0)).all().item()
    ):
        raise ValueError(f"{field} must contain only zero or one")
    return values


def _canonical_json_mapping(value: Any, *, field: str) -> dict[str, Any]:
    """Detach a mapping through canonical JSON or reject unsafe metadata."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    try:
        encoded = json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        payload = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{field} must be canonical-JSON serializable"
        ) from error
    if not isinstance(payload, dict):
        raise TypeError(f"{field} must encode a JSON object")
    return payload


def _validated_git_identity(value: Any) -> dict[str, Any]:
    payload = _canonical_json_mapping(value, field="training_identity git_identity")
    if set(payload) != _GIT_IDENTITY_KEYS:
        missing = sorted(_GIT_IDENTITY_KEYS - set(payload))
        unexpected = sorted(set(payload) - _GIT_IDENTITY_KEYS)
        raise ValueError(
            "training_identity git_identity fields do not match schema; "
            f"missing={missing}, unexpected={unexpected}"
        )
    commit = payload["commit"]
    if (
        not isinstance(commit, str)
        or len(commit) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise ValueError(
            "training_identity git_identity commit must be a full lowercase Git SHA"
        )
    if not isinstance(payload["tracked_dirty"], bool):
        raise TypeError(
            "training_identity git_identity tracked_dirty must be bool"
        )
    untracked = payload["untracked_source_files"]
    if not isinstance(untracked, list) or any(
        not isinstance(path, str) or not path for path in untracked
    ):
        raise TypeError(
            "training_identity git_identity untracked_source_files must be strings"
        )
    if untracked != sorted(set(untracked)):
        raise ValueError(
            "training_identity git_identity untracked_source_files must be "
            "sorted and unique"
        )
    return payload


def _validate_training_identity(value: Any) -> dict[str, Any]:
    payload = _canonical_json_mapping(value, field="training_identity")
    if set(payload) != _TRAINING_IDENTITY_KEYS:
        missing = sorted(_TRAINING_IDENTITY_KEYS - set(payload))
        unexpected = sorted(set(payload) - _TRAINING_IDENTITY_KEYS)
        raise ValueError(
            "training_identity fields do not match schema; "
            f"missing={missing}, unexpected={unexpected}"
        )
    for field in sorted(_TRAINING_IDENTITY_SHA_FIELDS):
        payload[field] = require_sha256(
            payload[field],
            field=f"training_identity {field}",
        )
    payload["git_identity"] = _validated_git_identity(payload["git_identity"])
    return payload


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _optimizer_parameter_ids(
    optimizer: torch.optim.Optimizer,
) -> list[int]:
    return [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu").clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _gate_state(gate: BinaryVideoGate) -> dict[str, torch.Tensor]:
    state = _cpu_tree(gate.state_dict())
    for name, value in state.items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"Gate parameter {name!r} is non-finite")
    return state


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_initialized()
            else []
        ),
    }


def _restore_rng_state(payload: Mapping[str, Any]) -> None:
    if set(payload) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise ValueError("Gate training RNG state fields do not match schema")
    cpu_state = payload["torch_cpu"]
    cuda_states = payload["torch_cuda"]
    if not isinstance(cpu_state, torch.Tensor) or cpu_state.dtype != torch.uint8:
        raise ValueError("Gate training torch CPU RNG state is invalid")
    if not isinstance(cuda_states, list) or any(
        not isinstance(state, torch.Tensor) or state.dtype != torch.uint8
        for state in cuda_states
    ):
        raise ValueError("Gate training torch CUDA RNG state is invalid")
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError("cannot restore CUDA Gate state without CUDA")
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError("Gate training CUDA device count changed on resume")
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(cpu_state.to(device="cpu"))
    if cuda_states:
        torch.cuda.set_rng_state_all(
            [state.to(device="cpu") for state in cuda_states]
        )


class GateTrainer:
    """Train only a :class:`BinaryVideoGate` from precomputed labels.

    ``train_labels`` is mandatory so class balance is fixed from the complete
    training split, rather than recomputed from individual mini-batches.
    This module intentionally has no dependency on any WAM implementation.
    """

    def __init__(
        self,
        gate: BinaryVideoGate,
        *,
        train_labels: Sequence[int | bool | float] | torch.Tensor,
        training_identity: Mapping[str, Any],
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        max_grad_norm: float = 1.0,
        optimizer: torch.optim.Optimizer | None = None,
    ):
        if not isinstance(gate, BinaryVideoGate):
            raise TypeError("gate must be a BinaryVideoGate")
        if not math.isfinite(float(lr)) or float(lr) <= 0.0:
            raise ValueError("lr must be finite and positive")
        if not math.isfinite(float(weight_decay)) or float(weight_decay) < 0.0:
            raise ValueError("weight_decay must be finite and non-negative")
        if not math.isfinite(float(max_grad_norm)) or float(max_grad_norm) <= 0.0:
            raise ValueError("max_grad_norm must be finite and positive")
        validated_training_identity = _validate_training_identity(training_identity)

        labels = _binary_labels(train_labels, field="train_labels")
        positives = int(labels.sum().item())
        negatives = int(labels.numel() - positives)
        if positives == 0 or negatives == 0:
            raise ValueError("Gate training split must contain both label classes")

        self.gate = gate
        self.gate.requires_grad_(True)
        parameters = tuple(self.gate.parameters())
        if not parameters:
            raise ValueError("Gate has no trainable parameters")
        self.optimizer = optimizer or torch.optim.AdamW(
            parameters,
            lr=float(lr),
            weight_decay=float(weight_decay),
        )
        optimizer_ids = _optimizer_parameter_ids(self.optimizer)
        expected_ids = {id(parameter) for parameter in parameters}
        if (
            len(optimizer_ids) != len(set(optimizer_ids))
            or set(optimizer_ids) != expected_ids
        ):
            raise ValueError("optimizer parameters must be exactly Gate parameters")

        self.train_positive_count = positives
        self.train_negative_count = negatives
        self._training_identity = _freeze_json(validated_training_identity)
        self.pos_weight = float(negatives / positives)
        self.max_grad_norm = float(max_grad_norm)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.global_step = 0
        self.epoch = 0
        self.best_epoch = -1
        self.best_global_step = 0
        self.best_val_bce = math.inf
        self.best_metrics: dict[str, Any] = {}
        self.epochs_without_improvement = 0
        self._best_gate_state: dict[str, torch.Tensor] | None = None
        self.gate.zero_grad(set_to_none=True)

    @property
    def device(self) -> torch.device:
        return next(self.gate.parameters()).device

    @property
    def training_identity(self) -> Mapping[str, Any]:
        """Immutable identity of every artifact/config used for this run."""

        return self._training_identity

    def _validate_batch(self, batch: Mapping[str, Any]) -> _ValidatedBatch:
        if not isinstance(batch, Mapping):
            raise TypeError("Gate batch must be a mapping")
        keys = set(batch)
        if keys != _BATCH_KEYS:
            missing = sorted(_BATCH_KEYS - keys)
            unexpected = sorted(keys - _BATCH_KEYS)
            raise ValueError(
                "Gate batch fields do not match the current-only schema; "
                f"missing={missing}, unexpected={unexpected}"
            )
        model_inputs: dict[str, torch.Tensor] = {}
        for name in ("input_image", "context", "context_mask", "proprio"):
            value = batch[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Gate batch {name} must be a torch.Tensor")
            model_inputs[name] = value.to(device=self.device)
        batch_size = int(model_inputs["input_image"].shape[0])
        labels = _binary_labels(batch["label"], field="Gate batch label")
        if labels.shape != (batch_size,):
            raise ValueError("Gate batch label must have shape [B]")

        weights = batch["sample_weight"]
        if not isinstance(weights, torch.Tensor):
            raise TypeError("Gate batch sample_weight must be a torch.Tensor")
        if weights.ndim != 1 or tuple(weights.shape) != (batch_size,):
            raise ValueError("Gate batch sample_weight must have shape [B]")
        if not weights.is_floating_point():
            raise TypeError("Gate batch sample_weight must be floating point")
        weights = weights.detach().to(dtype=torch.float32, device="cpu")
        if not torch.isfinite(weights).all() or bool((weights < 0.0).any().item()):
            raise ValueError("Gate batch sample_weight must be finite and non-negative")
        if not float(weights.sum().item()) > 0.0:
            raise ValueError("Gate batch sample_weight sum must be positive")

        sample_ids = batch["sample_id"]
        if isinstance(sample_ids, str):
            sample_ids = (sample_ids,)
        elif isinstance(sample_ids, Sequence):
            sample_ids = tuple(sample_ids)
        else:
            raise TypeError("Gate batch sample_id must be a string sequence")
        if len(sample_ids) != batch_size or any(
            not isinstance(value, str) or not value for value in sample_ids
        ):
            raise ValueError("Gate batch sample_id must contain B non-empty strings")
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("Gate batch sample_id values must be unique")
        return _ValidatedBatch(
            model_inputs=model_inputs,
            labels=labels.to(device=self.device),
            sample_weights=weights.to(device=self.device),
            sample_ids=tuple(sample_ids),
        )

    def _forward_loss(
        self,
        batch: Mapping[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        validated = self._validate_batch(batch)
        logits = self.gate(**validated.model_inputs)
        labels = validated.labels.to(dtype=logits.dtype)
        weights = validated.sample_weights.to(dtype=logits.dtype)
        elementwise = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=torch.as_tensor(
                self.pos_weight,
                dtype=logits.dtype,
                device=logits.device,
            ),
            reduction="none",
        )
        numerator = (elementwise * weights).sum()
        denominator = weights.sum()
        loss = numerator / denominator
        if not torch.isfinite(loss):
            raise RuntimeError("Gate objective became non-finite")
        return loss, logits, labels, weights

    def train_batch(self, batch: Mapping[str, Any]) -> float:
        """Apply one optimizer update and return its weighted objective BCE."""

        self.gate.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss, _, _, _ = self._forward_loss(batch)
        loss.backward()
        if any(
            parameter.grad is not None
            and not torch.isfinite(parameter.grad).all()
            for parameter in self.gate.parameters()
        ):
            raise RuntimeError("Gate gradient became non-finite")
        torch.nn.utils.clip_grad_norm_(
            self.gate.parameters(),
            self.max_grad_norm,
            error_if_nonfinite=True,
        )
        self.optimizer.step()
        self.global_step += 1
        return float(loss.detach().cpu().item())

    def _epoch(
        self,
        batches: Iterable[Mapping[str, Any]],
        *,
        training: bool,
        threshold: float,
        num_calibration_bins: int,
    ) -> GateEpochResult:
        self.gate.train(mode=training)
        all_logits: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        objective_numerator = 0.0
        objective_denominator = 0.0
        num_batches = 0
        for batch in batches:
            if training:
                self.optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                loss, logits, labels, weights = self._forward_loss(batch)
            if training:
                loss.backward()
                if any(
                    parameter.grad is not None
                    and not torch.isfinite(parameter.grad).all()
                    for parameter in self.gate.parameters()
                ):
                    raise RuntimeError("Gate gradient became non-finite")
                torch.nn.utils.clip_grad_norm_(
                    self.gate.parameters(),
                    self.max_grad_norm,
                    error_if_nonfinite=True,
                )
                self.optimizer.step()
                self.global_step += 1
            weight_sum = float(weights.detach().double().sum().cpu().item())
            objective_numerator += float(loss.detach().cpu().item()) * weight_sum
            objective_denominator += weight_sum
            all_logits.append(logits.detach().to(device="cpu", dtype=torch.float64))
            all_labels.append(labels.detach().to(device="cpu", dtype=torch.float64))
            num_batches += 1
        if num_batches == 0:
            raise ValueError("Gate epoch requires at least one batch")
        metrics = compute_gate_binary_metrics(
            logits=torch.cat(all_logits),
            labels=torch.cat(all_labels),
            threshold=threshold,
            num_calibration_bins=num_calibration_bins,
        )
        return GateEpochResult(
            objective_bce=objective_numerator / objective_denominator,
            metrics=metrics,
            num_batches=num_batches,
        )

    def train_epoch(
        self,
        batches: Iterable[Mapping[str, Any]],
        *,
        threshold: float = 0.5,
        num_calibration_bins: int = 10,
    ) -> GateEpochResult:
        """Train for one complete epoch and aggregate pre-update predictions."""

        result = self._epoch(
            batches,
            training=True,
            threshold=threshold,
            num_calibration_bins=num_calibration_bins,
        )
        self.epoch += 1
        return result

    def evaluate_epoch(
        self,
        batches: Iterable[Mapping[str, Any]],
        *,
        threshold: float = 0.5,
        num_calibration_bins: int = 10,
    ) -> GateEpochResult:
        """Evaluate without gradients or optimizer mutation."""

        return self._epoch(
            batches,
            training=False,
            threshold=threshold,
            num_calibration_bins=num_calibration_bins,
        )

    def fit(
        self,
        train_batches: Iterable[Mapping[str, Any]],
        val_batches: Iterable[Mapping[str, Any]],
        *,
        num_epochs: int = 5,
        early_stop_patience: int = 2,
        min_delta: float = 0.0,
        threshold: float = 0.5,
        num_calibration_bins: int = 10,
    ) -> GateFitResult:
        """Run additional epochs and early-stop on validation metric BCE."""

        if isinstance(num_epochs, bool) or not isinstance(num_epochs, int):
            raise TypeError("num_epochs must be an integer")
        if num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if (
            isinstance(early_stop_patience, bool)
            or not isinstance(early_stop_patience, int)
            or early_stop_patience < 0
        ):
            raise ValueError("early_stop_patience must be a non-negative integer")
        if not math.isfinite(float(min_delta)) or float(min_delta) < 0.0:
            raise ValueError("min_delta must be finite and non-negative")

        history: list[dict[str, Any]] = []
        stopped_early = False
        for _ in range(num_epochs):
            train_result = self.train_epoch(
                train_batches,
                threshold=threshold,
                num_calibration_bins=num_calibration_bins,
            )
            val_result = self.evaluate_epoch(
                val_batches,
                threshold=threshold,
                num_calibration_bins=num_calibration_bins,
            )
            record = {
                "epoch": self.epoch,
                "global_step": self.global_step,
                "train": train_result.to_dict(),
                "val": val_result.to_dict(),
            }
            history.append(record)
            val_bce = float(val_result.objective_bce)
            if val_bce < self.best_val_bce - float(min_delta):
                self.best_val_bce = val_bce
                self.best_epoch = self.epoch
                self.best_global_step = self.global_step
                self.best_metrics = val_result.to_dict()
                self._best_gate_state = _gate_state(self.gate)
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= early_stop_patience:
                    stopped_early = True
                    break
        return GateFitResult(
            epochs=tuple(history),
            stopped_early=stopped_early,
            best_epoch=self.best_epoch,
            best_val_bce=self.best_val_bce,
        )

    def save_training_state(self, path: str | Path) -> Path:
        """Atomically save exact optimizer, progress, best model, and RNG state."""

        payload = {
            "schema_version": GATE_TRAINING_STATE_SCHEMA_VERSION,
            "kind": GATE_TRAINING_STATE_KIND,
            "training_identity": _thaw_json(self.training_identity),
            "gate_config": self.gate.config(),
            "gate_state_dict": _gate_state(self.gate),
            "optimizer_state_dict": _cpu_tree(self.optimizer.state_dict()),
            "epoch": self.epoch,
            "global_step": self.global_step,
            "train_positive_count": self.train_positive_count,
            "train_negative_count": self.train_negative_count,
            "pos_weight": self.pos_weight,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "best_epoch": self.best_epoch,
            "best_global_step": self.best_global_step,
            "best_val_bce": self.best_val_bce,
            "best_metrics": dict(self.best_metrics),
            "epochs_without_improvement": self.epochs_without_improvement,
            "best_gate_state_dict": _cpu_tree(self._best_gate_state),
            "rng_state": _capture_rng_state(),
        }
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp")
        torch.save(payload, temporary)
        temporary.replace(output)
        return output

    def load_training_state(self, path: str | Path) -> None:
        """Restore a state saved by :meth:`save_training_state` fail-closed."""

        payload = torch.load(path, map_location="cpu", weights_only=False)
        required = {
            "schema_version",
            "kind",
            "training_identity",
            "gate_config",
            "gate_state_dict",
            "optimizer_state_dict",
            "epoch",
            "global_step",
            "train_positive_count",
            "train_negative_count",
            "pos_weight",
            "lr",
            "weight_decay",
            "max_grad_norm",
            "best_epoch",
            "best_global_step",
            "best_val_bce",
            "best_metrics",
            "epochs_without_improvement",
            "best_gate_state_dict",
            "rng_state",
        }
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError("Gate training state fields do not match schema")
        if (
            payload["schema_version"] != GATE_TRAINING_STATE_SCHEMA_VERSION
            or payload["kind"] != GATE_TRAINING_STATE_KIND
        ):
            raise ValueError("unsupported Gate training state")
        recorded_training_identity = _validate_training_identity(
            payload["training_identity"]
        )
        expected_training_identity = _thaw_json(self.training_identity)
        if recorded_training_identity != expected_training_identity:
            raise ValueError("Gate training state training_identity mismatch")
        if payload["gate_config"] != self.gate.config():
            raise ValueError("Gate training state architecture mismatch")
        expected_contract = {
            "train_positive_count": self.train_positive_count,
            "train_negative_count": self.train_negative_count,
            "pos_weight": self.pos_weight,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
        }
        for field, expected in expected_contract.items():
            recorded = payload[field]
            if isinstance(expected, float):
                matches = math.isclose(
                    float(recorded), expected, rel_tol=0.0, abs_tol=0.0
                )
            else:
                matches = recorded == expected
            if not matches:
                raise ValueError(f"Gate training state {field} mismatch")

        integer_fields = (
            "epoch",
            "global_step",
            "best_global_step",
            "epochs_without_improvement",
        )
        for field in integer_fields:
            value = payload[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Gate training state {field} is invalid")
        best_epoch = payload["best_epoch"]
        if (
            isinstance(best_epoch, bool)
            or not isinstance(best_epoch, int)
            or best_epoch < -1
        ):
            raise ValueError("Gate training state best_epoch is invalid")
        best_val_bce = float(payload["best_val_bce"])
        if math.isnan(best_val_bce) or best_val_bce < 0.0:
            raise ValueError("Gate training state best_val_bce is invalid")
        if not isinstance(payload["best_metrics"], Mapping):
            raise ValueError("Gate training state best_metrics must be a mapping")
        best_state = payload["best_gate_state_dict"]
        if (best_epoch == -1) != (best_state is None):
            raise ValueError("Gate training best-state presence is inconsistent")

        self.gate.load_state_dict(dict(payload["gate_state_dict"]), strict=True)
        self.optimizer.load_state_dict(dict(payload["optimizer_state_dict"]))
        self.epoch = payload["epoch"]
        self.global_step = payload["global_step"]
        self.best_epoch = best_epoch
        self.best_global_step = payload["best_global_step"]
        self.best_val_bce = best_val_bce
        self.best_metrics = dict(payload["best_metrics"])
        self.epochs_without_improvement = payload["epochs_without_improvement"]
        self._best_gate_state = (
            None if best_state is None else dict(best_state)
        )
        _restore_rng_state(payload["rng_state"])
        self.gate.zero_grad(set_to_none=True)

    def export_checkpoint(
        self,
        path: str | Path,
        *,
        selection: str,
        label_manifest_sha256: str,
        adapter_checkpoint_sha256: str,
        data_manifest_sha256: str,
        episode_split_assignment_sha256: str,
        training_config_sha256: str,
        git_identity: Mapping[str, Any],
    ) -> Path:
        """Export ``best`` or ``last`` with the exact training identity."""

        expected_identity = _thaw_json(self.training_identity)
        export_identity = _validate_training_identity(
            {
                "label_manifest_sha256": label_manifest_sha256,
                "adapter_checkpoint_sha256": adapter_checkpoint_sha256,
                "base_checkpoint_sha256": expected_identity[
                    "base_checkpoint_sha256"
                ],
                "data_manifest_sha256": data_manifest_sha256,
                "episode_split_assignment_sha256": (
                    episode_split_assignment_sha256
                ),
                "training_config_sha256": training_config_sha256,
                "git_identity": git_identity,
            }
        )
        if export_identity != expected_identity:
            raise ValueError(
                "Gate export identity differs from the training identity"
            )
        if selection == "last":
            gate = self.gate
            epoch = self.epoch
            global_step = self.global_step
            metrics = self.best_metrics
        elif selection == "best":
            if self._best_gate_state is None:
                raise RuntimeError("no best Gate state is available for export")
            # Deep-copying avoids consuming global initialization RNG merely
            # because a best checkpoint is exported during training.
            gate = copy.deepcopy(self.gate)
            gate.load_state_dict(self._best_gate_state, strict=True)
            epoch = self.best_epoch
            global_step = self.best_global_step
            metrics = self.best_metrics
        else:
            raise ValueError("selection must be 'best' or 'last'")
        return save_gate_checkpoint(
            path,
            gate,
            label_manifest_sha256=label_manifest_sha256,
            adapter_checkpoint_sha256=adapter_checkpoint_sha256,
            base_checkpoint_sha256=expected_identity[
                "base_checkpoint_sha256"
            ],
            data_manifest_sha256=data_manifest_sha256,
            episode_split_assignment_sha256=(
                episode_split_assignment_sha256
            ),
            training_config_sha256=training_config_sha256,
            git_identity=git_identity,
            global_step=global_step,
            epoch=epoch,
            best_metrics=metrics,
        )


__all__ = [
    "GATE_TRAINING_STATE_KIND",
    "GATE_TRAINING_STATE_SCHEMA_VERSION",
    "GateEpochResult",
    "GateFitResult",
    "GateTrainer",
]
