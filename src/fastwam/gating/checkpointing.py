"""Portable, identity-bound checkpoints for the independent Stage 2 Gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch

from fastwam.models.video_gate import BinaryVideoGate

from .contracts import require_sha256


GATE_CHECKPOINT_SCHEMA_VERSION = 2
GATE_CHECKPOINT_KIND = "stage2_binary_video_gate_export"
_REQUIRED_KEYS = {
    "schema_version",
    "kind",
    "gate_config",
    "gate_state_dict",
    "parameter_count",
    "label_manifest_sha256",
    "adapter_checkpoint_sha256",
    "base_checkpoint_sha256",
    "data_manifest_sha256",
    "episode_split_assignment_sha256",
    "training_config_sha256",
    "git_identity",
    "global_step",
    "epoch",
    "best_metrics",
}
_GIT_IDENTITY_KEYS = {
    "commit",
    "tracked_dirty",
    "untracked_source_files",
}


def _nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value < 0:
        raise ValueError(f"{field} must be non-negative")
    return int(value)


def _json_mapping(value: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    payload = dict(value)
    try:
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be canonical-JSON serializable") from error
    return payload


def _git_identity(
    value: Mapping[str, Any], *, field: str
) -> dict[str, Any]:
    payload = _json_mapping(value, field=field)
    if set(payload) != _GIT_IDENTITY_KEYS:
        raise ValueError(f"{field} fields do not match the Git identity schema")
    commit = payload["commit"]
    if not isinstance(commit, str) or len(commit) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise ValueError(f"{field} commit must be a full lowercase Git SHA")
    if not isinstance(payload["tracked_dirty"], bool):
        raise TypeError(f"{field} tracked_dirty must be bool")
    files = payload["untracked_source_files"]
    if not isinstance(files, list) or any(
        not isinstance(path, str) or not path for path in files
    ):
        raise TypeError(f"{field} untracked_source_files must be strings")
    if files != sorted(set(files)):
        raise ValueError(f"{field} untracked_source_files must be sorted and unique")
    return payload


def _finite_gate_state(gate: BinaryVideoGate) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, value in gate.state_dict().items():
        tensor = value.detach().to(device="cpu").clone()
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(f"Gate parameter {name!r} contains a non-finite value")
        state[name] = tensor
    return state


def save_gate_checkpoint(
    path: str | Path,
    gate: BinaryVideoGate,
    *,
    label_manifest_sha256: str,
    adapter_checkpoint_sha256: str,
    base_checkpoint_sha256: str,
    data_manifest_sha256: str,
    episode_split_assignment_sha256: str,
    training_config_sha256: str,
    git_identity: Mapping[str, Any],
    global_step: int,
    epoch: int,
    best_metrics: Mapping[str, Any],
) -> Path:
    if not isinstance(gate, BinaryVideoGate):
        raise TypeError("gate must be a BinaryVideoGate")
    identities = {
        "label_manifest_sha256": require_sha256(
            label_manifest_sha256,
            field="label_manifest_sha256",
        ),
        "adapter_checkpoint_sha256": require_sha256(
            adapter_checkpoint_sha256,
            field="adapter_checkpoint_sha256",
        ),
        "base_checkpoint_sha256": require_sha256(
            base_checkpoint_sha256,
            field="base_checkpoint_sha256",
        ),
        "data_manifest_sha256": require_sha256(
            data_manifest_sha256,
            field="data_manifest_sha256",
        ),
        "episode_split_assignment_sha256": require_sha256(
            episode_split_assignment_sha256,
            field="episode_split_assignment_sha256",
        ),
        "training_config_sha256": require_sha256(
            training_config_sha256,
            field="training_config_sha256",
        ),
    }
    payload = {
        "schema_version": GATE_CHECKPOINT_SCHEMA_VERSION,
        "kind": GATE_CHECKPOINT_KIND,
        "gate_config": gate.config(),
        "gate_state_dict": _finite_gate_state(gate),
        "parameter_count": gate.parameter_count(),
        **identities,
        "git_identity": _git_identity(git_identity, field="git_identity"),
        "global_step": _nonnegative_int(global_step, field="global_step"),
        "epoch": _nonnegative_int(epoch, field="epoch"),
        "best_metrics": _json_mapping(best_metrics, field="best_metrics"),
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    return output


def _expected_identity(
    payload: Mapping[str, Any],
    *,
    field: str,
    expected: str,
) -> None:
    recorded = require_sha256(payload.get(field), field=f"Gate checkpoint {field}")
    expected = require_sha256(expected, field=f"expected {field}")
    if recorded != expected:
        raise ValueError(f"Gate checkpoint {field} mismatch")


def load_gate_checkpoint(
    path: str | Path,
    *,
    expected_label_manifest_sha256: str,
    expected_adapter_checkpoint_sha256: str,
    expected_base_checkpoint_sha256: str,
    expected_data_manifest_sha256: str,
    expected_episode_split_assignment_sha256: str,
    expected_training_config_sha256: str,
    expected_git_identity: Mapping[str, Any],
    map_location: str | torch.device = "cpu",
) -> tuple[BinaryVideoGate, dict[str, Any]]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("Gate checkpoint must contain a mapping")
    payload = dict(payload)
    if set(payload) != _REQUIRED_KEYS:
        raise ValueError("Gate checkpoint fields do not match its schema")
    if (
        payload.get("schema_version") != GATE_CHECKPOINT_SCHEMA_VERSION
        or payload.get("kind") != GATE_CHECKPOINT_KIND
    ):
        raise ValueError("unsupported Gate checkpoint")
    _expected_identity(
        payload,
        field="label_manifest_sha256",
        expected=expected_label_manifest_sha256,
    )
    _expected_identity(
        payload,
        field="adapter_checkpoint_sha256",
        expected=expected_adapter_checkpoint_sha256,
    )
    _expected_identity(
        payload,
        field="base_checkpoint_sha256",
        expected=expected_base_checkpoint_sha256,
    )
    _expected_identity(
        payload,
        field="data_manifest_sha256",
        expected=expected_data_manifest_sha256,
    )
    _expected_identity(
        payload,
        field="episode_split_assignment_sha256",
        expected=expected_episode_split_assignment_sha256,
    )
    _expected_identity(
        payload,
        field="training_config_sha256",
        expected=expected_training_config_sha256,
    )
    _nonnegative_int(payload.get("global_step"), field="Gate checkpoint global_step")
    _nonnegative_int(payload.get("epoch"), field="Gate checkpoint epoch")
    recorded_git_identity = _git_identity(
        payload.get("git_identity"), field="Gate checkpoint git_identity"
    )
    validated_expected_git_identity = _git_identity(
        expected_git_identity, field="expected Gate checkpoint git_identity"
    )
    if recorded_git_identity != validated_expected_git_identity:
        raise ValueError("Gate checkpoint git_identity mismatch")
    _json_mapping(payload.get("best_metrics"), field="Gate checkpoint best_metrics")

    config = payload.get("gate_config")
    if not isinstance(config, Mapping):
        raise ValueError("Gate checkpoint is missing gate_config")
    gate = BinaryVideoGate.from_config(config)
    recorded_count = _nonnegative_int(
        payload.get("parameter_count"),
        field="Gate checkpoint parameter_count",
    )
    if recorded_count != gate.parameter_count():
        raise ValueError("Gate checkpoint parameter count disagrees with config")
    state = payload.get("gate_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("Gate checkpoint is missing gate_state_dict")
    gate.load_state_dict(dict(state), strict=True)
    for name, value in gate.state_dict().items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"loaded Gate parameter {name!r} is non-finite")
    gate.eval()
    gate.requires_grad_(False)
    return gate, payload


__all__ = [
    "GATE_CHECKPOINT_KIND",
    "GATE_CHECKPOINT_SCHEMA_VERSION",
    "load_gate_checkpoint",
    "save_gate_checkpoint",
]
