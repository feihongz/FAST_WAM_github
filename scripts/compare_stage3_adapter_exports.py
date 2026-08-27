#!/usr/bin/env python3
"""Require two formal Stage 3 Adapter exports to be tensor-exact."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
from typing import Any

import torch

from fastwam.models.wan22.video_action_alignment import (
    VideoActionResidualAdapter,
)


EXPORT_SCHEMA_VERSION = 2
EXPORT_KIND = "stage3_alignment_export"
MAX_EXPORT_SIZE_BYTES = 256 * 1024 * 1024
MAX_ADAPTER_PARAMETER_COUNT = MAX_EXPORT_SIZE_BYTES // torch.float32.itemsize
EXPORT_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "base_checkpoint",
        "base_checkpoint_sha256",
        "data_manifest_sha256",
        "alignment_config",
        "global_step",
        "git_commit",
        "training_contract_sha256",
        "asset_identities",
        "adapter",
    }
)
ASSET_IDENTITY_KEYS = frozenset({"path", "sha256", "size_bytes"})
REQUIRED_ASSET_NAMES = frozenset({"vae", "normalization_stats"})
ALIGNMENT_CONFIG_INTEGER_KEYS = frozenset(
    {
        "action_hidden_dim",
        "video_hidden_dim",
        "action_dim",
        "bottleneck_dim",
        "num_heads",
        "ffn_multiplier",
    }
)
ALIGNMENT_CONFIG_BOOLEAN_KEYS = frozenset(
    {"drop_first_video_frame", "zero_init_output"}
)
ALIGNMENT_CONFIG_KEYS = (
    ALIGNMENT_CONFIG_INTEGER_KEYS | ALIGNMENT_CONFIG_BOOLEAN_KEYS
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
GIT_COMMIT_PATTERN = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")


@dataclass(frozen=True)
class _ExportSnapshot:
    path: Path
    payload: dict[str, Any]
    size_bytes: int
    sha256: str
    st_dev: int
    st_ino: int


def _require_nonempty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Stage 3 export {field} must be a non-empty string")
    return value


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"Stage 3 export {field} must contain 64 lowercase hex characters"
        )
    return value


def _validate_asset_identities(value: Any, *, path: Path) -> None:
    if not isinstance(value, Mapping) or set(value) != REQUIRED_ASSET_NAMES:
        raise ValueError(
            "Stage 3 export asset_identities must contain exactly vae and "
            f"normalization_stats: {path}"
        )
    for name, identity in value.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Stage 3 export asset identity name is invalid: {path}")
        if not isinstance(identity, Mapping) or set(identity) != ASSET_IDENTITY_KEYS:
            raise ValueError(
                f"Stage 3 export asset identity schema is invalid: {path}:{name}"
            )
        _require_nonempty_string(
            identity["path"], field=f"asset_identities.{name}.path"
        )
        _require_sha256(
            identity["sha256"], field=f"asset_identities.{name}.sha256"
        )
        size_bytes = identity["size_bytes"]
        if (
            isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes <= 0
        ):
            raise ValueError(
                "Stage 3 export asset identity size_bytes must be a positive "
                f"integer: {path}:{name}"
            )


def _expected_adapter_state(
    value: Any, *, path: Path
) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping) or set(value) != ALIGNMENT_CONFIG_KEYS:
        raise ValueError(f"Stage 3 export alignment_config is invalid: {path}")
    config = dict(value)
    for name in ALIGNMENT_CONFIG_INTEGER_KEYS:
        item = config[name]
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ValueError(
                "Stage 3 export alignment_config integer is invalid: "
                f"{path}:{name}"
            )
    for name in ALIGNMENT_CONFIG_BOOLEAN_KEYS:
        if not isinstance(config[name], bool):
            raise ValueError(
                "Stage 3 export alignment_config boolean is invalid: "
                f"{path}:{name}"
            )
    try:
        # Validate constructor constraints and derive the authoritative
        # state_dict contract without allocating attacker-sized parameters.
        with torch.device("meta"):
            adapter = VideoActionResidualAdapter(**config)
        expected_state = dict(adapter.state_dict())
    except (OverflowError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(
            f"Stage 3 export alignment_config cannot instantiate Adapter: {path}"
        ) from error
    parameter_count = sum(tensor.numel() for tensor in expected_state.values())
    if parameter_count > MAX_ADAPTER_PARAMETER_COUNT:
        raise ValueError(
            "Stage 3 export Adapter parameter count exceeds the maximum: "
            f"{path} ({parameter_count} > {MAX_ADAPTER_PARAMETER_COUNT})"
        )
    return expected_state


def _validate_payload(payload: Any, *, path: Path) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != EXPORT_KEYS:
        raise ValueError(f"Stage 3 export schema is invalid: {path}")
    if (
        isinstance(payload["schema_version"], bool)
        or not isinstance(payload["schema_version"], int)
        or payload["schema_version"] != EXPORT_SCHEMA_VERSION
        or not isinstance(payload["kind"], str)
        or payload["kind"] != EXPORT_KIND
    ):
        raise ValueError(f"Stage 3 export header is invalid: {path}")

    _require_nonempty_string(payload["base_checkpoint"], field="base_checkpoint")
    _require_sha256(
        payload["base_checkpoint_sha256"], field="base_checkpoint_sha256"
    )
    _require_sha256(
        payload["data_manifest_sha256"], field="data_manifest_sha256"
    )
    _require_sha256(
        payload["training_contract_sha256"],
        field="training_contract_sha256",
    )
    git_commit = payload["git_commit"]
    if (
        not isinstance(git_commit, str)
        or GIT_COMMIT_PATTERN.fullmatch(git_commit) is None
    ):
        raise ValueError(
            "Stage 3 export git_commit must be 40 or 64 lowercase hex "
            f"characters: {path}"
        )

    global_step = payload["global_step"]
    if (
        isinstance(global_step, bool)
        or not isinstance(global_step, int)
        or global_step < 0
    ):
        raise ValueError(
            f"Stage 3 export global_step must be a non-negative integer: {path}"
        )

    _validate_asset_identities(payload["asset_identities"], path=path)
    expected_state = _expected_adapter_state(
        payload["alignment_config"], path=path
    )
    adapter = payload["adapter"]
    if not isinstance(adapter, Mapping) or not adapter:
        raise ValueError(f"Stage 3 export Adapter is empty or invalid: {path}")
    if set(adapter) != set(expected_state):
        raise ValueError(
            f"Stage 3 export Adapter tensor names are incomplete: {path}"
        )
    for name, expected_tensor in expected_state.items():
        tensor = adapter[name]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(
                f"Stage 3 export Adapter entry is invalid: {path}:{name}"
            )
        if (
            tensor.layout != torch.strided
            or tensor.device.type != "cpu"
            or tensor.dtype != torch.float32
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                "Stage 3 export Adapter tensor must be dense CPU FP32 and "
                "contiguous: "
                f"{path}:{name}"
            )
        if tensor.shape != expected_tensor.shape:
            raise ValueError(
                f"Stage 3 export Adapter tensor shape is invalid: {path}:{name}"
            )
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(
                f"Stage 3 export Adapter tensor is non-finite: {path}:{name}"
            )
    return payload


def _load_export(path: Path) -> _ExportSnapshot:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NONBLOCK", 0
    )
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        file_handle = os.fdopen(descriptor, "rb")
        descriptor = None  # file_handle now owns it.
        with file_handle as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f"Stage 3 export is not a regular file: {path}")
            if before.st_size <= 0:
                raise ValueError(f"Stage 3 export is empty: {path}")
            if before.st_size > MAX_EXPORT_SIZE_BYTES:
                raise ValueError(
                    "Stage 3 export exceeds the maximum allowed size: "
                    f"{path} ({before.st_size} > {MAX_EXPORT_SIZE_BYTES})"
                )
            immutable_bytes = handle.read(MAX_EXPORT_SIZE_BYTES + 1)
            after = os.fstat(handle.fileno())
    except OSError as error:
        raise ValueError(f"Cannot read Stage 3 export: {path}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)

    stable_fields_before = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    stable_fields_after = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if stable_fields_before != stable_fields_after:
        raise RuntimeError(f"Stage 3 export changed while being read: {path}")
    if len(immutable_bytes) != before.st_size:
        raise RuntimeError(f"Stage 3 export size changed while being read: {path}")
    if len(immutable_bytes) > MAX_EXPORT_SIZE_BYTES:
        raise ValueError(f"Stage 3 export exceeds the maximum allowed size: {path}")

    digest = hashlib.sha256(immutable_bytes).hexdigest()
    try:
        payload = torch.load(
            io.BytesIO(immutable_bytes),
            map_location="cpu",
            weights_only=True,
        )
    except Exception as error:
        raise ValueError(
            f"Cannot deserialize Stage 3 export safely: {path}"
        ) from error
    validated = _validate_payload(payload, path=path)
    return _ExportSnapshot(
        path=path,
        payload=validated,
        size_bytes=len(immutable_bytes),
        sha256=digest,
        st_dev=int(before.st_dev),
        st_ino=int(before.st_ino),
    )


def _validate_expected_sha256(
    value: str | None, *, field: str
) -> str | None:
    if value is None:
        return None
    return _require_sha256(value, field=field)


def compare_stage3_adapter_exports(
    left: str | Path,
    right: str | Path,
    *,
    expected_step: int | None = None,
    expected_base_checkpoint_sha256: str | None = None,
    expected_data_manifest_sha256: str | None = None,
    expected_training_contract_sha256: str | None = None,
    expected_git_commit: str | None = None,
) -> dict[str, Any]:
    """Compare formal export metadata and every Adapter tensor exactly."""

    if expected_step is not None and (
        isinstance(expected_step, bool)
        or not isinstance(expected_step, int)
        or expected_step < 0
    ):
        raise ValueError("expected_step must be a non-negative integer")
    expectations = {
        "base_checkpoint_sha256": _validate_expected_sha256(
            expected_base_checkpoint_sha256,
            field="expected_base_checkpoint_sha256",
        ),
        "data_manifest_sha256": _validate_expected_sha256(
            expected_data_manifest_sha256,
            field="expected_data_manifest_sha256",
        ),
        "training_contract_sha256": _validate_expected_sha256(
            expected_training_contract_sha256,
            field="expected_training_contract_sha256",
        ),
    }
    if expected_git_commit is not None and (
        not isinstance(expected_git_commit, str)
        or GIT_COMMIT_PATTERN.fullmatch(expected_git_commit) is None
    ):
        raise ValueError(
            "expected_git_commit must contain 40 or 64 lowercase hex characters"
        )

    try:
        left_path = Path(left).expanduser().resolve(strict=True)
        right_path = Path(right).expanduser().resolve(strict=True)
    except OSError as error:
        raise ValueError("Stage 3 export path does not exist") from error
    if left_path == right_path:
        raise ValueError("Stage 3 exports must be distinct resolved paths")

    left_snapshot = _load_export(left_path)
    right_snapshot = _load_export(right_path)
    if (left_snapshot.st_dev, left_snapshot.st_ino) == (
        right_snapshot.st_dev,
        right_snapshot.st_ino,
    ):
        raise ValueError("Stage 3 exports must not reference the same inode")

    left_payload = left_snapshot.payload
    right_payload = right_snapshot.payload
    left_metadata = {
        key: value for key, value in left_payload.items() if key != "adapter"
    }
    right_metadata = {
        key: value for key, value in right_payload.items() if key != "adapter"
    }
    if left_metadata != right_metadata:
        differing = sorted(
            key
            for key in left_metadata
            if left_metadata[key] != right_metadata.get(key)
        )
        raise ValueError(f"Stage 3 export metadata differs: {differing}")
    if (
        expected_step is not None
        and left_payload["global_step"] != expected_step
    ):
        raise ValueError(
            "Stage 3 export global_step differs from expected_step: "
            f"actual={left_payload['global_step']} expected={expected_step}"
        )
    for field, expected in expectations.items():
        if expected is not None and left_payload[field] != expected:
            raise ValueError(
                f"Stage 3 export {field} differs from expectation: "
                f"actual={left_payload[field]} expected={expected}"
            )
    if (
        expected_git_commit is not None
        and left_payload["git_commit"] != expected_git_commit
    ):
        raise ValueError(
            "Stage 3 export git_commit differs from expectation: "
            f"actual={left_payload['git_commit']} expected={expected_git_commit}"
        )

    left_adapter = left_payload["adapter"]
    right_adapter = right_payload["adapter"]
    parameter_count = 0
    for name in sorted(left_adapter):
        left_tensor = left_adapter[name]
        right_tensor = right_adapter[name]
        # Each export already passed the independent Adapter state contract;
        # retain explicit checks here to make comparison semantics self-evident.
        if (
            left_tensor.shape != right_tensor.shape
            or left_tensor.dtype != right_tensor.dtype
        ):
            raise ValueError(f"Stage 3 Adapter tensor contract differs: {name}")
        if not torch.equal(left_tensor, right_tensor):
            raise ValueError(f"Stage 3 Adapter tensor values differ: {name}")
        parameter_count += left_tensor.numel()

    def receipt_identity(snapshot: _ExportSnapshot) -> dict[str, Any]:
        return {
            "path": str(snapshot.path),
            "size_bytes": snapshot.size_bytes,
            "sha256": snapshot.sha256,
            "st_dev": snapshot.st_dev,
            "st_ino": snapshot.st_ino,
        }

    return {
        "status": "ok",
        "global_step": left_payload["global_step"],
        "base_checkpoint_sha256": left_payload["base_checkpoint_sha256"],
        "training_contract_sha256": left_payload[
            "training_contract_sha256"
        ],
        "data_manifest_sha256": left_payload["data_manifest_sha256"],
        "tensor_count": len(left_adapter),
        "parameter_count": parameter_count,
        "left": receipt_identity(left_snapshot),
        "right": receipt_identity(right_snapshot),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare uninterrupted and strict-resume Stage 3 exports."
    )
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--expected-step", type=int, default=None)
    parser.add_argument("--expected-base-checkpoint-sha256", default=None)
    parser.add_argument("--expected-data-manifest-sha256", default=None)
    parser.add_argument("--expected-training-contract-sha256", default=None)
    parser.add_argument("--expected-git-commit", default=None)
    args = parser.parse_args()
    receipt = compare_stage3_adapter_exports(
        args.left,
        args.right,
        expected_step=args.expected_step,
        expected_base_checkpoint_sha256=args.expected_base_checkpoint_sha256,
        expected_data_manifest_sha256=args.expected_data_manifest_sha256,
        expected_training_contract_sha256=(
            args.expected_training_contract_sha256
        ),
        expected_git_commit=args.expected_git_commit,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
