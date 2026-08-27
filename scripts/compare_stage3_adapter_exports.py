#!/usr/bin/env python3
"""Require two Stage 3 Adapter exports to be byte-semantically identical."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


EXPORT_SCHEMA_VERSION = 2
EXPORT_KIND = "stage3_alignment_export"
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


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_export(path: str | Path) -> tuple[Path, dict[str, Any]]:
    candidate = Path(path).expanduser().resolve(strict=True)
    if not candidate.is_file():
        raise ValueError(f"Stage 3 export is not a regular file: {candidate}")
    payload = torch.load(candidate, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != EXPORT_KEYS:
        raise ValueError(f"Stage 3 export schema is invalid: {candidate}")
    if (
        payload["schema_version"] != EXPORT_SCHEMA_VERSION
        or payload["kind"] != EXPORT_KIND
    ):
        raise ValueError(f"Stage 3 export header is invalid: {candidate}")
    global_step = payload["global_step"]
    if isinstance(global_step, bool) or not isinstance(global_step, int):
        raise ValueError(f"Stage 3 export global_step is invalid: {candidate}")
    if global_step < 0:
        raise ValueError(f"Stage 3 export global_step is negative: {candidate}")
    adapter = payload["adapter"]
    if not isinstance(adapter, Mapping) or not adapter:
        raise ValueError(f"Stage 3 export Adapter is empty or invalid: {candidate}")
    for name, tensor in adapter.items():
        if not isinstance(name, str) or not name or not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Stage 3 export Adapter entry is invalid: {candidate}")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(
                f"Stage 3 export Adapter tensor is non-finite: {candidate}:{name}"
            )
    return candidate, payload


def compare_stage3_adapter_exports(
    left: str | Path,
    right: str | Path,
    *,
    expected_step: int | None = None,
) -> dict[str, Any]:
    """Compare all export metadata and every Adapter tensor exactly."""

    left_path, left_payload = _load_export(left)
    right_path, right_payload = _load_export(right)
    if expected_step is not None and (
        isinstance(expected_step, bool)
        or not isinstance(expected_step, int)
        or expected_step < 0
    ):
        raise ValueError("expected_step must be a non-negative integer")

    left_metadata = {key: value for key, value in left_payload.items() if key != "adapter"}
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
    if expected_step is not None and left_payload["global_step"] != expected_step:
        raise ValueError(
            "Stage 3 export global_step differs from expected_step: "
            f"actual={left_payload['global_step']} expected={expected_step}"
        )

    left_adapter = left_payload["adapter"]
    right_adapter = right_payload["adapter"]
    if set(left_adapter) != set(right_adapter):
        raise ValueError("Stage 3 Adapter tensor names differ")
    parameter_count = 0
    for name in sorted(left_adapter):
        left_tensor = left_adapter[name]
        right_tensor = right_adapter[name]
        if left_tensor.shape != right_tensor.shape or left_tensor.dtype != right_tensor.dtype:
            raise ValueError(f"Stage 3 Adapter tensor contract differs: {name}")
        if not torch.equal(left_tensor, right_tensor):
            raise ValueError(f"Stage 3 Adapter tensor values differ: {name}")
        parameter_count += left_tensor.numel()

    return {
        "status": "ok",
        "global_step": left_payload["global_step"],
        "training_contract_sha256": left_payload["training_contract_sha256"],
        "data_manifest_sha256": left_payload["data_manifest_sha256"],
        "tensor_count": len(left_adapter),
        "parameter_count": parameter_count,
        "left": {
            "path": str(left_path),
            "size_bytes": left_path.stat().st_size,
            "sha256": _sha256_file(left_path),
        },
        "right": {
            "path": str(right_path),
            "size_bytes": right_path.stat().st_size,
            "sha256": _sha256_file(right_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare uninterrupted and strict-resume Stage 3 exports."
    )
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--expected-step", type=int, default=None)
    args = parser.parse_args()
    receipt = compare_stage3_adapter_exports(
        args.left,
        args.right,
        expected_step=args.expected_step,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
