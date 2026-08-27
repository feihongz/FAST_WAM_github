from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from scripts.compare_stage3_adapter_exports import (
    compare_stage3_adapter_exports,
)


def _payload(*, value: float = 1.0, step: int = 2) -> dict:
    return {
        "schema_version": 2,
        "kind": "stage3_alignment_export",
        "base_checkpoint": "/formal/base.pt",
        "base_checkpoint_sha256": "a" * 64,
        "data_manifest_sha256": "b" * 64,
        "alignment_config": {"bottleneck_dim": 256},
        "global_step": step,
        "git_commit": "c" * 40,
        "training_contract_sha256": "d" * 64,
        "asset_identities": {"vae": {"sha256": "e" * 64}},
        "adapter": {
            "projection.weight": torch.full((2, 3), value),
            "projection.bias": torch.tensor([value, value + 1]),
        },
    }


def _write(path, payload) -> None:
    torch.save(payload, path)


def test_compare_stage3_adapter_exports_accepts_exact_equivalence(tmp_path):
    left = tmp_path / "left.pt"
    right = tmp_path / "right.pt"
    _write(left, _payload())
    _write(right, _payload())

    receipt = compare_stage3_adapter_exports(left, right, expected_step=2)

    assert receipt["status"] == "ok"
    assert receipt["global_step"] == 2
    assert receipt["tensor_count"] == 2
    assert receipt["parameter_count"] == 8
    assert len(receipt["left"]["sha256"]) == 64
    assert len(receipt["right"]["sha256"]) == 64


def test_compare_stage3_adapter_exports_rejects_metadata_drift(tmp_path):
    left = tmp_path / "left.pt"
    right = tmp_path / "right.pt"
    _write(left, _payload())
    drifted = _payload()
    drifted["training_contract_sha256"] = "f" * 64
    _write(right, drifted)

    with pytest.raises(ValueError, match="metadata differs"):
        compare_stage3_adapter_exports(left, right)


def test_compare_stage3_adapter_exports_rejects_tensor_drift(tmp_path):
    left = tmp_path / "left.pt"
    right = tmp_path / "right.pt"
    _write(left, _payload())
    _write(right, _payload(value=2.0))

    with pytest.raises(ValueError, match="tensor values differ"):
        compare_stage3_adapter_exports(left, right)


def test_compare_stage3_adapter_exports_rejects_wrong_step_and_schema(tmp_path):
    left = tmp_path / "left.pt"
    right = tmp_path / "right.pt"
    payload = _payload()
    _write(left, payload)
    _write(right, payload)
    with pytest.raises(ValueError, match="expected_step"):
        compare_stage3_adapter_exports(left, right, expected_step=3)

    invalid = deepcopy(payload)
    invalid["unexpected"] = True
    _write(right, invalid)
    with pytest.raises(ValueError, match="schema is invalid"):
        compare_stage3_adapter_exports(left, right)
