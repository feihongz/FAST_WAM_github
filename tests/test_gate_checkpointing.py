import torch
import pytest

from fastwam.gating.checkpointing import (
    load_gate_checkpoint,
    save_gate_checkpoint,
)
from fastwam.models.video_gate import BinaryVideoGate


IDENTITIES = {
    "label_manifest_sha256": "a" * 64,
    "adapter_checkpoint_sha256": "b" * 64,
    "base_checkpoint_sha256": "f" * 64,
    "data_manifest_sha256": "c" * 64,
    "episode_split_assignment_sha256": "d" * 64,
    "training_config_sha256": "e" * 64,
}
GIT_IDENTITY = {
    "commit": "1" * 40,
    "tracked_dirty": False,
    "untracked_source_files": [],
}


def _save(tmp_path, gate=None):
    gate = BinaryVideoGate(proprio_dim=8) if gate is None else gate
    path = tmp_path / "gate.pt"
    save_gate_checkpoint(
        path,
        gate,
        **IDENTITIES,
        git_identity=GIT_IDENTITY,
        global_step=9,
        epoch=2,
        best_metrics={"bce": 0.25, "auroc": 0.75},
    )
    return path, gate


def _load(path, **overrides):
    expected = {
        "expected_label_manifest_sha256": IDENTITIES["label_manifest_sha256"],
        "expected_adapter_checkpoint_sha256": IDENTITIES[
            "adapter_checkpoint_sha256"
        ],
        "expected_base_checkpoint_sha256": IDENTITIES["base_checkpoint_sha256"],
        "expected_data_manifest_sha256": IDENTITIES["data_manifest_sha256"],
        "expected_episode_split_assignment_sha256": IDENTITIES[
            "episode_split_assignment_sha256"
        ],
        "expected_training_config_sha256": IDENTITIES["training_config_sha256"],
        "expected_git_identity": GIT_IDENTITY,
    }
    expected.update(overrides)
    return load_gate_checkpoint(path, **expected)


def _inputs():
    return {
        "input_image": torch.randn(2, 3, 16, 16),
        "context": torch.randn(2, 3, 4096),
        "context_mask": torch.ones(2, 3, dtype=torch.bool),
        "proprio": torch.randn(2, 8),
    }


def test_gate_checkpoint_roundtrip_restores_exact_logits_and_identity(tmp_path):
    torch.manual_seed(31)
    path, original = _save(tmp_path)
    inputs = _inputs()
    original.eval()
    with torch.no_grad():
        expected = original(**inputs)

    restored, payload = _load(path)
    with torch.no_grad():
        actual = restored(**inputs)

    assert torch.equal(actual, expected)
    assert restored.config() == original.config()
    assert not restored.training
    assert not any(parameter.requires_grad for parameter in restored.parameters())
    assert payload["adapter_checkpoint_sha256"] == "b" * 64


@pytest.mark.parametrize(
    ("argument", "message"),
    [
        ("expected_label_manifest_sha256", "label_manifest_sha256 mismatch"),
        ("expected_adapter_checkpoint_sha256", "adapter_checkpoint_sha256 mismatch"),
        ("expected_base_checkpoint_sha256", "base_checkpoint_sha256 mismatch"),
        ("expected_data_manifest_sha256", "data_manifest_sha256 mismatch"),
        ("expected_training_config_sha256", "training_config_sha256 mismatch"),
        (
            "expected_episode_split_assignment_sha256",
            "episode_split_assignment_sha256 mismatch",
        ),
    ],
)
def test_gate_checkpoint_rejects_identity_mismatch(tmp_path, argument, message):
    path, _ = _save(tmp_path)
    with pytest.raises(ValueError, match=message):
        _load(path, **{argument: "9" * 64})


def test_gate_checkpoint_rejects_git_identity_mismatch(tmp_path):
    path, _ = _save(tmp_path)
    mismatched = {**GIT_IDENTITY, "commit": "2" * 40}
    with pytest.raises(ValueError, match="git_identity mismatch"):
        _load(path, expected_git_identity=mismatched)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("schema", "unsupported"),
        ("missing", "fields"),
        ("count", "parameter count"),
        ("state", "state_dict"),
        ("non_finite", "non-finite"),
    ],
)
def test_gate_checkpoint_rejects_tampered_payload(tmp_path, mutation, message):
    path, _ = _save(tmp_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if mutation == "schema":
        payload["schema_version"] = 99
    elif mutation == "missing":
        del payload["data_manifest_sha256"]
    elif mutation == "count":
        payload["parameter_count"] += 1
    elif mutation == "state":
        del payload["gate_state_dict"][next(iter(payload["gate_state_dict"]))]
    elif mutation == "non_finite":
        first = next(iter(payload["gate_state_dict"].values()))
        first.reshape(-1)[0] = torch.nan
    torch.save(payload, path)

    with pytest.raises((RuntimeError, ValueError), match=message):
        _load(path)


def test_gate_checkpoint_refuses_non_finite_weights_on_save(tmp_path):
    gate = BinaryVideoGate(proprio_dim=8)
    next(gate.parameters()).data.reshape(-1)[0] = torch.inf

    with pytest.raises(ValueError, match="non-finite"):
        _save(tmp_path, gate)
