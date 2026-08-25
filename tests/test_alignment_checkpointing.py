import json
import random
import subprocess

import numpy as np
import pytest
import torch

from fastwam.alignment.checkpointing import (
    TRAINING_STATE_KIND,
    TRAINING_STATE_SCHEMA_VERSION,
    canonical_json_sha256,
    hash_state_tree,
    read_git_identity,
    resolve_base_checkpoint,
    sha256_file,
    validate_rng_state_files,
    validate_training_state,
    write_json_atomic,
)


def test_base_checkpoint_requires_exact_sha256(tmp_path):
    checkpoint = tmp_path / "base.pt"
    checkpoint.write_bytes(b"frozen-base")
    expected = sha256_file(checkpoint)

    identity = resolve_base_checkpoint(
        checkpoint,
        expected_sha256=expected,
    )

    assert identity.sha256 == expected
    assert identity.size_bytes == len(b"frozen-base")
    with pytest.raises(ValueError, match="mismatch"):
        resolve_base_checkpoint(checkpoint, expected_sha256="0" * 64)


def test_canonical_contract_hash_ignores_mapping_order():
    assert canonical_json_sha256({"a": 1, "b": [2, 3]}) == canonical_json_sha256(
        {"b": [2, 3], "a": 1}
    )


def _write_state(tmp_path):
    state = tmp_path / "step_000001"
    accelerator = state / "accelerator"
    accelerator.mkdir(parents=True)
    torch.save(
        {
            "step": 1,
            "random_state": random.getstate(),
            "numpy_random_seed": np.random.get_state(),
            "torch_manual_seed": torch.get_rng_state(),
        },
        accelerator / "random_states_0.pkl",
    )
    (accelerator / "model.safetensors").write_bytes(b"adapter-only")
    contract = {
        "base_checkpoint_sha256": "a" * 64,
        "training_contract_sha256": "b" * 64,
        "world_size": 1,
    }
    torch.save(
        {
            "schema_version": 1,
            "kind": "stage3_alignment_export",
            "global_step": 1,
            "base_checkpoint_sha256": contract["base_checkpoint_sha256"],
            "training_contract_sha256": contract[
                "training_contract_sha256"
            ],
            "adapter": {},
        },
        state / "adapter_export.pt",
    )
    manifest = {
        "schema_version": TRAINING_STATE_SCHEMA_VERSION,
        "kind": TRAINING_STATE_KIND,
        "complete": True,
        "global_step": 1,
        "epoch": 0,
        "batch_in_epoch": 1,
        "micro_batches_per_epoch": 4,
        "micro_step_in_accumulation": 0,
        "scheduler_last_epoch": 1,
        "device_type": "cpu",
        **contract,
    }
    manifest["files"] = hash_state_tree(state)
    write_json_atomic(state / "manifest.json", manifest)
    write_json_atomic(
        state / "COMPLETE",
        {
            "schema_version": TRAINING_STATE_SCHEMA_VERSION,
            "kind": TRAINING_STATE_KIND,
            "manifest_sha256": sha256_file(state / "manifest.json"),
        },
    )
    return state, contract


def test_training_state_validates_contract_rng_and_file_hashes(tmp_path):
    state, contract = _write_state(tmp_path)

    manifest = validate_training_state(state, expected_contract=contract)

    assert manifest["global_step"] == 1
    with pytest.raises(ValueError, match="contract mismatch"):
        validate_training_state(
            state,
            expected_contract={**contract, "world_size": 2},
        )

    (state / "accelerator" / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="inventory"):
        validate_training_state(state, expected_contract=contract)


def test_training_state_rejects_missing_complete_marker(tmp_path):
    state, contract = _write_state(tmp_path)
    (state / "COMPLETE").unlink()

    with pytest.raises(ValueError, match="incomplete"):
        validate_training_state(state, expected_contract=contract)


def test_training_state_rejects_manifest_tampering(tmp_path):
    state, contract = _write_state(tmp_path)
    manifest = json.loads((state / "manifest.json").read_text(encoding="utf-8"))
    manifest["global_step"] = 9
    write_json_atomic(state / "manifest.json", manifest)

    with pytest.raises(ValueError, match="manifest SHA256"):
        validate_training_state(state, expected_contract=contract)


def test_training_state_rejects_unloadable_rng(tmp_path):
    state, contract = _write_state(tmp_path)
    rng_path = state / "accelerator" / "random_states_0.pkl"
    rng = torch.load(rng_path, map_location="cpu", weights_only=False)
    rng["random_state"] = ("not", "a", "python-state")
    torch.save(rng, rng_path)

    manifest = json.loads((state / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"] = hash_state_tree(state)
    write_json_atomic(state / "manifest.json", manifest)
    write_json_atomic(
        state / "COMPLETE",
        {
            "schema_version": TRAINING_STATE_SCHEMA_VERSION,
            "kind": TRAINING_STATE_KIND,
            "manifest_sha256": sha256_file(state / "manifest.json"),
        },
    )

    with pytest.raises(ValueError, match="not loadable"):
        validate_training_state(state, expected_contract=contract)


@pytest.mark.parametrize("batch_in_epoch", [4, 5])
def test_training_state_rejects_batch_cursor_outside_epoch(
    tmp_path,
    batch_in_epoch,
):
    state, contract = _write_state(tmp_path)
    manifest_path = state / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["batch_in_epoch"] = batch_in_epoch
    write_json_atomic(manifest_path, manifest)
    write_json_atomic(
        state / "COMPLETE",
        {
            "schema_version": TRAINING_STATE_SCHEMA_VERSION,
            "kind": TRAINING_STATE_KIND,
            "manifest_sha256": sha256_file(manifest_path),
        },
    )

    with pytest.raises(ValueError, match="batch cursor is outside its epoch"):
        validate_training_state(state, expected_contract=contract)


@pytest.mark.parametrize("gradient_accumulation_steps", [0, -1])
def test_rng_validation_rejects_nonpositive_accumulation(
    tmp_path,
    gradient_accumulation_steps,
):
    with pytest.raises(ValueError, match="gradient_accumulation_steps.*positive"):
        validate_rng_state_files(
            tmp_path,
            world_size=1,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )


def test_git_identity_reports_only_untracked_source_contract_paths(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "tracked.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Stage3 Test",
            "-c",
            "user.email=stage3@example.invalid",
            "commit",
            "-qm",
            "initial",
        ],
        check=True,
    )
    expected = []
    for directory, filename in (
        ("src", "new_module.py"),
        ("configs", "new_config.yaml"),
        ("scripts", "new_launcher.py"),
        ("tests", "test_new_contract.py"),
    ):
        path = repo / directory / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("untracked\n", encoding="utf-8")
        expected.append(f"{directory}/{filename}")
    artifact = repo / "artifacts" / "local-output.bin"
    artifact.parent.mkdir()
    artifact.write_bytes(b"not source")

    identity = read_git_identity(repo)

    assert len(identity.commit) == 40
    assert identity.tracked_dirty is False
    assert identity.untracked_source_files == tuple(sorted(expected))
    assert "artifacts/local-output.bin" not in identity.untracked_source_files
