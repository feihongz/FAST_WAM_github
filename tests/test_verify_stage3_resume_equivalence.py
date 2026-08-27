from __future__ import annotations

import json
import random
import shutil

import numpy as np
import pytest
import torch

from fastwam.alignment.checkpointing import (
    LEGACY_TRAINING_STATE_SCHEMA_VERSION,
    STRICT_RESUME_PROVENANCE_KIND,
    STRICT_RESUME_PROVENANCE_SCHEMA_VERSION,
    TRAINING_STATE_KIND,
    TRAINING_STATE_SCHEMA_VERSION,
    canonical_json_sha256,
    hash_state_tree,
    sha256_file,
    write_json_atomic,
)
from fastwam.models.wan22.video_action_alignment import (
    VideoActionResidualAdapter,
    save_alignment_checkpoint,
)
from scripts.verify_stage3_resume_equivalence import (
    verify_stage3_resume_equivalence,
)


BASE_SHA256 = "a" * 64
DATA_SHA256 = "b" * 64
GIT_COMMIT = "c" * 40
ASSET_SHA256 = "e" * 64
STATS_SHA256 = "f" * 64
TRAINING_CONTRACT = {"effective_deepspeed_config": None, "name": "formal-test"}
TRAINING_SHA256 = canonical_json_sha256(TRAINING_CONTRACT)


def _adapter() -> VideoActionResidualAdapter:
    torch.manual_seed(42)
    return VideoActionResidualAdapter(
        action_hidden_dim=4,
        video_hidden_dim=6,
        action_dim=2,
        bottleneck_dim=4,
        num_heads=2,
        ffn_multiplier=1,
    )


def _write_state(
    root,
    *,
    step: int,
    provenance,
):
    state = root / f"step_{step:06d}"
    accelerator = state / "accelerator"
    accelerator.mkdir(parents=True)
    torch.save(
        {
            "step": step,
            "random_state": random.getstate(),
            "numpy_random_seed": np.random.get_state(),
            "torch_manual_seed": torch.get_rng_state(),
        },
        accelerator / "random_states_0.pkl",
    )
    (accelerator / "model.safetensors").write_bytes(b"adapter-only")
    save_alignment_checkpoint(
        state / "adapter_export.pt",
        _adapter(),
        base_checkpoint="/formal/base.pt",
        base_checkpoint_sha256=BASE_SHA256,
        data_manifest_sha256=DATA_SHA256,
        global_step=step,
        git_commit=GIT_COMMIT,
        training_contract_sha256=TRAINING_SHA256,
        asset_identities={
            "normalization_stats": {
                "path": "/formal/stats.json",
                "sha256": STATS_SHA256,
                "size_bytes": 321,
            },
            "vae": {
                "path": "/formal/vae.pt",
                "sha256": ASSET_SHA256,
                "size_bytes": 123,
            }
        },
    )
    manifest = {
        "schema_version": TRAINING_STATE_SCHEMA_VERSION,
        "kind": TRAINING_STATE_KIND,
        "complete": True,
        "global_step": step,
        "epoch": 0,
        "batch_in_epoch": step,
        "micro_batches_per_epoch": 4,
        "micro_step_in_accumulation": 0,
        "scheduler_last_epoch": step,
        "base_checkpoint": "/formal/base.pt",
        "base_checkpoint_sha256": BASE_SHA256,
        "base_checkpoint_size_bytes": 456,
        "alignment_config": _adapter().config(),
        "training_contract": TRAINING_CONTRACT,
        "training_contract_sha256": TRAINING_SHA256,
        "git_commit": GIT_COMMIT,
        "world_size": 1,
        "distributed_type": "DistributedType.NO",
        "zero_stage": 0,
        "mixed_precision": "no",
        "device_type": "cpu",
        "gradient_accumulation_steps": 1,
        "batch_size_per_rank": 1,
        "dataset_length": 4,
        "drop_last": True,
        "deepspeed_config_sha256": None,
        "asset_sha256": {
            "normalization_stats": STATS_SHA256,
            "vae": ASSET_SHA256,
        },
        "data_manifest_sha256": DATA_SHA256,
        "git_tracked_dirty": False,
        "git_untracked_source_files": [],
        "versions": {"torch": torch.__version__},
        "dataloader_contract": {
            "split_batches": False,
            "even_batches": True,
            "use_stateful_dataloader": False,
        },
        "strict_resume_provenance": provenance,
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
    return state


def _provenance(source):
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    return {
        "schema_version": STRICT_RESUME_PROVENANCE_SCHEMA_VERSION,
        "kind": STRICT_RESUME_PROVENANCE_KIND,
        "source_manifest_sha256": sha256_file(source / "manifest.json"),
        "source_complete_sha256": sha256_file(source / "COMPLETE"),
        "source_global_step": manifest["global_step"],
        "source_epoch": manifest["epoch"],
        "source_batch_in_epoch": manifest["batch_in_epoch"],
        "source_scheduler_last_epoch": manifest["scheduler_last_epoch"],
        "source_training_contract_sha256": manifest[
            "training_contract_sha256"
        ],
        "source_world_size": manifest["world_size"],
        "source_zero_stage": manifest["zero_stage"],
    }


def _artifacts(tmp_path):
    source = _write_state(tmp_path / "baseline", step=1, provenance=None)
    uninterrupted = _write_state(
        tmp_path / "baseline",
        step=2,
        provenance=None,
    )
    resumed = _write_state(
        tmp_path / "replay",
        step=2,
        provenance=_provenance(source),
    )
    uninterrupted_export = tmp_path / "uninterrupted-step2.pt"
    resumed_export = tmp_path / "resumed-step2.pt"
    shutil.copy2(uninterrupted / "adapter_export.pt", uninterrupted_export)
    shutil.copy2(resumed / "adapter_export.pt", resumed_export)
    return source, uninterrupted, resumed, uninterrupted_export, resumed_export


def _verify(artifacts):
    source, uninterrupted, resumed, uninterrupted_export, resumed_export = artifacts
    return verify_stage3_resume_equivalence(
        uninterrupted,
        source,
        resumed,
        uninterrupted_export,
        resumed_export,
        expected_final_step=2,
        expected_resume_step=1,
        expected_world_size=1,
        expected_zero_stage=0,
        expected_batch_size_per_rank=1,
        expected_gradient_accumulation_steps=1,
        expected_base_checkpoint_sha256=BASE_SHA256,
        expected_data_manifest_sha256=DATA_SHA256,
        expected_training_contract_sha256=TRAINING_SHA256,
        expected_git_commit=GIT_COMMIT,
    )


def _rewrite_manifest(state, mutate):
    manifest_path = state / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(manifest)
    write_json_atomic(manifest_path, manifest)
    write_json_atomic(
        state / "COMPLETE",
        {
            "schema_version": manifest["schema_version"],
            "kind": TRAINING_STATE_KIND,
            "manifest_sha256": sha256_file(manifest_path),
        },
    )


def test_verify_stage3_resume_equivalence_accepts_complete_lineage(tmp_path):
    receipt = _verify(_artifacts(tmp_path))

    assert receipt["status"] == "ok"
    assert receipt["expected_final_step"] == 2
    assert receipt["states"]["resume_source"]["global_step"] == 1
    assert receipt["states"]["resumed"]["world_size"] == 1
    assert len(receipt["states"]["resumed"]["rng_files"]) == 1
    assert receipt["adapter_equivalence"]["tensor_count"] == 16


def test_verify_stage3_resume_equivalence_rejects_missing_provenance(tmp_path):
    artifacts = _artifacts(tmp_path)
    resumed = artifacts[2]
    _rewrite_manifest(
        resumed,
        lambda manifest: manifest.update(strict_resume_provenance=None),
    )

    with pytest.raises(ValueError, match="provenance does not match"):
        _verify(artifacts)


def test_verify_stage3_resume_equivalence_rejects_forged_source_hash(tmp_path):
    artifacts = _artifacts(tmp_path)
    resumed = artifacts[2]

    def forge(manifest):
        manifest["strict_resume_provenance"]["source_manifest_sha256"] = "f" * 64

    _rewrite_manifest(resumed, forge)
    with pytest.raises(ValueError, match="provenance does not match"):
        _verify(artifacts)


def test_verify_stage3_resume_equivalence_rejects_wrong_topology(tmp_path):
    artifacts = _artifacts(tmp_path)
    with pytest.raises(ValueError, match="contract mismatch for world_size"):
        source, uninterrupted, resumed, left, right = artifacts
        verify_stage3_resume_equivalence(
            uninterrupted,
            source,
            resumed,
            left,
            right,
            expected_final_step=2,
            expected_resume_step=1,
            expected_world_size=8,
            expected_zero_stage=2,
            expected_batch_size_per_rank=2,
            expected_gradient_accumulation_steps=3,
            expected_base_checkpoint_sha256=BASE_SHA256,
            expected_data_manifest_sha256=DATA_SHA256,
            expected_training_contract_sha256=TRAINING_SHA256,
            expected_git_commit=GIT_COMMIT,
        )


def test_verify_stage3_resume_equivalence_rejects_legacy_state(tmp_path):
    artifacts = _artifacts(tmp_path)
    source = artifacts[0]

    def make_legacy(manifest):
        manifest["schema_version"] = LEGACY_TRAINING_STATE_SCHEMA_VERSION
        del manifest["strict_resume_provenance"]

    _rewrite_manifest(source, make_legacy)
    with pytest.raises(ValueError, match="schema v2"):
        _verify(artifacts)


def test_verify_stage3_resume_equivalence_rejects_incomplete_state(tmp_path):
    artifacts = _artifacts(tmp_path)
    (artifacts[2] / "COMPLETE").unlink()

    with pytest.raises(ValueError, match="incomplete"):
        _verify(artifacts)


def test_verify_stage3_resume_equivalence_requires_byte_exact_external_copy(
    tmp_path,
):
    artifacts = _artifacts(tmp_path)
    external = artifacts[3]
    payload = torch.load(external, map_location="cpu", weights_only=True)
    torch.save(payload, external, _use_new_zipfile_serialization=False)
    assert sha256_file(external) != sha256_file(
        artifacts[1] / "adapter_export.pt"
    )

    with pytest.raises(ValueError, match="external export SHA256"):
        _verify(artifacts)
