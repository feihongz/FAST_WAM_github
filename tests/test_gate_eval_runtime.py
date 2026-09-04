import json

import pytest
import torch

from fastwam.alignment.checkpointing import sha256_file
from fastwam.alignment.data_identity import canonical_data_manifest_sha256
from fastwam.alignment.text_cache_index import prompt_sha256
from fastwam.gating.checkpointing import save_gate_checkpoint
from fastwam.gating.eval_runtime import (
    ManifestBoundPromptContextProvider,
    load_gate_for_evaluation,
)
from fastwam.models.video_gate import BinaryVideoGate


PROMPT_TEMPLATE = "Instruction: {task}"
PROMPT = PROMPT_TEMPLATE.format(task="put the mug on the plate")
IDENTITIES = {
    "label_manifest_sha256": "a" * 64,
    "adapter_checkpoint_sha256": "b" * 64,
    "base_checkpoint_sha256": "c" * 64,
    "data_manifest_sha256": "d" * 64,
    "episode_split_assignment_sha256": "e" * 64,
    "training_config_sha256": "f" * 64,
}
GIT_IDENTITY = {
    "commit": "1" * 40,
    "tracked_dirty": False,
    "untracked_source_files": [],
}


def _write_v1_prompt_artifact(tmp_path):
    cache_root = tmp_path / "text-cache"
    cache_root.mkdir()
    cache_path = cache_root / "prompt.pt"
    context = torch.arange(20, dtype=torch.float32).reshape(5, 4).to(torch.bfloat16)
    mask = torch.tensor([True, True, False, True, False])
    torch.save({"context": context, "mask": mask}, cache_path)

    manifest = {
        "schema_version": 1,
        "kind": "stage3_libero_data_manifest",
        "sampling": {},
        "num_frames": 0,
        "dataset_roots": [],
        "text_embedding_cache": {
            "root": str(cache_root.resolve()),
            "prompt_template": PROMPT_TEMPLATE,
            "context_len": context.shape[0],
            "files": [
                {
                    "prompt_sha256": prompt_sha256(PROMPT),
                    "relative_path": cache_path.name,
                    "role": "text_embedding",
                    "sha256": sha256_file(cache_path),
                    "size_bytes": cache_path.stat().st_size,
                }
            ],
        },
        "normalization_stats": {},
        "decoder": {},
    }
    manifest["manifest_sha256"] = canonical_data_manifest_sha256(manifest)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, manifest, cache_path, context, mask


def _write_gate_checkpoint(tmp_path):
    gate = BinaryVideoGate(proprio_dim=8)
    path = tmp_path / "gate.pt"
    save_gate_checkpoint(
        path,
        gate,
        **IDENTITIES,
        git_identity=GIT_IDENTITY,
        global_step=19,
        epoch=3,
        best_metrics={"bce": 0.25, "auroc": 0.75},
    )
    return path, gate


def _load_gate(path, **overrides):
    expected = {
        "expected_checkpoint_sha256": sha256_file(path),
        **{f"expected_{key}": value for key, value in IDENTITIES.items()},
        "expected_git_identity": GIT_IDENTITY,
        "device": "cpu",
    }
    expected.update(overrides)
    return load_gate_for_evaluation(path, **expected)


def test_v1_prompt_context_preserves_gate_mask_and_zeros_padding(tmp_path):
    manifest_path, manifest, _, source_context, source_mask = (
        _write_v1_prompt_artifact(tmp_path)
    )

    with ManifestBoundPromptContextProvider(
        manifest_path,
        expected_manifest_sha256=manifest["manifest_sha256"],
        expected_prompt_template=PROMPT_TEMPLATE,
    ) as provider:
        bound = provider.load(PROMPT, device="cpu")

    expected_context = source_context.clone()
    expected_context[~source_mask] = 0
    assert torch.equal(bound.context, expected_context.unsqueeze(0))
    assert torch.equal(bound.gate_context_mask, source_mask.unsqueeze(0))
    assert torch.equal(
        bound.model_context_mask,
        torch.ones_like(source_mask).unsqueeze(0),
    )
    assert not torch.equal(bound.gate_context_mask, bound.model_context_mask)
    assert torch.count_nonzero(bound.context[0, ~source_mask]).item() == 0
    assert bound.identity["manifest_sha256"] == manifest["manifest_sha256"]
    assert bound.identity["prompt_sha256"] == prompt_sha256(PROMPT)
    assert bound.identity["valid_token_count"] == int(source_mask.sum().item())


def test_v1_prompt_context_rejects_payload_sha_drift(tmp_path):
    manifest_path, manifest, cache_path, _, _ = _write_v1_prompt_artifact(tmp_path)
    provider = ManifestBoundPromptContextProvider(
        manifest_path,
        expected_manifest_sha256=manifest["manifest_sha256"],
        expected_prompt_template=PROMPT_TEMPLATE,
    )
    payload = bytearray(cache_path.read_bytes())
    payload[len(payload) // 2] ^= 1
    cache_path.write_bytes(payload)

    with pytest.raises(ValueError, match="payload SHA256 mismatch"):
        provider.load(PROMPT, device="cpu")


def test_load_gate_for_evaluation_verifies_bytes_and_returns_frozen_eval_module(
    tmp_path,
):
    path, original = _write_gate_checkpoint(tmp_path)

    loaded = _load_gate(path)

    assert loaded.identity == {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "schema_version": 2,
        "kind": "stage2_binary_video_gate_export",
        "parameter_count": original.parameter_count(),
        "global_step": 19,
        "epoch": 3,
        "best_metrics": {"bce": 0.25, "auroc": 0.75},
        **IDENTITIES,
        "git_identity": GIT_IDENTITY,
    }
    assert not loaded.gate.training
    assert all(not module.training for module in loaded.gate.modules())
    assert all(not parameter.requires_grad for parameter in loaded.gate.parameters())
    for name, value in original.state_dict().items():
        assert torch.equal(loaded.gate.state_dict()[name], value)


def test_load_gate_for_evaluation_rejects_checkpoint_sha_mismatch(tmp_path):
    path, _ = _write_gate_checkpoint(tmp_path)

    with pytest.raises(ValueError, match="Gate checkpoint SHA256 mismatch"):
        _load_gate(path, expected_checkpoint_sha256="0" * 64)


@pytest.mark.parametrize(
    "field",
    [
        "label_manifest_sha256",
        "adapter_checkpoint_sha256",
        "base_checkpoint_sha256",
        "data_manifest_sha256",
        "episode_split_assignment_sha256",
        "training_config_sha256",
    ],
)
def test_load_gate_for_evaluation_rejects_each_identity_mismatch(tmp_path, field):
    path, _ = _write_gate_checkpoint(tmp_path)

    with pytest.raises(ValueError, match=rf"{field} mismatch"):
        _load_gate(path, **{f"expected_{field}": "9" * 64})
