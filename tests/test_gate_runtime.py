import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from fastwam.alignment.checkpointing import sha256_file
from fastwam.gating import runtime
from fastwam.models.wan22.fastwam_unified_aligned import FastWAMUnifiedAligned


DATA_SHA = "d" * 64


class TinyAdapter(nn.Module):
    def __init__(self, *, width: int = 3):
        super().__init__()
        self.projection = nn.Linear(width, width)
        self.width = width

    def config(self):
        return {"width": self.width, "drop_first_video_frame": True}


class TinyAligned(FastWAMUnifiedAligned):
    def __init__(self):
        nn.Module.__init__(self)
        self.backbone = nn.Linear(3, 3)
        self.alignment_adapter = TinyAdapter()
        self.load_events = []

    def load_frozen_base_checkpoint(self, path):
        self.load_events.append(("base", str(path)))
        self.train()
        self.requires_grad_(True)
        return {"step": 7, "torch_dtype": "torch.float32"}


def _write_artifacts(tmp_path, *, metadata_updates=None):
    base_path = tmp_path / "base.pt"
    base_path.write_bytes(b"tiny frozen base")
    base_sha = sha256_file(base_path)
    model = TinyAligned()
    payload = {
        "schema_version": 2,
        "kind": "stage3_alignment_export",
        "adapter": model.alignment_adapter.state_dict(),
        "base_checkpoint": "original/base.pt",
        "base_checkpoint_sha256": base_sha,
        "data_manifest_sha256": DATA_SHA,
        "alignment_config": model.alignment_adapter.config(),
        "global_step": 19,
        "git_commit": "abc123",
        "training_contract_sha256": "c" * 64,
        "asset_identities": {"text_encoder": {"sha256": "e" * 64}},
    }
    payload.update(metadata_updates or {})
    export_path = tmp_path / "adapter.pt"
    torch.save(payload, export_path)
    return model, base_path, base_sha, export_path, sha256_file(export_path)


def test_inspect_alignment_export_is_read_only_and_serializable(tmp_path):
    model, _, base_sha, export_path, export_sha = _write_artifacts(tmp_path)
    before = {
        name: value.detach().clone()
        for name, value in model.alignment_adapter.state_dict().items()
    }

    identity = runtime.inspect_alignment_export(
        export_path,
        expected_sha256=export_sha,
        expected_base_checkpoint_sha256=base_sha,
        expected_data_manifest_sha256=DATA_SHA,
    )

    json.dumps(identity)
    assert identity["path"] == str(export_path.resolve())
    assert identity["sha256"] == export_sha
    assert identity["size_bytes"] == export_path.stat().st_size
    assert identity["export_metadata"]["alignment_config"] == {
        "drop_first_video_frame": True,
        "width": 3,
    }
    assert identity["export_metadata"]["global_step"] == 19
    assert model.load_events == []
    for name, value in before.items():
        assert torch.equal(value, model.alignment_adapter.state_dict()[name])


def test_label_model_loads_base_then_adapter_and_finishes_frozen(
    tmp_path, monkeypatch
):
    model, base_path, base_sha, export_path, export_sha = _write_artifacts(tmp_path)
    original_loader = runtime.load_alignment_checkpoint

    def recording_loader(path, adapter, **kwargs):
        assert model.load_events == [("base", str(base_path.resolve()))]
        model.load_events.append(("adapter", str(Path(path))))
        return original_loader(path, adapter, **kwargs)

    monkeypatch.setattr(runtime, "load_alignment_checkpoint", recording_loader)
    identity = runtime.load_stage2_label_model(
        model,
        base_checkpoint_path=base_path,
        expected_base_checkpoint_sha256=base_sha,
        alignment_export_path=export_path,
        expected_alignment_export_sha256=export_sha,
        expected_data_manifest_sha256=DATA_SHA,
    )

    assert [event[0] for event in model.load_events] == ["base", "adapter"]
    assert not model.training
    assert all(not module.training for module in model.modules())
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert identity["kind"] == "stage2_label_model_identity"
    assert identity["base_checkpoint"] == {
        "path": str(base_path.resolve()),
        "sha256": base_sha,
        "size_bytes": base_path.stat().st_size,
    }
    assert identity["alignment_export"]["sha256"] == export_sha
    assert identity["data_manifest_sha256"] == DATA_SHA
    json.dumps(identity)


@pytest.mark.parametrize(
    ("metadata_updates", "expected_data_sha", "message"),
    [
        ({"base_checkpoint_sha256": "b" * 64}, DATA_SHA, "base checkpoint"),
        ({"data_manifest_sha256": "e" * 64}, DATA_SHA, "data manifest"),
        ({"alignment_config": {"width": 9}}, DATA_SHA, "config"),
    ],
)
def test_label_model_rejects_export_binding_before_mutating_model(
    tmp_path, metadata_updates, expected_data_sha, message
):
    model, base_path, base_sha, export_path, export_sha = _write_artifacts(
        tmp_path,
        metadata_updates=metadata_updates,
    )

    with pytest.raises(ValueError, match=message):
        runtime.load_stage2_label_model(
            model,
            base_checkpoint_path=base_path,
            expected_base_checkpoint_sha256=base_sha,
            alignment_export_path=export_path,
            expected_alignment_export_sha256=export_sha,
            expected_data_manifest_sha256=expected_data_sha,
        )

    assert model.load_events == []


def test_label_model_rejects_file_hashes_before_loading(tmp_path):
    model, base_path, base_sha, export_path, export_sha = _write_artifacts(tmp_path)
    with pytest.raises(ValueError, match="base checkpoint SHA256 mismatch"):
        runtime.load_stage2_label_model(
            model,
            base_checkpoint_path=base_path,
            expected_base_checkpoint_sha256="0" * 64,
            alignment_export_path=export_path,
            expected_alignment_export_sha256=export_sha,
            expected_data_manifest_sha256=DATA_SHA,
        )
    assert model.load_events == []

    with pytest.raises(ValueError, match="alignment export SHA256 mismatch"):
        runtime.load_stage2_label_model(
            model,
            base_checkpoint_path=base_path,
            expected_base_checkpoint_sha256=base_sha,
            alignment_export_path=export_path,
            expected_alignment_export_sha256="0" * 64,
            expected_data_manifest_sha256=DATA_SHA,
        )
    assert model.load_events == []


def test_label_model_rejects_non_aligned_model(tmp_path):
    _, base_path, base_sha, export_path, export_sha = _write_artifacts(tmp_path)
    with pytest.raises(TypeError, match="FastWAMUnifiedAligned"):
        runtime.load_stage2_label_model(
            nn.Linear(2, 2),
            base_checkpoint_path=base_path,
            expected_base_checkpoint_sha256=base_sha,
            alignment_export_path=export_path,
            expected_alignment_export_sha256=export_sha,
            expected_data_manifest_sha256=DATA_SHA,
        )


def test_adapter_load_failure_never_returns_success_identity(tmp_path, monkeypatch):
    model, base_path, base_sha, export_path, export_sha = _write_artifacts(tmp_path)

    def fail_adapter_load(*args, **kwargs):
        del args, kwargs
        model.load_events.append(("adapter_failed", ""))
        raise RuntimeError("strict Adapter load failed")

    monkeypatch.setattr(runtime, "load_alignment_checkpoint", fail_adapter_load)
    with pytest.raises(RuntimeError, match="strict Adapter load failed"):
        runtime.load_stage2_label_model(
            model,
            base_checkpoint_path=base_path,
            expected_base_checkpoint_sha256=base_sha,
            alignment_export_path=export_path,
            expected_alignment_export_sha256=export_sha,
            expected_data_manifest_sha256=DATA_SHA,
        )
    assert [event[0] for event in model.load_events] == ["base", "adapter_failed"]


def _formal_dataset():
    class FormalDataset:
        def __init__(self, base):
            self.strict_data_mode = True
            self.skip_padding_as_possible = False
            self.lerobot_dataset = base

        def __len__(self):
            return 23

    parts = [
        SimpleNamespace(
            root=Path("dataset/a"),
            video_backend="torchcodec",
            allow_video_backend_fallback=False,
        ),
        SimpleNamespace(
            root=Path("dataset/b"),
            video_backend="torchcodec",
            allow_video_backend_fallback=False,
        ),
    ]
    base = SimpleNamespace(
        strict_data_mode=True,
        processor=SimpleNamespace(normalizer=object()),
        multi_dataset=SimpleNamespace(_datasets=parts, num_episodes=7),
    )
    return FormalDataset(base)


def test_stage2_label_dataset_is_strict_normalized_and_manifest_bound(
    tmp_path,
    monkeypatch,
):
    dataset = _formal_dataset()
    manifest = {"manifest_sha256": DATA_SHA}
    stats_path = tmp_path / "stats.json"
    stats_path.write_text("{}", encoding="utf-8")
    calls = []

    def validate(candidate, payload, **kwargs):
        calls.append((candidate, payload, kwargs))
        return dict(payload)

    monkeypatch.setattr(runtime, "validate_data_manifest", validate)
    identity = runtime.validate_stage2_label_dataset(
        dataset,
        manifest,
        normalization_stats_path=stats_path,
        expected_data_manifest_sha256=DATA_SHA,
    )

    assert identity["kind"] == "stage2_label_data_identity"
    assert identity["dataset_length"] == 23
    assert identity["dataset_episodes"] == 7
    assert identity["content_verified"] is True
    assert identity["normalized_action_space"] is True
    assert calls[0][2]["full_content_verify"] is True
    assert calls[0][2]["normalization_stats_path"] == stats_path.resolve()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("outer_strict", "strict_data_mode"),
        ("padding_retry", "skip_padding_as_possible"),
        ("base_strict", "propagate strict"),
        ("normalizer", "normalizer"),
        ("backend", "video_backend=torchcodec"),
        ("fallback", "forbid"),
    ],
)
def test_stage2_label_dataset_rejects_nonformal_runtime(
    tmp_path,
    monkeypatch,
    mutation,
    message,
):
    dataset = _formal_dataset()
    if mutation == "outer_strict":
        dataset.strict_data_mode = False
    elif mutation == "padding_retry":
        dataset.skip_padding_as_possible = True
    elif mutation == "base_strict":
        dataset.lerobot_dataset.strict_data_mode = False
    elif mutation == "normalizer":
        dataset.lerobot_dataset.processor = None
    elif mutation == "backend":
        dataset.lerobot_dataset.multi_dataset._datasets[0].video_backend = "pyav"
    elif mutation == "fallback":
        dataset.lerobot_dataset.multi_dataset._datasets[
            0
        ].allow_video_backend_fallback = True
    monkeypatch.setattr(
        runtime,
        "validate_data_manifest",
        lambda *args, **kwargs: {"manifest_sha256": DATA_SHA},
    )

    with pytest.raises(RuntimeError, match=message):
        runtime.validate_stage2_label_dataset(
            dataset,
            {"manifest_sha256": DATA_SHA},
            normalization_stats_path=tmp_path / "stats.json",
            expected_data_manifest_sha256=DATA_SHA,
        )
