import json
from pathlib import Path

import pytest
import torch
from torch import nn

from fastwam.alignment import eval_loading
from fastwam.alignment.checkpointing import sha256_file
from fastwam.models.wan22.fastwam_unified_aligned import FastWAMUnifiedAligned


DATA_MANIFEST_SHA256 = "d" * 64
TRAINING_CONTRACT_SHA256 = "c" * 64
GLOBAL_STEP = 200


class TinyAdapter(nn.Module):
    def __init__(self, *, width: int = 3):
        super().__init__()
        self.projection = nn.Linear(width, width)
        self.width = int(width)

    def config(self):
        return {
            "width": self.width,
            "drop_first_video_frame": True,
        }


class TinyAligned(FastWAMUnifiedAligned):
    def __init__(self, *, adapter_width: int = 3, vae_path: Path | None = None):
        nn.Module.__init__(self)
        self.backbone = nn.Linear(3, 3)
        self.alignment_adapter = TinyAdapter(width=adapter_width)
        self.load_events: list[tuple[str, str]] = []
        if vae_path is not None:
            self.model_paths = {"vae": str(vae_path)}

    def load_frozen_base_checkpoint(self, path):
        self.load_events.append(("base", str(path)))
        with torch.no_grad():
            self.backbone.weight.fill_(7.0)
            self.backbone.bias.fill_(5.0)
        # Exercise the public loader's final inference-only freeze rather than
        # letting this tiny stand-in accidentally satisfy it in advance.
        self.train()
        self.requires_grad_(True)
        return {"step": 234830, "torch_dtype": "torch.float32"}


def _parameter_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }


def _assert_parameters_unchanged(
    model: nn.Module,
    before: dict[str, torch.Tensor],
) -> None:
    assert model.load_events == []
    assert set(model.state_dict()) == set(before)
    for name, value in before.items():
        assert torch.equal(model.state_dict()[name], value), name


def _write_artifacts(tmp_path: Path) -> dict[str, object]:
    base_path = tmp_path / "base.pt"
    base_path.write_bytes(b"strict tiny frozen base")
    base_sha256 = sha256_file(base_path)

    asset_paths = {
        "normalization_stats": tmp_path / "dataset_stats.json",
        "vae": tmp_path / "vae.bin",
    }
    asset_paths["normalization_stats"].write_bytes(b'{"mean": 0, "std": 1}\n')
    asset_paths["vae"].write_bytes(b"strict tiny vae")
    asset_sha256 = {
        name: sha256_file(path)
        for name, path in asset_paths.items()
    }

    source_adapter = TinyAdapter()
    with torch.no_grad():
        source_adapter.projection.weight.fill_(0.25)
        source_adapter.projection.bias.fill_(-0.5)
    source_adapter_state = {
        name: value.detach().clone()
        for name, value in source_adapter.state_dict().items()
    }
    export_path = tmp_path / "adapter.pt"
    torch.save(
        {
            "schema_version": 2,
            "kind": "stage3_alignment_export",
            "adapter": source_adapter_state,
            "base_checkpoint": "original/base.pt",
            "base_checkpoint_sha256": base_sha256,
            "data_manifest_sha256": DATA_MANIFEST_SHA256,
            "alignment_config": source_adapter.config(),
            "global_step": GLOBAL_STEP,
            "git_commit": "a" * 40,
            "training_contract_sha256": TRAINING_CONTRACT_SHA256,
            "asset_identities": {
                name: {
                    "path": f"original/{path.name}",
                    "sha256": asset_sha256[name],
                    "size_bytes": path.stat().st_size,
                }
                for name, path in asset_paths.items()
            },
        },
        export_path,
    )
    return {
        "base_path": base_path,
        "base_sha256": base_sha256,
        "export_path": export_path,
        "export_sha256": sha256_file(export_path),
        "asset_paths": asset_paths,
        "asset_sha256": asset_sha256,
        "source_adapter_state": source_adapter_state,
    }


def _load_kwargs(artifacts: dict[str, object]) -> dict[str, object]:
    return {
        "base_checkpoint_path": artifacts["base_path"],
        "expected_base_checkpoint_sha256": artifacts["base_sha256"],
        "alignment_export_path": artifacts["export_path"],
        "expected_alignment_export_sha256": artifacts["export_sha256"],
        "expected_data_manifest_sha256": DATA_MANIFEST_SHA256,
        "expected_training_contract_sha256": TRAINING_CONTRACT_SHA256,
        "expected_global_step": GLOBAL_STEP,
        "asset_paths": artifacts["asset_paths"],
        "expected_asset_sha256": artifacts["asset_sha256"],
    }


def test_eval_loader_binds_full_contract_loads_base_then_adapter_and_freezes(
    tmp_path,
):
    artifacts = _write_artifacts(tmp_path)
    model = TinyAligned(vae_path=artifacts["asset_paths"]["vae"])
    before_adapter = {
        name: value.detach().clone()
        for name, value in model.alignment_adapter.state_dict().items()
    }

    artifact_identity = eval_loading.inspect_aligned_model_artifacts(
        **_load_kwargs(artifacts)
    )
    assert model.load_events == []
    json.dumps(artifact_identity)
    assert artifact_identity["kind"] == "stage3_aligned_artifact_identity"
    assert artifact_identity["runtime_assets"].keys() == {
        "normalization_stats",
        "vae",
    }
    metadata = artifact_identity["alignment_export"]["export_metadata"]
    assert metadata["training_contract_sha256"] == TRAINING_CONTRACT_SHA256
    assert metadata["global_step"] == GLOBAL_STEP

    original_loader = eval_loading.load_alignment_checkpoint

    def recording_adapter_loader(path, adapter, **kwargs):
        assert model.load_events == [
            ("base", str(Path(artifacts["base_path"]).resolve()))
        ]
        model.load_events.append(("adapter", str(Path(path))))
        return original_loader(path, adapter, **kwargs)

    identity = eval_loading.load_prepared_aligned_model(
        model,
        artifact_identity,
        _alignment_loader=recording_adapter_loader,
    )

    assert [event[0] for event in model.load_events] == ["base", "adapter"]
    assert not model.training
    assert all(not module.training for module in model.modules())
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert identity["kind"] == "stage3_aligned_model_identity"
    assert identity["base_checkpoint"]["sha256"] == artifacts["base_sha256"]
    assert identity["alignment_export"]["sha256"] == artifacts["export_sha256"]
    assert identity["data_manifest_sha256"] == DATA_MANIFEST_SHA256
    assert identity["runtime_assets"] == artifact_identity["runtime_assets"]
    json.dumps(identity)

    source_state = artifacts["source_adapter_state"]
    assert any(
        not torch.equal(before_adapter[name], source_state[name])
        for name in source_state
    )
    for name, value in source_state.items():
        assert torch.equal(model.alignment_adapter.state_dict()[name], value)


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("base_sha256", "base checkpoint SHA256 mismatch"),
        ("export_sha256", "alignment export SHA256 mismatch"),
        ("data_manifest_sha256", "data manifest hash mismatch"),
        ("training_contract_sha256", "training contract hash mismatch"),
        ("global_step", "global step mismatch"),
        ("asset_sha256", "alignment export vae SHA256 mismatch"),
    ],
)
def test_eval_loader_rejects_contract_errors_before_model_mutation(
    tmp_path,
    failure,
    message,
):
    artifacts = _write_artifacts(tmp_path)
    model = TinyAligned()
    before = _parameter_snapshot(model)
    kwargs = _load_kwargs(artifacts)
    if failure == "base_sha256":
        kwargs["expected_base_checkpoint_sha256"] = "0" * 64
    elif failure == "export_sha256":
        kwargs["expected_alignment_export_sha256"] = "0" * 64
    elif failure == "data_manifest_sha256":
        kwargs["expected_data_manifest_sha256"] = "0" * 64
    elif failure == "training_contract_sha256":
        kwargs["expected_training_contract_sha256"] = "0" * 64
    elif failure == "global_step":
        kwargs["expected_global_step"] = GLOBAL_STEP + 1
    elif failure == "asset_sha256":
        expected_assets = dict(kwargs["expected_asset_sha256"])
        expected_assets["vae"] = "0" * 64
        kwargs["expected_asset_sha256"] = expected_assets
    else:  # pragma: no cover - keeps the parameter table fail closed.
        raise AssertionError(f"unknown failure fixture: {failure}")

    with pytest.raises(ValueError, match=message):
        eval_loading.load_frozen_aligned_model(model, **kwargs)

    _assert_parameters_unchanged(model, before)


def test_eval_loader_rejects_export_asset_binding_before_model_mutation(tmp_path):
    artifacts = _write_artifacts(tmp_path)
    export_path = Path(artifacts["export_path"])
    payload = torch.load(export_path, map_location="cpu", weights_only=False)
    payload["asset_identities"]["normalization_stats"]["sha256"] = "0" * 64
    torch.save(payload, export_path)
    artifacts["export_sha256"] = sha256_file(export_path)
    model = TinyAligned()
    before = _parameter_snapshot(model)

    with pytest.raises(
        ValueError,
        match="alignment export normalization_stats SHA256 mismatch",
    ):
        eval_loading.load_frozen_aligned_model(model, **_load_kwargs(artifacts))

    _assert_parameters_unchanged(model, before)


def test_eval_loader_rejects_runtime_asset_drift_before_model_mutation(tmp_path):
    artifacts = _write_artifacts(tmp_path)
    vae_path = Path(artifacts["asset_paths"]["vae"])
    vae_path.write_bytes(b"tampered tiny vae")
    model = TinyAligned()
    before = _parameter_snapshot(model)

    with pytest.raises(ValueError, match="vae SHA256 mismatch"):
        eval_loading.load_frozen_aligned_model(model, **_load_kwargs(artifacts))

    _assert_parameters_unchanged(model, before)


def test_eval_loader_rejects_adapter_config_before_base_mutation(tmp_path):
    artifacts = _write_artifacts(tmp_path)
    model = TinyAligned(adapter_width=4)
    before = _parameter_snapshot(model)

    with pytest.raises(ValueError, match="config does not match"):
        eval_loading.load_frozen_aligned_model(model, **_load_kwargs(artifacts))

    _assert_parameters_unchanged(model, before)


def test_eval_loader_rejects_model_that_loaded_a_different_vae(tmp_path):
    artifacts = _write_artifacts(tmp_path)
    model = TinyAligned(vae_path=tmp_path / "different-vae.bin")
    before = _parameter_snapshot(model)

    with pytest.raises(RuntimeError, match="contract-bound VAE"):
        eval_loading.load_frozen_aligned_model(model, **_load_kwargs(artifacts))

    _assert_parameters_unchanged(model, before)


def test_eval_loader_rechecks_stats_after_consumer_read(tmp_path):
    artifacts = _write_artifacts(tmp_path)
    artifact_identity = eval_loading.inspect_aligned_model_artifacts(
        **_load_kwargs(artifacts)
    )
    stats_path = Path(artifacts["asset_paths"]["normalization_stats"])
    stats_path.write_bytes(b'{"mean": 999, "std": 1}\n')

    with pytest.raises(
        RuntimeError,
        match="runtime asset normalization_stats changed",
    ):
        eval_loading.verify_aligned_runtime_asset(
            artifact_identity,
            "normalization_stats",
            phase="while normalization stats were being loaded",
        )
