from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fastwam.alignment.checkpointing import canonical_json_sha256, sha256_file
from fastwam.alignment.data_identity import canonical_data_manifest_sha256
from fastwam.gating.artifacts import (
    build_label_artifact_context,
    build_label_contract,
    build_label_row_from_context,
    publish_label_chunk_atomic_from_context,
)
from fastwam.gating.contracts import build_episode_split
from fastwam.gating.label_job import (
    enumerate_label_samples,
    plan_label_chunks,
)
from scripts import generate_gate_labels as generate_cli
from scripts import merge_gate_labels as merge_cli
from scripts import train_video_gate as train_cli


GIT_IDENTITY = {
    "commit": "e" * 40,
    "tracked_dirty": False,
    "untracked_source_files": [],
}

class _SourceSnapshotDouble:
    def __init__(self, *, fail_content_call=None):
        self.fail_content_call = fail_content_call
        self.stat_calls = 0
        self.content_calls = 0

    def check_stats(self):
        self.stat_calls += 1

    def check_content(self):
        self.content_calls += 1
        if self.content_calls == self.fail_content_call:
            raise RuntimeError("synthetic selected-source drift")


def _patch_source_snapshot(monkeypatch, module, snapshot=None):
    snapshot = snapshot or _SourceSnapshotDouble()
    monkeypatch.setattr(
        module,
        "capture_selected_source_snapshot",
        lambda _manifest: snapshot,
    )
    return snapshot


def _data_manifest() -> dict:
    manifest = {
        "schema_version": 1,
        "kind": "stage3_libero_data_manifest",
        "sampling": {},
        "num_frames": 7,
        "dataset_roots": [
            {
                "dataset_index": 0,
                "root": "/data/a",
                "selected_episodes": [2, 5],
                "num_frames": 4,
                "episode_boundaries": [
                    {"episode_index": 2, "from": 0, "to": 2, "length": 2},
                    {"episode_index": 5, "from": 2, "to": 4, "length": 2},
                ],
                "video_keys": [],
                "files": [],
            },
            {
                "dataset_index": 1,
                "root": "/data/b",
                "selected_episodes": [1, 4],
                "num_frames": 3,
                "episode_boundaries": [
                    {"episode_index": 1, "from": 0, "to": 1, "length": 1},
                    {"episode_index": 4, "from": 1, "to": 3, "length": 2},
                ],
                "video_keys": [],
                "files": [],
            },
        ],
        "text_embedding_cache": {},
        "normalization_stats": {},
        "decoder": {},
    }
    manifest["manifest_sha256"] = canonical_data_manifest_sha256(manifest)
    return manifest


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _merge_job(tmp_path: Path) -> dict:
    data_manifest = _data_manifest()
    episode_split = build_episode_split(
        data_manifest,
        validation_fraction=0.5,
        split_seed=9,
    )
    contract = build_label_contract(
        data_manifest=data_manifest,
        episode_split=episode_split,
        base_checkpoint_sha256="a" * 64,
        adapter_checkpoint_sha256="b" * 64,
        normalization_stats_sha256="c" * 64,
        data_config_sha256="d" * 64,
        git_identity=GIT_IDENTITY,
        base_seed=42,
        num_seed_pairs=2,
        relative_margin=0.05,
        num_shards=1,
        chunk_size=2,
        vae_sha256="9" * 64,
        label_runtime_config_sha256="8" * 64,
    )
    context = build_label_artifact_context(
        contract=contract,
        data_manifest=data_manifest,
        episode_split=episode_split,
    )
    rows_by_id = {}
    for index, sample in enumerate(enumerate_label_samples(context)):
        e10 = 0.8 if index % 2 == 0 else 1.0
        rows_by_id[sample.sample_id] = build_label_row_from_context(
            context=context,
            identity=sample.identity,
            e0=1.0,
            e10=e10,
            relative_gain=1.0 - e10,
            label=e10 < 0.95,
            sample_weight=1.0,
            num_video_frames=5,
        )

    job_dir = tmp_path / "job"
    plans = plan_label_chunks(
        context=context,
        output_dir=job_dir,
        chunk_size=2,
    )
    for plan in plans:
        published = publish_label_chunk_atomic_from_context(
            plan.path,
            context=context,
            shard_index=plan.shard_index,
            chunk_index=plan.chunk_index,
            planned_sample_ids=plan.planned_sample_ids,
            rows=[rows_by_id[sample_id] for sample_id in plan.planned_sample_ids],
        )
        assert published

    manifest_path = tmp_path / "data_manifest.json"
    split_path = tmp_path / "episode_split.json"
    contract_path = tmp_path / "label_contract.json"
    _write_json(manifest_path, data_manifest)
    _write_json(split_path, episode_split)
    _write_json(contract_path, contract)
    output_dir = tmp_path / "merged"
    config = {
        "label_job_dir": str(job_dir),
        "data_manifest": {
            "path": str(manifest_path),
            "expected_sha256": data_manifest["manifest_sha256"],
        },
        "episode_split": {
            "path": str(split_path),
            "expected_assignment_sha256": episode_split["assignment_sha256"],
        },
        "label_contract": {
            "path": str(contract_path),
            "expected_sha256": contract["contract_sha256"],
        },
        "output": {
            "directory": str(output_dir),
            "rows_file": "labels.jsonl",
            "manifest_file": "manifest.json",
            "expected_manifest_sha256": "",
        },
        "runtime": {"repo_dir": str(tmp_path), "require_clean_git": True},
    }
    return {
        "config": config,
        "plans": plans,
        "output_dir": output_dir,
        "contract": contract,
    }


def _allow_contract_git(monkeypatch) -> None:
    monkeypatch.setattr(
        merge_cli,
        "_validated_git_identity",
        lambda _runtime: deepcopy(GIT_IDENTITY),
    )
    _patch_source_snapshot(monkeypatch, merge_cli)


def test_merge_cli_validates_exact_plans_and_recovers_rows_only(
    tmp_path,
    monkeypatch,
):
    job = _merge_job(tmp_path)
    _allow_contract_git(monkeypatch)

    first = merge_cli.run_merge_gate_labels(job["config"])
    manifest_path = job["output_dir"] / "manifest.json"
    expected_manifest_bytes = manifest_path.read_bytes()
    manifest_path.unlink()

    resumed = merge_cli.run_merge_gate_labels(job["config"])
    assert resumed == first
    assert manifest_path.read_bytes() == expected_manifest_bytes


def test_merge_source_drift_before_publish_leaves_no_artifact(
    tmp_path,
    monkeypatch,
):
    job = _merge_job(tmp_path)
    _allow_contract_git(monkeypatch)
    snapshot = _SourceSnapshotDouble(fail_content_call=2)
    _patch_source_snapshot(monkeypatch, merge_cli, snapshot)

    with pytest.raises(RuntimeError, match="selected-source drift"):
        merge_cli.run_merge_gate_labels(job["config"])

    assert not (job["output_dir"] / "labels.jsonl").exists()
    assert not (job["output_dir"] / "manifest.json").exists()

def test_merge_cli_rejects_swapped_chunk_files(tmp_path, monkeypatch):
    job = _merge_job(tmp_path)
    _allow_contract_git(monkeypatch)
    first, second = job["plans"][:2]
    first_bytes = first.path.read_bytes()
    second_bytes = second.path.read_bytes()
    first.path.write_bytes(second_bytes)
    second.path.write_bytes(first_bytes)

    with pytest.raises(ValueError, match="external chunk plan"):
        merge_cli.run_merge_gate_labels(job["config"])


def test_merge_cli_rejects_forged_chunk_coordinates(tmp_path, monkeypatch):
    job = _merge_job(tmp_path)
    _allow_contract_git(monkeypatch)
    path = job["plans"][0].path
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["chunk_index"] += 100
    unhashed = dict(payload)
    unhashed.pop("chunk_sha256")
    payload["chunk_sha256"] = canonical_json_sha256(unhashed)
    _write_json(path, payload)

    with pytest.raises(ValueError, match="coordinates"):
        merge_cli.run_merge_gate_labels(job["config"])


def test_merge_cli_rejects_git_identity_drift(tmp_path, monkeypatch):
    job = _merge_job(tmp_path)
    drifted = deepcopy(GIT_IDENTITY)
    drifted["commit"] = "f" * 40
    monkeypatch.setattr(
        merge_cli,
        "_validated_git_identity",
        lambda _runtime: drifted,
    )

    with pytest.raises(RuntimeError, match="Git identity"):
        merge_cli.run_merge_gate_labels(job["config"])


def test_generate_rank_partition_and_immutable_json(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "3")
    monkeypatch.setenv("RANK", "1")
    assert generate_cli._rank_shard_indices(None, num_shards=8) == (1, 4, 7)
    with pytest.raises(ValueError, match="cannot be combined with torchrun"):
        generate_cli._rank_shard_indices([0, 5], num_shards=8)

    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    assert generate_cli._rank_shard_indices([0, 5], num_shards=8) == (0, 5)

    path = tmp_path / "identity.json"
    generate_cli._write_immutable_json(path, {"value": 1}, label="test identity")
    generate_cli._write_immutable_json(path, {"value": 1}, label="test identity")
    with pytest.raises(RuntimeError, match="different artifact"):
        generate_cli._write_immutable_json(
            path,
            {"value": 2},
            label="test identity",
        )


def test_generate_eight_rank_partition_covers_every_formal_shard(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")
    rank_assignments = []

    for rank in range(8):
        monkeypatch.setenv("RANK", str(rank))
        expected = tuple(range(rank, 64, 8))
        actual = generate_cli._rank_shard_indices(None, num_shards=64)
        assert actual == expected
        rank_assignments.append(set(actual))

    assert set().union(*rank_assignments) == set(range(64))
    for left_rank, left_assignment in enumerate(rank_assignments):
        for right_assignment in rank_assignments[left_rank + 1 :]:
            assert left_assignment.isdisjoint(right_assignment)


def test_generate_torchrun_device_rejects_explicit_cuda_index(
    monkeypatch,
):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(generate_cli.torch.cuda, "is_available", lambda: True)

    with pytest.raises(ValueError, match="runtime.device=cuda"):
        generate_cli._device({"device": "cuda:0", "require_cuda": True})


def test_generate_rechecks_vae_after_the_model_load_window(tmp_path):
    vae_path = tmp_path / "vae.safetensors"
    vae_path.write_bytes(b"verified VAE bytes")
    verified_sha256 = sha256_file(vae_path)
    vae_path.write_bytes(b"replaced VAE bytes")

    with pytest.raises(RuntimeError, match="changed while.*model was loading"):
        generate_cli._require_asset_unchanged(
            vae_path,
            expected_sha256=verified_sha256,
            label="the contract-bound VAE",
        )


def test_generate_snapshots_sources_before_dataset_construction(monkeypatch):
    events = []

    class Snapshot:
        def check_stats(self):
            events.append("stat")

    snapshot = Snapshot()
    dataset = object()
    monkeypatch.setattr(
        generate_cli,
        "capture_selected_source_snapshot",
        lambda _manifest: events.append("capture") or snapshot,
    )
    monkeypatch.setattr(
        generate_cli,
        "instantiate",
        lambda _config: events.append("instantiate") or dataset,
    )

    actual_dataset, actual_snapshot = (
        generate_cli._instantiate_label_dataset_under_source_guard(
            {"_target_": "tests.fake.Dataset"},
            {"manifest_sha256": "a" * 64},
        )
    )

    assert events == ["capture", "instantiate", "stat"]
    assert actual_dataset is dataset
    assert actual_snapshot is snapshot


def test_registered_vae_loader_rejects_incomplete_state(monkeypatch):
    from fastwam.models.wan22.helpers import loader as loader_module

    class TinyVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

    vae_entry = next(
        entry
        for entry in loader_module.WAN22_MODEL_REGISTRY
        if entry["model_name"] == "wan_video_vae"
    )
    assert vae_entry["strict_load"] is True
    monkeypatch.setattr(
        loader_module,
        "WAN22_MODEL_REGISTRY",
        [
            {
                **vae_entry,
                "model_hash": "tiny",
                "model_class": TinyVAE,
                "state_dict_converter": None,
            }
        ],
    )
    monkeypatch.setattr(loader_module, "hash_model_file", lambda _path: "tiny")
    monkeypatch.setattr(
        loader_module,
        "load_state_dict",
        lambda *_args, **_kwargs: {},
    )

    with pytest.raises(RuntimeError, match="Missing key"):
        loader_module._load_registered_model(
            "unused",
            "wan_video_vae",
            torch.float32,
            "cpu",
        )


def test_generate_runtime_contract_normalizes_precision_and_binds_model():
    model = {"_target_": "tests.fake.Model", "infer_shift": 5.0}
    runtime_environment = {
        "versions": {
            "torch": "2.8.0",
            "torchcodec": "0.7.0",
            "torchvision": "0.23.0",
            "datasets": "4.0.0",
            "pyarrow": "21.0.0",
            "av": "15.0.0",
        },
        "device": {
            "type": "cuda",
            "cuda_version": "12.8",
            "cudnn_version": 91002,
            "capability": [9, 0],
            "name": "Fake H100",
        },
    }
    first = generate_cli.build_label_runtime_config(
        model=model,
        mixed_precision="bfloat16",
        runtime_environment=runtime_environment,
    )
    alias = generate_cli.build_label_runtime_config(
        model=model,
        mixed_precision="bf16",
        runtime_environment=runtime_environment,
    )
    changed = generate_cli.build_label_runtime_config(
        model={**model, "infer_shift": 7.0},
        mixed_precision="bf16",
        runtime_environment=runtime_environment,
    )

    assert first == {
        "schema_version": 1,
        "kind": "stage2_label_runtime_config",
        "model": model,
        "mixed_precision": "bf16",
        "numerical_runtime": runtime_environment,
        "required_environment": {},
    }
    assert first == alias
    assert canonical_json_sha256(first) != canonical_json_sha256(changed)
    required = generate_cli.build_label_runtime_config(
        model=model,
        mixed_precision="bf16",
        runtime_environment=runtime_environment,
        required_environment={"CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
    )
    assert canonical_json_sha256(first) != canonical_json_sha256(required)


def test_generate_runtime_environment_is_injectable_without_cuda():
    ffmpeg_runtime = {
        "executable_version": "ffmpeg version 7.1 deterministic-test",
        "torchcodec_runtime": {
            "ffmpeg_version": "7.1",
            "libraries": {
                "libavcodec": [61, 3, 100],
                "libavformat": [61, 1, 100],
                "libavutil": [59, 8, 100],
            },
        },
    }

    fake_device = SimpleNamespace(type="cuda")
    fake_torch = SimpleNamespace(
        __version__="2.8.0+cu128",
        device=lambda _value: fake_device,
        version=SimpleNamespace(cuda="12.8"),
        backends=SimpleNamespace(
            cudnn=SimpleNamespace(version=lambda: 91002),
        ),
        cuda=SimpleNamespace(
            get_device_capability=lambda _device: (9, 0),
            get_device_name=lambda _device: "Fake H100",
        ),
    )

    environment = generate_cli.collect_label_runtime_environment(
        "cuda:7",
        package_version_resolver=lambda name: f"{name}-version",
        torch_runtime=fake_torch,
        ffmpeg_runtime_resolver=lambda: ffmpeg_runtime,
        nvidia_driver_version_resolver=lambda: "580.173.02",
    )

    assert environment["versions"] == {
        "torch": "2.8.0+cu128",
        "torchcodec": "torchcodec-version",
        "torchvision": "torchvision-version",
        "datasets": "datasets-version",
        "pyarrow": "pyarrow-version",
        "av": "av-version",
        "numpy": "numpy-version",
        "accelerate": "accelerate-version",
        "lerobot": "lerobot-version",
    }
    assert environment["ffmpeg"] == ffmpeg_runtime
    assert environment["device"] == {
        "type": "cuda",
        "cuda_version": "12.8",
        "cudnn_version": 91002,
        "nvidia_driver_version": "580.173.02",
        "capability": [9, 0],
        "name": "Fake H100",
    }
    assert environment["backend"] == {
        "deterministic_algorithms": False,
        "deterministic_warn_only": False,
        "cudnn_benchmark": None,
        "cudnn_deterministic": None,
        "cudnn_allow_tf32": None,
        "cuda_matmul_allow_tf32": None,
    }


def _training_semantics() -> tuple[dict, dict, dict, dict]:
    data = {"train": {"dataset_dirs": ["/data/a"]}, "val": None}
    gate = {
        "proprio_dim": 8,
        "context_dim": 16,
        "cnn_channels": [4, 8, 12],
        "context_feature_dim": 8,
        "proprio_hidden_dim": 8,
        "proprio_feature_dim": 4,
        "fusion_hidden_dim": 8,
    }
    training = {
        "seed": 42,
        "batch_size": 4,
        "num_workers": 0,
        "pin_memory": False,
        "shuffle": True,
        "learning_rate": 1.0e-4,
        "weight_decay": 1.0e-4,
        "max_grad_norm": 1.0,
        "num_epochs": 3,
        "early_stop_patience": 2,
        "min_delta": 0.0,
        "threshold": 0.5,
        "num_calibration_bins": 10,
    }
    runtime = {
        "repo_dir": "/repo/a",
        "require_clean_git": True,
        "device": "cpu",
        "require_cuda": False,
        "deterministic_algorithms": True,
    }
    return data, gate, training, runtime


def test_gate_training_contract_excludes_locators_and_binds_semantics():
    data, gate, training, runtime = _training_semantics()
    numerical_runtime = {
        "versions": {"torch": "2.8.0"},
        "device": {"type": "cpu"},
        "backend": {"deterministic_algorithms": True},
    }
    first = train_cli.build_training_config_contract(
        data=data,
        gate=gate,
        training=training,
        runtime=runtime,
        numerical_runtime=numerical_runtime,
    )
    changed_locator = dict(runtime, repo_dir="/different/repo")
    second = train_cli.build_training_config_contract(
        data=data,
        gate=gate,
        training=training,
        runtime=changed_locator,
        numerical_runtime=numerical_runtime,
    )
    assert first == second
    assert "repo_dir" not in first["runtime"]

    changed_training = dict(training, num_epochs=4)
    third = train_cli.build_training_config_contract(
        data=data,
        gate=gate,
        training=changed_training,
        runtime=runtime,
        numerical_runtime=numerical_runtime,
    )
    assert canonical_json_sha256(first) != canonical_json_sha256(third)

    changed_runtime = {
        **numerical_runtime,
        "versions": {"torch": "2.9.0"},
    }
    fourth = train_cli.build_training_config_contract(
        data=data,
        gate=gate,
        training=training,
        runtime=runtime,
        numerical_runtime=changed_runtime,
    )
    assert canonical_json_sha256(first) != canonical_json_sha256(fourth)


class _IndexDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 24

    def __getitem__(self, index):
        return index


def _loader_order(loader) -> list[int]:
    return [int(value) for batch in loader for value in batch]


def test_gate_epoch_loader_order_is_resume_stable():
    _, _, training, _ = _training_semantics()
    dataset = _IndexDataset()
    first, _ = train_cli._epoch_loaders(
        train_dataset=dataset,
        val_dataset=dataset,
        training=training,
        epoch_index=2,
    )
    resumed, _ = train_cli._epoch_loaders(
        train_dataset=dataset,
        val_dataset=dataset,
        training=training,
        epoch_index=2,
    )
    next_epoch, _ = train_cli._epoch_loaders(
        train_dataset=dataset,
        val_dataset=dataset,
        training=training,
        epoch_index=3,
    )
    assert _loader_order(first) == _loader_order(resumed)
    assert _loader_order(resumed) != _loader_order(next_epoch)


def test_gate_output_names_must_be_pairwise_distinct(tmp_path):
    with pytest.raises(ValueError, match="pairwise distinct"):
        train_cli._validate_output_paths(
            {
                "run_identity_file": tmp_path / "identity.json",
                "state_file": tmp_path / "state.pt",
                "best_file": tmp_path / "same.pt",
                "last_file": tmp_path / "same.pt",
                "summary_file": tmp_path / "summary.json",
            }
        )


def test_gate_training_rejects_distributed_launch(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")

    with pytest.raises(RuntimeError, match="single-process only"):
        train_cli._require_single_process_environment()


def test_gate_output_directory_has_one_writer(tmp_path):
    output_dir = tmp_path / "gate-run"

    with train_cli._exclusive_output_writer(output_dir) as lock_path:
        assert lock_path.is_file()
        with pytest.raises(RuntimeError, match="another Gate writer"):
            with train_cli._exclusive_output_writer(output_dir):
                pytest.fail("a second writer unexpectedly acquired the lock")

    with train_cli._exclusive_output_writer(output_dir):
        pass


def test_gate_history_trims_summary_ahead_of_resumed_state(tmp_path):
    summary_path = tmp_path / "summary.json"
    training_identity = {"label_manifest_sha256": "f" * 64}
    _write_json(
        summary_path,
        {
            "schema_version": 1,
            "kind": "stage2_binary_video_gate_training_summary",
            "training_identity": training_identity,
            "history_complete": True,
            "epoch_history": [{"epoch": 1}, {"epoch": 2}],
        },
    )

    history, complete = train_cli._load_prior_epoch_history(
        summary_path,
        training_identity=training_identity,
        resumed_epoch=1,
    )

    assert history == [{"epoch": 1}]
    assert complete


def test_train_cli_runs_small_gate_only_and_saves_every_epoch(
    tmp_path,
    monkeypatch,
):
    stats_path = tmp_path / "stats.json"
    stats_path.write_text("{}\n", encoding="utf-8")
    stats_sha = sha256_file(stats_path)
    raw_data = {
        "train": {
            "_target_": "tests.fake.StrictDataset",
            "dataset_dirs": ["raw-data"],
            "pretrained_norm_stats": str(stats_path),
        },
        "val": None,
    }
    canonical_data = train_cli._canonicalize_data_paths(
        raw_data,
        repo_dir=tmp_path,
    )
    data_sha = "d" * 64
    split_sha = "e" * 64
    contract_sha = "c" * 64
    label_sha = "f" * 64
    data_manifest = {"manifest_sha256": data_sha}
    episode_split = {"assignment_sha256": split_sha}
    contract = {
        "contract_sha256": contract_sha,
        "base_checkpoint_sha256": "a" * 64,
        "adapter_checkpoint_sha256": "b" * 64,
        "normalization_stats_sha256": stats_sha,
        "data_manifest_sha256": data_sha,
        "episode_assignment_sha256": split_sha,
        "data_config_sha256": canonical_json_sha256(canonical_data),
    }
    manifest_path = tmp_path / "data.json"
    split_path = tmp_path / "split.json"
    contract_path = tmp_path / "contract.json"
    _write_json(manifest_path, data_manifest)
    _write_json(split_path, episode_split)
    _write_json(contract_path, contract)

    rows = (
        {"split": "train", "label": False},
        {"split": "train", "label": True},
        {"split": "validation", "label": False},
        {"split": "validation", "label": True},
    )
    merged = SimpleNamespace(
        manifest={"manifest_sha256": label_sha},
        rows=rows,
    )
    monkeypatch.setattr(
        train_cli,
        "_validated_git_identity",
        lambda _runtime: deepcopy(GIT_IDENTITY),
    )
    monkeypatch.setattr(
        train_cli,
        "load_validated_merged_label_artifact",
        lambda *_args, **_kwargs: merged,
    )
    raw_dataset = SimpleNamespace(
        current_only=lambda: SimpleNamespace(kind="current-only")
    )
    monkeypatch.setattr(train_cli, "instantiate", lambda _config: raw_dataset)
    monkeypatch.setattr(
        train_cli,
        "_validate_formal_label_dataset",
        lambda *_args, **_kwargs: {},
    )
    _patch_source_snapshot(monkeypatch, train_cli)

    class TinyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, index):
            return index

    monkeypatch.setattr(
        train_cli,
        "Stage2GateDataset",
        lambda *_args, **_kwargs: TinyDataset(),
    )

    class TinyGate(torch.nn.Module):
        def __init__(self, **_kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

    monkeypatch.setattr(train_cli, "BinaryVideoGate", TinyGate)

    class FakeTrainer:
        instances = []

        def __init__(self, _gate, *, training_identity, **_kwargs):
            self.training_identity = training_identity
            self.epoch = 0
            self.global_step = 0
            self.best_epoch = -1
            self.best_val_bce = float("inf")
            self.best_metrics = {}
            self.epochs_without_improvement = 0
            self.saved_epochs = []
            self.exports = []
            self.__class__.instances.append(self)

        def load_training_state(self, path):
            saved = Path(path).read_text(encoding="utf-8")
            self.epoch = int(saved.removeprefix("epoch="))
            self.global_step = self.epoch * 2
            self.best_epoch = self.epoch
            self.best_val_bce = 1.0 / max(self.epoch, 1)
            self.best_metrics = {"bce": self.best_val_bce}

        def fit(self, _train, _val, **_kwargs):
            self.epoch += 1
            self.global_step += 2
            self.best_epoch = self.epoch
            self.best_val_bce = 1.0 / self.epoch
            self.best_metrics = {"bce": self.best_val_bce}
            return SimpleNamespace(
                epochs=({"epoch": self.epoch},),
                stopped_early=False,
            )

        def save_training_state(self, path):
            Path(path).write_bytes(f"epoch={self.epoch}".encode())
            self.saved_epochs.append(self.epoch)

        def export_checkpoint(self, path, *, selection, **identity):
            Path(path).write_bytes(selection.encode())
            self.exports.append((selection, identity))

    monkeypatch.setattr(train_cli, "GateTrainer", FakeTrainer)
    output_dir = tmp_path / "gate-run"
    config = {
        "output_dir": str(output_dir),
        "data_manifest": {
            "path": str(manifest_path),
            "expected_sha256": data_sha,
        },
        "episode_split": {
            "path": str(split_path),
            "expected_assignment_sha256": split_sha,
        },
        "label_contract": {
            "path": str(contract_path),
            "expected_sha256": contract_sha,
        },
        "label_manifest": {"path": "unused.json", "expected_sha256": label_sha},
        "source_identities": {
            "base_checkpoint_sha256": "a" * 64,
            "adapter_checkpoint_sha256": "b" * 64,
        },
        "assets": {
            "normalization_stats": {
                "path": str(stats_path),
                "expected_sha256": stats_sha,
            }
        },
        "data": raw_data,
        "gate": _training_semantics()[1],
        "training": dict(_training_semantics()[2], num_epochs=2),
        "checkpoint": {
            "strict_resume": True,
            "resume": None,
            "run_identity_file": "run_identity.json",
            "state_file": "state.pt",
            "best_file": "best.pt",
            "last_file": "last.pt",
            "summary_file": "summary.json",
        },
        "runtime": {
            "repo_dir": str(tmp_path),
            "require_clean_git": False,
            "device": "cpu",
            "require_cuda": False,
            "deterministic_algorithms": True,
        },
    }

    summary = train_cli.run_train_video_gate(config)
    trainer = FakeTrainer.instances[-1]
    assert summary["final_epoch"] == 2
    assert trainer.saved_epochs == [0, 1, 2, 2]
    assert summary["history_complete"]
    assert [record["epoch"] for record in summary["epoch_history"]] == [1, 2]
    assert [selection for selection, _ in trainer.exports] == ["best", "last"]
    assert all(
        identity["label_manifest_sha256"] == label_sha
        and identity["adapter_checkpoint_sha256"] == "b" * 64
        for _, identity in trainer.exports
    )
    assert trainer.training_identity["base_checkpoint_sha256"] == "a" * 64
    run_identity = json.loads(
        (output_dir / "run_identity.json").read_text(encoding="utf-8")
    )
    assert run_identity["training_config_sha256"] == canonical_json_sha256(
        run_identity["training_config"]
    )

    external_state = tmp_path / "external-state.pt"
    external_state.write_text("epoch=2", encoding="utf-8")

    drift_config = deepcopy(config)
    drift_output = tmp_path / "drifted-gate-run"
    drift_config["output_dir"] = str(drift_output)
    failing_snapshot = _SourceSnapshotDouble(fail_content_call=1)
    _patch_source_snapshot(monkeypatch, train_cli, failing_snapshot)
    with pytest.raises(RuntimeError, match="selected-source drift"):
        train_cli.run_train_video_gate(drift_config)
    drifted_trainer = FakeTrainer.instances[-1]
    assert drifted_trainer.exports == []
    assert not (drift_output / "best.pt").exists()
    assert not (drift_output / "last.pt").exists()
    external_config = deepcopy(config)
    external_config["checkpoint"]["resume"] = str(external_state)
    with pytest.raises(RuntimeError, match="external Gate resume"):
        train_cli.run_train_video_gate(external_config)

    resume_config = deepcopy(config)
    resume_config["checkpoint"]["resume"] = str(output_dir / "state.pt")
    resumed_summary = train_cli.run_train_video_gate(resume_config)
    resumed_trainer = FakeTrainer.instances[-1]
    assert resumed_trainer.saved_epochs == [2, 2]
    assert resumed_summary["new_epoch_history"] == []
    assert resumed_summary["history_complete"]
    assert [
        record["epoch"] for record in resumed_summary["epoch_history"]
    ] == [1, 2]


def test_stage2_configs_have_no_machine_specific_paths():
    repo = Path(__file__).resolve().parents[1]
    for name in (
        "generate_gate_labels.yaml",
        "merge_gate_labels.yaml",
        "train_video_gate.yaml",
    ):
        text = (repo / "configs" / name).read_text(encoding="utf-8")
        assert "/root/" not in text
        assert "hydra:" in text
    for name in ("generate_gate_labels.yaml", "train_video_gate.yaml"):
        text = (repo / "configs" / name).read_text(encoding="utf-8")
        assert "save_stats_copy: false" in text
