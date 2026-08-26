import json
from pathlib import Path
from types import SimpleNamespace

from accelerate import Accelerator
from hydra import compose, initialize_config_dir
import pytest
import torch

from fastwam.alignment.runtime import (
    _activate_accelerator_cuda_device,
    _all_rank_value,
    _canonicalize_data_paths,
    _main_rank_value,
    _resolve_asset_identities,
    _resolve_data_identity,
    _resolved_config,
    _run_all_rank_phase,
    _validate_required_environment,
    _validate_formal_dataset,
)
from fastwam.alignment.checkpointing import sha256_file
from fastwam.alignment.data_identity import canonical_data_manifest_sha256
from fastwam.utils.config_resolvers import register_default_resolvers


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_libero_stage3_hydra_contract_resolves():
    register_default_resolvers()
    with initialize_config_dir(
        config_dir=str(REPO_ROOT / "configs"),
        version_base="1.3",
    ):
        config = compose(
            config_name="train_stage3_alignment",
            overrides=["task=libero_stage3_alignment_2cam224_1e-4"],
        )

    resolved = _resolved_config(config)
    assert resolved["base"]["expected_sha256"] == (
        "17a5588cc2b8d162219c9daf818614f614ee4a7921933a4a26c5d678111330e9"
    )
    assert resolved["assets"]["vae"]["expected_sha256"] == (
        "20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36"
    )
    assert resolved["model"]["_target_"] == (
        "fastwam.runtime.create_fastwam_unified_aligned"
    )
    assert resolved["model"]["skip_dit_load_from_pretrain"] is True
    assert resolved["model"]["redirect_common_files"] is False
    assert resolved["training"]["drop_last"] is True
    assert resolved["training"]["num_workers"] == 0
    batch_size = int(resolved["training"]["batch_size"])
    accumulation = int(resolved["training"]["gradient_accumulation_steps"])
    micro_batches = 273_465 // (8 * batch_size)
    assert (batch_size, accumulation) == (2, 3)
    assert micro_batches == 17_091
    assert micro_batches % accumulation == 0
    assert batch_size * 8 * accumulation == 48
    assert 273_465 - micro_batches * 8 * batch_size == 9
    assert resolved["stage3"]["num_solver_steps"] == 10
    assert set(resolved["data_manifest"]) == {
        "path",
        "expected_sha256",
        "full_content_verify",
    }
    assert resolved["data_manifest"]["path"] == (
        "/root/feihong/FastWAM/formal_runs/contracts/stage3/"
        "libero_current_273465f_1693e/libero_stage3_data_manifest.json"
    )
    assert resolved["data_manifest"]["expected_sha256"] == (
        "08da49109a57b55c67f3fa4ac31fbfa44e44dd541a194a5d3420838537d0d320"
    )
    assert resolved["data_manifest"]["full_content_verify"] is True
    assert resolved["data"]["train"]["video_backend"] == "torchcodec"
    assert resolved["data"]["train"]["strict_data_mode"] is True
    assert int(resolved["runtime"]["expected_dataset_length"]) == 273465
    assert int(resolved["runtime"]["expected_dataset_episodes"]) == 1693


def test_runtime_environment_contract_is_fail_closed(monkeypatch):
    monkeypatch.setenv("STAGE3_TEST_ENV", "locked")
    assert _validate_required_environment(
        {"required_environment": {"STAGE3_TEST_ENV": "locked"}}
    ) == {"STAGE3_TEST_ENV": "locked"}

    monkeypatch.setenv("STAGE3_TEST_ENV", "wrong")
    with pytest.raises(RuntimeError, match="STAGE3_TEST_ENV"):
        _validate_required_environment(
            {"required_environment": {"STAGE3_TEST_ENV": "locked"}}
        )


def test_activate_accelerator_cuda_device_resolves_unindexed_single_gpu(
    monkeypatch,
):
    selected = []
    monkeypatch.setattr("torch.cuda.current_device", lambda: 2)
    monkeypatch.setattr("torch.cuda.set_device", selected.append)

    assert _activate_accelerator_cuda_device(torch.device("cuda")) == torch.device(
        "cuda:2"
    )
    assert selected == [2]


def test_activate_accelerator_cuda_device_preserves_explicit_index(monkeypatch):
    selected = []
    monkeypatch.setattr(
        "torch.cuda.current_device",
        lambda: pytest.fail("explicit CUDA device must not query current_device"),
    )
    monkeypatch.setattr("torch.cuda.set_device", selected.append)

    assert _activate_accelerator_cuda_device(torch.device("cuda:5")) == torch.device(
        "cuda:5"
    )
    assert selected == [5]


def test_activate_accelerator_cuda_device_requires_rank_local_multi_gpu_index(
    monkeypatch,
):
    selected = []
    monkeypatch.setattr("torch.cuda.device_count", lambda: 8)
    monkeypatch.setattr("torch.cuda.set_device", selected.append)

    assert _activate_accelerator_cuda_device(
        torch.device("cuda:3"),
        num_processes=8,
        local_process_index=3,
    ) == torch.device("cuda:3")
    assert selected == [3]

    for device in (torch.device("cuda"), torch.device("cuda:0")):
        with pytest.raises(RuntimeError, match="rank-local device"):
            _activate_accelerator_cuda_device(
                device,
                num_processes=8,
                local_process_index=3,
            )
    assert selected == [3]


def test_runtime_hashes_both_external_assets(tmp_path):
    vae = tmp_path / "vae.pt"
    stats = tmp_path / "stats.json"
    vae.write_bytes(b"tiny-vae")
    stats.write_bytes(b"tiny-stats")
    accelerator = Accelerator(
        cpu=True,
        step_scheduler_with_optimizer=False,
    )

    identities = _resolve_asset_identities(
        accelerator,
        {
            "vae": {"path": str(vae), "expected_sha256": sha256_file(vae)},
            "normalization_stats": {
                "path": str(stats),
                "expected_sha256": sha256_file(stats),
            },
        },
    )

    assert identities["vae"]["sha256"] == sha256_file(vae)
    assert identities["normalization_stats"]["sha256"] == sha256_file(stats)


def test_resolve_data_identity_validates_manifest_on_single_rank(
    tmp_path,
    monkeypatch,
):
    manifest_path = tmp_path / "data-manifest.json"
    manifest_payload = {
        "format": "stage3-test-manifest",
        "num_frames": 17,
        "dataset_roots": [
            {"root": "/datasets/libero-a"},
            {"root": "/datasets/libero-b"},
        ],
    }
    expected_sha256 = canonical_data_manifest_sha256(manifest_payload)
    manifest_payload["manifest_sha256"] = expected_sha256
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
    train_dataset = object()
    calls = []

    def fake_validate_data_manifest(
        dataset,
        manifest,
        *,
        normalization_stats_path,
        full_content_verify,
    ):
        calls.append(
            {
                "dataset": dataset,
                "manifest": manifest,
                "normalization_stats_path": normalization_stats_path,
                "full_content_verify": full_content_verify,
            }
        )
        return {
            "manifest_sha256": expected_sha256,
            "num_frames": 17,
            "dataset_roots": [
                {"root": "/datasets/libero-a"},
                {"root": "/datasets/libero-b"},
            ],
        }

    monkeypatch.setattr(
        "fastwam.alignment.runtime.validate_data_manifest",
        fake_validate_data_manifest,
    )
    binding_calls = []

    def fake_bind_text_cache(dataset, manifest):
        binding_calls.append((dataset, manifest))
        return {"integrity_mode": "test-v1"}

    monkeypatch.setattr(
        "fastwam.alignment.runtime.bind_validated_text_cache_integrity",
        fake_bind_text_cache,
    )
    accelerator = Accelerator(
        cpu=True,
        step_scheduler_with_optimizer=False,
    )

    identity = _resolve_data_identity(
        accelerator,
        train_dataset,
        manifest_config={
            "path": str(manifest_path),
            "expected_sha256": expected_sha256,
            "full_content_verify": True,
        },
        normalization_stats_path="/assets/dataset_stats.json",
    )

    assert identity == {
        "path": str(manifest_path.resolve()),
        "sha256": expected_sha256,
        "num_frames": 17,
        "dataset_roots": [
            "/datasets/libero-a",
            "/datasets/libero-b",
        ],
        "full_content_verified": True,
        "text_cache_verification": {"integrity_mode": "test-v1"},
    }
    assert binding_calls == [(train_dataset, manifest_payload)]
    assert calls == [
        {
            "dataset": train_dataset,
            "manifest": manifest_payload,
            "normalization_stats_path": "/assets/dataset_stats.json",
            "full_content_verify": True,
        },
        {
            "dataset": train_dataset,
            "manifest": manifest_payload,
            "normalization_stats_path": "/assets/dataset_stats.json",
            "full_content_verify": False,
        },
    ]


def test_main_rank_value_broadcasts_success_without_non_main_reexecution(
    monkeypatch,
):
    broadcast_status = []

    def main_broadcast(objects, *, from_process):
        assert from_process == 0
        broadcast_status[:] = objects

    monkeypatch.setattr(
        "fastwam.alignment.runtime.broadcast_object_list",
        main_broadcast,
    )
    monkeypatch.setattr(
        "fastwam.alignment.runtime.gather_object",
        lambda objects: objects * 8,
    )
    main_calls = []
    main = SimpleNamespace(is_main_process=True, num_processes=8)
    expected = {"sha256": "a" * 64}

    assert _main_rank_value(
        main,
        lambda: main_calls.append(True) or expected,
        label="distributed probe",
    ) == expected
    assert main_calls == [True]

    def non_main_broadcast(objects, *, from_process):
        assert from_process == 0
        assert objects == [None]
        objects[:] = broadcast_status

    monkeypatch.setattr(
        "fastwam.alignment.runtime.broadcast_object_list",
        non_main_broadcast,
    )
    non_main = SimpleNamespace(is_main_process=False, num_processes=8)

    assert _main_rank_value(
        non_main,
        lambda: pytest.fail("non-main rank must not execute the operation"),
        label="distributed probe",
    ) == expected


def test_main_rank_value_broadcasts_main_failure_to_non_main(
    monkeypatch,
):
    broadcast_status = []

    def main_broadcast(objects, *, from_process):
        assert from_process == 0
        broadcast_status[:] = objects

    monkeypatch.setattr(
        "fastwam.alignment.runtime.broadcast_object_list",
        main_broadcast,
    )
    monkeypatch.setattr(
        "fastwam.alignment.runtime.gather_object",
        lambda objects: objects * 8,
    )
    main = SimpleNamespace(is_main_process=True, num_processes=8)

    def fail():
        raise OSError("rank-zero hash failed")

    with pytest.raises(RuntimeError, match="OSError: rank-zero hash failed"):
        _main_rank_value(main, fail, label="distributed probe")

    def non_main_broadcast(objects, *, from_process):
        assert from_process == 0
        objects[:] = broadcast_status

    monkeypatch.setattr(
        "fastwam.alignment.runtime.broadcast_object_list",
        non_main_broadcast,
    )
    non_main = SimpleNamespace(is_main_process=False, num_processes=8)

    with pytest.raises(RuntimeError, match="OSError: rank-zero hash failed"):
        _main_rank_value(
            non_main,
            lambda: pytest.fail("non-main rank must not execute the operation"),
            label="distributed probe",
        )


def test_resolve_data_identity_non_main_skips_full_hash_and_binds_locally(
    tmp_path,
    monkeypatch,
):
    manifest_path = tmp_path / "data-manifest.json"
    manifest = {
        "format": "stage3-test-manifest",
        "num_frames": 23,
        "dataset_roots": [{"root": "/datasets/robotwin"}],
    }
    manifest_sha256 = canonical_data_manifest_sha256(manifest)
    manifest["manifest_sha256"] = manifest_sha256
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    main_identity = {
        "path": str(manifest_path.resolve()),
        "sha256": manifest_sha256,
        "num_frames": 23,
        "dataset_roots": ["/datasets/robotwin"],
        "full_content_verified": True,
    }

    def broadcast_main_identity(objects, *, from_process):
        assert from_process == 0
        assert objects == [None]
        objects[0] = {"ok": True, "value": main_identity}

    monkeypatch.setattr(
        "fastwam.alignment.runtime.broadcast_object_list",
        broadcast_main_identity,
    )
    monkeypatch.setattr(
        "fastwam.alignment.runtime.gather_object",
        lambda objects: objects * 8,
    )
    dataset = object()
    validation_calls = []
    events = []

    def validate_fast(
        dataset_arg,
        manifest_arg,
        *,
        normalization_stats_path,
        full_content_verify,
    ):
        events.append("validate")
        validation_calls.append(
            {
                "dataset": dataset_arg,
                "manifest": manifest_arg,
                "normalization_stats_path": normalization_stats_path,
                "full_content_verify": full_content_verify,
            }
        )
        assert full_content_verify is False
        return manifest_arg

    monkeypatch.setattr(
        "fastwam.alignment.runtime.validate_data_manifest",
        validate_fast,
    )
    binding_calls = []

    def bind(dataset_arg, manifest_arg):
        events.append("bind")
        binding_calls.append((dataset_arg, manifest_arg))
        return {"integrity_mode": "sha256_index_v1"}

    monkeypatch.setattr(
        "fastwam.alignment.runtime.bind_validated_text_cache_integrity",
        bind,
    )
    non_main = SimpleNamespace(is_main_process=False, num_processes=8)

    identity = _resolve_data_identity(
        non_main,
        dataset,
        manifest_config={
            "path": str(manifest_path),
            "expected_sha256": manifest_sha256,
            "full_content_verify": True,
        },
        normalization_stats_path="/assets/dataset_stats.json",
    )

    assert identity == {
        **main_identity,
        "text_cache_verification": {"integrity_mode": "sha256_index_v1"},
    }
    assert validation_calls == [
        {
            "dataset": dataset,
            "manifest": manifest,
            "normalization_stats_path": "/assets/dataset_stats.json",
            "full_content_verify": False,
        }
    ]
    assert binding_calls == [(dataset, manifest)]
    assert events == ["validate", "bind"]


def test_resolve_data_identity_non_main_rejects_local_manifest_drift_before_bind(
    tmp_path,
    monkeypatch,
):
    manifest_path = tmp_path / "data-manifest.json"
    local_manifest = {
        "num_frames": 24,
        "dataset_roots": [{"root": "/datasets/robotwin"}],
    }
    local_sha256 = canonical_data_manifest_sha256(local_manifest)
    local_manifest["manifest_sha256"] = local_sha256
    manifest_path.write_text(json.dumps(local_manifest), encoding="utf-8")
    main_sha256 = "a" * 64

    def broadcast_main_identity(objects, *, from_process):
        objects[0] = {
            "ok": True,
            "value": {
                "path": str(manifest_path.resolve()),
                "sha256": main_sha256,
                "num_frames": 23,
                "dataset_roots": ["/datasets/robotwin"],
                "full_content_verified": True,
            },
        }

    monkeypatch.setattr(
        "fastwam.alignment.runtime.broadcast_object_list",
        broadcast_main_identity,
    )
    monkeypatch.setattr(
        "fastwam.alignment.runtime.gather_object",
        lambda objects: objects * 8,
    )
    validation_calls = []

    def reject_drift(
        dataset,
        manifest,
        *,
        normalization_stats_path,
        full_content_verify,
    ):
        validation_calls.append(full_content_verify)
        assert full_content_verify is False
        raise ValueError("selected data identity drifted from manifest")

    monkeypatch.setattr(
        "fastwam.alignment.runtime.validate_data_manifest",
        reject_drift,
    )
    monkeypatch.setattr(
        "fastwam.alignment.runtime.bind_validated_text_cache_integrity",
        lambda *args, **kwargs: pytest.fail(
            "a drifted local manifest must never be bound"
        ),
    )
    non_main = SimpleNamespace(is_main_process=False, num_processes=8)

    with pytest.raises(
        RuntimeError,
        match="selected data identity drifted from manifest",
    ):
        _resolve_data_identity(
            non_main,
            object(),
            manifest_config={
                "path": str(manifest_path),
                "expected_sha256": main_sha256,
                "full_content_verify": True,
            },
            normalization_stats_path="/assets/dataset_stats.json",
        )
    assert validation_calls == [False]


def test_resolve_data_identity_rejects_wrong_expected_sha256(
    tmp_path,
    monkeypatch,
):
    manifest_path = tmp_path / "data-manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "fastwam.alignment.runtime.validate_data_manifest",
        lambda *args, **kwargs: {
            "manifest_sha256": "a" * 64,
            "num_frames": 1,
            "dataset_roots": [{"root": "/datasets/libero"}],
        },
    )
    accelerator = Accelerator(
        cpu=True,
        step_scheduler_with_optimizer=False,
    )

    with pytest.raises(RuntimeError, match="manifest SHA256 mismatch"):
        _resolve_data_identity(
            accelerator,
            object(),
            manifest_config={
                "path": str(manifest_path),
                "expected_sha256": "b" * 64,
                "full_content_verify": True,
            },
            normalization_stats_path="/assets/dataset_stats.json",
        )


@pytest.mark.parametrize("missing", ["path", "expected_sha256"])
def test_resolve_data_identity_requires_manifest_path_and_hash(
    tmp_path,
    monkeypatch,
    missing,
):
    manifest_path = tmp_path / "data-manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")

    def unexpected_validation(*args, **kwargs):
        raise AssertionError("manifest validation must not run")

    monkeypatch.setattr(
        "fastwam.alignment.runtime.validate_data_manifest",
        unexpected_validation,
    )
    manifest_config = {
        "path": str(manifest_path),
        "expected_sha256": "a" * 64,
        "full_content_verify": True,
    }
    del manifest_config[missing]
    accelerator = Accelerator(
        cpu=True,
        step_scheduler_with_optimizer=False,
    )

    with pytest.raises(
        RuntimeError,
        match="requires data_manifest.path and data_manifest.expected_sha256",
    ):
        _resolve_data_identity(
            accelerator,
            object(),
            manifest_config=manifest_config,
            normalization_stats_path="/assets/dataset_stats.json",
        )


def test_resolve_data_identity_requires_full_content_verification(
    tmp_path,
    monkeypatch,
):
    manifest_path = tmp_path / "data-manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")

    def unexpected_validation(*args, **kwargs):
        raise AssertionError("manifest validation must not run")

    monkeypatch.setattr(
        "fastwam.alignment.runtime.validate_data_manifest",
        unexpected_validation,
    )
    accelerator = Accelerator(
        cpu=True,
        step_scheduler_with_optimizer=False,
    )

    with pytest.raises(
        RuntimeError,
        match="data_manifest.full_content_verify=true",
    ):
        _resolve_data_identity(
            accelerator,
            object(),
            manifest_config={
                "path": str(manifest_path),
                "expected_sha256": "a" * 64,
                "full_content_verify": False,
            },
            normalization_stats_path="/assets/dataset_stats.json",
        )


def test_all_rank_value_propagates_single_rank_operation_error():
    accelerator = Accelerator(
        cpu=True,
        step_scheduler_with_optimizer=False,
    )

    def fail():
        raise OSError("identity probe failed")

    with pytest.raises(
        RuntimeError,
        match="test identity.*OSError: identity probe failed",
    ):
        _all_rank_value(accelerator, fail, label="test")


def test_run_all_rank_phase_returns_single_rank_operation_value():
    accelerator = Accelerator(
        cpu=True,
        step_scheduler_with_optimizer=False,
    )
    expected = object()

    assert (
        _run_all_rank_phase(
            accelerator,
            lambda: expected,
            label="test phase",
        )
        is expected
    )


def test_run_all_rank_phase_propagates_single_rank_operation_error():
    accelerator = Accelerator(
        cpu=True,
        step_scheduler_with_optimizer=False,
    )

    def fail():
        raise OSError("phase probe failed")

    with pytest.raises(
        RuntimeError,
        match="test phase.*OSError: phase probe failed",
    ):
        _run_all_rank_phase(accelerator, fail, label="test phase")


@pytest.mark.parametrize(
    ("helper", "operation"),
    [
        (_run_all_rank_phase, lambda: "done"),
        (_all_rank_value, lambda: {"value": "done"}),
    ],
    ids=["phase", "value"],
)
def test_all_rank_helpers_reject_non_mapping_gather_status(
    monkeypatch,
    helper,
    operation,
):
    monkeypatch.setattr(
        "fastwam.alignment.runtime.gather_object",
        lambda _: [None],
    )
    accelerator = Accelerator(
        cpu=True,
        step_scheduler_with_optimizer=False,
    )

    with pytest.raises(RuntimeError) as error:
        helper(accelerator, operation, label="corrupt gather")

    assert "corrupt gather" in str(error.value)
    assert "invalid" in str(error.value).lower()


def test_canonicalize_data_paths_resolves_all_training_paths(tmp_path):
    repo_dir = (tmp_path / "repo").resolve()
    data = {
        "train": {
            "dataset_dirs": ["data/one", "../shared/two"],
            "pretrained_norm_stats": "runs/stats.json",
            "text_embedding_cache_dir": "cache/text",
            "strict_data_mode": True,
        }
    }

    resolved = _canonicalize_data_paths(data, repo_dir=repo_dir)

    assert resolved["train"]["dataset_dirs"] == [
        str((repo_dir / "data/one").resolve()),
        str((repo_dir / "../shared/two").resolve()),
    ]
    assert resolved["train"]["pretrained_norm_stats"] == str(
        (repo_dir / "runs/stats.json").resolve()
    )
    assert resolved["train"]["text_embedding_cache_dir"] == str(
        (repo_dir / "cache/text").resolve()
    )
    assert data["train"]["dataset_dirs"] == ["data/one", "../shared/two"]


class _FormalDataset:
    def __init__(self, *, length: int = 277713, episodes: int = 1712):
        self.length = length
        self.strict_data_mode = True
        self.skip_padding_as_possible = False
        datasets = [
            SimpleNamespace(
                root=Path("relative/root-a"),
                video_backend="torchcodec",
                allow_video_backend_fallback=False,
            ),
            SimpleNamespace(
                root=Path("relative/root-b"),
                video_backend="torchcodec",
                allow_video_backend_fallback=False,
            ),
        ]
        multi = SimpleNamespace(_datasets=datasets, num_episodes=episodes)
        self.lerobot_dataset = SimpleNamespace(
            strict_data_mode=True,
            multi_dataset=multi,
        )

    def __len__(self):
        return self.length


def _formal_runtime_contract() -> dict[str, int]:
    return {
        "expected_dataset_length": 277713,
        "expected_dataset_episodes": 1712,
    }


def test_validate_formal_dataset_returns_canonical_identity():
    dataset = _FormalDataset()

    identity = _validate_formal_dataset(
        dataset,
        runtime_config=_formal_runtime_contract(),
    )

    assert identity == {
        "dataset_length": 277713,
        "dataset_episodes": 1712,
        "dataset_roots": [
            str(Path("relative/root-a").resolve()),
            str(Path("relative/root-b").resolve()),
        ],
        "video_backend": "torchcodec",
        "strict_data_mode": True,
        "skip_padding_as_possible": False,
    }


@pytest.mark.parametrize(
    ("mismatch", "message"),
    [
        ("outer_strict", "strict_data_mode"),
        ("padding_retry", "skip_padding_as_possible"),
        ("base_strict", "propagate strict"),
        ("backend", "video_backend=torchcodec"),
        ("fallback", "forbids video decoder fallback"),
        ("length", "cardinality mismatch"),
        ("episodes", "cardinality mismatch"),
    ],
)
def test_validate_formal_dataset_rejects_contract_mismatch(mismatch, message):
    dataset = _FormalDataset()
    if mismatch == "outer_strict":
        dataset.strict_data_mode = False
    elif mismatch == "padding_retry":
        dataset.skip_padding_as_possible = True
    elif mismatch == "base_strict":
        dataset.lerobot_dataset.strict_data_mode = False
    elif mismatch == "backend":
        dataset.lerobot_dataset.multi_dataset._datasets[0].video_backend = "pyav"
    elif mismatch == "fallback":
        dataset.lerobot_dataset.multi_dataset._datasets[
            0
        ].allow_video_backend_fallback = True
    elif mismatch == "length":
        dataset.length -= 1
    elif mismatch == "episodes":
        dataset.lerobot_dataset.multi_dataset.num_episodes -= 1

    with pytest.raises(RuntimeError, match=message):
        _validate_formal_dataset(
            dataset,
            runtime_config=_formal_runtime_contract(),
        )
