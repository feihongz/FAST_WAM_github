import json

import pytest
import torch
from omegaconf import OmegaConf

from experiments.libero.gate.analyze_demo_utility import (
    analyze,
    utility_statistics,
)
from experiments.libero.gate.demo_utility import (
    current_state_input_hashes,
    extract_current_state,
)
from experiments.libero.gate.collect_demo_utility import (
    _assert_directory_tree_provenance_unchanged,
    _build_manifest,
    _checkpoint_payload_provenance,
    _dataset_instantiation_path_overrides,
    _directory_tree_provenance,
    _validate_endpoint_config,
    _validate_existing_records_against_dataset,
    _validate_manifest_integrity,
    build_stratified_sample_plan,
    ensure_immutable_manifest,
    load_existing_record_index,
    resolve_dataset_stats_path,
)


def test_directory_tree_provenance_detects_content_change(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    member = source / "member.bin"
    member.write_bytes(b"before")
    provenance = _directory_tree_provenance(source, label="test source")

    _assert_directory_tree_provenance_unchanged(provenance, label="test source")
    member.write_bytes(b"after")
    with pytest.raises(RuntimeError, match="changed during collection"):
        _assert_directory_tree_provenance_unchanged(provenance, label="test source")


def test_explicit_dataset_stats_path_is_authoritative(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint-run"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "latest.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    fallback_path = checkpoint_dir / "dataset_stats.json"
    fallback_path.write_text("{}\n", encoding="utf-8")

    missing_explicit = tmp_path / "missing-stats.json"
    with pytest.raises(FileNotFoundError, match="Explicit COLLECTOR.dataset_stats_path"):
        resolve_dataset_stats_path(checkpoint_path, missing_explicit)

    assert resolve_dataset_stats_path(checkpoint_path) == fallback_path.resolve()
    explicit_path = tmp_path / "explicit-stats.json"
    explicit_path.write_text("{}\n", encoding="utf-8")
    assert resolve_dataset_stats_path(checkpoint_path, explicit_path) == explicit_path.resolve()


def test_dataset_instantiation_uses_fingerprinted_absolute_paths(tmp_path):
    source_a = (tmp_path / "suite-a").resolve()
    source_b = (tmp_path / "suite-b").resolve()
    cache = (tmp_path / "cache").resolve()
    overrides = _dataset_instantiation_path_overrides(
        [{"path": str(source_a)}, {"path": str(source_b)}],
        {"path": str(cache)},
    )
    assert overrides == {
        "dataset_dirs": [str(source_a), str(source_b)],
        "text_embedding_cache_dir": str(cache),
    }

    with pytest.raises(ValueError, match="not absolute"):
        _dataset_instantiation_path_overrides(
            [{"path": "relative/suite"}], {"path": str(cache)}
        )


def _ranges():
    return [
        {
            "dataset_index": 0,
            "dataset_id": "suite_a",
            "dataset_name": "suite_a",
            "start": 0,
            "stop": 10,
            "population": 10,
        },
        {
            "dataset_index": 1,
            "dataset_id": "suite_b",
            "dataset_name": "suite_b",
            "start": 10,
            "stop": 40,
            "population": 30,
        },
        {
            "dataset_index": 2,
            "dataset_id": "suite_c",
            "dataset_name": "suite_c",
            "start": 40,
            "stop": 60,
            "population": 20,
        },
    ]


def test_stratified_plan_is_deterministic_shuffled_and_without_replacement():
    selected, strata = build_stratified_sample_plan(_ranges(), num_samples=18, seed=42)
    repeated, repeated_strata = build_stratified_sample_plan(
        reversed(_ranges()), num_samples=18, seed=42
    )
    changed_seed, _ = build_stratified_sample_plan(_ranges(), num_samples=18, seed=43)

    assert selected == repeated
    assert strata == repeated_strata
    assert selected != changed_seed
    assert len(selected) == len(set(selected)) == 18
    assert [item["allocated"] for item in strata] == [3, 9, 6]
    assert all(item["allocated"] >= 1 for item in strata)
    assert selected != sorted(selected)
    assert all(0 <= index < 60 for index in selected)
    assert all(len(item["ordered_selected_source_indices_sha256"]) == 64 for item in strata)


def test_stratified_plan_rejects_oversampling():
    with pytest.raises(ValueError, match="without replacement"):
        build_stratified_sample_plan(_ranges(), num_samples=61, seed=42)


def test_collector_rejects_non_endpoint_full_prefix():
    cfg = OmegaConf.create(
        {
            "COLLECTOR": {
                "num_inference_steps": 10,
                "full_prefix_steps": 9,
                "force_custom_prefix": True,
                "num_video_frames": 9,
            },
            "data": {"train": {"num_frames": 33, "action_video_freq_ratio": 4}},
        }
    )
    with pytest.raises(ValueError, match="must equal num_inference_steps"):
        _validate_endpoint_config(cfg)


def test_manifest_rejects_a_stripped_legacy_payload(tmp_path):
    path = tmp_path / "manifest.json"
    payload = {"compatibility_fingerprint": "abc", "created_at_utc": "first"}
    with pytest.raises(ValueError, match="compatibility must be a mapping"):
        ensure_immutable_manifest(path, payload)
    assert not path.exists()


def test_existing_record_index_detects_duplicate_sample_id(tmp_path):
    path = tmp_path / "records.jsonl"
    rows = [
        {
            "sample_id": "suite/episode_000001/frame_000002",
            "source_metadata": {"requested_sample_idx": 7},
        },
        {
            "sample_id": "suite/episode_000001/frame_000002",
            "source_metadata": {"requested_sample_idx": 8},
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate sample_id"):
        load_existing_record_index(path)


def test_existing_record_index_checks_manifest_fingerprint(tmp_path):
    path = tmp_path / "records.jsonl"
    row = {
        "sample_id": "suite/episode_000001/frame_000002",
        "source_metadata": {"requested_sample_idx": 7},
        "manifest_compatibility_fingerprint": "old",
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_existing_record_index(path, expected_manifest_fingerprint="new")


def _valid_completed_record(*, fingerprint="manifest-fingerprint", source_index=7):
    zero_hash = "0" * 64
    component_hashes = {
        "action_is_pad": zero_hash,
        "context": zero_hash,
        "context_mask": zero_hash,
        "input_image": zero_hash,
        "proprio": zero_hash,
        "valid_target_action": zero_hash,
    }
    import hashlib

    combined = hashlib.sha256(
        json.dumps(
            component_hashes, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 1,
        "collector_record_schema_version": 1,
        "sample_id": "libero_goal_no_noops_lerobot/episode_000001/frame_000002",
        "dataset_id": "libero_goal_no_noops_lerobot",
        "dataset_name": "libero_goal_no_noops_lerobot",
        "suite": "libero_goal",
        "episode_index": 1,
        "episode_id": 1,
        "frame_index": 2,
        "task_index": 0,
        "task_id": 0,
        "task_id_source": "lerobot_task_index",
        "task": "open drawer",
        "seed": 123,
        "num_inference_steps": 10,
        "n0": 0,
        "nfull": 10,
        "e0": 0.2,
        "efull": 0.1,
        "utility": 0.1,
        "n0_latency_ms": 1.0,
        "nfull_latency_ms": 2.0,
        "total_latency_ms": 3.0,
        "valid_length": 2,
        "target_action_shape": [2, 7],
        "pred_n0_shape": [2, 7],
        "pred_nfull_shape": [2, 7],
        "input_hashes": {**component_hashes, "combined": combined},
        "n0_route": {
            "inference_mode": "prefix",
            "video_prefix_steps": 0,
            "num_inference_steps": 10,
            "force_custom_prefix": True,
        },
        "nfull_route": {
            "inference_mode": "prefix",
            "video_prefix_steps": 10,
            "num_inference_steps": 10,
            "force_custom_prefix": True,
        },
        "current_proprio": [0.0, 1.0],
        "source_metadata": {
            "requested_sample_idx": source_index,
            "source_sample_idx": source_index,
            "dataset_name": "libero_goal_no_noops_lerobot",
            "episode_index": 1,
            "frame_index": 2,
            "task_index": 0,
            "task": "open drawer",
        },
        "manifest_compatibility_fingerprint": fingerprint,
        "checkpoint_sha256": "c" * 64,
        "dataset_stats_sha256": "s" * 64,
        "vae_sha256": "v" * 64,
        "git_sha": "g" * 40,
    }


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.pop("utility"),
        lambda row: row["nfull_route"].__setitem__("video_prefix_steps", 9),
        lambda row: row["source_metadata"].__setitem__("source_sample_idx", 8),
        lambda row: row.__setitem__("checkpoint_sha256", "x" * 64),
        lambda row: row["input_hashes"].__setitem__("combined", "f" * 64),
        lambda row: row.pop("current_proprio"),
        lambda row: row.__setitem__("n0_latency_ms", -1.0),
        lambda row: row.update(e0=-0.2, efull=-0.3, utility=0.1),
    ],
)
def test_resume_rejects_incomplete_or_tampered_completed_record(tmp_path, mutation):
    path = tmp_path / "records.jsonl"
    row = _valid_completed_record()
    mutation(row)
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_existing_record_index(
            path,
            expected_manifest_fingerprint="manifest-fingerprint",
            expected_full_steps=10,
            expected_checkpoint_sha256="c" * 64,
            expected_dataset_stats_sha256="s" * 64,
            expected_vae_sha256="v" * 64,
            expected_git_sha="g" * 40,
        )


def test_analyzer_rejects_an_invalid_production_record(tmp_path):
    path = tmp_path / "records.jsonl"
    row = _valid_completed_record()
    row.update(e0=-0.2, efull=-0.3, utility=0.1)
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="MSE values.*non-negative"):
        analyze(
            path,
            tmp_path / "analysis",
            near_zero_epsilon=1e-4,
            bins=5,
            make_plot=False,
        )


def test_resume_accepts_a_complete_validated_record(tmp_path):
    path = tmp_path / "records.jsonl"
    row = _valid_completed_record()
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    sample_ids, source_indices, count = load_existing_record_index(
        path,
        expected_manifest_fingerprint="manifest-fingerprint",
        expected_full_steps=10,
        expected_checkpoint_sha256="c" * 64,
        expected_dataset_stats_sha256="s" * 64,
        expected_vae_sha256="v" * 64,
        expected_git_sha="g" * 40,
    )
    assert sample_ids == {row["sample_id"]}
    assert source_indices == {7: row["sample_id"]}
    assert count == 1


def test_resume_rows_are_rebound_to_the_actual_dataset_state(tmp_path):
    sample = {
        "video": torch.zeros(3, 9, 2, 2),
        "proprio": torch.tensor([[0.0, 1.0], [2.0, 3.0]]),
        "context": torch.zeros(3, 4),
        "context_mask": torch.tensor([True, True, False]),
        "action": torch.zeros(2, 7),
        "action_is_pad": torch.tensor([False, False]),
        "metadata": {
            "requested_sample_idx": 7,
            "source_sample_idx": 7,
            "dataset_index": 0,
            "dataset_id": "libero_goal_no_noops_lerobot",
            "dataset_name": "libero_goal_no_noops_lerobot",
            "episode_index": 1,
            "frame_index": 2,
            "task_index": 0,
            "task": "open drawer",
        },
    }

    class FakeDataset:
        def __getitem__(self, index):
            assert index == 7
            return sample

    record = _valid_completed_record()
    record["input_hashes"] = current_state_input_hashes(extract_current_state(sample))
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    ranges = [{"dataset_index": 0, "start": 0, "stop": 10}]
    task_tables = {0: {0: "open drawer"}}
    assert (
        _validate_existing_records_against_dataset(
            records_path,
            dataset=FakeDataset(),
            ranges=ranges,
            task_tables=task_tables,
        )
        == 1
    )

    disguised = json.loads(json.dumps(record))
    disguised.update(
        sample_id="libero_goal_no_noops_lerobot/episode_000003/frame_000002",
        episode_index=3,
        episode_id=3,
    )
    disguised["source_metadata"]["episode_index"] = 3
    records_path.write_text(json.dumps(disguised) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="real dataset identity"):
        _validate_existing_records_against_dataset(
            records_path,
            dataset=FakeDataset(),
            ranges=ranges,
            task_tables=task_tables,
        )


def _manifest_cfg():
    return OmegaConf.create(
        {
            "model": {"_target_": "fake.Model", "hidden": 8},
            "data": {
                "train": {
                    "dataset_dirs": ["suite_a"],
                    "text_embedding_cache_dir": "cache",
                    "num_frames": 33,
                }
            },
            "mixed_precision": "bf16",
            "COLLECTOR": {
                "seed": 42,
                "device": "cuda",
                "num_inference_steps": 10,
                "full_prefix_steps": 10,
                "num_video_frames": 9,
                "rand_device": "cpu",
                "sigma_shift": None,
                "tiled": False,
                "force_custom_prefix": True,
                "resume": True,
                "continue_on_error": True,
            },
        }
    )


def test_untracked_status_is_provenance_but_not_resume_compatibility(tmp_path):
    cfg = _manifest_cfg()
    checkpoint_path = tmp_path / "checkpoint.pt"
    stats_path = tmp_path / "dataset_stats.json"
    checkpoint_path.write_bytes(b"checkpoint")
    stats_path.write_text("{}\n", encoding="utf-8")
    common = {
        "cfg": cfg,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": "c" * 64,
        "stats_path": stats_path,
        "stats_sha256": "s" * 64,
        "vae_artifact": {"path": "vae.pt", "sha256": "a" * 64},
        "dataset_source_artifacts": [
            {
                "dataset_name": "suite_a",
                "sha256": "b" * 64,
                "file_count": 1,
                "total_size_bytes": 1,
            }
        ],
        "context_cache_artifact": {"path": "cache", "sha256": "c" * 64},
        "ranges": _ranges(),
        "task_tables": {0: {0: "a"}, 1: {0: "b"}, 2: {0: "c"}},
        "plan_strata": [],
        "selected_indices": [1, 11, 41],
    }
    clean_git = {
        "commit": "a" * 40,
        "branch": "branch",
        "dirty": False,
        "status_porcelain": [],
        "status_sha256": "0" * 64,
        "tracked_diff_sha256": "d" * 64,
    }
    untracked_git = {
        **clean_git,
        "dirty": True,
        "status_porcelain": ["?? unrelated.bin"],
        "status_sha256": "1" * 64,
    }
    clean = _build_manifest(**common, git=clean_git)
    untracked = _build_manifest(**common, git=untracked_git)

    assert clean["compatibility_fingerprint"] == untracked["compatibility_fingerprint"]
    assert clean["git"] != untracked["git"]
    assert clean["scientific_source_files"]
    assert clean["artifacts"]["checkpoint"]["size_bytes"] == len(b"checkpoint")
    environment = clean["compatibility"]["execution_environment"]
    assert environment["device"] == "cuda"
    assert environment["torch_version"] == str(torch.__version__)
    assert environment["cuda_version"] == (
        None if torch.version.cuda is None else str(torch.version.cuda)
    )

    relocated_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    relocated_cfg.data.train.dataset_dirs = ["/different/mount/suite_a"]
    relocated_cfg.data.train.text_embedding_cache_dir = "/different/mount/cache"
    relocated = _build_manifest(
        **{**common, "cfg": relocated_cfg},
        git=clean_git,
    )
    assert relocated["compatibility_fingerprint"] == clean["compatibility_fingerprint"]
    assert relocated["resolved_config_sha256"] != clean["resolved_config_sha256"]

    relocated_ranges = [
        {**item, "dataset_id": f"/different/mount/{item['dataset_name']}"}
        for item in common["ranges"]
    ]
    relocated_range_manifest = _build_manifest(
        **{**common, "ranges": relocated_ranges},
        git=clean_git,
    )
    assert (
        relocated_range_manifest["compatibility_fingerprint"]
        == clean["compatibility_fingerprint"]
    )
    assert relocated_range_manifest["dataset_index_ranges"] != clean["dataset_index_ranges"]

    changed_science_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    changed_science_cfg.data.train.num_frames = 65
    changed_science = _build_manifest(
        **{**common, "cfg": changed_science_cfg},
        git=clean_git,
    )
    assert changed_science["compatibility_fingerprint"] != clean["compatibility_fingerprint"]

    changed_device_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    changed_device_cfg.COLLECTOR.device = "cpu"
    changed_device = _build_manifest(
        **{**common, "cfg": changed_device_cfg},
        git=clean_git,
    )
    assert changed_device["compatibility_fingerprint"] != clean["compatibility_fingerprint"]

    changed_code = _build_manifest(
        **common,
        git={**clean_git, "tracked_diff_sha256": "e" * 64},
    )
    assert changed_code["compatibility_fingerprint"] != clean["compatibility_fingerprint"]

    changed_vae = _build_manifest(
        **{
            **common,
            "vae_artifact": {"path": "/different/stage/vae.pt", "sha256": "f" * 64},
        },
        git=clean_git,
    )
    assert changed_vae["compatibility_fingerprint"] != clean["compatibility_fingerprint"]

    same_vae_bytes_at_new_path = _build_manifest(
        **{
            **common,
            "vae_artifact": {"path": "/different/stage/vae.pt", "sha256": "a" * 64},
        },
        git=clean_git,
    )
    assert (
        same_vae_bytes_at_new_path["compatibility_fingerprint"]
        == clean["compatibility_fingerprint"]
    )

    changed_dataset = _build_manifest(
        **{
            **common,
            "dataset_source_artifacts": [
                {
                    "dataset_name": "suite_a",
                    "sha256": "e" * 64,
                    "file_count": 1,
                    "total_size_bytes": 1,
                }
            ],
        },
        git=clean_git,
    )
    assert changed_dataset["compatibility_fingerprint"] != clean["compatibility_fingerprint"]

    changed_context = _build_manifest(
        **{
            **common,
            "context_cache_artifact": {"path": "cache", "sha256": "d" * 64},
        },
        git=clean_git,
    )
    assert changed_context["compatibility_fingerprint"] != clean["compatibility_fingerprint"]


def test_checkpoint_payload_provenance_checks_keys_and_shapes():
    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mot = torch.nn.Linear(3, 2)

    model = FakeModel()
    payload = {"mot": model.mot.state_dict(), "step": 123, "torch_dtype": "bfloat16"}
    info = _checkpoint_payload_provenance(model, payload)

    assert info["state_key"] == "mot"
    assert info["step"] == 123
    assert info["overlap_key_count"] == 2
    assert info["missing_key_count"] == 0
    assert info["unexpected_key_count"] == 0

    wrong_shape = {
        "mot": {
            "weight": torch.zeros(4, 3),
            "bias": torch.zeros(2),
        }
    }
    with pytest.raises(ValueError, match="shape mismatch"):
        _checkpoint_payload_provenance(model, wrong_shape)

    partial = {"mot": {"weight": model.mot.weight.detach().clone()}}
    with pytest.raises(ValueError, match="partial MoT checkpoint"):
        _checkpoint_payload_provenance(model, partial)


def test_checkpoint_payload_requires_complete_proprio_encoder():
    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mot = torch.nn.Linear(3, 2)
            self.proprio_encoder = torch.nn.Linear(8, 4)

    model = FakeModel()
    with pytest.raises(ValueError, match="missing proprio_encoder"):
        _checkpoint_payload_provenance(model, {"mot": model.mot.state_dict()})

    info = _checkpoint_payload_provenance(
        model,
        {
            "mot": model.mot.state_dict(),
            "proprio_encoder": model.proprio_encoder.state_dict(),
        },
    )
    assert info["proprio_encoder"]["state_key_count"] == 2


def test_statistics_partition_positive_negative_and_nearzero():
    stats = utility_statistics([-0.2, -0.00001, 0.0, 0.00001, 0.4], near_zero_epsilon=1e-4)
    assert stats["count"] == 5
    assert stats["positive_count"] == 1
    assert stats["negative_count"] == 1
    assert stats["nearzero_count"] == 3
    assert stats["mean_abs"] == pytest.approx(0.120004)
    assert stats["q50"] == pytest.approx(0.0)


def test_analysis_writes_json_and_overall_suite_task_csv(tmp_path):
    records_path = tmp_path / "records.jsonl"
    rows = [
        {
            "sample_id": "spatial/episode_000001/frame_000001",
            "suite": "libero_spatial",
            "task_index": 0,
            "task": "pick object",
            "e0": 0.3,
            "efull": 0.1,
            "utility": 0.2,
        },
        {
            "sample_id": "goal/episode_000001/frame_000001",
            "suite": "libero_goal",
            "task_index": 1,
            "task": "open drawer",
            "e0": 0.1,
            "efull": 0.2,
            "utility": -0.1,
        },
    ]
    records_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output_dir = tmp_path / "analysis"

    report = analyze(
        records_path,
        output_dir,
        near_zero_epsilon=1e-4,
        bins=5,
        make_plot=False,
    )

    assert report["overall"]["count"] == 2
    assert len(report["by_suite"]) == 2
    assert len(report["by_task"]) == 2
    for filename in (
        "summary.json",
        "overall.csv",
        "by_suite.csv",
        "by_task.csv",
        "histogram.csv",
    ):
        assert (output_dir / filename).is_file()

    assert report["completeness"] == {
        "verified_against_manifest": False,
        "allow_incomplete": False,
        "is_complete": None,
        "status": "unverified_no_manifest",
        "expected_count": None,
        "completed_count": 2,
        "missing_count": None,
        "coverage_fraction": None,
        "missing_source_index_examples": [],
        "selection_sha256": None,
    }


def _write_manifest_plan(tmp_path, plan, *, fingerprint="manifest-fingerprint"):
    manifest = {
        "compatibility_fingerprint": fingerprint,
        "selection": {
            "num_samples": len(plan),
            "ordered_selected_source_indices": list(plan),
        },
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest) + "\n",
        encoding="utf-8",
    )


def _write_manifest_records(
    tmp_path, source_indices, *, fingerprint="manifest-fingerprint"
):
    records_path = tmp_path / "records.jsonl"
    rows = [
        {
            "sample_id": f"libero/episode_000001/frame_{source_index:06d}",
            "suite": "libero_goal",
            "task_index": 0,
            "task": "open drawer",
            "e0": 0.2,
            "efull": 0.1,
            "utility": 0.1,
            "manifest_compatibility_fingerprint": fingerprint,
            "source_metadata": {"requested_sample_idx": source_index},
        }
        for source_index in source_indices
    ]
    records_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return records_path


def test_analysis_requires_complete_manifest_plan_by_default(tmp_path):
    records_path = _write_manifest_records(tmp_path, [7])
    _write_manifest_plan(tmp_path, [7, 8])

    with pytest.raises(ValueError, match="Collection is incomplete"):
        analyze(
            records_path,
            tmp_path / "analysis-default",
            near_zero_epsilon=1e-4,
            bins=5,
            make_plot=False,
        )

    report = analyze(
        records_path,
        tmp_path / "analysis-explicit-partial",
        near_zero_epsilon=1e-4,
        bins=5,
        make_plot=False,
        allow_incomplete=True,
    )
    assert report["completeness"] == {
        "verified_against_manifest": True,
        "allow_incomplete": True,
        "is_complete": False,
        "status": "incomplete_allowed",
        "expected_count": 2,
        "completed_count": 1,
        "missing_count": 1,
        "coverage_fraction": 0.5,
        "missing_source_index_examples": [8],
        "selection_sha256": report["completeness"]["selection_sha256"],
    }


def test_analysis_complete_manifest_plan_reports_full_coverage(tmp_path):
    records_path = _write_manifest_records(tmp_path, [8, 7])
    _write_manifest_plan(tmp_path, [7, 8])

    report = analyze(
        records_path,
        tmp_path / "analysis",
        near_zero_epsilon=1e-4,
        bins=5,
        make_plot=False,
    )
    assert report["completeness"]["is_complete"] is True
    assert report["completeness"]["status"] == "complete"
    assert report["completeness"]["coverage_fraction"] == 1.0


def test_analysis_rejects_out_of_plan_record_even_when_incomplete_allowed(tmp_path):
    records_path = _write_manifest_records(tmp_path, [7, 99])
    _write_manifest_plan(tmp_path, [7, 8])

    with pytest.raises(ValueError, match="outside the immutable manifest plan"):
        analyze(
            records_path,
            tmp_path / "analysis",
            near_zero_epsilon=1e-4,
            bins=5,
            make_plot=False,
            allow_incomplete=True,
        )


def test_analysis_requires_fingerprint_on_every_manifest_record(tmp_path):
    records_path = _write_manifest_records(tmp_path, [7], fingerprint="")
    _write_manifest_plan(tmp_path, [7])

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        analyze(
            records_path,
            tmp_path / "analysis",
            near_zero_epsilon=1e-4,
            bins=5,
            make_plot=False,
        )


def test_manifest_integrity_rejects_tampered_compatibility_or_selection(tmp_path):
    cfg = _manifest_cfg()
    checkpoint_path = tmp_path / "checkpoint.pt"
    stats_path = tmp_path / "dataset_stats.json"
    checkpoint_path.write_bytes(b"checkpoint")
    stats_path.write_text("{}\n", encoding="utf-8")
    manifest = _build_manifest(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256="c" * 64,
        stats_path=stats_path,
        stats_sha256="s" * 64,
        vae_artifact={"path": "vae.pt", "sha256": "a" * 64},
        dataset_source_artifacts=[
            {
                "dataset_name": "suite_a",
                "sha256": "b" * 64,
                "file_count": 1,
                "total_size_bytes": 1,
            }
        ],
        context_cache_artifact={"path": "cache", "sha256": "d" * 64},
        git={
            "commit": "a" * 40,
            "branch": "branch",
            "dirty": False,
            "status_porcelain": [],
            "status_sha256": "0" * 64,
            "tracked_diff_sha256": "e" * 64,
        },
        ranges=_ranges(),
        task_tables={0: {0: "a"}, 1: {0: "b"}, 2: {0: "c"}},
        plan_strata=[],
        selected_indices=[1, 11, 41],
    )
    _validate_manifest_integrity(manifest)
    path = tmp_path / "manifest.json"
    assert ensure_immutable_manifest(path, manifest) == manifest
    original = path.read_bytes()
    compatible = json.loads(json.dumps(manifest))
    compatible["created_at_utc"] = "different-operational-timestamp"
    assert ensure_immutable_manifest(path, compatible) == manifest
    assert path.read_bytes() == original

    stripped = {"compatibility_fingerprint": manifest["compatibility_fingerprint"]}
    path.write_text(json.dumps(stripped), encoding="utf-8")
    with pytest.raises(ValueError, match="compatibility must be a mapping"):
        ensure_immutable_manifest(path, manifest)
    path.write_bytes(original)

    tampered_compatibility = json.loads(json.dumps(manifest))
    tampered_compatibility["compatibility"]["mixed_precision"] = "fp16"
    with pytest.raises(ValueError, match="fingerprint does not match"):
        _validate_manifest_integrity(tampered_compatibility)

    tampered_selection = json.loads(json.dumps(manifest))
    tampered_selection["selection"]["ordered_selected_source_indices"][0] = 2
    with pytest.raises(ValueError, match="selection digest"):
        _validate_manifest_integrity(tampered_selection)
